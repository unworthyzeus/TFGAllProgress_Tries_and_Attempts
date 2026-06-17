"""Recalculate Try 80 per city metrics with explicit native clipping.

This script is intentionally scoped to thesis result files. It reuses the
active Try 80 dataset, split, prior and model code, then writes compact
city tables for path loss / calibrated attenuation, delay spread and
angular spread.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
TFGPRACTICE = ROOT.parent
SCRIPTS = ROOT / "scripts"
DEFAULT_EXPERIMENT = TFGPRACTICE / "cluster_outputs" / "TFGEightiethTry80" / "try80_joint_huge_pathloss_finetune"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "thesis_revisions_per_city_clipped_dml_20260615"

for path in (ROOT, SCRIPTS):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

from compare_prior_try80_rmse_mae_mapcorr import (  # noqa: E402
    Report,
    batch_string,
    build_data_cfg,
    json_clean,
    make_metric_update,
    resolve_device,
    tensor_np,
    to_device,
)
from src.config_try80 import Try80Cfg  # noqa: E402
from src.data_utils import (  # noqa: E402
    HeightEmbedding,
    Try80JointDataset,
    list_hdf5_samples,
    split_city_holdout,
)
from src.metrics_try80 import TASKS, inverse_transform, transform_target  # noqa: E402
from src.model_try80 import Try80Model, Try80ModelConfig  # noqa: E402


SCOPES = ("overall", "los", "nlos")
TASK_LABELS = {
    "path_loss": "PL_CA",
    "delay_spread": "DS",
    "angular_spread": "AS",
}
EVAL_CLIPS = {
    "path_loss": (20.0, 185.0),
    "delay_spread": (0.0, 910.0),
    "angular_spread": (0.0, 180.0),
}
PAIR_NAMES = ("prior", "model", "prior_vs_model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test", "all"], default=["val", "test"])
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT / "resolved_config.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_EXPERIMENT / "best_model.pt")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--hdf5-path", type=Path, default=TFGPRACTICE / "Datasets" / "CKM_Dataset_270326.h5")
    parser.add_argument("--try78-los-calibration-json", type=Path, default=ROOT / "calibrations" / "try78_los_two_ray_calibration.json")
    parser.add_argument("--try78-nlos-calibration-json", type=Path, default=ROOT / "calibrations" / "try78_nlos_regime_calibration.json")
    parser.add_argument("--try79-calibration-json", type=Path, default=ROOT / "calibrations" / "try79_calibration.json")
    parser.add_argument("--device", default="directml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    cfg = Try80Cfg.load(args.config)
    cfg.data.hdf5_path = args.hdf5_path.resolve()
    cfg.prior.try78_los_calibration_json = args.try78_los_calibration_json.resolve()
    cfg.prior.try78_nlos_calibration_json = args.try78_nlos_calibration_json.resolve()
    cfg.prior.try79_calibration_json = args.try79_calibration_json.resolve()
    cfg.data.precomputed_priors_hdf5_path = None

    device = resolve_device(args.device)
    refs = list_hdf5_samples(cfg.data.hdf5_path)
    train_refs, val_refs, test_refs = split_city_holdout(
        refs,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        split_seed=cfg.data.split_seed,
    )
    refs_by_split = {
        "train": train_refs,
        "val": val_refs,
        "test": test_refs,
        "all": list(train_refs) + list(val_refs) + list(test_refs),
    }

    model = load_model(args.checkpoint, cfg, device)
    height_embed = HeightEmbedding()
    split_counts = {name: len(split_refs) for name, split_refs in refs_by_split.items()}
    split_cities = {name: sorted({city for city, _sample in split_refs}) for name, split_refs in refs_by_split.items()}
    metadata = build_metadata(args, cfg, device, split_counts, split_cities)
    (args.out_root / "run_metadata.json").write_text(
        json.dumps(json_clean(metadata), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    for split_name in args.splits:
        ordered_refs = list(refs_by_split[split_name])
        if args.limit > 0:
            ordered_refs = ordered_refs[: args.limit]
        evaluate_split(
            cfg=cfg,
            model=model,
            height_embed=height_embed,
            device=device,
            split_name=split_name,
            ordered_refs=ordered_refs,
            out_dir=args.out_root / split_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            progress_every=args.progress_every,
        )


def load_model(checkpoint: Path, cfg: Try80Cfg, device: torch.device) -> Try80Model:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_cfg_raw = state.get("model_cfg") if isinstance(state, dict) else None
    model_cfg = Try80ModelConfig(**(model_cfg_raw or cfg.model.__dict__))
    model = Try80Model(model_cfg)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()
    return model


def build_metadata(
    args: argparse.Namespace,
    cfg: Try80Cfg,
    device: torch.device,
    split_counts: Mapping[str, int],
    split_cities: Mapping[str, Sequence[str]],
) -> Dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "splits": args.splits,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "hdf5_path": str(cfg.data.hdf5_path.resolve()),
        "try78_los_calibration_json": str(cfg.prior.try78_los_calibration_json.resolve()),
        "try78_nlos_calibration_json": str(cfg.prior.try78_nlos_calibration_json.resolve()),
        "try79_calibration_json": str(cfg.prior.try79_calibration_json.resolve()),
        "device_requested": args.device,
        "device_resolved": str(device),
        "directml_used": str(device).startswith("privateuseone"),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "limit": args.limit,
        "split_protocol": {
            "mode": "city_holdout",
            "split_seed": cfg.data.split_seed,
            "val_ratio": cfg.data.val_ratio,
            "test_ratio": cfg.data.test_ratio,
            "counts": dict(split_counts),
            "cities": {name: list(cities) for name, cities in split_cities.items()},
        },
        "mask_policy": {
            "source": str((ROOT / "src" / "data_utils.py").resolve()),
            "path_loss": "finite path_loss >= 20 dB, optional path_loss_no_data_mask excluded, ground pixels only when derive_no_data_from_non_ground is true",
            "delay_spread": "ground pixels with finite target >= 0",
            "angular_spread": "ground pixels with finite target >= 0",
            "config_path_loss_no_data_mask_column": cfg.data.path_loss_no_data_mask_column,
            "config_derive_no_data_from_non_ground": cfg.data.derive_no_data_from_non_ground,
        },
        "clip_policy": {
            "applied_to": "prior and model native predictions before metrics; targets use dataset values after the valid masks",
            "eval_prediction_clips": {
                task: {"min": lo, "max": hi}
                for task, (lo, hi) in EVAL_CLIPS.items()
            },
            "active_code_sources": {
                "path_loss": "src/priors_try80.py PATH_LOSS_MIN_DB=20 and PATH_LOSS_MAX_DB=185; src/data_utils.py normalizes path priors by 185",
                "delay_spread": "src/priors_try80.py METRIC_SPECS['delay_spread']['clip_hi']=910 and src/data_utils.py LOG1P_DELAY_NORM=log1p(910)",
                "angular_spread": "src/priors_try80.py caps calibrated Try79 angular prior to the dataset range 0..180 for both LoS and NLoS",
            },
        },
        "metrics": {
            "rmse_pw": "Pixel weighted RMSE over finite valid pixels.",
            "mae_pw": "Pixel weighted MAE over finite valid pixels.",
            "map_corr": "Per sample Pearson correlation over finite valid pixels, aggregated by valid pixel count.",
        },
    }


def evaluate_split(
    *,
    cfg: Try80Cfg,
    model: Try80Model,
    height_embed: HeightEmbedding,
    device: torch.device,
    split_name: str,
    ordered_refs: Sequence[Tuple[str, str]],
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    progress_every: int,
) -> None:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Try80JointDataset(build_data_cfg(cfg), ordered_refs, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
    )
    report = Report()
    with torch.no_grad():
        for step, raw_batch in enumerate(loader, start=1):
            batch = to_device(raw_batch, device)
            priors_native = {task: batch[f"{task}_prior"] for task in TASKS}
            priors_trans = {task: transform_target(task, priors_native[task]) for task in TASKS}
            outputs = model(batch["inputs"], height_embed(batch["antenna_height_m"]), priors_trans)
            preds_native = {task: inverse_transform(task, outputs[task]["pred_trans"]) for task in TASKS}
            bsz = int(preds_native["path_loss"].shape[0])
            for bi in range(bsz):
                target, prior, model_pred, masks = extract_arrays_clipped(batch, priors_native, preds_native, bi)
                report.update_sample(
                    city=batch_string(batch["city"], bi),
                    top3=batch_string(batch["topology_class_3"], bi),
                    top6=batch_string(batch["topology_class_6"], bi),
                    antenna_bin=batch_string(batch["antenna_bin"], bi),
                    refs={"target": target, "prior": prior, "model": model_pred},
                    masks=masks,
                )
            if progress_every > 0 and (step % progress_every == 0 or step == len(loader)):
                print(
                    f"[{split_name} {step:05d}/{len(loader):05d}] "
                    f"samples={report.global_stats.n_samples} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    summary = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": split_name,
            "n_samples": len(ordered_refs),
            "elapsed_seconds": time.time() - started,
            "eval_prediction_clips": {
                task: {"min": lo, "max": hi}
                for task, (lo, hi) in EVAL_CLIPS.items()
            },
        },
        "comparison": report.to_json(),
    }
    (out_dir / "per_city_metrics_summary.json").write_text(
        json.dumps(json_clean(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_global_csv(out_dir / "global_metrics.csv", report)
    write_city_long_csv(out_dir / "per_city_metrics_long.csv", report)
    write_city_compact_csv(out_dir / "per_city_metrics_compact.csv", report)
    write_markdown(out_dir / "per_city_metrics_summary.md", summary)


def extract_arrays_clipped(
    batch: Mapping[str, object],
    priors: Mapping[str, torch.Tensor],
    preds: Mapping[str, torch.Tensor],
    bi: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, np.ndarray]],
]:
    los = tensor_np(batch["los_mask"][bi, 0]) > 0.5
    nlos = tensor_np(batch["nlos_mask"][bi, 0]) > 0.5
    target: Dict[str, np.ndarray] = {}
    prior: Dict[str, np.ndarray] = {}
    model_pred: Dict[str, np.ndarray] = {}
    masks: Dict[str, Dict[str, np.ndarray]] = {}
    for task in TASKS:
        lo, hi = EVAL_CLIPS[task]
        valid = tensor_np(batch[f"{task}_mask"][bi, 0]) > 0.5
        target[task] = tensor_np(batch[f"{task}_target"][bi, 0])
        prior[task] = clip_native(tensor_np(priors[task][bi, 0]), lo, hi)
        model_pred[task] = clip_native(tensor_np(preds[task][bi, 0]), lo, hi)
        masks[task] = {
            "overall": valid,
            "los": valid & los,
            "nlos": valid & nlos,
        }
    return target, prior, model_pred, masks


def clip_native(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(values.astype(np.float32, copy=False), lo, hi)


def write_global_csv(path: Path, report: Report) -> None:
    rows = metric_rows("global", "all", report.global_stats)
    write_rows(path, rows)


def write_city_long_csv(path: Path, report: Report) -> None:
    rows = []
    for city, group_stats in sorted(report.by_city.items()):
        rows.extend(metric_rows("city", city, group_stats))
    write_rows(path, rows)


def metric_rows(group_type: str, group_name: str, group_stats) -> Iterable[Dict[str, object]]:
    for task in TASKS:
        for scope in SCOPES:
            prior = group_stats.stats["prior"][task][scope].to_json()
            model = group_stats.stats["model"][task][scope].to_json()
            prior_model = group_stats.stats["prior_vs_model"][task][scope].to_json()
            yield {
                "group_type": group_type,
                "group": group_name,
                "task": task,
                "task_label": TASK_LABELS[task],
                "scope": scope,
                "n_samples": group_stats.n_samples,
                "gt_prior_rmse": prior["rmse_pw"],
                "gt_model_rmse": model["rmse_pw"],
                "delta_model_minus_prior_rmse": sub_optional(model["rmse_pw"], prior["rmse_pw"]),
                "prior_model_rmse": prior_model["rmse_pw"],
                "gt_prior_mae": prior["mae_pw"],
                "gt_model_mae": model["mae_pw"],
                "delta_model_minus_prior_mae": sub_optional(model["mae_pw"], prior["mae_pw"]),
                "prior_model_mae": prior_model["mae_pw"],
                "gt_prior_map_corr": prior["map_corr"],
                "gt_model_map_corr": model["map_corr"],
                "delta_model_minus_prior_map_corr": sub_optional(model["map_corr"], prior["map_corr"]),
                "prior_model_map_corr": prior_model["map_corr"],
                "n_pixels": model["n_pixels"],
            }


def write_city_compact_csv(path: Path, report: Report) -> None:
    rows = []
    for city, group_stats in sorted(report.by_city.items()):
        row: Dict[str, object] = {
            "city": city,
            "n_samples": group_stats.n_samples,
        }
        for task in TASKS:
            prior = group_stats.stats["prior"][task]["overall"].to_json()
            model = group_stats.stats["model"][task]["overall"].to_json()
            label = TASK_LABELS[task].lower()
            row[f"{label}_prior_rmse"] = prior["rmse_pw"]
            row[f"{label}_model_rmse"] = model["rmse_pw"]
            row[f"{label}_delta_rmse"] = sub_optional(model["rmse_pw"], prior["rmse_pw"])
            row[f"{label}_prior_mae"] = prior["mae_pw"]
            row[f"{label}_model_mae"] = model["mae_pw"]
            row[f"{label}_delta_mae"] = sub_optional(model["mae_pw"], prior["mae_pw"])
        rows.append(row)
    fields = [
        "city",
        "n_samples",
        "pl_ca_prior_rmse",
        "pl_ca_model_rmse",
        "pl_ca_delta_rmse",
        "pl_ca_prior_mae",
        "pl_ca_model_mae",
        "pl_ca_delta_mae",
        "ds_prior_rmse",
        "ds_model_rmse",
        "ds_delta_rmse",
        "ds_prior_mae",
        "ds_model_mae",
        "ds_delta_mae",
        "as_prior_rmse",
        "as_model_rmse",
        "as_delta_rmse",
        "as_prior_mae",
        "as_model_mae",
        "as_delta_mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_rows(path: Path, rows_iter: Iterable[Dict[str, object]]) -> None:
    rows = list(rows_iter)
    fields = [
        "group_type",
        "group",
        "task",
        "task_label",
        "scope",
        "n_samples",
        "gt_prior_rmse",
        "gt_model_rmse",
        "delta_model_minus_prior_rmse",
        "prior_model_rmse",
        "gt_prior_mae",
        "gt_model_mae",
        "delta_model_minus_prior_mae",
        "prior_model_mae",
        "gt_prior_map_corr",
        "gt_model_map_corr",
        "delta_model_minus_prior_map_corr",
        "prior_model_map_corr",
        "n_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    split = summary["metadata"]["split"]
    comp = summary["comparison"]
    global_node = comp["global"]
    lines = [
        f"# Per City Metrics, {split}",
        "",
        "Negative dRMSE means the model improves over the frozen prior.",
        "",
        "## Global Overall",
        "",
        "| Output | Prior RMSE | Model RMSE | dRMSE | Prior MAE | Model MAE | dMAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        prior = global_node["metrics"]["prior"][task]["overall"]
        model = global_node["metrics"]["model"][task]["overall"]
        delta = global_node["delta_model_minus_prior"][task]["overall"]
        lines.append(
            f"| {TASK_LABELS[task]} | {fmt(prior['rmse_pw'])} | {fmt(model['rmse_pw'])} | "
            f"{fmt(delta['rmse_pw'])} | {fmt(prior['mae_pw'])} | {fmt(model['mae_pw'])} | "
            f"{fmt(delta['mae_pw'])} |"
        )
    lines.extend(
        [
            "",
            "## City Overall RMSE",
            "",
            "| City | n | PL/CA model | DS model | AS model |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for city, group_stats in sorted(comp["by_city"].items()):
        metrics = group_stats["metrics"]["model"]
        lines.append(
            f"| {city} | {group_stats['n_samples']} | "
            f"{fmt(metrics['path_loss']['overall']['rmse_pw'])} | "
            f"{fmt(metrics['delay_spread']['overall']['rmse_pw'])} | "
            f"{fmt(metrics['angular_spread']['overall']['rmse_pw'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sub_optional(a: object, b: object) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    main()
