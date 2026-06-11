"""Evaluate Try80 priors/model with RMSE, MAE and MapCorr.

This runner is intentionally narrower than the all quality metric script:
it does not compute SSIM or gradient correlation, because the paper and thesis
report RMSE, MAE and map correlation. It keeps the same city holdout protocol,
checkpoint, priors, masks and grouping metadata, and adds antenna height and
topology plus antenna height breakdowns.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TFGPRACTICE = ROOT.parent
DEFAULT_EXPERIMENT = TFGPRACTICE / "cluster_outputs" / "TFGEightiethTry80" / "try80_joint_huge_pathloss_finetune"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_try80 import Try80Cfg  # noqa: E402
from src.data_utils import (  # noqa: E402
    HeightEmbedding,
    Try80DataConfig,
    Try80JointDataset,
    list_hdf5_samples,
    split_city_holdout,
)
from src.metrics_try80 import TASKS, inverse_transform, transform_target  # noqa: E402
from src.model_try80 import Try80Model, Try80ModelConfig  # noqa: E402


SCOPES = ("overall", "los", "nlos")
PAIRS = {
    "prior": ("target", "prior"),
    "model": ("target", "model"),
    "prior_vs_model": ("prior", "model"),
}
PAIR_DISPLAY = {
    "prior": "GT-prior",
    "model": "GT-model",
    "prior_vs_model": "prior-model",
}
SampleRef = Tuple[str, str]


@dataclass
class MetricUpdate:
    sse: float = 0.0
    sae: float = 0.0
    n_error: int = 0
    map_corr_weighted_sum: float = 0.0
    n_corr: int = 0


@dataclass
class MetricStat(MetricUpdate):
    def add(self, update: MetricUpdate) -> None:
        self.sse += update.sse
        self.sae += update.sae
        self.n_error += update.n_error
        self.map_corr_weighted_sum += update.map_corr_weighted_sum
        self.n_corr += update.n_corr

    def to_json(self) -> Dict[str, float | int | None]:
        return {
            "rmse_pw": math.sqrt(self.sse / self.n_error) if self.n_error else None,
            "mae_pw": self.sae / self.n_error if self.n_error else None,
            "n_pixels": self.n_error,
            "map_corr": weighted_mean_or_none(self.map_corr_weighted_sum, self.n_corr),
            "map_corr_pixels": self.n_corr,
        }


@dataclass
class GroupStats:
    n_samples: int = 0
    city_counts: Dict[str, int] = field(default_factory=dict)
    topology_class_3_counts: Dict[str, int] = field(default_factory=dict)
    topology_class_6_counts: Dict[str, int] = field(default_factory=dict)
    antenna_bin_counts: Dict[str, int] = field(default_factory=dict)
    stats: Dict[str, Dict[str, Dict[str, MetricStat]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stats:
            self.stats = {
                pair: {task: {scope: MetricStat() for scope in SCOPES} for task in TASKS}
                for pair in PAIRS
            }

    def add_meta(self, city: str, top3: str, top6: str, antenna_bin: str) -> None:
        self.n_samples += 1
        inc(self.city_counts, city)
        inc(self.topology_class_3_counts, top3)
        inc(self.topology_class_6_counts, top6)
        inc(self.antenna_bin_counts, antenna_bin)

    def add_update(self, pair: str, task: str, scope: str, update: MetricUpdate) -> None:
        self.stats[pair][task][scope].add(update)

    def to_json(self) -> Dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "city_counts": dict(sorted(self.city_counts.items())),
            "topology_class_3_counts": dict(sorted(self.topology_class_3_counts.items())),
            "topology_class_6_counts": dict(sorted(self.topology_class_6_counts.items())),
            "antenna_bin_counts": dict(sorted(self.antenna_bin_counts.items())),
            "metrics": {
                pair: {
                    task: {scope: self.stats[pair][task][scope].to_json() for scope in SCOPES}
                    for task in TASKS
                }
                for pair in PAIRS
            },
            "delta_model_minus_prior": self.delta_model_minus_prior(),
        }

    def delta_model_minus_prior(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for task in TASKS:
            out[task] = {}
            for scope in SCOPES:
                prior = self.stats["prior"][task][scope].to_json()
                model = self.stats["model"][task][scope].to_json()
                out[task][scope] = {
                    "rmse_pw": sub_optional(model["rmse_pw"], prior["rmse_pw"]),
                    "mae_pw": sub_optional(model["mae_pw"], prior["mae_pw"]),
                    "map_corr": sub_optional(model["map_corr"], prior["map_corr"]),
                }
        return out


class Report:
    def __init__(self) -> None:
        self.global_stats = GroupStats()
        self.by_topology_class_6: Dict[str, GroupStats] = {}
        self.by_topology_class_3: Dict[str, GroupStats] = {}
        self.by_antenna_bin: Dict[str, GroupStats] = {}
        self.by_topology_class_6_antenna_bin: Dict[str, GroupStats] = {}
        self.by_topology_class_3_antenna_bin: Dict[str, GroupStats] = {}
        self.by_city: Dict[str, GroupStats] = {}

    def update_sample(
        self,
        *,
        city: str,
        top3: str,
        top6: str,
        antenna_bin: str,
        refs: Mapping[str, Mapping[str, np.ndarray]],
        masks: Mapping[str, Mapping[str, np.ndarray]],
    ) -> None:
        groups = [
            self.global_stats,
            group(self.by_topology_class_6, top6),
            group(self.by_topology_class_3, top3),
            group(self.by_antenna_bin, antenna_bin),
            group(self.by_topology_class_6_antenna_bin, f"{top6}|{antenna_bin}"),
            group(self.by_topology_class_3_antenna_bin, f"{top3}|{antenna_bin}"),
            group(self.by_city, city),
        ]
        updates: Dict[Tuple[str, str, str], MetricUpdate] = {}
        for task in TASKS:
            for scope in SCOPES:
                mask = masks[task][scope]
                for pair, (left_name, right_name) in PAIRS.items():
                    updates[(pair, task, scope)] = make_metric_update(
                        refs[left_name][task],
                        refs[right_name][task],
                        mask,
                    )
        for stats_group in groups:
            stats_group.add_meta(city, top3, top6, antenna_bin)
            for (pair, task, scope), update in updates.items():
                stats_group.add_update(pair, task, scope, update)

    def to_json(self) -> Dict[str, object]:
        return {
            "global": self.global_stats.to_json(),
            "by_topology_class_6": groups_to_json(self.by_topology_class_6),
            "by_topology_class_3": groups_to_json(self.by_topology_class_3),
            "by_antenna_bin": groups_to_json(self.by_antenna_bin),
            "by_topology_class_6_antenna_bin": groups_to_json(self.by_topology_class_6_antenna_bin),
            "by_topology_class_3_antenna_bin": groups_to_json(self.by_topology_class_3_antenna_bin),
            "by_city": groups_to_json(self.by_city),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test", "all"], default=["val", "test"])
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT / "resolved_config.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_EXPERIMENT / "best_model.pt")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "rmse_mae_mapcorr_val_test_dml_b1_after_range_update")
    parser.add_argument("--hdf5-path", type=Path, default=TFGPRACTICE / "Datasets" / "CKM_Dataset_270326.h5")
    parser.add_argument("--try78-los-calibration-json", type=Path, default=ROOT / "calibrations" / "try78_los_two_ray_calibration.json")
    parser.add_argument("--try78-nlos-calibration-json", type=Path, default=ROOT / "calibrations" / "try78_nlos_regime_calibration.json")
    parser.add_argument("--try79-calibration-json", type=Path, default=ROOT / "calibrations" / "try79_calibration.json")
    parser.add_argument("--device", default="directml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
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

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg_raw = state.get("model_cfg") if isinstance(state, dict) else None
    model_cfg = Try80ModelConfig(**(model_cfg_raw or cfg.model.__dict__))
    model = Try80Model(model_cfg)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()
    height_embed = HeightEmbedding()

    all_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits": args.splits,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "hdf5_path": str(cfg.data.hdf5_path.resolve()),
        "try78_los_calibration_json": str(cfg.prior.try78_los_calibration_json.resolve()),
        "try78_nlos_calibration_json": str(cfg.prior.try78_nlos_calibration_json.resolve()),
        "try79_calibration_json": str(cfg.prior.try79_calibration_json.resolve()),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "split_protocol": {
            "mode": "city_holdout",
            "split_seed": cfg.data.split_seed,
            "val_ratio": cfg.data.val_ratio,
            "test_ratio": cfg.data.test_ratio,
            "counts": {
                "train": len(train_refs),
                "val": len(val_refs),
                "test": len(test_refs),
                "all": len(refs),
            },
        },
        "metrics": {
            "rmse_pw": "Pixel weighted RMSE over finite valid pixels.",
            "mae_pw": "Pixel weighted MAE over finite valid pixels.",
            "map_corr": "Per sample Pearson correlation over finite valid pixels, aggregated by valid pixel count.",
        },
    }
    (args.out_root / "run_metadata.json").write_text(
        json.dumps(json_clean(all_metadata), indent=2),
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


def evaluate_split(
    *,
    cfg: Try80Cfg,
    model: Try80Model,
    height_embed: HeightEmbedding,
    device: torch.device,
    split_name: str,
    ordered_refs: Sequence[SampleRef],
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
                target, prior, model_pred, masks = extract_arrays(batch, priors_native, preds_native, bi)
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
        },
        "comparison": report.to_json(),
    }
    (out_dir / "rmse_mae_mapcorr_summary.json").write_text(
        json.dumps(json_clean(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "rmse_mae_mapcorr_summary.csv", report)
    write_model_prior_comparison_csv(out_dir / "rmse_mae_mapcorr_model_prior_comparison.csv", report)
    write_markdown(out_dir / "rmse_mae_mapcorr_summary.md", summary)


def build_data_cfg(cfg: Try80Cfg) -> Try80DataConfig:
    return Try80DataConfig(
        hdf5_path=cfg.data.hdf5_path,
        try78_los_calibration_json=cfg.prior.try78_los_calibration_json,
        try78_nlos_calibration_json=cfg.prior.try78_nlos_calibration_json,
        try79_calibration_json=cfg.prior.try79_calibration_json,
        image_size=cfg.data.image_size,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        split_seed=cfg.data.split_seed,
        topology_norm_m=cfg.data.topology_norm_m,
        path_loss_no_data_mask_column=cfg.data.path_loss_no_data_mask_column,
        derive_no_data_from_non_ground=cfg.data.derive_no_data_from_non_ground,
        augment_d4=False,
        precomputed_priors_hdf5_path=None,
    )


def extract_arrays(
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
        valid = tensor_np(batch[f"{task}_mask"][bi, 0]) > 0.5
        target[task] = tensor_np(batch[f"{task}_target"][bi, 0])
        prior[task] = tensor_np(priors[task][bi, 0])
        model_pred[task] = tensor_np(preds[task][bi, 0])
        masks[task] = {
            "overall": valid,
            "los": valid & los,
            "nlos": valid & nlos,
        }
    return target, prior, model_pred, masks


def make_metric_update(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> MetricUpdate:
    valid = mask.astype(bool, copy=False) & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return MetricUpdate()
    xv = x[valid].astype(np.float64, copy=False)
    yv = y[valid].astype(np.float64, copy=False)
    diff = xv - yv
    corr = corr_from_arrays(xv, yv)
    n_corr = int(xv.size) if corr is not None else 0
    return MetricUpdate(
        sse=float(np.sum(diff * diff, dtype=np.float64)),
        sae=float(np.sum(np.abs(diff), dtype=np.float64)),
        n_error=int(xv.size),
        map_corr_weighted_sum=float(corr * n_corr) if corr is not None else 0.0,
        n_corr=n_corr,
    )


def corr_from_arrays(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2:
        return None
    x_centered = x - float(np.mean(x, dtype=np.float64))
    y_centered = y - float(np.mean(y, dtype=np.float64))
    denom = math.sqrt(float(np.sum(x_centered * x_centered, dtype=np.float64))) * math.sqrt(
        float(np.sum(y_centered * y_centered, dtype=np.float64))
    )
    if denom <= 1.0e-12:
        return None
    return float(np.sum(x_centered * y_centered, dtype=np.float64) / denom)


def iter_rows(report: Report) -> Iterable[Dict[str, object]]:
    groups = grouped_report_items(report)
    for group_type, group_name, group_stats in groups:
        for pair in PAIRS:
            for task in TASKS:
                for scope in SCOPES:
                    yield {
                        "group_type": group_type,
                        "group": group_name,
                        "pair": pair,
                        "pair_label": PAIR_DISPLAY[pair],
                        "task": task,
                        "scope": scope,
                        "n_samples": group_stats.n_samples,
                        **group_stats.stats[pair][task][scope].to_json(),
                    }


def write_csv(path: Path, report: Report) -> None:
    fields = [
        "group_type",
        "group",
        "pair",
        "pair_label",
        "task",
        "scope",
        "n_samples",
        "rmse_pw",
        "mae_pw",
        "n_pixels",
        "map_corr",
        "map_corr_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(iter_rows(report))


def iter_model_prior_comparison_rows(report: Report) -> Iterable[Dict[str, object]]:
    metric_names = (
        ("rmse_pw", "rmse"),
        ("mae_pw", "mae"),
        ("map_corr", "map_corr"),
    )
    for group_type, group_name, group_stats in grouped_report_items(report):
        node = group_stats.to_json()
        for task in TASKS:
            for scope in SCOPES:
                row: Dict[str, object] = {
                    "group_type": group_type,
                    "group": group_name,
                    "task": task,
                    "scope": scope,
                    "n_samples": group_stats.n_samples,
                }
                for raw_metric, out_metric in metric_names:
                    prior = node["metrics"]["prior"][task][scope][raw_metric]
                    model = node["metrics"]["model"][task][scope][raw_metric]
                    prior_vs_model = node["metrics"]["prior_vs_model"][task][scope][raw_metric]
                    row[f"gt_prior_{out_metric}"] = prior
                    row[f"gt_model_{out_metric}"] = model
                    row[f"delta_model_minus_prior_{out_metric}"] = sub_optional(model, prior)
                    row[f"prior_model_{out_metric}"] = prior_vs_model
                yield row


def write_model_prior_comparison_csv(path: Path, report: Report) -> None:
    fields = [
        "group_type",
        "group",
        "task",
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(iter_model_prior_comparison_rows(report))


def write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    comp = summary["comparison"]
    tasks = (("path_loss", "PL"), ("delay_spread", "DS"), ("angular_spread", "AS"))
    lines = [
        "# RMSE, MAE And MapCorr Comparison",
        "",
        "Negative dRMSE/dMAE is better; positive dMapCorr is better.",
        "",
        "## Global Overall",
        "",
        "| Output | Prior RMSE | Model RMSE | dRMSE | Prior MAE | Model MAE | dMAE | Prior MapCorr | Model MapCorr | dMapCorr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    global_node = comp["global"]
    for task, label in tasks:
        prior = global_node["metrics"]["prior"][task]["overall"]
        model = global_node["metrics"]["model"][task]["overall"]
        dlt = global_node["delta_model_minus_prior"][task]["overall"]
        lines.append(
            f"| {label} | {fmt(prior['rmse_pw'])} | {fmt(model['rmse_pw'])} | {fmt(dlt['rmse_pw'])} | "
            f"{fmt(prior['mae_pw'])} | {fmt(model['mae_pw'])} | {fmt(dlt['mae_pw'])} | "
            f"{fmt(prior['map_corr'], 6)} | {fmt(model['map_corr'], 6)} | {fmt(dlt['map_corr'], 6)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grouped_report_items(report: Report) -> list[Tuple[str, str, GroupStats]]:
    groups: list[Tuple[str, str, GroupStats]] = [("global", "all", report.global_stats)]
    groups += [("topology_class_6", key, value) for key, value in sorted(report.by_topology_class_6.items())]
    groups += [("topology_class_3", key, value) for key, value in sorted(report.by_topology_class_3.items())]
    groups += [("antenna_bin", key, value) for key, value in sorted(report.by_antenna_bin.items())]
    groups += [
        ("topology_class_6_antenna_bin", key, value)
        for key, value in sorted(report.by_topology_class_6_antenna_bin.items())
    ]
    groups += [
        ("topology_class_3_antenna_bin", key, value)
        for key, value in sorted(report.by_topology_class_3_antenna_bin.items())
    ]
    groups += [("city", key, value) for key, value in sorted(report.by_city.items())]
    return groups


def resolve_device(name: str | None) -> torch.device:
    requested = (name or "auto").lower()
    if requested in {"directml", "dml"}:
        import torch_directml

        return torch_directml.device()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml

            return torch_directml.device()
        except Exception:
            return torch.device("cpu")
    return torch.device(name)


def to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def tensor_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def batch_string(value: object, idx: int) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[idx])
    return str(value)


def inc(store: MutableMapping[str, int], key: str) -> None:
    store[key] = store.get(key, 0) + 1


def group(store: MutableMapping[str, GroupStats], key: str) -> GroupStats:
    if key not in store:
        store[key] = GroupStats()
    return store[key]


def groups_to_json(store: Mapping[str, GroupStats]) -> Dict[str, object]:
    return {key: group_stats.to_json() for key, group_stats in sorted(store.items())}


def sub_optional(a: object, b: object) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def weighted_mean_or_none(total: float, n: int) -> Optional[float]:
    return total / n if n else None


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def json_clean(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    return value


if __name__ == "__main__":
    main()
