"""Compare Try80 priors/model with RMSE, SSIM, MapCorr, and GradCorr.

This is the combined quality-metric runner. It evaluates the same three pairs
in one pass:

- GT-prior: ground truth vs frozen calibrated prior
- GT-model: ground truth vs frozen calibrated prior + Try80 residual model
- prior-model: frozen calibrated prior vs prior + Try80 residual model

Metrics:

- RMSE over finite valid pixels
- task-valid masked SSIM, reported by scope
- per-map valid-pixel MapCorr
- task-valid gradient-magnitude GradCorr, reported by scope
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

from compare_prior_try80_ssim_rmse import (  # noqa: E402
    DATA_RANGES,
    autocast_context,
    batch_string,
    build_torch_ssim_maps,
    compute_task_data_ranges,
    json_clean,
    masked_ssim_map,
    resolve_device,
    split_cities,
    tensor_np,
    to_device,
)
from compare_prior_try80_structure_metrics import (  # noqa: E402
    make_pair_update as make_corr_update,
    weighted_mean_or_none,
)


SCOPES = ("overall", "los", "nlos")
PAIRS = {
    "prior": ("GT", "prior"),
    "model": ("GT", "prior+Try80"),
    "prior_vs_model": ("prior", "prior+Try80"),
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
    n_rmse: int = 0
    ssim_sum: float = 0.0
    ssim_pixels: int = 0
    ssim_sample_sum: float = 0.0
    ssim_sample_count: int = 0
    map_corr_weighted_sum: float = 0.0
    n_corr: int = 0
    grad_corr_weighted_sum: float = 0.0
    n_grad_corr: int = 0


@dataclass
class MetricStat(MetricUpdate):
    def add(self, update: MetricUpdate) -> None:
        self.sse += update.sse
        self.n_rmse += update.n_rmse
        self.ssim_sum += update.ssim_sum
        self.ssim_pixels += update.ssim_pixels
        self.ssim_sample_sum += update.ssim_sample_sum
        self.ssim_sample_count += update.ssim_sample_count
        self.map_corr_weighted_sum += update.map_corr_weighted_sum
        self.n_corr += update.n_corr
        self.grad_corr_weighted_sum += update.grad_corr_weighted_sum
        self.n_grad_corr += update.n_grad_corr

    def to_json(self) -> Dict[str, float | int | None]:
        return {
            "rmse_pw": math.sqrt(self.sse / self.n_rmse) if self.n_rmse else None,
            "n_pixels": self.n_rmse,
            "ssim_pw": self.ssim_sum / self.ssim_pixels if self.ssim_pixels else None,
            "ssim_pixels": self.ssim_pixels,
            "ssim_sample_mean": (
                self.ssim_sample_sum / self.ssim_sample_count
                if self.ssim_sample_count
                else None
            ),
            "ssim_sample_count": self.ssim_sample_count,
            "map_corr": weighted_mean_or_none(self.map_corr_weighted_sum, self.n_corr),
            "map_corr_pixels": self.n_corr,
            "grad_mag_corr": weighted_mean_or_none(
                self.grad_corr_weighted_sum,
                self.n_grad_corr,
            ),
            "grad_corr_pixels": self.n_grad_corr,
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
                    "ssim_pw": sub_optional(model["ssim_pw"], prior["ssim_pw"]),
                    "ssim_sample_mean": sub_optional(
                        model["ssim_sample_mean"],
                        prior["ssim_sample_mean"],
                    ),
                    "map_corr": sub_optional(model["map_corr"], prior["map_corr"]),
                    "grad_mag_corr": sub_optional(
                        model["grad_mag_corr"],
                        prior["grad_mag_corr"],
                    ),
                }
        return out


class Report:
    def __init__(self) -> None:
        self.global_stats = GroupStats()
        self.by_topology_class_6: Dict[str, GroupStats] = {}
        self.by_topology_class_3: Dict[str, GroupStats] = {}
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
        ssim_maps: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    ) -> None:
        groups = [
            self.global_stats,
            group(self.by_topology_class_6, top6),
            group(self.by_topology_class_3, top3),
            group(self.by_city, city),
        ]
        updates: Dict[Tuple[str, str, str], MetricUpdate] = {}
        for task in TASKS:
            local_mask = masks[task]["overall"]
            for scope in SCOPES:
                mask = masks[task][scope]
                updates[("prior", task, scope)] = make_metric_update(
                    refs["target"][task],
                    refs["prior"][task],
                    mask,
                    ssim_maps["prior"][task][scope],
                    local_mask=local_mask,
                )
                updates[("model", task, scope)] = make_metric_update(
                    refs["target"][task],
                    refs["model"][task],
                    mask,
                    ssim_maps["model"][task][scope],
                    local_mask=local_mask,
                )
                updates[("prior_vs_model", task, scope)] = make_metric_update(
                    refs["prior"][task],
                    refs["model"][task],
                    mask,
                    ssim_maps["prior_vs_model"][task][scope],
                    local_mask=local_mask,
                )
        for g in groups:
            g.add_meta(city, top3, top6, antenna_bin)
            for (pair, task, scope), update in updates.items():
                g.add_update(pair, task, scope, update)

    def to_json(self) -> Dict[str, object]:
        return {
            "global": self.global_stats.to_json(),
            "by_topology_class_6": groups_to_json(self.by_topology_class_6),
            "by_topology_class_3": groups_to_json(self.by_topology_class_3),
            "by_city": groups_to_json(self.by_city),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--hdf5-path", type=Path, default=None)
    parser.add_argument("--try78-los-calibration-json", type=Path, default=None)
    parser.add_argument("--try78-nlos-calibration-json", type=Path, default=None)
    parser.add_argument("--try79-calibration-json", type=Path, default=None)
    parser.add_argument("--precomputed-priors-hdf5-path", type=Path, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use: auto default, cpu, cuda, cuda:0, directml, or dml.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--ssim-win-size", type=int, default=11)
    parser.add_argument("--ssim-backend", choices=["torch", "skimage"], default="torch")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    started = time.time()
    cfg = Try80Cfg.load(args.config)
    if args.hdf5_path is not None:
        cfg.data.hdf5_path = args.hdf5_path.resolve()
    if args.try78_los_calibration_json is not None:
        cfg.prior.try78_los_calibration_json = args.try78_los_calibration_json.resolve()
    if args.try78_nlos_calibration_json is not None:
        cfg.prior.try78_nlos_calibration_json = args.try78_nlos_calibration_json.resolve()
    if args.try79_calibration_json is not None:
        cfg.prior.try79_calibration_json = args.try79_calibration_json.resolve()
    cfg.data.precomputed_priors_hdf5_path = (
        args.precomputed_priors_hdf5_path.resolve()
        if args.precomputed_priors_hdf5_path
        else None
    )

    device = resolve_device(args.device)
    refs = list_hdf5_samples(cfg.data.hdf5_path)
    train_refs, val_refs, test_refs = split_city_holdout(
        refs,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        split_seed=cfg.data.split_seed,
    )
    refs_by_split = {"train": train_refs, "val": val_refs, "test": test_refs}
    ordered_refs = (
        list(train_refs) + list(val_refs) + list(test_refs)
        if args.split == "all"
        else list(refs_by_split[args.split])
    )
    if args.limit > 0:
        ordered_refs = ordered_refs[: args.limit]

    ssim_data_ranges, ssim_data_range_stats = compute_task_data_ranges(
        cfg.data.hdf5_path,
        ordered_refs,
        path_loss_no_data_mask_column=cfg.data.path_loss_no_data_mask_column,
        derive_no_data_from_non_ground=cfg.data.derive_no_data_from_non_ground,
    )
    print(f"Resolved SSIM data ranges: {ssim_data_ranges}", flush=True)

    dataset = Try80JointDataset(build_data_cfg(cfg), ordered_refs, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg_raw = state.get("model_cfg") if isinstance(state, dict) else None
    model_cfg = Try80ModelConfig(**(model_cfg_raw or cfg.model.__dict__))
    model = Try80Model(model_cfg)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    height_embed = HeightEmbedding()
    report = Report()

    with torch.no_grad():
        for step, raw_batch in enumerate(loader, start=1):
            batch = to_device(raw_batch, device)
            priors_native = {task: batch[f"{task}_prior"] for task in TASKS}
            priors_trans = {task: transform_target(task, priors_native[task]) for task in TASKS}
            with autocast_context(device, args.mixed_precision):
                outputs = model(batch["inputs"], height_embed(batch["antenna_height_m"]), priors_trans)
            preds_native = {task: inverse_transform(task, outputs[task]["pred_trans"]) for task in TASKS}

            ssim_maps_batch = None
            if args.ssim_backend == "torch":
                ssim_maps_batch = build_torch_ssim_maps(
                    batch,
                    priors_native,
                    preds_native,
                    args.ssim_win_size,
                    ssim_data_ranges,
                )

            bsz = int(preds_native["path_loss"].shape[0])
            for bi in range(bsz):
                target, prior, model_pred, masks = extract_arrays(batch, priors_native, preds_native, bi)
                if ssim_maps_batch is not None:
                    ssim_maps = {
                        pair: {
                            task: {
                                scope: tensor_np(ssim_maps_batch[pair][task][scope][bi, 0])
                                for scope in SCOPES
                            }
                            for task in TASKS
                        }
                        for pair in PAIRS
                    }
                else:
                    ssim_maps = build_numpy_ssim_maps(
                        target,
                        prior,
                        model_pred,
                        masks,
                        args.ssim_win_size,
                        ssim_data_ranges,
                    )
                report.update_sample(
                    city=batch_string(batch["city"], bi),
                    top3=batch_string(batch["topology_class_3"], bi),
                    top6=batch_string(batch["topology_class_6"], bi),
                    antenna_bin=batch_string(batch["antenna_bin"], bi),
                    refs={"target": target, "prior": prior, "model": model_pred},
                    masks=masks,
                    ssim_maps=ssim_maps,
                )

            if args.progress_every > 0 and (step % args.progress_every == 0 or step == len(loader)):
                print(
                    f"[{step:05d}/{len(loader):05d}] "
                    f"samples={report.global_stats.n_samples} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "n_samples": len(ordered_refs),
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "hdf5_path": str(cfg.data.hdf5_path.resolve()),
            "try78_los_calibration_json": str(cfg.prior.try78_los_calibration_json.resolve()),
            "try78_nlos_calibration_json": str(cfg.prior.try78_nlos_calibration_json.resolve()),
            "try79_calibration_json": str(cfg.prior.try79_calibration_json.resolve()),
            "precomputed_priors_hdf5_path": (
                str(cfg.data.precomputed_priors_hdf5_path.resolve())
                if cfg.data.precomputed_priors_hdf5_path
                else None
            ),
            "device": str(device),
            "mixed_precision": bool(args.mixed_precision),
            "pairs": PAIRS,
            "pair_display": PAIR_DISPLAY,
            "metrics": {
                "rmse_pw": "Pixel-weighted RMSE over finite valid pair pixels.",
                "ssim_pw": (
                    "Pixel-weighted masked SSIM. Local SSIM windows use only finite "
                    "task-valid pixels; LoS/NLoS scopes select center pixels for aggregation."
                ),
                "map_corr": (
                    "Per-sample/per-scope Pearson correlation over finite valid pixels, "
                    "aggregated by valid-pixel count."
                ),
                "grad_mag_corr": (
                    "Per-sample/per-scope Pearson correlation between gradient "
                    "magnitudes. Gradients use finite task-valid neighborhoods; "
                    "LoS/NLoS scopes select center pixels for aggregation."
                ),
            },
            "ssim": {
                "win_size": args.ssim_win_size,
                "backend": args.ssim_backend,
                "data_range_mode": "evaluated_split_target_valid_max_minus_min",
                "data_ranges": ssim_data_ranges,
                "data_range_stats": ssim_data_range_stats,
                "fallback_data_ranges": DATA_RANGES,
                "directml_note": (
                    "When the model device is DirectML/non-CPU/non-CUDA, SSIM local-window "
                    "maps are computed on CPU for backend compatibility."
                ),
            },
            "split_protocol": {
                "mode": "city_holdout",
                "split_seed": cfg.data.split_seed,
                "val_ratio": cfg.data.val_ratio,
                "test_ratio": cfg.data.test_ratio,
                "raw_counts": {
                    "train": len(train_refs),
                    "val": len(val_refs),
                    "test": len(test_refs),
                    "all": len(refs),
                    "evaluated": len(ordered_refs),
                },
                "cities": {
                    "train": split_cities(train_refs),
                    "val": split_cities(val_refs),
                    "test": split_cities(test_refs),
                },
            },
            "elapsed_seconds": time.time() - started,
        },
        "comparison": report.to_json(),
    }
    json_path = args.out_dir / "all_quality_metrics_summary.json"
    json_path.write_text(json.dumps(json_clean(summary), indent=2, allow_nan=False), encoding="utf-8")
    write_csv(args.out_dir / "all_quality_metrics_summary.csv", report)
    write_model_prior_comparison_csv(
        args.out_dir / "all_quality_metrics_model_prior_comparison.csv",
        report,
    )
    write_markdown(args.out_dir / "all_quality_metrics_summary.md", summary)
    print(f"Wrote {json_path.resolve()}", flush=True)


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
        precomputed_priors_hdf5_path=cfg.data.precomputed_priors_hdf5_path,
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


def build_numpy_ssim_maps(
    target: Mapping[str, np.ndarray],
    prior: Mapping[str, np.ndarray],
    model_pred: Mapping[str, np.ndarray],
    masks: Mapping[str, Mapping[str, np.ndarray]],
    win_size: int,
    data_ranges: Mapping[str, float],
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    return {
        "prior": {
            task: {
                scope: masked_ssim_map(
                    target=target[task],
                    pred=prior[task],
                    valid_mask=masks[task]["overall"],
                    data_range=data_ranges[task],
                    win_size=win_size,
                )
                for scope in SCOPES
            }
            for task in TASKS
        },
        "model": {
            task: {
                scope: masked_ssim_map(
                    target=target[task],
                    pred=model_pred[task],
                    valid_mask=masks[task]["overall"],
                    data_range=data_ranges[task],
                    win_size=win_size,
                )
                for scope in SCOPES
            }
            for task in TASKS
        },
        "prior_vs_model": {
            task: {
                scope: masked_ssim_map(
                    target=prior[task],
                    pred=model_pred[task],
                    valid_mask=masks[task]["overall"],
                    data_range=data_ranges[task],
                    win_size=win_size,
                )
                for scope in SCOPES
            }
            for task in TASKS
        },
    }


def make_metric_update(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    ssim_map: np.ndarray,
    *,
    local_mask: np.ndarray | None = None,
) -> MetricUpdate:
    corr = make_corr_update(x, y, mask, gradient_mask=local_mask)
    valid = mask.astype(bool, copy=False) & np.isfinite(x) & np.isfinite(y)
    finite_ssim = valid & np.isfinite(ssim_map)
    sample_ssim: Optional[float] = None
    ssim_sum = 0.0
    ssim_pixels = 0
    if np.any(finite_ssim):
        values = ssim_map[finite_ssim].astype(np.float64, copy=False)
        ssim_sum = float(np.sum(values, dtype=np.float64))
        ssim_pixels = int(values.size)
        sample_ssim = float(np.mean(values, dtype=np.float64))
    return MetricUpdate(
        sse=corr.sse,
        n_rmse=corr.n_rmse,
        ssim_sum=ssim_sum,
        ssim_pixels=ssim_pixels,
        ssim_sample_sum=sample_ssim if sample_ssim is not None else 0.0,
        ssim_sample_count=1 if sample_ssim is not None else 0,
        map_corr_weighted_sum=corr.map_corr_weighted_sum,
        n_corr=corr.n_corr,
        grad_corr_weighted_sum=corr.grad_corr_weighted_sum,
        n_grad_corr=corr.n_grad_corr,
    )


def iter_rows(report: Report) -> Iterable[Dict[str, object]]:
    groups = [("global", "all", report.global_stats)]
    groups += [("topology_class_6", key, value) for key, value in sorted(report.by_topology_class_6.items())]
    groups += [("topology_class_3", key, value) for key, value in sorted(report.by_topology_class_3.items())]
    groups += [("city", key, value) for key, value in sorted(report.by_city.items())]
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
    rows = list(iter_rows(report))
    fields = [
        "group_type",
        "group",
        "pair",
        "pair_label",
        "task",
        "scope",
        "n_samples",
        "rmse_pw",
        "n_pixels",
        "ssim_pw",
        "ssim_pixels",
        "ssim_sample_mean",
        "ssim_sample_count",
        "map_corr",
        "map_corr_pixels",
        "grad_mag_corr",
        "grad_corr_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iter_model_prior_comparison_rows(report: Report) -> Iterable[Dict[str, object]]:
    groups = [("global", "all", report.global_stats)]
    groups += [("topology_class_6", key, value) for key, value in sorted(report.by_topology_class_6.items())]
    groups += [("topology_class_3", key, value) for key, value in sorted(report.by_topology_class_3.items())]
    groups += [("city", key, value) for key, value in sorted(report.by_city.items())]
    metric_names = (
        ("rmse_pw", "rmse"),
        ("ssim_pw", "ssim"),
        ("map_corr", "map_corr"),
        ("grad_mag_corr", "grad_corr"),
    )
    for group_type, group_name, group_stats in groups:
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
    rows = list(iter_model_prior_comparison_rows(report))
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
        "gt_prior_ssim",
        "gt_model_ssim",
        "delta_model_minus_prior_ssim",
        "prior_model_ssim",
        "gt_prior_map_corr",
        "gt_model_map_corr",
        "delta_model_minus_prior_map_corr",
        "prior_model_map_corr",
        "gt_prior_grad_corr",
        "gt_model_grad_corr",
        "delta_model_minus_prior_grad_corr",
        "prior_model_grad_corr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    comp = summary["comparison"]
    tasks = (("path_loss", "PL"), ("delay_spread", "DS"), ("angular_spread", "AS"))
    lines = [
        "# All Quality Metric Comparison",
        "",
        "Negative dRMSE is better; positive dSSIM/dMapCorr/dGradCorr are better.",
        "",
        "## Global Overall: Values And Deltas",
        "",
        "| Output | GT-prior RMSE | GT-model RMSE | dRMSE | GT-prior SSIM | GT-model SSIM | dSSIM | GT-prior MapCorr | GT-model MapCorr | dMapCorr | GT-prior GradCorr | GT-model GradCorr | dGradCorr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    global_node = comp["global"]
    for task, label in tasks:
        dlt = global_node["delta_model_minus_prior"][task]["overall"]
        prior = global_node["metrics"]["prior"][task]["overall"]
        model = global_node["metrics"]["model"][task]["overall"]
        lines.append(
            f"| {label} | {fmt(prior['rmse_pw'])} | {fmt(model['rmse_pw'])} | "
            f"{fmt(dlt['rmse_pw'])} | {fmt(prior['ssim_pw'], 6)} | "
            f"{fmt(model['ssim_pw'], 6)} | {fmt(dlt['ssim_pw'], 6)} | "
            f"{fmt(prior['map_corr'], 6)} | {fmt(model['map_corr'], 6)} | "
            f"{fmt(dlt['map_corr'], 6)} | {fmt(prior['grad_mag_corr'], 6)} | "
            f"{fmt(model['grad_mag_corr'], 6)} | {fmt(dlt['grad_mag_corr'], 6)} |"
        )
    lines.extend([
        "",
        "## Global Overall: Absolute Pair Metrics",
        "",
        "| Pair | Output | RMSE | SSIM | MapCorr | GradCorr |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for pair in PAIRS:
        for task, label in tasks:
            row = global_node["metrics"][pair][task]["overall"]
            lines.append(
                f"| {PAIR_DISPLAY[pair]} | {label} | "
                f"{fmt(row['rmse_pw'])} | {fmt(row['ssim_pw'], 6)} | "
                f"{fmt(row['map_corr'], 6)} | {fmt(row['grad_mag_corr'], 6)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


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


if __name__ == "__main__":
    main()
