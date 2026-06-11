"""Compare frozen priors vs Try80 residual model with SSIM and RMSE.

The script evaluates both predictors in the same forward pass over a split.
RMSE and SSIM are accumulated over the same finite valid center pixels for each
task/scope. SSIM local windows use finite task-valid pixels, so no-data pixels
do not enter local statistics. LoS/NLoS scopes select center pixels for
aggregation but do not act as SSIM-window boundaries.
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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_try80 import Try80Cfg  # noqa: E402
from src.data_utils import (  # noqa: E402
    HeightEmbedding,
    PATH_LOSS_MIN_DB,
    Try80DataConfig,
    Try80JointDataset,
    list_hdf5_samples,
    read_field,
    split_city_holdout,
)
from src.metrics_try80 import TASKS, inverse_transform, transform_target  # noqa: E402
from src.model_try80 import Try80Model, Try80ModelConfig  # noqa: E402


SCOPES = ("overall", "los", "nlos")
KINDS = ("prior", "model", "prior_vs_model")
DATA_RANGES = {
    "path_loss": 185.0,
    "delay_spread": 400.0,
    "angular_spread": 90.0,
}
MIN_SSIM_DATA_RANGE = 1.0
SampleRef = Tuple[str, str]


@dataclass
class MetricStat:
    sse: float = 0.0
    n_pixels: int = 0
    ssim_sum: float = 0.0
    ssim_pixels: int = 0
    ssim_sample_sum: float = 0.0
    ssim_sample_count: int = 0

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        ssim_map: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        valid = mask.astype(bool, copy=False) & np.isfinite(pred) & np.isfinite(target)
        if not np.any(valid):
            return None, None

        diff = pred[valid] - target[valid]
        self.sse += float(np.sum(diff * diff, dtype=np.float64))
        self.n_pixels += int(valid.sum())
        sample_rmse = float(np.sqrt(np.mean(diff * diff, dtype=np.float64)))

        finite_ssim = valid & np.isfinite(ssim_map)
        sample_ssim: Optional[float] = None
        if np.any(finite_ssim):
            values = ssim_map[finite_ssim]
            sample_ssim = float(np.mean(values, dtype=np.float64))
            self.ssim_sum += float(np.sum(values, dtype=np.float64))
            self.ssim_pixels += int(finite_ssim.sum())
            self.ssim_sample_sum += sample_ssim
            self.ssim_sample_count += 1

        return sample_rmse, sample_ssim

    def to_json(self) -> Dict[str, float | int | None]:
        return {
            "rmse_pw": math.sqrt(self.sse / self.n_pixels) if self.n_pixels else None,
            "n_pixels": self.n_pixels,
            "ssim_pw": self.ssim_sum / self.ssim_pixels if self.ssim_pixels else None,
            "ssim_pixels": self.ssim_pixels,
            "ssim_sample_mean": (
                self.ssim_sample_sum / self.ssim_sample_count if self.ssim_sample_count else None
            ),
            "ssim_sample_count": self.ssim_sample_count,
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
                kind: {task: {scope: MetricStat() for scope in SCOPES} for task in TASKS}
                for kind in KINDS
            }

    def add_meta(self, city: str, top3: str, top6: str, antenna_bin: str) -> None:
        self.n_samples += 1
        _inc(self.city_counts, city)
        _inc(self.topology_class_3_counts, top3)
        _inc(self.topology_class_6_counts, top6)
        _inc(self.antenna_bin_counts, antenna_bin)

    def update_metric(
        self,
        kind: str,
        task: str,
        scope: str,
        pred: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        ssim_map: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        return self.stats[kind][task][scope].update(pred, target, mask, ssim_map)

    def to_json(self) -> Dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "city_counts": dict(sorted(self.city_counts.items())),
            "topology_class_3_counts": dict(sorted(self.topology_class_3_counts.items())),
            "topology_class_6_counts": dict(sorted(self.topology_class_6_counts.items())),
            "antenna_bin_counts": dict(sorted(self.antenna_bin_counts.items())),
            "metrics": {
                kind: {
                    task: {scope: self.stats[kind][task][scope].to_json() for scope in SCOPES}
                    for task in TASKS
                }
                for kind in KINDS
            },
            "delta_model_minus_prior": self._delta_json(),
        }

    def _delta_json(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for task in TASKS:
            out[task] = {}
            for scope in SCOPES:
                model = self.stats["model"][task][scope].to_json()
                prior = self.stats["prior"][task][scope].to_json()
                out[task][scope] = {
                    "rmse_pw": _sub_optional(model["rmse_pw"], prior["rmse_pw"]),
                    "ssim_pw": _sub_optional(model["ssim_pw"], prior["ssim_pw"]),
                    "ssim_sample_mean": _sub_optional(
                        model["ssim_sample_mean"], prior["ssim_sample_mean"]
                    ),
                    "n_pixels": model["n_pixels"],
                }
        return out


class ComparisonReport:
    def __init__(self) -> None:
        self.global_stats = GroupStats()
        self.by_city: Dict[str, GroupStats] = {}
        self.by_topology_class_6: Dict[str, GroupStats] = {}
        self.by_topology_class_3: Dict[str, GroupStats] = {}
        self.by_antenna_bin: Dict[str, GroupStats] = {}
        self.by_prior_expert: Dict[str, GroupStats] = {}
        self.per_sample: List[Dict[str, object]] = []

    def update_sample(
        self,
        *,
        split: str,
        city: str,
        sample: str,
        top3: str,
        top6: str,
        antenna_bin: str,
        targets: Mapping[str, np.ndarray],
        masks: Mapping[str, Mapping[str, np.ndarray]],
        priors: Mapping[str, np.ndarray],
        preds: Mapping[str, np.ndarray],
        ssim_maps: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
        store_per_sample: bool,
    ) -> None:
        prior_expert = f"{top3}|{antenna_bin}"
        groups = [
            self.global_stats,
            _group(self.by_city, city),
            _group(self.by_topology_class_6, top6),
            _group(self.by_topology_class_3, top3),
            _group(self.by_antenna_bin, antenna_bin),
            _group(self.by_prior_expert, prior_expert),
        ]
        for group in groups:
            group.add_meta(city, top3, top6, antenna_bin)

        row: Dict[str, object] = {
            "split": split,
            "city": city,
            "sample": sample,
            "topology_class_3": top3,
            "topology_class_6": top6,
            "antenna_bin": antenna_bin,
            "prior_expert": prior_expert,
        }

        for task in TASKS:
            for scope in SCOPES:
                mask = masks[task][scope]
                row[f"{task}_{scope}_n_pixels"] = int(mask.sum())
                pair_specs = (
                    ("prior", targets, priors),
                    ("model", targets, preds),
                    ("prior_vs_model", priors, preds),
                )
                for kind, target_by_task, pred_by_task in pair_specs:
                    sample_rmse = None
                    sample_ssim = None
                    for group in groups:
                        sample_rmse, sample_ssim = group.update_metric(
                            kind,
                            task,
                            scope,
                            pred_by_task[task],
                            target_by_task[task],
                            mask,
                            ssim_maps[kind][task][scope],
                        )
                    if store_per_sample:
                        row[f"{task}_{scope}_{kind}_rmse"] = sample_rmse
                        row[f"{task}_{scope}_{kind}_ssim"] = sample_ssim
                if store_per_sample:
                    row[f"{task}_{scope}_rmse_delta_model_minus_prior"] = _sub_optional(
                        row.get(f"{task}_{scope}_model_rmse"),
                        row.get(f"{task}_{scope}_prior_rmse"),
                    )
                    row[f"{task}_{scope}_ssim_delta_model_minus_prior"] = _sub_optional(
                        row.get(f"{task}_{scope}_model_ssim"),
                        row.get(f"{task}_{scope}_prior_ssim"),
                    )

        if store_per_sample:
            self.per_sample.append(row)

    def to_json(self, include_per_sample: bool) -> Dict[str, object]:
        return {
            "global": self.global_stats.to_json(),
            "by_city": _groups_to_json(self.by_city),
            "by_topology_class_6": _groups_to_json(self.by_topology_class_6),
            "by_topology_class_3": _groups_to_json(self.by_topology_class_3),
            "by_antenna_bin": _groups_to_json(self.by_antenna_bin),
            "by_prior_expert": _groups_to_json(self.by_prior_expert),
            "per_sample": self.per_sample if include_per_sample else [],
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
    parser.add_argument("--no-per-sample", action="store_true")
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
        args.precomputed_priors_hdf5_path.resolve() if args.precomputed_priors_hdf5_path else None
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
    split_by_ref: Dict[SampleRef, str] = {}
    for split_name, split_refs in refs_by_split.items():
        for ref in split_refs:
            split_by_ref[ref] = split_name
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
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    height_embed = HeightEmbedding()
    report = ComparisonReport()

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
                city = batch_string(batch["city"], bi)
                sample = batch_string(batch["sample"], bi)
                top3 = batch_string(batch["topology_class_3"], bi)
                top6 = batch_string(batch["topology_class_6"], bi)
                antenna_bin = batch_string(batch["antenna_bin"], bi)
                split = split_by_ref[(city, sample)]

                targets, masks, priors, preds = extract_sample_arrays(batch, priors_native, preds_native, bi)
                if ssim_maps_batch is not None:
                    ssim_maps = {
                        kind: {
                            task: {
                                scope: tensor_np(ssim_maps_batch[kind][task][scope][bi, 0])
                                for scope in SCOPES
                            }
                            for task in TASKS
                        }
                        for kind in KINDS
                    }
                else:
                    ssim_maps = {
                        "prior": {
                            task: {
                                scope: masked_ssim_map(
                                    target=targets[task],
                                    pred=priors[task],
                                    valid_mask=masks[task]["overall"],
                                    data_range=ssim_data_ranges[task],
                                    win_size=args.ssim_win_size,
                                )
                                for scope in SCOPES
                            }
                            for task in TASKS
                        },
                        "model": {
                            task: {
                                scope: masked_ssim_map(
                                    target=targets[task],
                                    pred=preds[task],
                                    valid_mask=masks[task]["overall"],
                                    data_range=ssim_data_ranges[task],
                                    win_size=args.ssim_win_size,
                                )
                                for scope in SCOPES
                            }
                            for task in TASKS
                        },
                        "prior_vs_model": {
                            task: {
                                scope: masked_ssim_map(
                                    target=priors[task],
                                    pred=preds[task],
                                    valid_mask=masks[task]["overall"],
                                    data_range=ssim_data_ranges[task],
                                    win_size=args.ssim_win_size,
                                )
                                for scope in SCOPES
                            }
                            for task in TASKS
                        },
                    }

                report.update_sample(
                    split=split,
                    city=city,
                    sample=sample,
                    top3=top3,
                    top6=top6,
                    antenna_bin=antenna_bin,
                    targets=targets,
                    masks=masks,
                    priors=priors,
                    preds=preds,
                    ssim_maps=ssim_maps,
                    store_per_sample=not args.no_per_sample,
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
            "precomputed_priors_used": bool(
                cfg.data.precomputed_priors_hdf5_path
                and cfg.data.precomputed_priors_hdf5_path.exists()
            ),
            "device": str(device),
            "mixed_precision": bool(args.mixed_precision),
            "torch_version": torch.__version__,
            "ssim": {
                "win_size": args.ssim_win_size,
                "backend": args.ssim_backend,
                "data_range_mode": "evaluated_split_target_valid_max_minus_min",
                "data_ranges": ssim_data_ranges,
                "data_range_stats": ssim_data_range_stats,
                "fallback_data_ranges": DATA_RANGES,
                "pairs": {
                    "prior": "GT vs frozen calibrated prior",
                    "model": "GT vs frozen calibrated prior + Try80 residual model",
                    "prior_vs_model": "frozen calibrated prior vs prior + Try80 residual model",
                },
                "method": (
                    "SSIM is computed with mask-aware local windows for each task. "
                    "Only finite task-valid pixels contribute to each local mean, variance, "
                    "and covariance. LoS/NLoS scopes select center pixels for aggregation "
                    "but do not act as SSIM-window boundaries. Finite center pixels with "
                    "fewer than two finite task-valid window pixels are excluded from SSIM aggregation."
                ),
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
        "comparison": report.to_json(include_per_sample=not args.no_per_sample),
    }
    summary_path = args.out_dir / "prior_vs_try80_ssim_rmse_summary.json"
    summary_path.write_text(json.dumps(json_clean(summary), indent=2, allow_nan=False), encoding="utf-8")
    write_summary_csv(args.out_dir / "prior_vs_try80_ssim_rmse_summary.csv", report)
    if not args.no_per_sample:
        write_per_sample_csv(args.out_dir / "prior_vs_try80_ssim_rmse_per_sample.csv", report.per_sample)
    print(f"Wrote {summary_path.resolve()}", flush=True)


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


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested is None or requested.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = requested.strip().lower()
    if name in {"directml", "dml"}:
        try:
            import torch_directml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "DirectML requested, but torch-directml is not installed. "
                "Install torch-directml or use --device cpu/cuda."
            ) from exc
        return torch_directml.device()
    return torch.device(requested)


def to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return torch.amp.autocast(device_type="cpu", enabled=False)


def extract_sample_arrays(
    batch: Mapping[str, object],
    priors_native: Mapping[str, torch.Tensor],
    preds_native: Mapping[str, torch.Tensor],
    bi: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    los_mask = tensor_np(batch["los_mask"][bi, 0]) > 0.5
    nlos_mask = tensor_np(batch["nlos_mask"][bi, 0]) > 0.5
    targets: Dict[str, np.ndarray] = {}
    masks: Dict[str, Dict[str, np.ndarray]] = {}
    priors: Dict[str, np.ndarray] = {}
    preds: Dict[str, np.ndarray] = {}
    for task in TASKS:
        target = tensor_np(batch[f"{task}_target"][bi, 0])
        valid = tensor_np(batch[f"{task}_mask"][bi, 0]) > 0.5
        targets[task] = target
        masks[task] = {
            "overall": valid,
            "los": valid & los_mask,
            "nlos": valid & nlos_mask,
        }
        priors[task] = tensor_np(priors_native[task][bi, 0])
        preds[task] = tensor_np(preds_native[task][bi, 0])
    return targets, masks, priors, preds


def compute_task_data_ranges(
    hdf5_path: Path,
    refs: Sequence[SampleRef],
    *,
    path_loss_no_data_mask_column: Optional[str] = None,
    derive_no_data_from_non_ground: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float | int | str | None]]]:
    """Compute SSIM dynamic ranges from valid target pixels in the evaluated split."""
    mins = {task: math.inf for task in TASKS}
    maxs = {task: -math.inf for task in TASKS}
    counts = {task: 0 for task in TASKS}
    min_refs = {task: None for task in TASKS}
    max_refs = {task: None for task in TASKS}

    with h5py.File(str(hdf5_path), "r") as handle:
        for city, sample in refs:
            grp = handle[city][sample]
            topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
            ground = topology == 0.0
            for task in TASKS:
                target = read_field(grp, task)
                if task == "path_loss":
                    valid = np.isfinite(target) & (target >= PATH_LOSS_MIN_DB)
                    if path_loss_no_data_mask_column:
                        key = str(path_loss_no_data_mask_column).strip()
                        if key and key in grp:
                            valid &= ~(np.asarray(grp[key][...], dtype=np.float32) > 0.5)
                    if derive_no_data_from_non_ground:
                        valid &= ground
                else:
                    valid = ground & np.isfinite(target) & (target >= 0.0)
                if not np.any(valid):
                    continue
                values = target[valid]
                local_min = float(np.min(values))
                local_max = float(np.max(values))
                counts[task] += int(values.size)
                if local_min < mins[task]:
                    mins[task] = local_min
                    min_refs[task] = f"{city}/{sample}"
                if local_max > maxs[task]:
                    maxs[task] = local_max
                    max_refs[task] = f"{city}/{sample}"

    data_ranges: Dict[str, float] = {}
    stats: Dict[str, Dict[str, float | int | str | None]] = {}
    for task in TASKS:
        fallback = float(DATA_RANGES[task])
        if counts[task] > 0 and math.isfinite(mins[task]) and math.isfinite(maxs[task]):
            resolved = max(float(maxs[task] - mins[task]), MIN_SSIM_DATA_RANGE)
            source = "evaluated_split_target_valid_max_minus_min"
            task_min: Optional[float] = float(mins[task])
            task_max: Optional[float] = float(maxs[task])
        else:
            resolved = fallback
            source = "fallback_no_valid_target_pixels"
            task_min = None
            task_max = None
        data_ranges[task] = float(resolved)
        stats[task] = {
            "source": source,
            "min": task_min,
            "max": task_max,
            "data_range": float(resolved),
            "fallback_data_range": fallback,
            "n_valid_pixels": int(counts[task]),
            "min_sample": min_refs[task],
            "max_sample": max_refs[task],
        }
    return data_ranges, stats


def build_torch_ssim_maps(
    batch: Mapping[str, object],
    priors_native: Mapping[str, torch.Tensor],
    preds_native: Mapping[str, torch.Tensor],
    win_size: int,
    data_ranges: Mapping[str, float] | None = None,
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    def ssim_device_tensor(tensor: torch.Tensor) -> torch.Tensor:
        # DirectML is useful for the model forward pass, but CPU is safer for
        # reflect padding and local-window SSIM reductions.
        return tensor.detach().cpu() if tensor.device.type not in {"cpu", "cuda"} else tensor

    resolved_ranges = DATA_RANGES if data_ranges is None else data_ranges
    maps: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {
        "prior": {},
        "model": {},
        "prior_vs_model": {},
    }
    for task in TASKS:
        target = ssim_device_tensor(batch[f"{task}_target"]).float()
        prior = ssim_device_tensor(priors_native[task]).float()
        pred = ssim_device_tensor(preds_native[task]).float()
        valid = ssim_device_tensor(batch[f"{task}_mask"]).float() > 0.5
        prior_map = torch_ssim_map(
            target,
            prior,
            valid,
            data_range=resolved_ranges[task],
            win_size=win_size,
        )
        model_map = torch_ssim_map(
            target,
            pred,
            valid,
            data_range=resolved_ranges[task],
            win_size=win_size,
        )
        prior_vs_model_map = torch_ssim_map(
            prior,
            pred,
            valid,
            data_range=resolved_ranges[task],
            win_size=win_size,
        )
        maps["prior"][task] = {
            scope: prior_map
            for scope in SCOPES
        }
        maps["model"][task] = {
            scope: model_map
            for scope in SCOPES
        }
        maps["prior_vs_model"][task] = {
            scope: prior_vs_model_map
            for scope in SCOPES
        }
    return maps


def torch_ssim_map(
    target: torch.Tensor,
    pred: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    data_range: float,
    win_size: int,
) -> torch.Tensor:
    finite = valid_mask & torch.isfinite(target) & torch.isfinite(pred)
    target_eval = torch.where(finite, target, torch.zeros_like(target)).float()
    pred_eval = torch.where(finite, pred, torch.zeros_like(pred)).float()
    win = odd_win_size(win_size, target.shape[-2:])
    pad = win // 2
    kernel = torch.full(
        (1, 1, win, win),
        fill_value=1.0,
        dtype=torch.float32,
        device=target.device,
    )

    def filt(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x.float(), (pad, pad, pad, pad), mode="reflect"), kernel)

    count = filt(finite.float())
    safe_count = torch.clamp(count, min=1.0)
    ux = filt(target_eval) / safe_count
    uy = filt(pred_eval) / safe_count
    uxx = filt(target_eval * target_eval) / safe_count
    uyy = filt(pred_eval * pred_eval) / safe_count
    uxy = filt(target_eval * pred_eval) / safe_count
    cov_norm = safe_count / torch.clamp(safe_count - 1.0, min=1.0)
    vx = torch.clamp(cov_norm * (uxx - ux * ux), min=0.0)
    vy = torch.clamp(cov_norm * (uyy - uy * uy), min=0.0)
    vxy = cov_norm * (uxy - ux * uy)
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    numerator = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
    denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    ssim = numerator / torch.clamp(denominator, min=1.0e-12)
    return torch.where(finite & (count >= 2.0), ssim, torch.full_like(ssim, float("nan")))


def masked_ssim_map(
    *,
    target: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    data_range: float,
    win_size: int,
) -> np.ndarray:
    target_t = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))[None, None]
    pred_t = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))[None, None]
    valid_t = torch.from_numpy(np.ascontiguousarray(valid_mask, dtype=bool))[None, None]
    ssim_map = torch_ssim_map(
        target_t,
        pred_t,
        valid_t,
        data_range=float(data_range),
        win_size=win_size,
    )
    return tensor_np(ssim_map[0, 0]).astype(np.float32, copy=False)


def odd_win_size(requested: int, shape: Sequence[int]) -> int:
    smallest = int(min(shape))
    win = max(3, min(int(requested), smallest if smallest % 2 else smallest - 1))
    return win if win % 2 else win - 1


def tensor_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def batch_string(value: object, idx: int) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[idx])
    return str(value)


def split_cities(refs: Iterable[SampleRef]) -> List[str]:
    return sorted({city for city, _ in refs})


def _inc(store: MutableMapping[str, int], key: str) -> None:
    store[key] = store.get(key, 0) + 1


def _group(store: MutableMapping[str, GroupStats], key: str) -> GroupStats:
    if key not in store:
        store[key] = GroupStats()
    return store[key]


def _groups_to_json(store: Mapping[str, GroupStats]) -> Dict[str, object]:
    return {key: group.to_json() for key, group in sorted(store.items())}


def _sub_optional(a: object, b: object) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def iter_summary_rows(report: ComparisonReport) -> Iterable[Dict[str, object]]:
    groups: List[Tuple[str, str, GroupStats]] = [("global", "all", report.global_stats)]
    groups += [("city", key, value) for key, value in sorted(report.by_city.items())]
    groups += [
        ("topology_class_6", key, value)
        for key, value in sorted(report.by_topology_class_6.items())
    ]
    groups += [
        ("topology_class_3", key, value)
        for key, value in sorted(report.by_topology_class_3.items())
    ]
    groups += [("antenna_bin", key, value) for key, value in sorted(report.by_antenna_bin.items())]
    groups += [("prior_expert", key, value) for key, value in sorted(report.by_prior_expert.items())]
    for group_type, group_name, group in groups:
        for kind in KINDS:
            for task in TASKS:
                for scope in SCOPES:
                    values = group.stats[kind][task][scope].to_json()
                    yield {
                        "group_type": group_type,
                        "group": group_name,
                        "kind": kind,
                        "task": task,
                        "scope": scope,
                        "n_samples": group.n_samples,
                        **values,
                    }


def write_summary_csv(path: Path, report: ComparisonReport) -> None:
    rows = list(iter_summary_rows(report))
    fieldnames = [
        "group_type",
        "group",
        "kind",
        "task",
        "scope",
        "n_samples",
        "rmse_pw",
        "n_pixels",
        "ssim_pw",
        "ssim_pixels",
        "ssim_sample_mean",
        "ssim_sample_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_per_sample_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "split",
        "city",
        "sample",
        "topology_class_3",
        "topology_class_6",
        "antenna_bin",
        "prior_expert",
    ]
    ordered = preferred + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def json_clean(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): json_clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    return value


if __name__ == "__main__":
    main()
