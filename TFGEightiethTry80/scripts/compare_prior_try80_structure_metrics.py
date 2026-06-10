"""Compare prior vs Try80 with RMSE and structure-focused correlations.

This is a companion to the SSIM/RMSE evaluator. It keeps the same split,
checkpoint, priors, masks, and output grouping, but measures structure with:

- per-map valid-pixel Pearson map correlation, aggregated by valid-pixel count
- per-map valid-pixel gradient-magnitude correlation, aggregated by gradient-pixel count

Those metrics separate spatial pattern alignment from SSIM's luminance/contrast
terms, which can be misleading for heavy-tailed delay/angular spread maps.
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
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

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


SCOPES = ("overall", "los", "nlos")
PAIRS = {
    "prior": ("GT", "prior"),
    "model": ("GT", "prior+Try80"),
    "prior_vs_model": ("prior", "prior+Try80"),
}
SampleRef = Tuple[str, str]


@dataclass
class PairUpdate:
    sse: float = 0.0
    n_rmse: int = 0
    map_corr_weighted_sum: float = 0.0
    n_corr: int = 0
    grad_corr_weighted_sum: float = 0.0
    n_grad_corr: int = 0


@dataclass
class PairStat(PairUpdate):
    def update(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        self.add(make_pair_update(x, y, mask))

    def add(self, update: PairUpdate) -> None:
        self.sse += update.sse
        self.n_rmse += update.n_rmse
        self.map_corr_weighted_sum += update.map_corr_weighted_sum
        self.n_corr += update.n_corr
        self.grad_corr_weighted_sum += update.grad_corr_weighted_sum
        self.n_grad_corr += update.n_grad_corr

    def to_json(self) -> Dict[str, float | int | None]:
        return {
            "rmse_pw": math.sqrt(self.sse / self.n_rmse) if self.n_rmse else None,
            "n_pixels": self.n_rmse,
            "map_corr": weighted_mean_or_none(self.map_corr_weighted_sum, self.n_corr),
            "grad_mag_corr": weighted_mean_or_none(
                self.grad_corr_weighted_sum,
                self.n_grad_corr,
            ),
        }


@dataclass
class GroupStats:
    n_samples: int = 0
    city_counts: Dict[str, int] = field(default_factory=dict)
    topology_class_3_counts: Dict[str, int] = field(default_factory=dict)
    topology_class_6_counts: Dict[str, int] = field(default_factory=dict)
    antenna_bin_counts: Dict[str, int] = field(default_factory=dict)
    stats: Dict[str, Dict[str, Dict[str, PairStat]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stats:
            self.stats = {
                pair: {task: {scope: PairStat() for scope in SCOPES} for task in TASKS}
                for pair in PAIRS
            }

    def add_meta(self, city: str, top3: str, top6: str, antenna_bin: str) -> None:
        self.n_samples += 1
        inc(self.city_counts, city)
        inc(self.topology_class_3_counts, top3)
        inc(self.topology_class_6_counts, top6)
        inc(self.antenna_bin_counts, antenna_bin)

    def update(
        self,
        pair: str,
        task: str,
        scope: str,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        self.stats[pair][task][scope].update(x, y, mask)

    def add_update(self, pair: str, task: str, scope: str, update: PairUpdate) -> None:
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
                    "map_corr": sub_optional(model["map_corr"], prior["map_corr"]),
                    "grad_mag_corr": sub_optional(model["grad_mag_corr"], prior["grad_mag_corr"]),
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
    ) -> None:
        groups = [
            self.global_stats,
            group(self.by_topology_class_6, top6),
            group(self.by_topology_class_3, top3),
            group(self.by_city, city),
        ]
        updates: Dict[Tuple[str, str, str], PairUpdate] = {}
        for task in TASKS:
            for scope in SCOPES:
                mask = masks[task][scope]
                updates[("prior", task, scope)] = make_pair_update(
                    refs["target"][task], refs["prior"][task], mask
                )
                updates[("model", task, scope)] = make_pair_update(
                    refs["target"][task], refs["model"][task], mask
                )
                updates[("prior_vs_model", task, scope)] = make_pair_update(
                    refs["prior"][task], refs["model"][task], mask
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
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--hdf5-path", type=Path, default=None)
    parser.add_argument("--try78-los-calibration-json", type=Path, default=None)
    parser.add_argument("--try78-nlos-calibration-json", type=Path, default=None)
    parser.add_argument("--try79-calibration-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    started = time.time()
    cfg = Try80Cfg.load(args.config)
    if args.hdf5_path:
        cfg.data.hdf5_path = args.hdf5_path.resolve()
    if args.try78_los_calibration_json:
        cfg.prior.try78_los_calibration_json = args.try78_los_calibration_json.resolve()
    if args.try78_nlos_calibration_json:
        cfg.prior.try78_nlos_calibration_json = args.try78_nlos_calibration_json.resolve()
    if args.try79_calibration_json:
        cfg.prior.try79_calibration_json = args.try79_calibration_json.resolve()
    cfg.data.precomputed_priors_hdf5_path = None

    all_refs = list_hdf5_samples(cfg.data.hdf5_path)
    train_refs, val_refs, test_refs = split_city_holdout(
        all_refs,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        split_seed=cfg.data.split_seed,
    )
    split_refs = {"train": train_refs, "val": val_refs, "test": test_refs}[args.split]
    if args.limit:
        split_refs = split_refs[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Try80JointDataset(build_data_cfg(cfg), split_refs, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Try80Model(Try80ModelConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"], strict=False)
    model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    height_embed = HeightEmbedding()
    report = Report()

    with torch.no_grad():
        for step, raw_batch in enumerate(loader, start=1):
            batch = to_device(raw_batch, device)
            priors = {task: batch[f"{task}_prior"] for task in TASKS}
            with autocast_context(device, args.mixed_precision):
                outputs = model(
                    batch["inputs"],
                    height_embed(batch["antenna_height_m"]),
                    {task: transform_target(task, priors[task]) for task in TASKS},
                )
            preds = {task: inverse_transform(task, outputs[task]["pred_trans"]) for task in TASKS}
            bsz = int(preds["path_loss"].shape[0])
            for bi in range(bsz):
                target, prior, model_pred, masks = extract_arrays(batch, priors, preds, bi)
                report.update_sample(
                    city=batch_string(batch["city"], bi),
                    top3=batch_string(batch["topology_class_3"], bi),
                    top6=batch_string(batch["topology_class_6"], bi),
                    antenna_bin=batch_string(batch["antenna_bin"], bi),
                    refs={"target": target, "prior": prior, "model": model_pred},
                    masks=masks,
                )
            if args.progress_every and (step % args.progress_every == 0 or step == len(loader)):
                print(
                    f"[{step:05d}/{len(loader):05d}] samples={report.global_stats.n_samples} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "n_samples": len(split_refs),
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "hdf5_path": str(cfg.data.hdf5_path.resolve()),
            "calibrations": {
                "try78_los": str(cfg.prior.try78_los_calibration_json.resolve()),
                "try78_nlos": str(cfg.prior.try78_nlos_calibration_json.resolve()),
                "try79": str(cfg.prior.try79_calibration_json.resolve()),
            },
            "device": str(device),
            "mixed_precision": bool(args.mixed_precision),
            "pairs": PAIRS,
            "metric_definition": {
                "map_corr": (
                    "Per-sample/per-scope Pearson correlation over finite valid pixels; "
                    "equivalent to z-scoring each map on that sample/scope and averaging "
                    "the resulting correlations by valid-pixel count."
                ),
                "grad_mag_corr": (
                    "Per-sample/per-scope Pearson correlation between finite-difference "
                    "gradient magnitudes. Gradients are computed only across same-mask "
                    "finite neighbor pixels, then averaged by gradient-pixel count."
                ),
                "nonfinite_values": (
                    "Pixels with non-finite target or prediction values are excluded "
                    "rather than converted to zero."
                ),
            },
            "elapsed_seconds": time.time() - started,
        },
        "comparison": report.to_json(),
    }
    (args.out_dir / "structure_metrics_summary.json").write_text(
        json.dumps(json_clean(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(args.out_dir / "structure_metrics_summary.csv", report)
    write_markdown(args.out_dir / "structure_metrics_summary.md", summary)
    print(f"Wrote {(args.out_dir / 'structure_metrics_summary.json').resolve()}", flush=True)


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


def make_pair_update(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> PairUpdate:
    valid = mask.astype(bool, copy=False) & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return PairUpdate()
    x_clean = x.astype(np.float32, copy=False)
    y_clean = y.astype(np.float32, copy=False)
    xv = x_clean[valid].astype(np.float64, copy=False)
    yv = y_clean[valid].astype(np.float64, copy=False)
    diff = yv - xv

    map_corr = corr_from_arrays(xv, yv)
    map_corr_weight = int(xv.size) if map_corr is not None else 0

    gx_map, grad_valid = masked_gradient_magnitude(x_clean, valid)
    gy_map, _ = masked_gradient_magnitude(y_clean, valid)
    gx = gx_map[grad_valid].astype(np.float64, copy=False)
    gy = gy_map[grad_valid].astype(np.float64, copy=False)
    grad_corr = corr_from_arrays(gx, gy)
    grad_corr_weight = int(gx.size) if grad_corr is not None else 0

    return PairUpdate(
        sse=float(np.sum(diff * diff, dtype=np.float64)),
        n_rmse=int(valid.sum()),
        map_corr_weighted_sum=float(map_corr * map_corr_weight) if map_corr is not None else 0.0,
        n_corr=map_corr_weight,
        grad_corr_weighted_sum=float(grad_corr * grad_corr_weight) if grad_corr is not None else 0.0,
        n_grad_corr=grad_corr_weight,
    )


def masked_gradient_magnitude(arr: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite-difference gradient magnitudes without crossing invalid pixels."""
    v = valid.astype(bool, copy=False)
    work = np.where(v, arr.astype(np.float32, copy=False), 0.0).astype(np.float32, copy=False)
    gx = np.zeros_like(work, dtype=np.float32)
    gy = np.zeros_like(work, dtype=np.float32)

    left = np.zeros_like(v, dtype=bool)
    right = np.zeros_like(v, dtype=bool)
    up = np.zeros_like(v, dtype=bool)
    down = np.zeros_like(v, dtype=bool)
    left[:, 1:] = v[:, 1:] & v[:, :-1]
    right[:, :-1] = v[:, :-1] & v[:, 1:]
    up[1:, :] = v[1:, :] & v[:-1, :]
    down[:-1, :] = v[:-1, :] & v[1:, :]

    both_x = left & right
    right_only = right & ~left
    left_only = left & ~right
    if work.shape[1] > 1:
        gx[:, :-1] = np.where(right_only[:, :-1], work[:, 1:] - work[:, :-1], gx[:, :-1])
        gx[:, 1:] = np.where(left_only[:, 1:], work[:, 1:] - work[:, :-1], gx[:, 1:])
    if work.shape[1] > 2:
        gx[:, 1:-1] = np.where(
            both_x[:, 1:-1],
            0.5 * (work[:, 2:] - work[:, :-2]),
            gx[:, 1:-1],
        )

    both_y = up & down
    down_only = down & ~up
    up_only = up & ~down
    if work.shape[0] > 1:
        gy[:-1, :] = np.where(down_only[:-1, :], work[1:, :] - work[:-1, :], gy[:-1, :])
        gy[1:, :] = np.where(up_only[1:, :], work[1:, :] - work[:-1, :], gy[1:, :])
    if work.shape[0] > 2:
        gy[1:-1, :] = np.where(
            both_y[1:-1, :],
            0.5 * (work[2:, :] - work[:-2, :]),
            gy[1:-1, :],
        )

    grad_valid = v & (left | right | up | down)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)
    return mag, grad_valid


def corr_from_sums(
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
    n: int,
) -> float | None:
    if n < 2:
        return None
    cov = sum_xy - (sum_x * sum_y / n)
    vx = sum_x2 - (sum_x * sum_x / n)
    vy = sum_y2 - (sum_y * sum_y / n)
    denom = math.sqrt(max(vx, 0.0) * max(vy, 0.0))
    if denom <= 1.0e-12:
        return None
    return cov / denom


def corr_from_arrays(x: np.ndarray, y: np.ndarray) -> float | None:
    return corr_from_sums(
        float(np.sum(x, dtype=np.float64)),
        float(np.sum(y, dtype=np.float64)),
        float(np.sum(x * x, dtype=np.float64)),
        float(np.sum(y * y, dtype=np.float64)),
        float(np.sum(x * y, dtype=np.float64)),
        int(x.size),
    )


def weighted_mean_or_none(weighted_sum: float, n: int) -> float | None:
    return weighted_sum / n if n > 0 else None


def to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return torch.amp.autocast(device_type="cpu", enabled=False)


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


def sub_optional(a: object, b: object) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


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
        "task",
        "scope",
        "n_samples",
        "rmse_pw",
        "n_pixels",
        "map_corr",
        "grad_mag_corr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    comp = summary["comparison"]
    tasks = (("path_loss", "PL"), ("delay_spread", "DS"), ("angular_spread", "AS"))
    lines = [
        "# Structure Metric Comparison",
        "",
        "Negative dRMSE is better; positive correlation deltas are better.",
        "",
        "## Global Overall",
        "",
        "| Output | dRMSE model-prior | dMapCorr model-prior | dGradCorr model-prior | prior-model RMSE | prior-model MapCorr | prior-model GradCorr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    global_node = comp["global"]
    for task, label in tasks:
        dlt = global_node["delta_model_minus_prior"][task]["overall"]
        pm = global_node["metrics"]["prior_vs_model"][task]["overall"]
        lines.append(
            f"| {label} | {fmt(dlt['rmse_pw'])} | {fmt(dlt['map_corr'], 6)} | "
            f"{fmt(dlt['grad_mag_corr'], 6)} | {fmt(pm['rmse_pw'])} | "
            f"{fmt(pm['map_corr'], 6)} | {fmt(pm['grad_mag_corr'], 6)} |"
        )
    lines.extend([
        "",
        "## By Environment Class 6",
        "",
        "| Environment | Samples | PL dMapCorr | DS dMapCorr | AS dMapCorr | PL dGradCorr | DS dGradCorr | AS dGradCorr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for env, node in comp["by_topology_class_6"].items():
        vals = []
        for task, _ in tasks:
            vals.append(node["delta_model_minus_prior"][task]["overall"]["map_corr"])
        for task, _ in tasks:
            vals.append(node["delta_model_minus_prior"][task]["overall"]["grad_mag_corr"])
        lines.append(
            f"| {env} | {node['n_samples']} | {fmt(vals[0], 6)} | {fmt(vals[1], 6)} | "
            f"{fmt(vals[2], 6)} | {fmt(vals[3], 6)} | {fmt(vals[4], 6)} | {fmt(vals[5], 6)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
