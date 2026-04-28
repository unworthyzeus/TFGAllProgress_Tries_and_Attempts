"""Deep Try 80 evaluation with city/prior-expert/LoS breakdowns.

The standard evaluator reports global and macro-expert metrics. This script
keeps those definitions, then expands them for thesis-style analysis:

- test split as unseen-city generalization
- all train+val+test samples
- by city
- by prior expert (topology_class_3 | antenna_bin)
- by city and prior expert simultaneously
- overall / LoS / NLoS for every output
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

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
from src.losses_try80 import LossWeights, combined_loss  # noqa: E402
from src.metrics_try80 import TASKS, inverse_transform, transform_target  # noqa: E402
from src.model_try80 import Try80Model, Try80ModelConfig  # noqa: E402


SCOPES = ("overall", "los", "nlos")
KINDS = ("model", "prior")
SampleRef = Tuple[str, str]


@dataclass
class Stat:
    sse: float = 0.0
    sae: float = 0.0
    n_pixels: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        if not torch.any(mask):
            return
        diff = pred[mask] - target[mask]
        self.sse += float((diff * diff).sum().item())
        self.sae += float(diff.abs().sum().item())
        self.n_pixels += int(mask.sum().item())

    def to_json(self) -> Dict[str, float | int | None]:
        rmse = math.sqrt(self.sse / self.n_pixels) if self.n_pixels > 0 else None
        mae = self.sae / self.n_pixels if self.n_pixels > 0 else None
        return {
            "rmse_pw": rmse,
            "mae_pw": mae,
            "n_pixels": self.n_pixels,
        }


@dataclass
class GroupStats:
    n_samples: int = 0
    split_counts: Dict[str, int] = field(default_factory=dict)
    city_counts: Dict[str, int] = field(default_factory=dict)
    prior_expert_counts: Dict[str, int] = field(default_factory=dict)
    topology_class_3_counts: Dict[str, int] = field(default_factory=dict)
    topology_class_6_counts: Dict[str, int] = field(default_factory=dict)
    antenna_bin_counts: Dict[str, int] = field(default_factory=dict)
    stats: Dict[str, Dict[str, Dict[str, Stat]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stats:
            self.stats = {
                kind: {task: {scope: Stat() for scope in SCOPES} for task in TASKS}
                for kind in KINDS
            }

    def add_sample_meta(
        self,
        split: str,
        city: str,
        prior_expert: str,
        topology_class_3: str,
        topology_class_6: str,
        antenna_bin: str,
    ) -> None:
        self.n_samples += 1
        _inc(self.split_counts, split)
        _inc(self.city_counts, city)
        _inc(self.prior_expert_counts, prior_expert)
        _inc(self.topology_class_3_counts, topology_class_3)
        _inc(self.topology_class_6_counts, topology_class_6)
        _inc(self.antenna_bin_counts, antenna_bin)

    def update_metrics(
        self,
        task: str,
        target: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
        model_pred: torch.Tensor,
        prior_pred: torch.Tensor,
    ) -> None:
        for scope, mask in masks.items():
            self.stats["model"][task][scope].update(model_pred, target, mask)
            self.stats["prior"][task][scope].update(prior_pred, target, mask)

    def to_json(self) -> Dict[str, object]:
        metrics = {
            kind: {
                task: {scope: self.stats[kind][task][scope].to_json() for scope in SCOPES}
                for task in TASKS
            }
            for kind in KINDS
        }
        return {
            "n_samples": self.n_samples,
            "split_counts": dict(sorted(self.split_counts.items())),
            "city_counts": dict(sorted(self.city_counts.items())),
            "prior_expert_counts": dict(sorted(self.prior_expert_counts.items())),
            "topology_class_3_counts": dict(sorted(self.topology_class_3_counts.items())),
            "topology_class_6_counts": dict(sorted(self.topology_class_6_counts.items())),
            "antenna_bin_counts": dict(sorted(self.antenna_bin_counts.items())),
            "metrics": metrics,
            "delta_model_minus_prior": _delta_metrics(metrics["model"], metrics["prior"]),
        }


class DeepBreakdown:
    def __init__(self, name: str) -> None:
        self.name = name
        self.global_stats = GroupStats()
        self.by_city: Dict[str, GroupStats] = {}
        self.by_prior_expert: Dict[str, GroupStats] = {}
        self.by_city_and_prior_expert: Dict[str, Dict[str, GroupStats]] = {}
        self.by_split: Dict[str, GroupStats] = {}
        self.by_split_and_prior_expert: Dict[str, Dict[str, GroupStats]] = {}
        self.per_sample_rows: List[Dict[str, object]] = []

    def update_sample(
        self,
        *,
        split: str,
        city: str,
        sample: str,
        prior_expert: str,
        topology_class_3: str,
        topology_class_6: str,
        antenna_bin: str,
        batch: Mapping[str, object],
        sample_index: int,
        preds_native: Mapping[str, torch.Tensor],
        priors_native: Mapping[str, torch.Tensor],
        store_per_sample: bool,
    ) -> None:
        groups = [
            self.global_stats,
            _group(self.by_city, city),
            _group(self.by_prior_expert, prior_expert),
            _nested_group(self.by_city_and_prior_expert, city, prior_expert),
            _group(self.by_split, split),
            _nested_group(self.by_split_and_prior_expert, split, prior_expert),
        ]
        for group in groups:
            group.add_sample_meta(
                split=split,
                city=city,
                prior_expert=prior_expert,
                topology_class_3=topology_class_3,
                topology_class_6=topology_class_6,
                antenna_bin=antenna_bin,
            )

        row: Dict[str, object] = {
            "split": split,
            "city": city,
            "sample": sample,
            "prior_expert": prior_expert,
            "topology_class_3": topology_class_3,
            "topology_class_6": topology_class_6,
            "antenna_bin": antenna_bin,
        }

        los_mask = batch["los_mask"][sample_index, 0] > 0.5
        nlos_mask = batch["nlos_mask"][sample_index, 0] > 0.5
        for task in TASKS:
            target = batch[f"{task}_target"][sample_index, 0]
            valid = batch[f"{task}_mask"][sample_index, 0] > 0.5
            masks = {
                "overall": valid,
                "los": valid & los_mask,
                "nlos": valid & nlos_mask,
            }
            model_pred = preds_native[task][sample_index, 0]
            prior_pred = priors_native[task][sample_index, 0]
            for group in groups:
                group.update_metrics(task, target, masks, model_pred, prior_pred)
            if store_per_sample:
                for scope, mask in masks.items():
                    row[f"{task}_{scope}_n_pixels"] = int(mask.sum().item())
                    row[f"{task}_{scope}_model_rmse"] = _sample_rmse(model_pred, target, mask)
                    row[f"{task}_{scope}_model_mae"] = _sample_mae(model_pred, target, mask)
                    row[f"{task}_{scope}_prior_rmse"] = _sample_rmse(prior_pred, target, mask)
                    row[f"{task}_{scope}_prior_mae"] = _sample_mae(prior_pred, target, mask)
                    row[f"{task}_{scope}_rmse_delta_model_minus_prior"] = _sub_optional(
                        row[f"{task}_{scope}_model_rmse"],
                        row[f"{task}_{scope}_prior_rmse"],
                    )

        if store_per_sample:
            self.per_sample_rows.append(row)

    def to_json(self, include_per_sample: bool) -> Dict[str, object]:
        return {
            "name": self.name,
            "global": self.global_stats.to_json(),
            "by_split": _groups_to_json(self.by_split),
            "by_city": _groups_to_json(self.by_city),
            "by_prior_expert": _groups_to_json(self.by_prior_expert),
            "by_split_and_prior_expert": _nested_groups_to_json(self.by_split_and_prior_expert),
            "by_city_and_prior_expert": _nested_groups_to_json(self.by_city_and_prior_expert),
            "per_sample": self.per_sample_rows if include_per_sample else [],
        }


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


def build_model_cfg(cfg: Try80Cfg) -> Try80ModelConfig:
    return Try80ModelConfig(
        in_channels=cfg.model.in_channels,
        cond_dim=cfg.model.cond_dim,
        height_embed_dim=cfg.model.height_embed_dim,
        base_width=cfg.model.base_width,
        num_components=cfg.model.num_components,
        decoder_dropout=cfg.model.decoder_dropout,
        alpha_bias=cfg.model.alpha_bias,
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        path_residual_los_max=cfg.model.path_residual_los_max,
        path_residual_nlos_max=cfg.model.path_residual_nlos_max,
        delay_residual_los_max=cfg.model.delay_residual_los_max,
        delay_residual_nlos_max=cfg.model.delay_residual_nlos_max,
        angular_residual_los_max=cfg.model.angular_residual_los_max,
        angular_residual_nlos_max=cfg.model.angular_residual_nlos_max,
    )


def to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, etc. Defaults to cuda if available.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test sample limit.")
    parser.add_argument("--no-per-sample", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    started = time.time()
    cfg = Try80Cfg.load(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    refs = list_hdf5_samples(cfg.data.hdf5_path)
    train_refs, val_refs, test_refs = split_city_holdout(
        refs,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        split_seed=cfg.data.split_seed,
    )
    split_by_ref: Dict[SampleRef, str] = {}
    for split, split_refs in (("train", train_refs), ("val", val_refs), ("test", test_refs)):
        for ref in split_refs:
            split_by_ref[ref] = split

    ordered_refs = list(train_refs) + list(val_refs) + list(test_refs)
    if args.limit > 0:
        ordered_refs = ordered_refs[: args.limit]

    data_cfg = build_data_cfg(cfg)
    dataset = Try80JointDataset(data_cfg, ordered_refs, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = Try80Model(build_model_cfg(cfg)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    height_embed = HeightEmbedding()
    weights = LossWeights(**cfg.losses.__dict__)
    all_report = DeepBreakdown("all_cities_all_splits")
    test_report = DeepBreakdown("test_unseen_cities")
    loss_totals = {
        "all_cities_all_splits": _empty_loss_totals(),
        "test_unseen_cities": _empty_loss_totals(),
    }
    loss_counts = {
        "all_cities_all_splits": 0,
        "test_unseen_cities": 0,
    }

    with torch.no_grad():
        for step, raw_batch in enumerate(loader, start=1):
            batch = to_device(raw_batch, device)
            priors_native = {task: batch[f"{task}_prior"] for task in TASKS}
            priors_trans = {task: transform_target(task, priors_native[task]) for task in TASKS}
            outputs = model(batch["inputs"], height_embed(batch["antenna_height_m"]), priors_trans)
            preds_native = {task: inverse_transform(task, outputs[task]["pred_trans"]) for task in TASKS}
            loss_terms = combined_loss(batch, outputs, preds_native, priors_native, weights=weights)

            bsz = int(preds_native["path_loss"].shape[0])
            for bi in range(bsz):
                city = _batch_string(batch["city"], bi)
                sample = _batch_string(batch["sample"], bi)
                split = split_by_ref[(city, sample)]
                top3 = _batch_string(batch["topology_class_3"], bi)
                top6 = _batch_string(batch["topology_class_6"], bi)
                ant = _batch_string(batch["antenna_bin"], bi)
                prior_expert = f"{top3}|{ant}"
                all_report.update_sample(
                    split=split,
                    city=city,
                    sample=sample,
                    prior_expert=prior_expert,
                    topology_class_3=top3,
                    topology_class_6=top6,
                    antenna_bin=ant,
                    batch=batch,
                    sample_index=bi,
                    preds_native=preds_native,
                    priors_native=priors_native,
                    store_per_sample=not args.no_per_sample,
                )
                _accumulate_losses(loss_totals["all_cities_all_splits"], loss_terms)
                loss_counts["all_cities_all_splits"] += 1
                if split == "test":
                    test_report.update_sample(
                        split=split,
                        city=city,
                        sample=sample,
                        prior_expert=prior_expert,
                        topology_class_3=top3,
                        topology_class_6=top6,
                        antenna_bin=ant,
                        batch=batch,
                        sample_index=bi,
                        preds_native=preds_native,
                        priors_native=priors_native,
                        store_per_sample=not args.no_per_sample,
                    )
                    _accumulate_losses(loss_totals["test_unseen_cities"], loss_terms)
                    loss_counts["test_unseen_cities"] += 1

            if args.progress_every > 0 and (step % args.progress_every == 0 or step == len(loader)):
                elapsed = time.time() - started
                print(
                    f"[{step:05d}/{len(loader):05d}] elapsed={elapsed:.1f}s "
                    f"all_samples={all_report.global_stats.n_samples} "
                    f"test_samples={test_report.global_stats.n_samples}",
                    flush=True,
                )

    for name, totals in loss_totals.items():
        count = max(loss_counts[name], 1)
        for key in totals:
            totals[key] /= count

    split_summary = {
        "train": _split_summary(train_refs),
        "val": _split_summary(val_refs),
        "test": _split_summary(test_refs),
    }

    report = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint.resolve()),
            "config": str(args.config.resolve()),
            "hdf5_path": str(cfg.data.hdf5_path.resolve()),
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
            "torch_version": torch.__version__,
            "tasks": list(TASKS),
            "scopes": list(SCOPES),
            "prior_expert_definition": "topology_class_3|antenna_bin",
            "metric_definition": {
                "rmse_pw": "pixel-weighted RMSE over valid target pixels",
                "mae_pw": "pixel-weighted MAE over valid target pixels",
                "delta_model_minus_prior": "model metric minus frozen-prior metric; negative is better than prior",
            },
            "split_protocol": {
                "mode": "city_holdout",
                "split_seed": cfg.data.split_seed,
                "val_ratio": cfg.data.val_ratio,
                "test_ratio": cfg.data.test_ratio,
                "test_is_unseen_city_generalization": True,
                "raw_counts": {
                    "train": len(train_refs),
                    "val": len(val_refs),
                    "test": len(test_refs),
                    "all": len(refs),
                },
                "city_counts": {
                    "train": len(split_summary["train"]["cities"]),
                    "val": len(split_summary["val"]["cities"]),
                    "test": len(split_summary["test"]["cities"]),
                    "all": len({city for city, _ in refs}),
                },
                "cities": {
                    "train": split_summary["train"]["cities"],
                    "val": split_summary["val"]["cities"],
                    "test_unseen": split_summary["test"]["cities"],
                },
            },
            "limit": args.limit,
            "elapsed_seconds": time.time() - started,
        },
        "test_unseen_cities": test_report.to_json(include_per_sample=not args.no_per_sample),
        "all_cities_all_splits": all_report.to_json(include_per_sample=not args.no_per_sample),
        "losses": loss_totals,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_clean(report), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}", flush=True)


def _empty_loss_totals() -> Dict[str, float]:
    return {
        "total": 0.0,
        "map_nll": 0.0,
        "dist_kl": 0.0,
        "moment_match": 0.0,
        "anchor": 0.0,
        "prior_guard": 0.0,
        "outlier_budget": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
    }


def _accumulate_losses(totals: MutableMapping[str, float], loss_terms: Mapping[str, torch.Tensor]) -> None:
    for key in totals:
        totals[key] += float(loss_terms[key].detach().item())


def _batch_string(value: object, idx: int) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[idx])
    return str(value)


def _split_summary(refs: Sequence[SampleRef]) -> Dict[str, object]:
    by_city: Dict[str, int] = {}
    for city, _ in refs:
        _inc(by_city, city)
    return {
        "n_samples": len(refs),
        "cities": sorted(by_city),
        "samples_by_city": dict(sorted(by_city.items())),
    }


def _inc(store: MutableMapping[str, int], key: str) -> None:
    store[key] = store.get(key, 0) + 1


def _group(store: MutableMapping[str, GroupStats], key: str) -> GroupStats:
    if key not in store:
        store[key] = GroupStats()
    return store[key]


def _nested_group(store: MutableMapping[str, Dict[str, GroupStats]], key1: str, key2: str) -> GroupStats:
    if key1 not in store:
        store[key1] = {}
    return _group(store[key1], key2)


def _groups_to_json(store: Mapping[str, GroupStats]) -> Dict[str, object]:
    return {key: group.to_json() for key, group in sorted(store.items())}


def _nested_groups_to_json(store: Mapping[str, Mapping[str, GroupStats]]) -> Dict[str, object]:
    return {
        key1: {key2: group.to_json() for key2, group in sorted(inner.items())}
        for key1, inner in sorted(store.items())
    }


def _delta_metrics(model_metrics: Mapping[str, object], prior_metrics: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for task in TASKS:
        out[task] = {}
        for scope in SCOPES:
            model_scope = model_metrics[task][scope]
            prior_scope = prior_metrics[task][scope]
            out[task][scope] = {
                "rmse_pw": _sub_optional(model_scope["rmse_pw"], prior_scope["rmse_pw"]),
                "mae_pw": _sub_optional(model_scope["mae_pw"], prior_scope["mae_pw"]),
                "n_pixels": model_scope["n_pixels"],
            }
    return out


def _sub_optional(a: object, b: object) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _sample_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not torch.any(mask):
        return None
    diff = pred[mask] - target[mask]
    return float(torch.sqrt((diff * diff).mean()).item())


def _sample_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not torch.any(mask):
        return None
    return float((pred[mask] - target[mask]).abs().mean().item())


def _json_clean(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    return value


if __name__ == "__main__":
    main()
