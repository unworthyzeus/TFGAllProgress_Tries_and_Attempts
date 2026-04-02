#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search compact A2G prior parameters for Try 45 using train/val discipline.")
    p.add_argument("--try-dir", default=str(PRACTICE_ROOT / "TFGFortyFifthTry45"))
    p.add_argument(
        "--config",
        default="experiments/fortyfifthtry45_pmnet_moe_enhanced_prior/fortyfifthtry45_pmnet_moe_enhanced_prior.yaml",
    )
    p.add_argument("--dataset", default=str(PRACTICE_ROOT / "Datasets" / "CKM_Dataset_270326.h5"))
    p.add_argument("--device", default="directml")
    p.add_argument("--max-train-samples", type=int, default=160)
    p.add_argument("--max-val-samples", type=int, default=96)
    p.add_argument("--pixel-sample", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nlos-bias-values", type=float, nargs="+", default=[-24.0, -20.0, -16.16, -12.0, -8.0, -4.0])
    p.add_argument("--nlos-amp-values", type=float, nargs="+", default=[8.0, 12.0436, 16.0, 20.0, 24.0])
    p.add_argument("--nlos-tau-values", type=float, nargs="+", default=[4.0, 6.0, 7.52, 10.0, 14.0])
    p.add_argument(
        "--out-json",
        default=str(PRACTICE_ROOT / "analysis" / "try45_a2g_parameter_grid.json"),
    )
    p.add_argument(
        "--out-md",
        default=str(PRACTICE_ROOT / "analysis" / "try45_a2g_parameter_grid.md"),
    )
    p.add_argument(
        "--out-best-params-json",
        default=str(PRACTICE_ROOT / "TFGFortyFifthTry45" / "prior_calibration" / "a2g_best_params_train_search.json"),
    )
    return p.parse_args()


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _resolve_device(name: str) -> torch.device | object:
    dev = str(name).lower()
    if dev == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if dev == "directml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    return torch.device("cpu")


def _denorm(t: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
    return t * float(meta.get("scale", 1.0)) + float(meta.get("offset", 0.0))


def _subset_indices(length: int, limit: int, seed: int) -> List[int]:
    idxs = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    if limit <= 0 or limit >= length:
        return idxs
    return idxs[:limit]


def _finalize(stats: Dict[str, float]) -> Dict[str, float]:
    count = max(int(stats["count"]), 1)
    mse = stats["sq"] / count
    return {
        "count": int(stats["count"]),
        "rmse_db": float(math.sqrt(mse)),
        "mae_db": float(stats["abs"] / count),
    }


def _init_stats() -> Dict[str, Dict[str, float]]:
    return {
        "overall": {"count": 0.0, "sq": 0.0, "abs": 0.0},
        "LoS": {"count": 0.0, "sq": 0.0, "abs": 0.0},
        "NLoS": {"count": 0.0, "sq": 0.0, "abs": 0.0},
    }


def _update_stats(bucket: Dict[str, Dict[str, float]], diff: torch.Tensor, los_mask: torch.Tensor, valid: torch.Tensor) -> None:
    all_diff = diff[valid]
    bucket["overall"]["count"] += float(all_diff.numel())
    bucket["overall"]["sq"] += float(torch.sum(all_diff * all_diff).item())
    bucket["overall"]["abs"] += float(torch.sum(torch.abs(all_diff)).item())
    for label, mask in [("LoS", valid & (los_mask > 0.5)), ("NLoS", valid & (los_mask <= 0.5))]:
        if torch.any(mask):
            sub = diff[mask]
            bucket[label]["count"] += float(sub.numel())
            bucket[label]["sq"] += float(torch.sum(sub * sub).item())
            bucket[label]["abs"] += float(torch.sum(torch.abs(sub)).item())


def main() -> None:
    args = parse_args()
    try_dir = _resolve(PRACTICE_ROOT, args.try_dir).resolve()
    config_path = _resolve(try_dir, args.config).resolve()
    sys.path.insert(0, str(try_dir))

    config_utils = importlib.import_module("config_utils")
    data_utils = importlib.import_module("data_utils")

    cfg = config_utils.load_config(str(config_path))
    cfg["data"]["hdf5_path"] = str(Path(args.dataset).resolve())
    cfg["augmentation"] = dict(cfg.get("augmentation", {}))
    cfg["augmentation"]["enable"] = False
    cfg["data"] = dict(cfg.get("data", {}))
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    formula_cfg["enabled"] = True
    formula_cfg["regime_calibration_json"] = None
    formula_cfg["formula"] = "a2g_paper_eq9_eq10_eq11"
    formula_cfg["include_shadow_sigma_channel"] = False
    cfg["data"]["path_loss_formula_input"] = formula_cfg

    splits = data_utils.build_dataset_splits_from_config(cfg)
    train_ds = splits["train"]
    val_ds = splits["val"]
    target_meta = dict(cfg["target_metadata"]["path_loss"])
    device = _resolve_device(args.device)
    rng = random.Random(args.seed)

    los_idx = 1 if cfg["data"].get("los_input_column") else None
    meters_per_pixel = float(formula_cfg.get("meters_per_pixel", 1.0))
    frequency_ghz = float(formula_cfg.get("frequency_ghz", 7.125))
    receiver_height_m = float(formula_cfg.get("receiver_height_m", 1.5))
    image_size = int(cfg["data"].get("image_size", 513))

    def collect_samples(ds: Any, indices: Sequence[int]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for ds_idx in indices:
            city, sample = ds.sample_refs[ds_idx]
            x, y, m = ds[ds_idx][:3]
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            target_db = _denorm(y[0], target_meta)
            valid = m[0] > 0.0
            if not torch.any(valid):
                continue
            los_map = x[los_idx] if los_idx is not None else torch.ones_like(target_db)
            antenna_height_m = float(ds._resolve_hdf5_scalar_value(city, sample, "antenna_height_m"))
            valid_idx = torch.nonzero(valid, as_tuple=False)
            if valid_idx.shape[0] > args.pixel_sample > 0:
                perm = torch.randperm(valid_idx.shape[0], generator=torch.Generator().manual_seed(args.seed + ds_idx))
                valid_idx = valid_idx[perm[: args.pixel_sample]]
                sampled_valid = torch.zeros_like(valid, dtype=torch.bool)
                sampled_valid[valid_idx[:, 0], valid_idx[:, 1]] = True
                valid = sampled_valid
            items.append(
                {
                    "target_db": target_db,
                    "valid": valid,
                    "los_map": los_map,
                    "antenna_height_m": antenna_height_m,
                    "city": city,
                    "sample": sample,
                }
            )
        return items

    train_items = collect_samples(train_ds, _subset_indices(len(train_ds), args.max_train_samples, args.seed))
    val_items = collect_samples(val_ds, _subset_indices(len(val_ds), args.max_val_samples, args.seed + 1))

    candidates: List[Dict[str, Any]] = []
    for nlos_bias in args.nlos_bias_values:
        for nlos_amp in args.nlos_amp_values:
            for nlos_tau in args.nlos_tau_values:
                a2g_params = {
                    "los_log_coeff": -20.0,
                    "los_bias": 0.0,
                    "nlos_bias": float(nlos_bias),
                    "nlos_amp": float(nlos_amp),
                    "nlos_tau": float(nlos_tau),
                }
                record: Dict[str, Any] = {"a2g_params": a2g_params}
                for split_name, items in [("train", train_items), ("val", val_items)]:
                    stats = _init_stats()
                    for item in items:
                        prior_db = data_utils._compute_formula_path_loss_db(
                            image_size=image_size,
                            antenna_height_m=float(item["antenna_height_m"]),
                            receiver_height_m=receiver_height_m,
                            frequency_ghz=frequency_ghz,
                            meters_per_pixel=meters_per_pixel,
                            formula_mode="a2g_paper_eq9_eq10_eq11",
                            los_tensor=item["los_map"],
                            a2g_params=a2g_params,
                            clip_min=float(target_meta.get("clip_min", 0.0)),
                            clip_max=float(target_meta.get("clip_max", 180.0)),
                        ).squeeze(0)
                        diff = prior_db - item["target_db"]
                        _update_stats(stats, diff, item["los_map"], item["valid"])
                    record[split_name] = {k: _finalize(v) for k, v in stats.items()}
                train_nlos = record["train"]["NLoS"]["rmse_db"]
                train_los = record["train"]["LoS"]["rmse_db"]
                record["train_priority_score"] = float(0.85 * train_nlos + 0.15 * train_los)
                candidates.append(record)

    candidates.sort(key=lambda x: (x["train_priority_score"], x["train"]["overall"]["rmse_db"]))
    best = candidates[0]

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_best = Path(args.out_best_params_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_best.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "search_space": {
            "nlos_bias_values": list(args.nlos_bias_values),
            "nlos_amp_values": list(args.nlos_amp_values),
            "nlos_tau_values": list(args.nlos_tau_values),
        },
        "selection_rule": "minimize train_priority_score = 0.85 * train_NLoS_RMSE + 0.15 * train_LoS_RMSE",
        "train_sample_count": len(train_items),
        "val_sample_count": len(val_items),
        "pixel_sample": int(args.pixel_sample),
        "best": best,
        "top10": candidates[:10],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_best.write_text(json.dumps(best["a2g_params"], indent=2), encoding="utf-8")

    lines = [
        "# Try 45 A2G parameter grid search",
        "",
        "This search keeps the paper-style A2G functional form and varies only a small number of global coefficients.",
        "",
        "Selection rule:",
        "",
        "- minimize `0.85 * train_NLoS_RMSE + 0.15 * train_LoS_RMSE`",
        "- train split for search",
        "- validation split only for later comparison",
        "",
        "## Best candidate",
        "",
        f"- params: `{json.dumps(best['a2g_params'])}`",
        f"- train overall: `{best['train']['overall']['rmse_db']:.4f} dB`",
        f"- train LoS: `{best['train']['LoS']['rmse_db']:.4f} dB`",
        f"- train NLoS: `{best['train']['NLoS']['rmse_db']:.4f} dB`",
        f"- val overall: `{best['val']['overall']['rmse_db']:.4f} dB`",
        f"- val LoS: `{best['val']['LoS']['rmse_db']:.4f} dB`",
        f"- val NLoS: `{best['val']['NLoS']['rmse_db']:.4f} dB`",
        "",
        "## Top 10",
        "",
    ]
    for idx, cand in enumerate(candidates[:10], start=1):
        lines.append(
            f"{idx}. `{json.dumps(cand['a2g_params'])}` | train NLoS `{cand['train']['NLoS']['rmse_db']:.4f}` | val NLoS `{cand['val']['NLoS']['rmse_db']:.4f}` | val overall `{cand['val']['overall']['rmse_db']:.4f}`"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["best"], indent=2))


if __name__ == "__main__":
    main()
