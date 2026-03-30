#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
from torch import amp
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze path-loss errors by regimes for a try/checkpoint.")
    p.add_argument("--try-dir", required=True, help="Absolute or relative path to the try folder")
    p.add_argument("--config", required=True, help="Config path relative to the try folder, or absolute")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path relative to the try folder, or absolute")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--device", default="cuda", help="cuda / cpu / directml")
    p.add_argument("--out-json", default=str(PRACTICE_ROOT / "analysis" / "pathloss_regime_analysis.json"))
    p.add_argument("--out-md", default=str(PRACTICE_ROOT / "analysis" / "pathloss_regime_analysis.md"))
    p.add_argument("--out-csv", default=str(PRACTICE_ROOT / "analysis" / "pathloss_regime_samples.csv"))
    return p.parse_args()


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def _nearest_resize(arr: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def _safe_label(low: float, high: float, unit: str = "") -> str:
    if math.isinf(high):
        return f">={low:.0f}{unit}"
    return f"{low:.0f}-{high:.0f}{unit}"


def _build_distance_map_m(image_size: int, meters_per_pixel: float) -> np.ndarray:
    half = (image_size - 1) / 2.0
    y = np.arange(image_size, dtype=np.float32) - half
    x = np.arange(image_size, dtype=np.float32) - half
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.sqrt(xx ** 2 + yy ** 2) * float(meters_per_pixel)


def _summarize_group(rows: Iterable[Dict[str, Any]], value_key: str = "sample_rmse_db") -> Dict[str, float]:
    values = [float(r[value_key]) for r in rows if np.isfinite(float(r[value_key]))]
    if not values:
        return {"count": 0, "rmse_mean": float("nan"), "rmse_median": float("nan")}
    return {
        "count": len(values),
        "rmse_mean": float(np.mean(values)),
        "rmse_median": float(np.median(values)),
    }


def _write_markdown(
    out_path: Path,
    payload: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Path-Loss Regime Error Analysis")
    lines.append("")
    lines.append(f"- Try: `{payload['try_dir']}`")
    lines.append(f"- Split: `{payload['split']}`")
    lines.append(f"- Checkpoint: `{payload['checkpoint']}`")
    lines.append(f"- Overall RMSE: `{payload['overall']['rmse_db']:.4f} dB`")
    lines.append(f"- Samples analyzed: `{payload['overall']['sample_count']}`")
    lines.append("")

    lines.append("## Distance bins")
    lines.append("")
    for label, stats in payload["pixel_bins"]["distance"].items():
        lines.append(f"- `{label}`: count `{stats['count']}`, RMSE `{stats['rmse_db']:.4f} dB`, MAE `{stats['mae_db']:.4f} dB`")
    lines.append("")

    lines.append("## LoS bins")
    lines.append("")
    for label, stats in payload["pixel_bins"]["los"].items():
        lines.append(f"- `{label}`: count `{stats['count']}`, RMSE `{stats['rmse_db']:.4f} dB`, MAE `{stats['mae_db']:.4f} dB`")
    lines.append("")

    lines.append("## Sample bins")
    lines.append("")
    for section in ("density", "building_height_proxy", "antenna_height"):
        lines.append(f"### {section.replace('_', ' ').title()}")
        lines.append("")
        for label, stats in payload["sample_bins"][section].items():
            lines.append(f"- `{label}`: count `{stats['count']}`, mean RMSE `{stats['rmse_mean']:.4f} dB`, median RMSE `{stats['rmse_median']:.4f} dB`")
        lines.append("")

    lines.append("## Worst Cities")
    lines.append("")
    for row in payload["worst_cities"]:
        lines.append(f"- `{row['city']}`: count `{row['count']}`, mean RMSE `{row['rmse_mean']:.4f} dB`")
    lines.append("")

    lines.append("## Hardest Samples")
    lines.append("")
    for row in payload["hardest_samples"]:
        lines.append(
            f"- `{row['city']}/{row['sample']}`: RMSE `{row['sample_rmse_db']:.4f} dB`, "
            f"density `{row['building_density']:.4f}`, building-height-proxy `{row['building_height_proxy']:.4f}`, "
            f"antenna `{row['antenna_height_m']:.2f} m`, LoS-ratio `{row['los_ratio_valid']:.4f}`"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try_dir = _resolve(PRACTICE_ROOT, args.try_dir).resolve()
    config_path = _resolve(try_dir, args.config).resolve()
    checkpoint_path = _resolve(try_dir, args.checkpoint).resolve()

    sys.path.insert(0, str(try_dir))
    config_utils = importlib.import_module("config_utils")
    data_utils = importlib.import_module("data_utils")
    evaluate_cgan = importlib.import_module("evaluate_cgan")
    model_cgan = importlib.import_module("model_cgan")
    train_cgan = importlib.import_module("train_cgan")

    cfg = config_utils.load_config(str(config_path))
    config_utils.anchor_data_paths_to_config_file(cfg, str(config_path))
    cfg["augmentation"] = dict(cfg.get("augmentation", {}))
    cfg["augmentation"]["enable"] = False
    cfg["runtime"] = dict(cfg.get("runtime", {}))
    cfg["runtime"]["device"] = args.device
    device = config_utils.resolve_device(cfg["runtime"]["device"])

    splits = data_utils.build_dataset_splits_from_config(cfg)
    dataset = splits[args.split]
    if not hasattr(dataset, "sample_refs"):
        raise RuntimeError("This analysis script expects an HDF5-backed dataset with sample_refs.")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=config_utils.is_cuda_device(device))
    target_columns = list(cfg["target_columns"])
    if target_columns != ["path_loss"]:
        raise RuntimeError("This analysis script currently supports path_loss-only tries.")
    target_metadata = dict(cfg.get("target_metadata", {}))

    sc_dim = int(data_utils.compute_scalar_cond_dim(cfg)) if data_utils.uses_scalar_film_conditioning(cfg) else 0
    generator = model_cgan.UNetGenerator(
        in_channels=data_utils.compute_input_channels(cfg),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        path_loss_hybrid=bool(dict(cfg.get("path_loss_hybrid", {})).get("enabled", False)),
        norm_type=str(cfg["model"].get("norm_type", "batch")),
        scalar_cond_dim=sc_dim,
        scalar_film_hidden=int(cfg["model"].get("scalar_film_hidden", 128)),
        upsample_mode=str(cfg["model"].get("upsample_mode", "transpose")),
    ).to(device)
    state = config_utils.load_torch_checkpoint(str(checkpoint_path), device)
    generator.load_state_dict(state["generator"] if "generator" in state else state)
    generator.eval()

    image_size = int(cfg["data"].get("image_size", 513))
    meters_per_pixel = float(dict(cfg["data"].get("path_loss_formula_input", {})).get("meters_per_pixel", 1.0))
    distance_map_m = _build_distance_map_m(image_size, meters_per_pixel)
    distance_bins = [(0.0, 64.0), (64.0, 128.0), (128.0, 192.0), (192.0, 256.0), (256.0, float("inf"))]
    distance_acc = { _safe_label(lo, hi, "m"): {"count": 0, "sq": 0.0, "abs": 0.0} for lo, hi in distance_bins }
    los_acc = {"LoS": {"count": 0, "sq": 0.0, "abs": 0.0}, "NLoS": {"count": 0, "sq": 0.0, "abs": 0.0}}

    sample_rows: List[Dict[str, Any]] = []
    total_count = 0
    total_sq = 0.0
    amp_enabled = bool(cfg["training"].get("amp", False)) and config_utils.is_cuda_device(device)

    with h5py.File(str(cfg["data"]["hdf5_path"]), "r") as handle:
        for idx, batch in enumerate(loader):
            city, sample = dataset.sample_refs[idx]
            x, y, m, sc = data_utils.unpack_cgan_batch(batch, device)
            with torch.no_grad(), amp.autocast(device_type="cuda", enabled=amp_enabled):
                raw_outputs = data_utils.forward_cgan_generator(generator, x, sc)
                pred, _ = train_cgan.decode_generator_outputs(raw_outputs, x, sc, target_columns, target_metadata, cfg)

            pred_phys = evaluate_cgan.denormalize_channel(pred[:, :1], target_metadata["path_loss"]).detach().cpu().numpy()[0, 0]
            tgt_phys = evaluate_cgan.denormalize_channel(y[:, :1], target_metadata["path_loss"]).detach().cpu().numpy()[0, 0]
            valid_mask = (m[:, :1].detach().cpu().numpy()[0, 0] > 0.0)
            err = pred_phys - tgt_phys

            grp = handle[city][sample]
            topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
            los = np.asarray(grp["los_mask"][...], dtype=np.float32)
            los_resized = _nearest_resize(los, image_size) > 0.5
            non_zero = topo[topo != 0]
            antenna_height = float(np.asarray(grp["uav_height"][...], dtype=np.float32).reshape(-1)[0])

            valid_err = err[valid_mask]
            total_count += int(valid_err.size)
            total_sq += float(np.sum(valid_err ** 2))

            abs_err = np.abs(err)
            for lo, hi in distance_bins:
                label = _safe_label(lo, hi, "m")
                bin_mask = valid_mask & (distance_map_m >= lo) & (distance_map_m < hi)
                vals = err[bin_mask]
                if vals.size:
                    distance_acc[label]["count"] += int(vals.size)
                    distance_acc[label]["sq"] += float(np.sum(vals ** 2))
                    distance_acc[label]["abs"] += float(np.sum(np.abs(vals)))

            for label, los_value in (("LoS", True), ("NLoS", False)):
                bin_mask = valid_mask & (los_resized == los_value)
                vals = err[bin_mask]
                if vals.size:
                    los_acc[label]["count"] += int(vals.size)
                    los_acc[label]["sq"] += float(np.sum(vals ** 2))
                    los_acc[label]["abs"] += float(np.sum(np.abs(vals)))

            sample_rows.append(
                {
                    "city": city,
                    "sample": sample,
                    "sample_rmse_db": float(np.sqrt(np.mean(valid_err ** 2))) if valid_err.size else float("nan"),
                    "sample_mae_db": float(np.mean(np.abs(valid_err))) if valid_err.size else float("nan"),
                    "building_density": float(np.mean(topo != 0)),
                    "building_height_proxy": float(np.mean(non_zero)) if non_zero.size else 0.0,
                    "los_ratio_valid": float(np.mean(los_resized[valid_mask])) if np.any(valid_mask) else float("nan"),
                    "antenna_height_m": antenna_height,
                    "valid_pixel_count": int(np.count_nonzero(valid_mask)),
                }
            )

    def _finalize_pixel(acc: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for label, stats in acc.items():
            count = int(stats["count"])
            if count <= 0:
                out[label] = {"count": 0, "rmse_db": float("nan"), "mae_db": float("nan")}
            else:
                out[label] = {
                    "count": count,
                    "rmse_db": float(np.sqrt(stats["sq"] / count)),
                    "mae_db": float(stats["abs"] / count),
                }
        return out

    sample_rmse = [r["sample_rmse_db"] for r in sample_rows if np.isfinite(r["sample_rmse_db"])]
    antenna_values = np.asarray([r["antenna_height_m"] for r in sample_rows], dtype=np.float32)
    building_height_values = np.asarray([r["building_height_proxy"] for r in sample_rows], dtype=np.float32)
    antenna_q = np.quantile(antenna_values, [1 / 3, 2 / 3]).tolist() if len(antenna_values) >= 3 else [80.0, 160.0]
    bh_q = np.quantile(building_height_values, [1 / 3, 2 / 3]).tolist() if len(building_height_values) >= 3 else [20.0, 60.0]

    sample_bins = {
        "density": {
            "open": _summarize_group([r for r in sample_rows if r["building_density"] < 0.15]),
            "mixed": _summarize_group([r for r in sample_rows if 0.15 <= r["building_density"] < 0.30]),
            "dense": _summarize_group([r for r in sample_rows if r["building_density"] >= 0.30]),
        },
        "building_height_proxy": {
            f"low(<{bh_q[0]:.2f})": _summarize_group([r for r in sample_rows if r["building_height_proxy"] < bh_q[0]]),
            f"mid({bh_q[0]:.2f}-{bh_q[1]:.2f})": _summarize_group([r for r in sample_rows if bh_q[0] <= r["building_height_proxy"] < bh_q[1]]),
            f"high(>={bh_q[1]:.2f})": _summarize_group([r for r in sample_rows if r["building_height_proxy"] >= bh_q[1]]),
        },
        "antenna_height": {
            f"low(<{antenna_q[0]:.2f}m)": _summarize_group([r for r in sample_rows if r["antenna_height_m"] < antenna_q[0]]),
            f"mid({antenna_q[0]:.2f}-{antenna_q[1]:.2f}m)": _summarize_group([r for r in sample_rows if antenna_q[0] <= r["antenna_height_m"] < antenna_q[1]]),
            f"high(>={antenna_q[1]:.2f}m)": _summarize_group([r for r in sample_rows if r["antenna_height_m"] >= antenna_q[1]]),
        },
    }

    city_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        city_groups[row["city"]].append(row)
    worst_cities = sorted(
        [{"city": city, **_summarize_group(rows)} for city, rows in city_groups.items()],
        key=lambda r: (-(r["rmse_mean"] if np.isfinite(r["rmse_mean"]) else -1e9), -r["count"]),
    )[:10]
    hardest_samples = sorted(sample_rows, key=lambda r: -(r["sample_rmse_db"] if np.isfinite(r["sample_rmse_db"]) else -1e9))[:15]

    payload = {
        "try_dir": str(try_dir),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "overall": {
            "sample_count": len(sample_rows),
            "rmse_db": float(np.sqrt(total_sq / max(total_count, 1))),
            "sample_rmse_mean_db": float(np.mean(sample_rmse)) if sample_rmse else float("nan"),
        },
        "pixel_bins": {
            "distance": _finalize_pixel(distance_acc),
            "los": _finalize_pixel(los_acc),
        },
        "sample_bins": sample_bins,
        "worst_cities": worst_cities,
        "hardest_samples": hardest_samples,
    }

    out_json = Path(args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()) if sample_rows else [])
        if sample_rows:
            writer.writeheader()
            writer.writerows(sample_rows)
    _write_markdown(out_md, payload)
    print(out_json)
    print(out_md)
    print(out_csv)


if __name__ == "__main__":
    main()
