#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare prior-only candidates for Try 45 on the official non-building mask.")
    p.add_argument("--try-dir", default=str(PRACTICE_ROOT / "TFGFortyFifthTry45"))
    p.add_argument(
        "--config",
        default="experiments/fortyfifthtry45_pmnet_moe_enhanced_prior/fortyfifthtry45_pmnet_moe_enhanced_prior.yaml",
    )
    p.add_argument("--dataset", default=str(PRACTICE_ROOT / "Datasets" / "CKM_Dataset_270326.h5"))
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--device", default="directml")
    p.add_argument(
        "--out-json",
        default=str(PRACTICE_ROOT / "analysis" / "try45_prior_candidates.json"),
    )
    p.add_argument(
        "--out-md",
        default=str(PRACTICE_ROOT / "analysis" / "try45_prior_candidates.md"),
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


def _finalize(total: Dict[str, float]) -> Dict[str, float]:
    count = max(int(total["count"]), 1)
    mse = total["sq"] / count
    return {
        "count": int(total["count"]),
        "rmse_db": float(math.sqrt(mse)),
        "mae_db": float(total["abs"] / count),
    }


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
    cfg["data"]["path_loss_formula_input"] = dict(cfg["data"].get("path_loss_formula_input", {}))
    cfg["data"]["path_loss_formula_input"]["regime_calibration_json"] = None

    device = _resolve_device(args.device)
    splits = data_utils.build_dataset_splits_from_config(cfg)
    ds = splits[args.split]
    target_meta = dict(cfg["target_metadata"]["path_loss"])

    formula_idx = 1
    if cfg["data"].get("los_input_column"):
        formula_idx += 1
    if cfg["data"].get("distance_map_channel", False):
        formula_idx += 1

    formula_modes = [
        "hybrid_two_ray_cost231",
        "hybrid_two_ray_cost231_a2g_nlos",
        "a2g_paper_eq9_eq10_eq11",
    ]
    totals = {
        mode: {
            "overall": {"count": 0.0, "sq": 0.0, "abs": 0.0},
            "LoS": {"count": 0.0, "sq": 0.0, "abs": 0.0},
            "NLoS": {"count": 0.0, "sq": 0.0, "abs": 0.0},
        }
        for mode in formula_modes
    }

    def denorm(t: torch.Tensor) -> torch.Tensor:
        return t * float(target_meta.get("scale", 1.0)) + float(target_meta.get("offset", 0.0))

    mode_datasets = {}
    for mode in formula_modes:
        cfg_local = dict(cfg)
        cfg_local["data"] = dict(cfg["data"])
        formula_cfg = dict(cfg_local["data"]["path_loss_formula_input"])
        formula_cfg["formula"] = mode
        cfg_local["data"]["path_loss_formula_input"] = formula_cfg
        mode_datasets[mode] = data_utils.build_dataset_splits_from_config(cfg_local)[args.split]

    for idx in range(len(ds)):
        x_ref, y, m = ds[idx][:3]
        x_ref = x_ref.to(device)
        y = y.to(device)
        m = m.to(device)
        valid = m[0] > 0.0
        los_map = x_ref[1] if cfg["data"].get("los_input_column") else torch.ones_like(valid, dtype=torch.float32, device=device)
        target_db = denorm(y[0])

        for mode in formula_modes:
            x_mode = mode_datasets[mode][idx][0].to(device)
            prior_db = denorm(x_mode[formula_idx])
            diff = (prior_db - target_db)[valid]
            totals[mode]["overall"]["count"] += float(diff.numel())
            totals[mode]["overall"]["sq"] += float(torch.sum(diff * diff).item())
            totals[mode]["overall"]["abs"] += float(torch.sum(torch.abs(diff)).item())
            for label, mask in [("LoS", valid & (los_map > 0.5)), ("NLoS", valid & (los_map <= 0.5))]:
                if torch.any(mask):
                    sub = (prior_db - target_db)[mask]
                    totals[mode][label]["count"] += float(sub.numel())
                    totals[mode][label]["sq"] += float(torch.sum(sub * sub).item())
                    totals[mode][label]["abs"] += float(torch.sum(torch.abs(sub)).item())

    payload = {
        mode: {section: _finalize(stats) for section, stats in sections.items()}
        for mode, sections in totals.items()
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Try 45 prior candidate comparison",
        "",
        "Official metric mask: `topology == 0` and dataset-valid pixels only.",
        "",
    ]
    for mode, sections in payload.items():
        lines.append(f"## `{mode}`")
        lines.append("")
        for section, stats in sections.items():
            lines.append(
                f"- `{section}`: RMSE `{stats['rmse_db']:.4f} dB`, MAE `{stats['mae_db']:.4f} dB`, count `{stats['count']}`"
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
