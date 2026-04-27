"""Evaluate all Try 76 experts on a split with an explicit device.

This is intentionally lightweight compared with ``evaluate_try76.py``: it
skips histogram CSV export and writes one aggregate JSON for thesis tables.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader


TRY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRY_ROOT.parent
if str(TRY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY_ROOT))

from src.config_try76 import Try76Cfg  # noqa: E402
from src.data_utils import HeightEmbedding, SampleRef, Try76Config, build_expert_datasets  # noqa: E402
from src.model_try76 import Try76Model, Try76ModelConfig  # noqa: E402


def resolve_device(name: str) -> torch.device:
    if name == "directml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_registry(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return list(raw.get("experts", []))


def evaluate_expert(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    device: torch.device,
    classify_cache: Dict[SampleRef, str],
) -> Dict[str, Any]:
    cfg = Try76Cfg.load(config_path)
    if not cfg.data.hdf5_path.is_file():
        local_hdf5 = PROJECT_ROOT / "Datasets" / "CKM_Dataset_270326.h5"
        if local_hdf5.is_file():
            cfg.data.hdf5_path = local_hdf5
        else:
            raise FileNotFoundError(f"HDF5 dataset not found: {cfg.data.hdf5_path}")
    data_cfg = Try76Config(
        hdf5_path=cfg.data.hdf5_path,
        topology_class=cfg.data.topology_class,
        region_mode=cfg.data.region_mode,
        image_size=cfg.data.image_size,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        split_seed=cfg.data.split_seed,
        path_loss_no_data_mask_column=cfg.data.path_loss_no_data_mask_column,
        derive_no_data_from_non_ground=cfg.data.derive_no_data_from_non_ground,
    )
    _, val_ds, test_ds = build_expert_datasets(data_cfg, classify_cache=classify_cache)
    ds = val_ds if split == "val" else test_ds
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    model_cfg = Try76ModelConfig(
        in_channels=4,
        cond_dim=64,
        height_embed_dim=32,
        base_width=cfg.model.base_width,
        K=cfg.model.K,
        clamp_lo=cfg.model.clamp_lo,
        clamp_hi=cfg.model.clamp_hi,
        outlier_sigma_floor=cfg.model.outlier_sigma_floor,
    )
    model = Try76Model(model_cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    model.eval()
    print(
        f"[try76] infer {cfg.data.topology_class}/{cfg.data.region_mode} "
        f"samples={len(ds)} model_device={next(model.parameters()).device}",
        flush=True,
    )
    height_embed = HeightEmbedding()

    sample_rmse_sum = 0.0
    sample_mae_sum = 0.0
    sample_count = 0
    pixel_sq_sum = 0.0
    pixel_abs_sum = 0.0
    pixel_count = 0.0

    started = time.perf_counter()
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        h_emb = height_embed(batch["antenna_height_m"])
        with torch.no_grad():
            pred = model(batch["inputs"], h_emb)["pred"]
        target = batch["target"]
        mask = batch["loss_mask"]
        valid = mask.sum().clamp_min(1.0)
        sq = ((pred - target) ** 2 * mask).sum()
        abs_err = ((pred - target).abs() * mask).sum()
        sample_rmse_sum += float((sq / valid).sqrt().detach().cpu().item())
        sample_mae_sum += float((abs_err / valid).detach().cpu().item())
        sample_count += 1
        pixel_sq_sum += float(sq.detach().cpu().item())
        pixel_abs_sum += float(abs_err.detach().cpu().item())
        pixel_count += float(mask.sum().detach().cpu().item())

    pixel_count_safe = max(pixel_count, 1.0)
    return {
        "expert_id": f"{cfg.data.topology_class}_{cfg.data.region_mode.replace('_only', '')}",
        "topology_class": cfg.data.topology_class,
        "region_mode": cfg.data.region_mode,
        "n_samples": sample_count,
        "pixels": int(pixel_count),
        "sample_mean_rmse_db": sample_rmse_sum / max(sample_count, 1),
        "sample_mean_mae_db": sample_mae_sum / max(sample_count, 1),
        "pixel_weighted_rmse_db": (pixel_sq_sum / pixel_count_safe) ** 0.5,
        "pixel_weighted_mae_db": pixel_abs_sum / pixel_count_safe,
        "seconds": time.perf_counter() - started,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=TRY_ROOT / "experiments" / "seventysixth_try76_experts" / "try76_expert_registry.yaml")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "cluster_outputs" / "TFGSeventySixthTry76")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="directml")
    parser.add_argument("--region", choices=["all", "los_only", "nlos_only"], default="all")
    parser.add_argument("--out", type=Path, default=TRY_ROOT / "tmp_try76_directml_test_metrics.json")
    args = parser.parse_args()

    device = resolve_device(args.device)
    rows: List[Dict[str, Any]] = []
    classify_cache: Dict[SampleRef, str] = {}
    started = time.perf_counter()
    for expert in load_registry(args.registry):
        config_path = (TRY_ROOT / expert["config"]).resolve()
        cfg = Try76Cfg.load(config_path)
        if args.region != "all" and cfg.data.region_mode != args.region:
            continue
        checkpoint_path = args.outputs_root / expert["expert_id"] / "best_model.pt"
        if not checkpoint_path.is_file():
            checkpoint_path = args.outputs_root / f"try76_expert_{expert['expert_id']}" / "best_model.pt"
        print(f"[try76] {expert['expert_id']} split={args.split} device={device}", flush=True)
        rows.append(evaluate_expert(config_path, checkpoint_path, args.split, device, classify_cache))

    by_region: Dict[str, Dict[str, float]] = {}
    for region in sorted({row["region_mode"] for row in rows}):
        subset = [row for row in rows if row["region_mode"] == region]
        total_pixels = sum(row["pixels"] for row in subset)
        by_region[region] = {
            "n_experts": len(subset),
            "total_samples": sum(row["n_samples"] for row in subset),
            "total_pixels": total_pixels,
            "global_sample_weighted_rmse_db": sum(row["sample_mean_rmse_db"] * row["n_samples"] for row in subset) / max(sum(row["n_samples"] for row in subset), 1),
            "global_pixel_weighted_rmse_db": (sum((row["pixel_weighted_rmse_db"] ** 2) * row["pixels"] for row in subset) / max(total_pixels, 1)) ** 0.5,
            "expert_macro_mean_sample_rmse_db": sum(row["sample_mean_rmse_db"] for row in subset) / max(len(subset), 1),
            "expert_macro_mean_pixel_rmse_db": sum(row["pixel_weighted_rmse_db"] for row in subset) / max(len(subset), 1),
        }

    result = {
        "device": str(device),
        "split": args.split,
        "elapsed_seconds": time.perf_counter() - started,
        "per_expert": rows,
        "by_region": by_region,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
