"""Evaluate all Try 77 spread experts on a split with an explicit device."""
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

from src.config_try77 import Try77Cfg  # noqa: E402
from src.data_utils import HeightEmbedding, SampleRef, Try77Config, build_expert_datasets  # noqa: E402
from src.model_try77 import Try77Model, Try77ModelConfig  # noqa: E402


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
    cfg = Try77Cfg.load(config_path)
    if not cfg.data.hdf5_path.is_file():
        local_hdf5 = PROJECT_ROOT / "Datasets" / "CKM_Dataset_270326.h5"
        if local_hdf5.is_file():
            cfg.data.hdf5_path = local_hdf5
        else:
            raise FileNotFoundError(f"HDF5 dataset not found: {cfg.data.hdf5_path}")
    data_cfg = Try77Config(
        hdf5_path=cfg.data.hdf5_path,
        topology_class=cfg.data.topology_class,
        metric=cfg.data.metric,
        image_size=cfg.data.image_size,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        split_seed=cfg.data.split_seed,
    )
    _, val_ds, test_ds = build_expert_datasets(data_cfg, classify_cache=classify_cache)
    ds = val_ds if split == "val" else test_ds
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    model_cfg = Try77ModelConfig(
        in_channels=4,
        cond_dim=64,
        height_embed_dim=32,
        base_width=cfg.model.base_width,
        K=cfg.model.K,
        clamp_lo=cfg.model.clamp_lo,
        clamp_hi=cfg.model.clamp_hi,
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        spike_mu_max=cfg.model.spike_mu_max,
        spike_sigma_min=cfg.model.spike_sigma_min,
        spike_sigma_max=cfg.model.spike_sigma_max,
    )
    model = Try77Model(model_cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    model.eval()
    print(
        f"[try77] infer {cfg.data.topology_class}/{cfg.data.metric} "
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
        "expert_id": f"{cfg.data.topology_class}_{cfg.data.metric}",
        "topology_class": cfg.data.topology_class,
        "metric": cfg.data.metric,
        "n_samples": sample_count,
        "pixels": int(pixel_count),
        "sample_mean_rmse": sample_rmse_sum / max(sample_count, 1),
        "sample_mean_mae": sample_mae_sum / max(sample_count, 1),
        "pixel_weighted_rmse": (pixel_sq_sum / pixel_count_safe) ** 0.5,
        "pixel_weighted_mae": pixel_abs_sum / pixel_count_safe,
        "seconds": time.perf_counter() - started,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=TRY_ROOT / "experiments" / "seventyseventh_try77_experts" / "try77_expert_registry.yaml")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "cluster_outputs" / "TFGSeventySeventhTry77")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="directml")
    parser.add_argument("--metric", choices=["all", "delay_spread", "angular_spread"], default="all")
    parser.add_argument("--out", type=Path, default=TRY_ROOT / "tmp_try77_directml_test_metrics.json")
    args = parser.parse_args()

    device = resolve_device(args.device)
    rows: List[Dict[str, Any]] = []
    classify_cache: Dict[SampleRef, str] = {}
    started = time.perf_counter()
    for expert in load_registry(args.registry):
        config_path = (TRY_ROOT / expert["config"]).resolve()
        cfg = Try77Cfg.load(config_path)
        if args.metric != "all" and cfg.data.metric != args.metric:
            continue
        checkpoint_path = args.outputs_root / expert["expert_id"] / "best_model.pt"
        if not checkpoint_path.is_file():
            checkpoint_path = args.outputs_root / f"try77_expert_{expert['expert_id']}" / "best_model.pt"
        print(f"[try77] {expert['expert_id']} split={args.split} device={device}", flush=True)
        rows.append(evaluate_expert(config_path, checkpoint_path, args.split, device, classify_cache))

    by_metric: Dict[str, Dict[str, float]] = {}
    for metric in sorted({row["metric"] for row in rows}):
        subset = [row for row in rows if row["metric"] == metric]
        total_pixels = sum(row["pixels"] for row in subset)
        by_metric[metric] = {
            "n_experts": len(subset),
            "total_samples": sum(row["n_samples"] for row in subset),
            "total_pixels": total_pixels,
            "global_sample_weighted_rmse": sum(row["sample_mean_rmse"] * row["n_samples"] for row in subset) / max(sum(row["n_samples"] for row in subset), 1),
            "global_pixel_weighted_rmse": (sum((row["pixel_weighted_rmse"] ** 2) * row["pixels"] for row in subset) / max(total_pixels, 1)) ** 0.5,
            "expert_macro_mean_sample_rmse": sum(row["sample_mean_rmse"] for row in subset) / max(len(subset), 1),
            "expert_macro_mean_pixel_rmse": sum(row["pixel_weighted_rmse"] for row in subset) / max(len(subset), 1),
        }

    result = {
        "device": str(device),
        "split": args.split,
        "elapsed_seconds": time.perf_counter() - started,
        "per_expert": rows,
        "by_metric": by_metric,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
