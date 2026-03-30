from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from config_utils import anchor_data_paths_to_config_file, load_config, load_torch_checkpoint, resolve_device
from data_utils import _compute_distance_map_2d, _normalize_array, build_dataset_splits_from_config, compute_input_channels
from model_pmnet import PMNetResidualRegressor


def denormalize(values: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    return values * scale + offset


def clip_to_target_range(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    clip_min = metadata.get("clip_min")
    clip_max = metadata.get("clip_max")
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    if clip_min is None or clip_max is None:
        return values
    min_norm = (float(clip_min) - offset) / max(scale, 1e-12)
    max_norm = (float(clip_max) - offset) / max(scale, 1e-12)
    return values.clamp(min=min_norm, max=max_norm)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr)
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if abs(arr_max - arr_min) < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - arr_min) / (arr_max - arr_min)
    return np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)


def build_model_input_from_dataset_sample(
    cfg: Dict[str, Any],
    city: str,
    sample: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    splits = build_dataset_splits_from_config(cfg)
    for split in splits.values():
        refs = getattr(split, "sample_refs", [])
        if (city, sample) in refs:
            idx = refs.index((city, sample))
            item = split[idx]
            if len(item) == 4:
                x, y, m, sc = item
            else:
                x, y, m = item
                sc = None
            return x.unsqueeze(0), y, m, sc
    raise KeyError(f"Sample {city}/{sample} was not found in any configured split.")


def parse_scalar_values(raw: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not raw:
        return result
    for part in [chunk.strip() for chunk in raw.split(",") if chunk.strip()]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = float(value.strip())
    return result


def build_model_input_from_images(
    cfg: Dict[str, Any],
    topology_path: str,
    los_path: Optional[str],
    scalar_values: Dict[str, float],
) -> torch.Tensor:
    image_size = int(cfg["data"]["image_size"])
    data_cfg = dict(cfg["data"])

    topo_img = Image.open(topology_path).convert("L")
    topo_arr = np.asarray(topo_img, dtype=np.float32)
    topo_tensor = _normalize_array(topo_arr, dict(data_cfg.get("input_metadata", {})))
    topo_tensor = torch.from_numpy(topo_tensor).unsqueeze(0)
    topo_tensor = TF.resize(topo_tensor, [image_size, image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)

    channels = [topo_tensor]
    if data_cfg.get("los_input_column"):
        if not los_path:
            raise ValueError("Config expects LoS input; provide --los-input or use --hdf5-city/sample.")
        los_img = Image.open(los_path).convert("L")
        los_arr = np.asarray(los_img, dtype=np.float32) / 255.0
        los_tensor = torch.from_numpy(los_arr).unsqueeze(0)
        los_tensor = TF.resize(los_tensor, [image_size, image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        channels.append(los_tensor)

    if data_cfg.get("distance_map_channel", False):
        channels.append(_compute_distance_map_2d(image_size))

    if bool(cfg["model"].get("use_scalar_channels", False)):
        scalar_columns = list(data_cfg.get("scalar_feature_columns", []))
        scalar_norms = dict(data_cfg.get("scalar_feature_norms", {}))
        values = []
        for col in scalar_columns:
            raw_value = float(scalar_values.get(col, 0.0))
            norm = max(float(scalar_norms.get(col, 1.0)), 1e-12)
            values.append(raw_value / norm)
        if values:
            scalar_tensor = torch.tensor(values, dtype=torch.float32).view(len(values), 1, 1).expand(len(values), image_size, image_size)
            channels.append(scalar_tensor)

    return torch.cat(channels, dim=0).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with Try 44 PMNet-v3-style direct path-loss model")
    parser.add_argument("--config", type=str, default="experiments/fortyfourthtry44_pmnet_v3_no_prior/fortyfourthtry44_pmnet_v3_no_prior.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hdf5-city", type=str, default="")
    parser.add_argument("--hdf5-sample", type=str, default="")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--los-input", type=str, default="")
    parser.add_argument("--scalar-values", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    device = resolve_device(cfg["runtime"]["device"])
    target_meta = dict(cfg["target_metadata"]["path_loss"])

    model = PMNetResidualRegressor(
        in_channels=compute_input_channels(cfg),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        gradient_checkpointing=False,
    ).to(device)

    state = load_torch_checkpoint(args.checkpoint, device)
    model.load_state_dict(state["model"] if "model" in state else state["generator"])
    model.eval()

    if args.hdf5_city and args.hdf5_sample:
        model_input, target, mask, scalar_cond = build_model_input_from_dataset_sample(cfg, args.hdf5_city, args.hdf5_sample)
        scalar_cond = scalar_cond.unsqueeze(0).to(device) if scalar_cond is not None else None
    else:
        if not args.input:
            raise ValueError("Provide either --hdf5-city/--hdf5-sample or --input.")
        model_input = build_model_input_from_images(
            cfg,
            topology_path=args.input,
            los_path=args.los_input or None,
            scalar_values=parse_scalar_values(args.scalar_values),
        )
        target = None
        mask = None
        scalar_cond = None

    model_input = model_input.to(device)
    with torch.no_grad():
        pred = model(model_input)
        pred = clip_to_target_range(pred, target_meta)

    pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
    pred_db = denormalize(pred_np, target_meta)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(to_uint8(pred_db)).save(out_dir / "prediction_path_loss.png")
        np.save(out_dir / "prediction_path_loss_db.npy", pred_db)
        if target is not None:
            target_db = denormalize(target.squeeze(0).cpu().numpy(), target_meta)
            Image.fromarray(to_uint8(target_db)).save(out_dir / "target_path_loss.png")
        if mask is not None:
            mask_np = (mask.squeeze(0).cpu().numpy() > 0).astype(np.uint8) * 255
            Image.fromarray(mask_np).save(out_dir / "valid_mask.png")
        print(f"Saved outputs to {out_dir}")
    else:
        print(
            {
                "prediction_min_db": float(np.nanmin(pred_db)),
                "prediction_max_db": float(np.nanmax(pred_db)),
                "prediction_mean_db": float(np.nanmean(pred_db)),
            }
        )


if __name__ == "__main__":
    main()
