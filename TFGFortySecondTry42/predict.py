from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from config_utils import anchor_data_paths_to_config_file, load_config, load_torch_checkpoint, resolve_device
from data_utils import (
    _apply_formula_regime_calibration,
    _compute_distance_map_2d,
    _compute_formula_path_loss_db,
    _normalize_array,
    _normalize_channel,
    build_dataset_splits_from_config,
    compute_input_channels,
)
from model_pmnet import PMNetResidualRegressor


def formula_channel_index(cfg: Dict[str, Any]) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):
        idx += 1
    if cfg["data"].get("distance_map_channel", False):
        idx += 1
    if not bool(cfg["data"].get("path_loss_formula_input", {}).get("enabled", False)):
        raise ValueError("This predictor expects data.path_loss_formula_input.enabled = true")
    return idx


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


def load_calibration(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    rel = formula_cfg.get("regime_calibration_json")
    if not rel:
        return None
    path = Path(__file__).resolve().parent / str(rel)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    city_name: str,
) -> torch.Tensor:
    image_size = int(cfg["data"]["image_size"])
    data_cfg = dict(cfg["data"])
    formula_cfg = dict(data_cfg.get("path_loss_formula_input", {}))
    calibration = load_calibration(cfg)

    topo_img = Image.open(topology_path).convert("L")
    topo_arr = np.asarray(topo_img, dtype=np.float32)
    topo_tensor = _normalize_array(topo_arr, dict(data_cfg.get("input_metadata", {})))
    topo_tensor = torch.from_numpy(topo_tensor).unsqueeze(0)
    topo_tensor = TF.resize(
        topo_tensor,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    channels = [topo_tensor]
    los_tensor: Optional[torch.Tensor] = None
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

    if bool(formula_cfg.get("enabled", False)):
        antenna_height_m = float(scalar_values.get("antenna_height_m", 60.0))
        raw_topology = np.asarray(topo_img.resize((image_size, image_size), Image.BILINEAR), dtype=np.float32)
        non_ground_threshold = float(data_cfg.get("non_ground_threshold", 0.0))
        non_ground = raw_topology != non_ground_threshold
        building_density = float(np.mean(non_ground))
        non_zero = raw_topology[non_ground]
        mean_height = float(np.mean(non_zero)) if non_zero.size else 0.0
        clip_min = float(cfg["target_metadata"]["path_loss"].get("clip_min", 0.0))
        clip_max = float(cfg["target_metadata"]["path_loss"].get("clip_max", 180.0))
        prior_db = _compute_formula_path_loss_db(
            image_size=image_size,
            antenna_height_m=antenna_height_m,
            receiver_height_m=float(formula_cfg.get("receiver_height_m", 1.5)),
            frequency_ghz=float(formula_cfg.get("frequency_ghz", 7.125)),
            meters_per_pixel=float(formula_cfg.get("meters_per_pixel", 1.0)),
            formula_mode=str(formula_cfg.get("formula", "hybrid_two_ray_cost231")),
            los_tensor=los_tensor,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        prior_db = _apply_formula_regime_calibration(
            prior_db,
            calibration,
            city=city_name,
            density=building_density,
            height=mean_height,
            antenna_height_m=antenna_height_m,
            los_tensor=los_tensor,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        prior_norm = _normalize_channel(prior_db, dict(cfg["target_metadata"]["path_loss"]))
        channels.append(prior_norm)

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
    parser = argparse.ArgumentParser(description="Predict with Try 42 PMNet-style prior+residual path-loss model")
    parser.add_argument("--config", type=str, default="experiments/fortysecondtry42_pmnet_prior_residual/fortysecondtry42_pmnet_prior_residual.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hdf5-city", type=str, default="")
    parser.add_argument("--hdf5-sample", type=str, default="")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--los-input", type=str, default="")
    parser.add_argument("--scalar-values", type=str, default="")
    parser.add_argument("--city-name", type=str, default="unknown_city")
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
            city_name=args.city_name,
        )
        target = None
        mask = None
        scalar_cond = None

    model_input = model_input.to(device)
    with torch.no_grad():
        residual = model(model_input)
        prior = model_input[:, formula_channel_index(cfg) : formula_channel_index(cfg) + 1]
        pred = clip_to_target_range(prior + residual, target_meta)

    residual_np = residual.squeeze(0).squeeze(0).cpu().numpy()
    prior_np = prior.squeeze(0).squeeze(0).cpu().numpy()
    pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
    prior_db = denormalize(prior_np, target_meta)
    pred_db = denormalize(pred_np, target_meta)
    residual_db = denormalize(residual_np, {"scale": float(target_meta.get("scale", 1.0)), "offset": 0.0})

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(cfg["runtime"]["output_dir"]) / "predict_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "residual_raw.npy", residual_np)
    np.save(out_dir / "prior_raw.npy", prior_np)
    np.save(out_dir / "prediction_raw.npy", pred_np)
    np.save(out_dir / "residual_db.npy", residual_db)
    np.save(out_dir / "prior_db.npy", prior_db)
    np.save(out_dir / "prediction_db.npy", pred_db)

    Image.fromarray(to_uint8(prior_db), mode="L").save(out_dir / "prior_db.png")
    Image.fromarray(to_uint8(pred_db), mode="L").save(out_dir / "prediction_db.png")
    Image.fromarray(to_uint8(residual_db), mode="L").save(out_dir / "residual_db.png")

    if target is not None:
        target_np = target.squeeze(0).squeeze(0).cpu().numpy()
        mask_np = mask.squeeze(0).squeeze(0).cpu().numpy() if mask is not None else None
        target_db = denormalize(target_np, target_meta)
        np.save(out_dir / "target_db.npy", target_db)
        Image.fromarray(to_uint8(target_db), mode="L").save(out_dir / "target_db.png")
        if mask_np is not None:
            error_db = np.where(mask_np > 0, pred_db - target_db, np.nan)
            np.save(out_dir / "error_db.npy", error_db)

    print(f"Try 42 prediction outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
