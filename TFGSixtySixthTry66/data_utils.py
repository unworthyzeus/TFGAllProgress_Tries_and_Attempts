from __future__ import annotations

import json
import hashlib
import math
import os
import random
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _resolve_path(root_dir: Path, rel_path: str) -> Path:
    return root_dir / rel_path.replace('\\', '/').replace('\r', '').replace('\n', '')


def _augmentation_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    aug_cfg = dict(cfg.get('augmentation', {}))
    return {
        'augment': bool(aug_cfg.get('enable', False)),
        'hflip_prob': float(aug_cfg.get('hflip_prob', 0.5)),
        'vflip_prob': float(aug_cfg.get('vflip_prob', 0.5)),
        'rot90_prob': float(aug_cfg.get('rot90_prob', 0.4)),
    }


def _normalize_array(arr: np.ndarray, metadata: Optional[Dict[str, Any]]) -> np.ndarray:
    if metadata is None:
        return arr.astype(np.float32, copy=False)

    scale = float(metadata.get('scale', 1.0))
    offset = float(metadata.get('offset', 0.0))
    if abs(scale) < 1e-12:
        scale = 1.0
    return ((arr.astype(np.float32, copy=False) - offset) / scale).astype(np.float32, copy=False)


def _normalize_channel(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    scale = float(metadata.get('scale', 1.0))
    offset = float(metadata.get('offset', 0.0))
    if abs(scale) < 1e-12:
        scale = 1.0
    return (values - offset) / scale


def _build_prior_confidence_channel(
    los_tensor: Optional[torch.Tensor],
    distance_tensor: torch.Tensor,
    topology_tensor: torch.Tensor,
    *,
    non_ground_threshold: float,
    kernel_size: int = 31,
) -> torch.Tensor:
    """Build a heuristic confidence map for the formula prior in [0, 1]."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    if los_tensor is None:
        los_prob = torch.full_like(distance_tensor, 0.5)
    else:
        los_prob = los_tensor.clamp(0.0, 1.0)

    distance_norm = distance_tensor.clamp(0.0, 1.0)
    blocker_mask = (topology_tensor != float(non_ground_threshold)).to(dtype=torch.float32)
    local_blocker = F.avg_pool2d(
        blocker_mask.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).squeeze(0).clamp(0.0, 1.0)

    # Confidence is lower where local blockage is dense and farther ranges dominate.
    conf_los = (1.0 - (0.35 * distance_norm + 0.45 * local_blocker)).clamp(0.05, 1.0)
    conf_nlos = (1.0 - (0.55 * local_blocker + 0.35 * distance_norm)).clamp(0.05, 1.0)
    return (los_prob * conf_los + (1.0 - los_prob) * conf_nlos).clamp(0.0, 1.0)


def _resize_array(arr: np.ndarray, image_size: int, metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    normalized = _normalize_array(np.asarray(arr), metadata)
    if normalized.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {normalized.shape}")
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    return TF.resize(
        tensor,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def _path_loss_db_to_linear_normalized(arr: np.ndarray, image_size: int) -> torch.Tensor:
    """Convert path loss dB to linear scale, normalize to [0,1] via log scale."""
    arr = np.asarray(arr, dtype=np.float32)
    linear = np.power(10.0, -arr / 10.0)
    linear = np.clip(linear, 1e-18, 1.0)
    log_linear = np.log10(linear)
    normalized = (log_linear + 18.0) / 18.0
    tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
    return TF.resize(
        tensor,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def _formula_cache_pack(tensor: torch.Tensor, dtype_name: str) -> dict[str, Any]:
    data = tensor.detach().cpu().contiguous()
    kind = str(dtype_name).lower()
    if kind == 'uint8':
        data = torch.clamp(torch.round(data.to(dtype=torch.float32) * 255.0), 0.0, 255.0).to(dtype=torch.uint8)
    elif kind == 'float16':
        data = data.to(dtype=torch.float16)
    elif kind in {'float32', 'fp32'}:
        data = data.to(dtype=torch.float32)
        kind = 'float32'
    else:
        data = torch.clamp(torch.round(data.to(dtype=torch.float32) * 255.0), 0.0, 255.0).to(dtype=torch.uint8)
        kind = 'uint8'
    return {'tensor': data, 'cache_dtype': kind}


def _formula_cache_unpack(payload: Any) -> torch.Tensor:
    kind = payload.get('cache_dtype') if isinstance(payload, dict) else None
    tensor = payload['tensor'] if isinstance(payload, dict) and 'tensor' in payload else payload
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if str(kind).lower() == 'uint8' or tensor.dtype == torch.uint8:
        return tensor.to(dtype=torch.float32) / 255.0
    return tensor.to(dtype=torch.float32)


def _formula_hdf5_unpack(array: np.ndarray, dataset_name: str, cache_dtype: Optional[str] = None) -> torch.Tensor:
    kind = str(cache_dtype or "").lower().strip()
    if not kind:
        if "u8" in dataset_name.lower():
            kind = "uint8"
        elif "f16" in dataset_name.lower():
            kind = "float16"
        else:
            kind = str(array.dtype)
    if kind == "uint8" or np.asarray(array).dtype == np.uint8:
        tensor = torch.from_numpy(np.asarray(array, dtype=np.uint8).astype(np.float32) / 255.0)
    elif kind in {"float16", "f16"}:
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float16).astype(np.float32))
    else:
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def path_loss_linear_normalized_to_db(tensor: torch.Tensor) -> torch.Tensor:
    """Convert normalized linear path loss back to dB."""
    # Metrics and confidence targets must avoid fp16 underflow near 1e-18.
    working = tensor.to(dtype=torch.float32)
    normalized = working.clamp(0.0, 1.0)
    log_linear = normalized * 18.0 - 18.0
    linear = torch.pow(10.0, log_linear).clamp(min=1e-18)
    return -10.0 * torch.log10(linear)


def _compute_scalar_norms(
    scalar_feature_columns: Sequence[str],
    constant_scalar_features: Dict[str, float],
    scalar_feature_norms: Dict[str, float],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    for col in scalar_feature_columns:
        if col in scalar_feature_norms:
            norms[col] = max(float(scalar_feature_norms[col]), 1.0)
        elif df is not None and col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            max_abs = float(values.abs().max()) if not values.empty else 1.0
            norms[col] = max(max_abs, 1.0)
        else:
            norms[col] = 1.0

    for col, value in constant_scalar_features.items():
        if col in scalar_feature_norms:
            norms[col] = max(float(scalar_feature_norms[col]), 1.0)
        else:
            norms[col] = max(abs(float(value)), 1.0)

    return norms


def _compute_distance_map_2d(image_size: int) -> torch.Tensor:
    """2D horizontal distance from map center (antenna at center). Normalized to [0, 1]."""
    half = (image_size - 1) / 2.0
    y = torch.arange(image_size, dtype=torch.float32) - half
    x = torch.arange(image_size, dtype=torch.float32) - half
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt(xx ** 2 + yy ** 2)
    max_dist = 256.0 * (2.0 ** 0.5)
    normalized = (dist / max_dist).clamp(0.0, 1.0)
    return normalized.unsqueeze(0)


def _fspl_db(d3d_m: torch.Tensor, freq_mhz: float) -> torch.Tensor:
    return 32.45 + 20.0 * torch.log10(d3d_m / 1000.0) + 20.0 * math.log10(freq_mhz)


def _elevation_angle_deg(d2d_m: torch.Tensor, h_tx: float, h_rx: float) -> torch.Tensor:
    return torch.rad2deg(
        torch.atan2(torch.full_like(d2d_m, h_tx - h_rx), d2d_m.clamp(min=1.0))
    ).clamp(0.1, 89.9)


def _fresnel_reflection_coefficient(
    d2d_m: torch.Tensor,
    h_tx: float,
    h_rx: float,
    eps_r: float,
) -> torch.Tensor:
    d_ref = torch.sqrt(d2d_m ** 2 + (h_tx + h_rx) ** 2).clamp(min=1.0)
    cos_i = ((h_tx + h_rx) / d_ref).clamp(1e-4, 1.0)
    sin_i_sq = (1.0 - cos_i * cos_i).clamp(0.0, 1.0)
    root_term = torch.sqrt((eps_r - sin_i_sq).clamp(min=1e-4))
    gamma_h = (cos_i - root_term) / (cos_i + root_term + 1e-8)
    gamma_v = ((eps_r * cos_i) - root_term) / ((eps_r * cos_i) + root_term + 1e-8)
    gamma = 0.5 * (gamma_h + gamma_v)
    return gamma.clamp(-0.95, 0.95)


def _coherent_two_ray_components_db(
    d2d_m: torch.Tensor,
    h_tx: float,
    h_rx: float,
    freq_ghz: float,
    eps_r: float = 5.0,
    roughness_m: float = 0.0,
) -> torch.Tensor:
    wavelength_m = 0.299792458 / max(freq_ghz, 0.1)
    d_los = torch.sqrt(d2d_m ** 2 + (h_tx - h_rx) ** 2).clamp(min=1.0)
    d_ref = torch.sqrt(d2d_m ** 2 + (h_tx + h_rx) ** 2).clamp(min=1.0)
    gamma = _fresnel_reflection_coefficient(d2d_m, h_tx, h_rx, eps_r=eps_r)
    if roughness_m > 0.0:
        cos_i = ((h_tx + h_rx) / d_ref).clamp(1e-4, 1.0)
        roughness_factor = torch.exp(-((4.0 * math.pi * roughness_m * cos_i) / wavelength_m) ** 2)
        gamma = gamma * roughness_factor
    k = (2.0 * math.pi) / wavelength_m
    delta = d_ref - d_los

    e_los_real = 1.0 / d_los
    e_los_imag = torch.zeros_like(d_los)
    e_ref_real = gamma * torch.cos(k * delta) / d_ref
    e_ref_imag = -gamma * torch.sin(k * delta) / d_ref

    power_rel = ((wavelength_m / (4.0 * math.pi)) ** 2) * (
        (e_los_real + e_ref_real) ** 2 + (e_los_imag + e_ref_imag) ** 2
    )
    coherent_db = -10.0 * torch.log10(power_rel.clamp(min=1e-18))
    fspl_direct_db = _fspl_db(d_los, freq_ghz * 1000.0)
    return coherent_db, fspl_direct_db


def _coherent_two_ray_path_loss_db(
    d2d_m: torch.Tensor,
    h_tx: float,
    h_rx: float,
    freq_ghz: float,
    eps_r: float = 5.0,
    roughness_m: float = 0.0,
) -> torch.Tensor:
    coherent_db, fspl_direct_db = _coherent_two_ray_components_db(
        d2d_m,
        h_tx,
        h_rx,
        freq_ghz,
        eps_r=eps_r,
        roughness_m=roughness_m,
    )

    # Preserve the interference structure while preventing unrealistically deep notches.
    return torch.maximum(coherent_db, fspl_direct_db - 6.0)


def _damped_coherent_two_ray_path_loss_db(
    d2d_m: torch.Tensor,
    h_tx: float,
    h_rx: float,
    freq_ghz: float,
    eps_r: float = 5.0,
    roughness_m: float = 0.02,
    excess_limit_db: float = 7.5,
    interference_decay_m: float = 450.0,
    min_interference_blend: float = 0.2,
) -> torch.Tensor:
    coherent_db, fspl_direct_db = _coherent_two_ray_components_db(
        d2d_m,
        h_tx,
        h_rx,
        freq_ghz,
        eps_r=eps_r,
        roughness_m=roughness_m,
    )
    excess_limit_db = max(float(excess_limit_db), 1e-3)
    excess_db = coherent_db - fspl_direct_db
    limited_excess_db = excess_limit_db * torch.tanh(excess_db / excess_limit_db)
    if interference_decay_m > 0.0:
        decay = torch.exp(-d2d_m / float(interference_decay_m))
    else:
        decay = torch.ones_like(d2d_m)
    blend = float(min_interference_blend) + (1.0 - float(min_interference_blend)) * decay
    blend = blend.clamp(0.0, 1.0)
    return fspl_direct_db + blend * limited_excess_db


_RAY_PROXY_CACHE: Dict[Tuple[int, int], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}


def _get_angle_bin_rays(image_size: int, angle_bins: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    key = (int(image_size), int(angle_bins))
    cached = _RAY_PROXY_CACHE.get(key)
    if cached is not None:
        return cached

    center = (float(image_size - 1) / 2.0, float(image_size - 1) / 2.0)
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    dy = yy - center[0]
    dx = xx - center[1]
    radius = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    angles = (np.arctan2(dy, dx) + math.pi) / (2.0 * math.pi)
    bin_ids = np.floor(angles * int(angle_bins)).astype(np.int32)
    bin_ids = np.clip(bin_ids, 0, int(angle_bins) - 1)

    rays: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for bin_idx in range(int(angle_bins)):
        ys, xs = np.where(bin_ids == bin_idx)
        if ys.size == 0:
            continue
        rr = radius[ys, xs]
        order = np.argsort(rr, kind="stable")
        rays.append((ys[order], xs[order], rr[order]))

    _RAY_PROXY_CACHE[key] = rays
    return rays


def _compute_ray_obstruction_proxy_features(
    raw_topology: np.ndarray,
    los_tensor: Optional[torch.Tensor],
    *,
    non_ground_threshold: float,
    meters_per_pixel: float,
    angle_bins: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw_topology.ndim != 2:
        raise ValueError(f"Expected raw_topology [H,W], got {tuple(raw_topology.shape)}")

    image_size = int(raw_topology.shape[0])
    rays = _get_angle_bin_rays(image_size, angle_bins)
    raw_topology = np.asarray(raw_topology, dtype=np.float32)
    building_mask = raw_topology != float(non_ground_threshold)
    max_height = float(np.max(raw_topology)) if raw_topology.size else 1.0
    max_height = max(max_height, 1.0)
    max_radius_px = max((image_size - 1) / 2.0 * math.sqrt(2.0), 1.0)
    max_radius_m = max_radius_px * max(float(meters_per_pixel), 1e-6)

    if los_tensor is None:
        los_map = np.ones_like(raw_topology, dtype=np.float32)
    else:
        los_map = np.asarray(los_tensor.squeeze(0).cpu(), dtype=np.float32)

    shadow_depth = np.zeros_like(raw_topology, dtype=np.float32)
    distance_since_los = np.zeros_like(raw_topology, dtype=np.float32)
    blocker_height = np.zeros_like(raw_topology, dtype=np.float32)
    blocker_count = np.zeros_like(raw_topology, dtype=np.float32)

    for ys, xs, rr_px in rays:
        rr_m = rr_px * float(meters_per_pixel)
        los_line = los_map[ys, xs] > 0.5
        building_line = building_mask[ys, xs]
        heights = np.where(building_line, raw_topology[ys, xs], 0.0).astype(np.float32)

        last_los_r = 0.0
        max_block = 0.0
        block_segments = 0
        prev_building = False
        for i in range(len(rr_m)):
            if building_line[i] and not prev_building:
                block_segments += 1
            prev_building = bool(building_line[i])
            max_block = max(max_block, float(heights[i]))

            if los_line[i]:
                last_los_r = float(rr_m[i])
                continue

            dist_since = max(float(rr_m[i]) - last_los_r, 0.0)
            shadow_severity = dist_since * (1.0 + max_block / max_height)
            shadow_depth[ys[i], xs[i]] = float(np.clip(shadow_severity / max(max_radius_m * 2.0, 1.0), 0.0, 1.0))
            distance_since_los[ys[i], xs[i]] = float(np.clip(dist_since / max(max_radius_m, 1.0), 0.0, 1.0))
            blocker_height[ys[i], xs[i]] = float(np.clip(max_block / max_height, 0.0, 1.0))
            blocker_count[ys[i], xs[i]] = float(np.clip(block_segments / 8.0, 0.0, 1.0))

    return (
        torch.from_numpy(shadow_depth).unsqueeze(0),
        torch.from_numpy(distance_since_los).unsqueeze(0),
        torch.from_numpy(blocker_height).unsqueeze(0),
        torch.from_numpy(blocker_count).unsqueeze(0),
    )


def _infer_city_type_simple(density: float, height: float) -> str:
    if density >= 0.34 or height >= 90.0:
        return "dense_highrise"
    if density <= 0.18 and height <= 30.0:
        return "open_lowrise"
    return "mixed_midrise"


def _nlos_shadow_sigma_proxy_db(d2d_m: torch.Tensor, h_tx: float, h_rx: float) -> torch.Tensor:
    theta_deg = _elevation_angle_deg(d2d_m, h_tx, h_rx)
    return 2.3197 * torch.pow((90.0 - theta_deg).clamp(min=1e-3), 0.2361)


def _vinogradov_height_nlos_mean_path_loss_db(
    d3d_m: torch.Tensor,
    h_tx: float,
    freq_ghz: float,
    city_type: str,
) -> torch.Tensor:
    freq_mhz = max(float(freq_ghz), 0.1) * 1000.0
    lambda0_db = 32.45 + 20.0 * math.log10(freq_mhz) - 60.0

    # Coarse, structurally motivated parameters from the Vinogradov/Saboor
    # height-dependent A2G LSF family. These are intentionally low-precision:
    # the gain should come from the model form, not fine coefficient fitting.
    params = {
        "open_lowrise": {"n0": 4.6, "n_inf": 2.85, "h0": 25.0},
        "mixed_midrise": {"n0": 5.0, "n_inf": 2.9, "h0": 40.0},
        "dense_highrise": {"n0": 5.4, "n_inf": 3.0, "h0": 120.0},
    }
    p = params.get(city_type, params["mixed_midrise"])
    n_h = p["n_inf"] + (p["n0"] - p["n_inf"]) * math.exp(-max(h_tx, 1.0) / p["h0"])
    return lambda0_db + 10.0 * n_h * torch.log10(d3d_m.clamp(min=1.0))


def _vinogradov_height_nlos_mean_path_loss_db_relaxed(
    d3d_m: torch.Tensor,
    h_tx: float,
    freq_ghz: float,
    city_type: str,
) -> torch.Tensor:
    freq_mhz = max(float(freq_ghz), 0.1) * 1000.0
    lambda0_db = 32.45 + 20.0 * math.log10(freq_mhz) - 60.0

    # Relaxed variant for the visual Try 48 refinement: preserve the
    # height-dependent family but back off the large-scale NLoS slope so the
    # deterministic shadow term carries the local severity instead of lifting
    # the whole NLoS map into saturation.
    params = {
        "open_lowrise": {"n0": 4.15, "n_inf": 2.65, "h0": 35.0},
        "mixed_midrise": {"n0": 4.45, "n_inf": 2.75, "h0": 55.0},
        "dense_highrise": {"n0": 4.70, "n_inf": 2.85, "h0": 95.0},
    }
    p = params.get(city_type, params["mixed_midrise"])
    n_h = p["n_inf"] + (p["n0"] - p["n_inf"]) * math.exp(-max(h_tx, 1.0) / p["h0"])
    return lambda0_db + 10.0 * n_h * torch.log10(d3d_m.clamp(min=1.0))


def _vinogradov_height_nlos_path_loss_db(
    d3d_m: torch.Tensor,
    d2d_m: torch.Tensor,
    h_tx: float,
    h_rx: float,
    freq_ghz: float,
    city_type: str,
) -> torch.Tensor:
    sigma_theta = _nlos_shadow_sigma_proxy_db(d2d_m, h_tx, h_rx)
    base_nlos = _vinogradov_height_nlos_mean_path_loss_db(d3d_m, h_tx, freq_ghz, city_type)
    # Use sigma as a severity proxy, not as a random term.
    return base_nlos + 0.35 * sigma_theta


def _obstruction_local_features(topology_norm: torch.Tensor, kernel_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    topo = topology_norm.to(dtype=torch.float32)
    if topo.ndim == 3:
        topo = topo.squeeze(0)
    if topo.ndim != 2:
        raise ValueError(f"Expected topology [H,W], got {tuple(topo.shape)}")
    topo4 = topo.unsqueeze(0).unsqueeze(0)
    building = (topo4 > 0.0).to(dtype=torch.float32)
    kernel = max(int(kernel_size), 1)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    density = F.avg_pool2d(building, kernel, stride=1, padding=pad).squeeze(0).squeeze(0)
    mean_height = F.avg_pool2d(topo4 * 255.0, kernel, stride=1, padding=pad).squeeze(0).squeeze(0)
    return density, mean_height


def _sigma_map(distance_norm: torch.Tensor, antenna_height_m: float, los_map: torch.Tensor, receiver_height_m: float = 1.5) -> torch.Tensor:
    dist = distance_norm.squeeze(0) if distance_norm.ndim == 3 else distance_norm
    los = los_map.squeeze(0) if los_map.ndim == 3 else los_map
    distance_scale_m = 256.0 * math.sqrt(2.0)
    ground_distance_m = dist * distance_scale_m
    theta_deg = torch.rad2deg(
        torch.atan2((antenna_height_m - receiver_height_m) * torch.ones_like(ground_distance_m), ground_distance_m.clamp(min=1.0))
    )
    sigma_los = 0.0272 * torch.pow((90.0 - theta_deg).clamp(min=0.0), 0.7475)
    sigma_nlos = 2.3197 * torch.pow((90.0 - theta_deg).clamp(min=0.0), 0.2361)
    return torch.where(los > 0.5, sigma_los, sigma_nlos)


def _compute_local_obstruction_features(
    topology_tensor: torch.Tensor,
    *,
    non_ground_threshold: float,
    kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if topology_tensor.ndim != 3:
        raise ValueError(f"Expected topology tensor [1,H,W], got {tuple(topology_tensor.shape)}")
    building = (topology_tensor != float(non_ground_threshold)).to(dtype=torch.float32)
    height = topology_tensor.to(dtype=torch.float32)
    kernel = max(int(kernel_size), 1)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    local_density = F.avg_pool2d(building.unsqueeze(0), kernel_size=kernel, stride=1, padding=pad).squeeze(0)
    local_height = F.avg_pool2d(height.unsqueeze(0), kernel_size=kernel, stride=1, padding=pad).squeeze(0)
    return local_density.clamp(0.0, 1.0), local_height.clamp(min=0.0)


def _compute_a2g_shadow_sigma_db(
    image_size: int,
    antenna_height_m: float,
    receiver_height_m: float,
    meters_per_pixel: float,
    *,
    los_tensor: Optional[torch.Tensor] = None,
    a2g_params: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    distance_norm = _compute_distance_map_2d(image_size)
    if los_tensor is not None:
        distance_norm = distance_norm.to(device=los_tensor.device, dtype=los_tensor.dtype)
    half = (image_size - 1) / 2.0
    max_dist_pixels = max(half * (2.0 ** 0.5), 1.0)
    ground_distance_m = distance_norm * float(max_dist_pixels * meters_per_pixel)
    h_tx = max(float(antenna_height_m), 1.0)
    h_rx = max(float(receiver_height_m), 0.5)
    theta_deg = torch.rad2deg(
        torch.atan2((h_tx - h_rx) * torch.ones_like(ground_distance_m), ground_distance_m.clamp(min=1.0))
    )
    a2g_params = dict(a2g_params or {})
    sigma_los_rho = float(a2g_params.get("sigma_los_rho", 0.0272))
    sigma_los_mu = float(a2g_params.get("sigma_los_mu", 0.7475))
    sigma_nlos_rho = float(a2g_params.get("sigma_nlos_rho", 2.3197))
    sigma_nlos_mu = float(a2g_params.get("sigma_nlos_mu", 0.2361))
    sigma_los = sigma_los_rho * torch.pow((90.0 - theta_deg).clamp(min=0.0), sigma_los_mu)
    sigma_nlos = sigma_nlos_rho * torch.pow((90.0 - theta_deg).clamp(min=0.0), sigma_nlos_mu)
    if los_tensor is None:
        return 0.5 * (sigma_los + sigma_nlos)
    los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
    return los_prob * sigma_los + (1.0 - los_prob) * sigma_nlos


def _theta_norm_map(distance_norm: torch.Tensor, antenna_height_m: float, receiver_height_m: float = 1.5) -> torch.Tensor:
    dist = distance_norm.squeeze(0) if distance_norm.ndim == 3 else distance_norm
    distance_scale_m = 256.0 * math.sqrt(2.0)
    ground_distance_m = dist * distance_scale_m
    theta_deg = torch.rad2deg(
        torch.atan2((antenna_height_m - receiver_height_m) * torch.ones_like(ground_distance_m), ground_distance_m.clamp(min=1.0))
    )
    return (theta_deg / 90.0).clamp(0.0, 1.0)


def _obstruction_feature_matrix(
    prior_db: torch.Tensor,
    distance_norm: torch.Tensor,
    local_density_small: torch.Tensor,
    local_density_large: torch.Tensor,
    local_height_small: torch.Tensor,
    local_height_large: torch.Tensor,
    nlos_support_small: torch.Tensor,
    nlos_support_large: torch.Tensor,
    shadow_sigma_db: torch.Tensor,
    theta_norm: torch.Tensor,
) -> torch.Tensor:
    prior_work = prior_db.squeeze(0) if prior_db.ndim == 3 else prior_db
    dist_work = distance_norm.squeeze(0) if distance_norm.ndim == 3 else distance_norm
    distance_scale_m = 256.0 * math.sqrt(2.0)
    logd = torch.log1p(dist_work * distance_scale_m)
    return torch.stack(
        [
            prior_work * prior_work,
            prior_work,
            logd,
            local_density_small,
            local_density_large,
            local_height_small / 255.0,
            local_height_large / 255.0,
            local_density_large * logd,
            nlos_support_small,
            nlos_support_large,
            nlos_support_large * logd,
            shadow_sigma_db,
            theta_norm,
            nlos_support_large * theta_norm,
        ],
        dim=-1,
    )


def _formula_cache_signature(
    image_size: int,
    formula_cfg: Dict[str, Any],
    path_meta: Dict[str, Any],
    non_ground_threshold: float,
) -> str:
    payload = {
        "image_size": int(image_size),
        "formula_cfg": formula_cfg,
        "path_meta": path_meta,
        "non_ground_threshold": float(non_ground_threshold),
        "formula_cache_version": str(formula_cfg.get("cache_version", "")),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _meters_to_odd_kernel(window_m: float, meters_per_pixel: float, max_kernel: int = 129) -> int:
    px = max(int(round(float(window_m) / max(float(meters_per_pixel), 1e-3))), 1)
    if px % 2 == 0:
        px += 1
    max_kernel = max(int(max_kernel), 3)
    if max_kernel % 2 == 0:
        max_kernel -= 1
    return min(px, max_kernel)


def _avg_pool_same(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pooled = F.avg_pool2d(
        x.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )
    return pooled.squeeze(0)


def _max_pool_same(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pooled = F.max_pool2d(
        x.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return pooled.squeeze(0)


@lru_cache(maxsize=8)
def _radial_ray_cache(image_size: int, angle_bins: int) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    half = (image_size - 1) / 2.0
    yy, xx = np.meshgrid(
        np.arange(image_size, dtype=np.float32) - half,
        np.arange(image_size, dtype=np.float32) - half,
        indexing="ij",
    )
    radii = np.sqrt(xx * xx + yy * yy).reshape(-1)
    angles = np.arctan2(yy, xx).reshape(-1)
    bins = np.floor(((angles + math.pi) / (2.0 * math.pi)) * float(angle_bins)).astype(np.int32)
    bins = np.clip(bins, 0, max(int(angle_bins) - 1, 0))

    ray_orders: List[np.ndarray] = []
    ray_radii: List[np.ndarray] = []
    for b in range(int(angle_bins)):
        idx = np.flatnonzero(bins == b)
        if idx.size == 0:
            continue
        order = idx[np.argsort(radii[idx], kind="stable")].astype(np.int32, copy=False)
        ray_orders.append(order)
        ray_radii.append(radii[order].astype(np.float32, copy=False))
    return tuple(ray_orders), tuple(ray_radii)


def _compute_radial_shadow_depth_m(
    los_tensor: Optional[torch.Tensor],
    meters_per_pixel: float,
    angle_bins: int = 720,
) -> torch.Tensor:
    if los_tensor is None:
        raise ValueError("los_tensor is required for radial shadow depth computation")

    los = los_tensor.to(dtype=torch.float32)
    if los.ndim == 3:
        los = los.squeeze(0)
    los_np = (los.detach().cpu().numpy() > 0.5)
    ray_orders, ray_radii = _radial_ray_cache(int(los_np.shape[-1]), max(int(angle_bins), 90))
    los_flat = los_np.reshape(-1)
    depth_flat = np.zeros_like(los_flat, dtype=np.float32)

    for order, radii in zip(ray_orders, ray_radii):
        los_sorted = los_flat[order]
        last_los_radius = np.maximum.accumulate(np.where(los_sorted, radii, 0.0)).astype(np.float32, copy=False)
        nlos_sorted = ~los_sorted
        depth_flat[order] = np.where(
            nlos_sorted,
            np.maximum(radii - last_los_radius, 0.0),
            0.0,
        ).astype(np.float32, copy=False)

    depth = torch.from_numpy(depth_flat.reshape(los_np.shape)) * float(max(meters_per_pixel, 1e-3))
    return depth.to(device=los_tensor.device, dtype=torch.float32)


def _compute_shadowed_ripple_db(
    coherent_db: torch.Tensor,
    smooth_reference_db: torch.Tensor,
    los_tensor: Optional[torch.Tensor],
    shadow_depth_m: Optional[torch.Tensor],
    near_occ: Optional[torch.Tensor],
    a2g_params: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    params = dict(a2g_params or {})
    ripple_limit_db = max(float(params.get("ripple_limit_db", 10.0)), 1e-3)
    ripple_raw_db = coherent_db - smooth_reference_db
    ripple_db = ripple_limit_db * torch.tanh(ripple_raw_db / ripple_limit_db)

    if los_tensor is None:
        return float(params.get("ripple_gain_los", 0.95)) * ripple_db

    los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
    nlos_prob = 1.0 - los_prob
    if shadow_depth_m is None:
        shadow_decay = torch.ones_like(ripple_db)
    else:
        decay_m = max(float(params.get("ripple_shadow_decay_m", 120.0)), 1e-3)
        shadow_decay = torch.exp(-shadow_depth_m / decay_m)
    openness = torch.ones_like(ripple_db) if near_occ is None else (1.0 - near_occ).clamp(0.0, 1.0)
    los_gain = float(params.get("ripple_gain_los", 0.95))
    nlos_gain = float(params.get("ripple_gain_nlos", 0.38))
    deep_floor = float(params.get("ripple_deep_floor", 0.08))
    nlos_weight = (deep_floor + nlos_gain * shadow_decay * (0.40 + 0.60 * openness)).clamp(0.0, 1.0)
    return (los_gain * los_prob + nlos_weight * nlos_prob) * ripple_db


def _compute_structural_nlos_obstruction_db(
    d2d_m: torch.Tensor,
    topology_tensor: Optional[torch.Tensor],
    los_tensor: Optional[torch.Tensor],
    *,
    h_tx: float,
    h_rx: float,
    city_type: str,
    meters_per_pixel: float,
    non_ground_threshold: float,
    a2g_params: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    params = dict(a2g_params or {})
    sigma_theta = _nlos_shadow_sigma_proxy_db(d2d_m, h_tx, h_rx)
    nlos_prob = 1.0 - los_tensor.clamp(0.0, 1.0) if los_tensor is not None else torch.ones_like(d2d_m)
    if topology_tensor is None:
        return nlos_prob * (0.35 * sigma_theta)

    topo = topology_tensor.to(dtype=torch.float32)
    if topo.ndim == 2:
        topo = topo.unsqueeze(0)
    building_mask = (topo > float(non_ground_threshold)).float()
    if float(torch.amax(building_mask).item()) <= 0.0:
        return nlos_prob * (0.35 * sigma_theta)

    topo_height = topo.clamp(min=0.0) * building_mask
    height_scale = max(float(torch.amax(topo_height).item()), 1.0)
    topo_height_norm = (topo_height / height_scale).clamp(0.0, 1.0)

    near_kernel = _meters_to_odd_kernel(float(params.get("obstruction_near_window_m", 15.0)), meters_per_pixel)
    far_kernel = _meters_to_odd_kernel(float(params.get("obstruction_far_window_m", 45.0)), meters_per_pixel)
    shadow_kernel = _meters_to_odd_kernel(float(params.get("shadow_support_window_m", 31.0)), meters_per_pixel)

    near_occ = _avg_pool_same(building_mask, near_kernel)
    far_occ = _avg_pool_same(building_mask, far_kernel)
    near_height = _avg_pool_same(topo_height_norm, near_kernel)
    far_height = _avg_pool_same(topo_height_norm, far_kernel)
    shadow_support = _avg_pool_same(nlos_prob, shadow_kernel)

    max_distance = max(float(torch.amax(d2d_m).item()), 1.0)
    distance_norm = (d2d_m / max_distance).clamp(0.0, 1.0)
    shadow_depth = nlos_prob * (0.45 * shadow_support + 0.55 * distance_norm)

    severity_score = (
        0.30 * near_occ
        + 0.18 * far_occ
        + 0.20 * near_height
        + 0.12 * far_height
        + 0.20 * shadow_support
    ).clamp(0.0, 1.5)

    city_params = {
        "open_lowrise": {"bias": 3.0, "scale": 7.5},
        "mixed_midrise": {"bias": 5.0, "scale": 10.0},
        "dense_highrise": {"bias": 7.0, "scale": 13.0},
    }
    cp = city_params.get(city_type, city_params["mixed_midrise"])
    sigma_weight = float(params.get("shadow_sigma_weight", 0.30))
    depth_weight = float(params.get("shadow_depth_weight", 4.0))
    max_obstruction_db = float(params.get("nlos_max_obstruction_db", 24.0))

    obstruction_db = (
        float(cp["bias"]) * nlos_prob
        + float(cp["scale"]) * severity_score * nlos_prob
        + sigma_weight * sigma_theta * nlos_prob
        + depth_weight * shadow_depth
    )
    return obstruction_db.clamp(min=0.0, max=max_obstruction_db)


def _compute_deepshadow_nlos_obstruction_db(
    d2d_m: torch.Tensor,
    topology_tensor: Optional[torch.Tensor],
    los_tensor: Optional[torch.Tensor],
    *,
    h_tx: float,
    h_rx: float,
    city_type: str,
    meters_per_pixel: float,
    non_ground_threshold: float,
    a2g_params: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    params = dict(a2g_params or {})
    sigma_theta = _nlos_shadow_sigma_proxy_db(d2d_m, h_tx, h_rx)
    nlos_prob = 1.0 - los_tensor.clamp(0.0, 1.0) if los_tensor is not None else torch.ones_like(d2d_m)
    if topology_tensor is None:
        return nlos_prob * (0.55 * sigma_theta)

    topo = topology_tensor.to(dtype=torch.float32)
    if topo.ndim == 2:
        topo = topo.unsqueeze(0)
    building_mask = (topo > float(non_ground_threshold)).float()
    if float(torch.amax(building_mask).item()) <= 0.0:
        return nlos_prob * (0.55 * sigma_theta)

    topo_height = topo.clamp(min=0.0) * building_mask
    height_scale = max(float(torch.amax(topo_height).item()), 1.0)
    topo_height_norm = (topo_height / height_scale).clamp(0.0, 1.0)

    near_kernel = _meters_to_odd_kernel(float(params.get("obstruction_near_window_m", 15.0)), meters_per_pixel)
    far_kernel = _meters_to_odd_kernel(float(params.get("obstruction_far_window_m", 45.0)), meters_per_pixel)
    shadow_kernel = _meters_to_odd_kernel(float(params.get("shadow_support_window_m", 31.0)), meters_per_pixel)
    max_kernel = _meters_to_odd_kernel(float(params.get("obstruction_max_window_m", 21.0)), meters_per_pixel)

    near_occ = _avg_pool_same(building_mask, near_kernel)
    far_occ = _avg_pool_same(building_mask, far_kernel)
    near_height = _avg_pool_same(topo_height_norm, near_kernel)
    far_height = _avg_pool_same(topo_height_norm, far_kernel)
    local_max_height = _max_pool_same(topo_height_norm, max_kernel)
    shadow_support_near = _avg_pool_same(nlos_prob, near_kernel)
    shadow_support_far = _avg_pool_same(nlos_prob, shadow_kernel)

    max_distance = max(float(torch.amax(d2d_m).item()), 1.0)
    distance_norm = (d2d_m / max_distance).clamp(0.0, 1.0)
    shadow_depth = nlos_prob * (0.25 * shadow_support_near + 0.35 * shadow_support_far + 0.40 * distance_norm)
    blocker_prominence = torch.relu(local_max_height - far_height)

    shallow_score = (
        0.24 * near_occ
        + 0.16 * far_occ
        + 0.16 * near_height
        + 0.08 * far_height
        + 0.18 * shadow_support_near
        + 0.18 * shadow_support_far
    ).clamp(0.0, 1.5)
    deep_score = (
        0.20 * far_occ
        + 0.24 * local_max_height
        + 0.18 * blocker_prominence
        + 0.18 * shadow_support_far
        + 0.20 * distance_norm
    ).clamp(0.0, 1.5)

    shallow_gate = torch.sigmoid((shallow_score - float(params.get("shallow_shadow_threshold", 0.28))) / max(float(params.get("shadow_transition_temp", 0.07)), 1e-3))
    deep_gate = torch.sigmoid((deep_score - float(params.get("deep_shadow_threshold", 0.42))) / max(float(params.get("deep_shadow_temp", 0.06)), 1e-3))
    low_altitude_gate = math.exp(-max(h_tx, 1.0) / max(float(params.get("low_altitude_decay_m", 70.0)), 1e-3))

    city_params = {
        "open_lowrise": {"bias": 3.0, "scale": 8.0, "deep_boost": 4.0, "low_altitude_boost": 2.0},
        "mixed_midrise": {"bias": 6.0, "scale": 11.0, "deep_boost": 6.5, "low_altitude_boost": 3.5},
        "dense_highrise": {"bias": 9.0, "scale": 15.0, "deep_boost": 9.0, "low_altitude_boost": 5.0},
    }
    cp = city_params.get(city_type, city_params["mixed_midrise"])
    sigma_weight = float(params.get("shadow_sigma_weight", 0.55))
    depth_weight = float(params.get("shadow_depth_weight", 6.5))
    blocker_weight = float(params.get("blocker_prominence_weight", 8.0))
    max_obstruction_db = float(params.get("nlos_max_obstruction_db", 32.0))

    obstruction_db = (
        float(cp["bias"]) * nlos_prob
        + float(cp["scale"]) * shallow_score * nlos_prob
        + float(cp["deep_boost"]) * deep_gate * nlos_prob
        + float(cp["low_altitude_boost"]) * low_altitude_gate * (0.6 * shallow_gate + deep_gate) * nlos_prob
        + sigma_weight * sigma_theta * nlos_prob
        + depth_weight * shadow_depth
        + blocker_weight * blocker_prominence * nlos_prob
    )
    return obstruction_db.clamp(min=0.0, max=max_obstruction_db)


def _compute_directional_ripple_nlos_obstruction_db(
    d2d_m: torch.Tensor,
    topology_tensor: Optional[torch.Tensor],
    los_tensor: Optional[torch.Tensor],
    *,
    h_tx: float,
    h_rx: float,
    city_type: str,
    meters_per_pixel: float,
    non_ground_threshold: float,
    a2g_params: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    params = dict(a2g_params or {})
    sigma_theta = _nlos_shadow_sigma_proxy_db(d2d_m, h_tx, h_rx)
    nlos_prob = 1.0 - los_tensor.clamp(0.0, 1.0) if los_tensor is not None else torch.ones_like(d2d_m)
    shadow_depth_m = (
        _compute_radial_shadow_depth_m(
            los_tensor,
            meters_per_pixel,
            angle_bins=int(params.get("shadow_angle_bins", 720)),
        )
        if los_tensor is not None
        else None
    )
    if topology_tensor is None:
        base = nlos_prob * (0.28 * sigma_theta)
        return base, shadow_depth_m, None

    topo = topology_tensor.to(dtype=torch.float32)
    if topo.ndim == 2:
        topo = topo.unsqueeze(0)
    building_mask = (topo > float(non_ground_threshold)).float()
    if float(torch.amax(building_mask).item()) <= 0.0:
        base = nlos_prob * (0.28 * sigma_theta)
        return base, shadow_depth_m, torch.zeros_like(d2d_m)

    topo_height = topo.clamp(min=0.0) * building_mask
    height_scale = max(float(torch.amax(topo_height).item()), 1.0)
    topo_height_norm = (topo_height / height_scale).clamp(0.0, 1.0)

    near_kernel = _meters_to_odd_kernel(float(params.get("obstruction_near_window_m", 15.0)), meters_per_pixel)
    far_kernel = _meters_to_odd_kernel(float(params.get("obstruction_far_window_m", 45.0)), meters_per_pixel)
    support_kernel = _meters_to_odd_kernel(float(params.get("shadow_support_window_m", 31.0)), meters_per_pixel)
    max_kernel = _meters_to_odd_kernel(float(params.get("obstruction_max_window_m", 21.0)), meters_per_pixel)

    near_occ = _avg_pool_same(building_mask, near_kernel)
    far_occ = _avg_pool_same(building_mask, far_kernel)
    near_height = _avg_pool_same(topo_height_norm, near_kernel)
    far_height = _avg_pool_same(topo_height_norm, far_kernel)
    local_max_height = _max_pool_same(topo_height_norm, max_kernel)
    shadow_support = _avg_pool_same(nlos_prob, support_kernel)
    blocker_prominence = torch.relu(local_max_height - 0.5 * (near_height + far_height))
    openness = (1.0 - near_occ).clamp(0.0, 1.0)

    max_distance = max(float(torch.amax(d2d_m).item()), 1.0)
    distance_norm = (d2d_m / max_distance).clamp(0.0, 1.0)
    if shadow_depth_m is None:
        shadow_depth_norm = nlos_prob * distance_norm
    else:
        depth_scale_m = max(float(params.get("shadow_depth_scale_m", 42.0)), 1e-3)
        shadow_depth_norm = (shadow_depth_m / depth_scale_m).clamp(0.0, 2.5)

    shallow_gate = torch.sigmoid(
        (shadow_depth_norm - float(params.get("shallow_shadow_depth_norm", 0.22)))
        / max(float(params.get("shadow_transition_temp", 0.10)), 1e-3)
    )
    deep_gate = torch.sigmoid(
        (shadow_depth_norm - float(params.get("deep_shadow_depth_norm", 0.85)))
        / max(float(params.get("deep_shadow_temp", 0.14)), 1e-3)
    )
    low_altitude_gate = math.exp(-max(h_tx, 1.0) / max(float(params.get("low_altitude_decay_m", 110.0)), 1e-3))

    city_params = {
        "open_lowrise": {"bias": 1.5, "occ": 2.8, "depth": 3.8, "deep": 1.5, "sigma": 0.18},
        "mixed_midrise": {"bias": 2.2, "occ": 3.6, "depth": 4.8, "deep": 2.2, "sigma": 0.22},
        "dense_highrise": {"bias": 3.0, "occ": 4.8, "depth": 5.6, "deep": 3.1, "sigma": 0.26},
    }
    cp = city_params.get(city_type, city_params["mixed_midrise"])
    open_relief_weight = float(params.get("nlos_open_relief_weight", 4.5))
    prominence_weight = float(params.get("blocker_prominence_weight", 4.0))
    low_altitude_weight = float(params.get("low_altitude_boost_weight", 1.8))
    max_obstruction_db = float(params.get("nlos_max_obstruction_db", 18.0))

    occupancy_score = (
        0.24 * near_occ
        + 0.12 * far_occ
        + 0.22 * near_height
        + 0.12 * far_height
        + 0.18 * local_max_height
        + 0.12 * shadow_support
    ).clamp(0.0, 1.3)

    obstruction_db = (
        float(cp["bias"]) * nlos_prob
        + float(cp["occ"]) * occupancy_score * nlos_prob
        + float(cp["depth"]) * shadow_depth_norm * nlos_prob
        + float(cp["deep"]) * deep_gate * shadow_depth_norm * nlos_prob
        + float(cp["sigma"]) * sigma_theta * nlos_prob
        + prominence_weight * blocker_prominence * nlos_prob
        + low_altitude_weight * low_altitude_gate * shallow_gate * nlos_prob
        - open_relief_weight * openness * shallow_gate * nlos_prob
    )
    return obstruction_db.clamp(min=0.0, max=max_obstruction_db), shadow_depth_m, near_occ


def _compute_formula_path_loss_db(
    image_size: int,
    antenna_height_m: float,
    receiver_height_m: float,
    frequency_ghz: float,
    meters_per_pixel: float,
    formula_mode: str,
    los_tensor: Optional[torch.Tensor] = None,
    topology_tensor: Optional[torch.Tensor] = None,
    a2g_params: Optional[Dict[str, float]] = None,
    building_density: Optional[float] = None,
    building_height_proxy: Optional[float] = None,
    non_ground_threshold: float = 0.0,
    clip_min: float = 0.0,
    clip_max: float = 180.0,
) -> torch.Tensor:
    distance_norm = _compute_distance_map_2d(image_size)
    half = (image_size - 1) / 2.0
    max_dist_pixels = max(half * (2.0 ** 0.5), 1.0)
    ground_distance_m = distance_norm * float(max_dist_pixels * meters_per_pixel)
    h_tx = max(float(antenna_height_m), 1.0)
    h_rx = max(float(receiver_height_m), 0.5)
    d2d_m = ground_distance_m.clamp(min=1.0)
    d3d_m = torch.sqrt(d2d_m ** 2 + (h_tx - h_rx) ** 2).clamp(min=1.0)
    freq_ghz = max(float(frequency_ghz), 0.1)
    freq_mhz = freq_ghz * 1000.0
    mode = str(formula_mode).lower()
    a2g_params = dict(a2g_params or {})
    city_type = _infer_city_type_simple(float(building_density or 0.0), float(building_height_proxy or 0.0))

    fspl_db = _fspl_db(d3d_m, freq_mhz)

    if mode == 'two_ray_ground':
        wavelength_m = 0.299792458 / freq_ghz
        crossover_m = max((4.0 * math.pi * h_tx * h_rx) / wavelength_m, 1.0)
        two_ray_db = 40.0 * torch.log10(d3d_m) - 20.0 * math.log10(h_tx) - 20.0 * math.log10(h_rx)
        path_db = torch.where(d3d_m <= crossover_m, fspl_db, two_ray_db)
    elif mode == 'coherent_two_ray_ground':
        path_db = _coherent_two_ray_path_loss_db(
            d2d_m,
            h_tx,
            h_rx,
            freq_ghz,
            eps_r=float(a2g_params.get("ground_eps_r", 5.0)),
            roughness_m=float(a2g_params.get("ground_roughness_m", 0.0)),
        )
    elif mode == 'cost231_hata':
        log_f = math.log10(freq_mhz)
        a_hm = (1.1 * log_f - 0.7) * h_rx - (1.56 * log_f - 0.8)
        c_m = 3.0
        d_km = (d2d_m / 1000.0).clamp(min=0.001)
        hb_log = math.log10(max(h_tx, 1.0))
        path_db = (
            46.3
            + 33.9 * log_f
            - 13.82 * hb_log
            - a_hm
            + (44.9 - 6.55 * hb_log) * torch.log10(d_km)
            + c_m
        )
    elif mode == 'hybrid_two_ray_cost231':
        log_f = math.log10(freq_mhz)
        a_hm = (1.1 * log_f - 0.7) * h_rx - (1.56 * log_f - 0.8)
        c_m = 3.0
        d_km = (d2d_m / 1000.0).clamp(min=0.001)
        hb_log = math.log10(max(h_tx, 1.0))
        cost231_db = (
            46.3
            + 33.9 * log_f
            - 13.82 * hb_log
            - a_hm
            + (44.9 - 6.55 * hb_log) * torch.log10(d_km)
            + c_m
        )
        wavelength_m = 0.299792458 / freq_ghz
        crossover_m = max((4.0 * math.pi * h_tx * h_rx) / wavelength_m, 1.0)
        two_ray_db = 40.0 * torch.log10(d3d_m) - 20.0 * math.log10(h_tx) - 20.0 * math.log10(h_rx)
        los_path_db = torch.where(d3d_m <= crossover_m, fspl_db, two_ray_db)
        if los_tensor is None:
            path_db = 0.5 * (los_path_db + torch.maximum(cost231_db, fspl_db))
        else:
            los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
            path_db = los_prob * los_path_db + (1.0 - los_prob) * torch.maximum(cost231_db, fspl_db)
    elif mode == 'hybrid_two_ray_cost231_a2g_nlos':
        log_f = math.log10(freq_mhz)
        a_hm = (1.1 * log_f - 0.7) * h_rx - (1.56 * log_f - 0.8)
        c_m = 3.0
        d_km = (d2d_m / 1000.0).clamp(min=0.001)
        hb_log = math.log10(max(h_tx, 1.0))
        cost231_db = (
            46.3
            + 33.9 * log_f
            - 13.82 * hb_log
            - a_hm
            + (44.9 - 6.55 * hb_log) * torch.log10(d_km)
            + c_m
        )
        wavelength_m = 0.299792458 / freq_ghz
        crossover_m = max((4.0 * math.pi * h_tx * h_rx) / wavelength_m, 1.0)
        two_ray_db = 40.0 * torch.log10(d3d_m) - 20.0 * math.log10(h_tx) - 20.0 * math.log10(h_rx)
        los_path_db = torch.where(d3d_m <= crossover_m, fspl_db, two_ray_db)

        theta_deg = torch.rad2deg(torch.atan2((h_tx - h_rx) * torch.ones_like(d2d_m), d2d_m.clamp(min=1.0)))
        sin_theta = torch.sin(torch.deg2rad(theta_deg)).clamp(min=1e-4)
        lambda0_db = 20.0 * math.log10((4.0 * math.pi * h_tx * freq_ghz * 1.0e9) / 299792458.0)
        los_log_coeff = float(a2g_params.get("los_log_coeff", -20.0))
        los_bias = float(a2g_params.get("los_bias", 0.0))
        nlos_bias = float(a2g_params.get("nlos_bias", -16.16))
        nlos_amp = float(a2g_params.get("nlos_amp", 12.0436))
        nlos_tau = max(float(a2g_params.get("nlos_tau", 7.52)), 1e-3)
        a2g_los_db = lambda0_db + los_bias + los_log_coeff * torch.log10(sin_theta)
        a2g_nlos_db = lambda0_db + (nlos_bias + nlos_amp * torch.exp(-(90.0 - theta_deg) / nlos_tau))
        nlos_path_db = torch.maximum(cost231_db, a2g_nlos_db)

        if los_tensor is None:
            path_db = 0.5 * (torch.minimum(los_path_db, a2g_los_db) + nlos_path_db)
        else:
            los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
            los_blend = 0.7 * los_path_db + 0.3 * torch.minimum(los_path_db, a2g_los_db)
            path_db = los_prob * los_blend + (1.0 - los_prob) * nlos_path_db
    elif mode == 'hybrid_coherent_two_ray_vinogradov_nlos':
        los_path_db = _coherent_two_ray_path_loss_db(d2d_m, h_tx, h_rx, freq_ghz)
        nlos_path_db = _vinogradov_height_nlos_path_loss_db(d3d_m, d2d_m, h_tx, h_rx, freq_ghz, city_type)
        if los_tensor is None:
            path_db = 0.5 * (los_path_db + nlos_path_db)
        else:
            los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
            path_db = los_prob * los_path_db + (1.0 - los_prob) * nlos_path_db
    elif mode == 'hybrid_damped_coherent_two_ray_obstruction_vinogradov_nlos':
        los_path_db = _damped_coherent_two_ray_path_loss_db(
            d2d_m,
            h_tx,
            h_rx,
            freq_ghz,
            eps_r=float(a2g_params.get("ground_eps_r", 5.0)),
            roughness_m=float(a2g_params.get("ground_roughness_m", 0.02)),
            excess_limit_db=float(a2g_params.get("interference_excess_limit_db", 7.5)),
            interference_decay_m=float(a2g_params.get("interference_decay_m", 450.0)),
            min_interference_blend=float(a2g_params.get("min_interference_blend", 0.2)),
        )
        nlos_base_db = _vinogradov_height_nlos_mean_path_loss_db(d3d_m, h_tx, freq_ghz, city_type)
        nlos_base_db = torch.maximum(
            nlos_base_db,
            fspl_db + float(a2g_params.get("nlos_min_margin_db", 4.0)),
        )
        obstruction_db = _compute_structural_nlos_obstruction_db(
            d2d_m,
            topology_tensor,
            los_tensor,
            h_tx=h_tx,
            h_rx=h_rx,
            city_type=city_type,
            meters_per_pixel=meters_per_pixel,
            non_ground_threshold=non_ground_threshold,
            a2g_params=a2g_params,
        )
        nlos_path_db = nlos_base_db + obstruction_db
        if los_tensor is None:
            path_db = 0.5 * (los_path_db + nlos_path_db)
        else:
            los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
            path_db = los_prob * los_path_db + (1.0 - los_prob) * nlos_path_db
    elif mode == 'hybrid_damped_coherent_two_ray_deepshadow_vinogradov_nlos':
        los_path_db = _damped_coherent_two_ray_path_loss_db(
            d2d_m,
            h_tx,
            h_rx,
            freq_ghz,
            eps_r=float(a2g_params.get("ground_eps_r", 5.0)),
            roughness_m=float(a2g_params.get("ground_roughness_m", 0.02)),
            excess_limit_db=float(a2g_params.get("interference_excess_limit_db", 7.5)),
            interference_decay_m=float(a2g_params.get("interference_decay_m", 450.0)),
            min_interference_blend=float(a2g_params.get("min_interference_blend", 0.2)),
        )
        nlos_base_db = _vinogradov_height_nlos_mean_path_loss_db(d3d_m, h_tx, freq_ghz, city_type)
        nlos_floor_db = torch.maximum(
            fspl_db + float(a2g_params.get("nlos_min_margin_db", 6.0)),
            los_path_db + float(a2g_params.get("nlos_los_margin_db", 7.0)),
        )
        nlos_base_db = torch.maximum(nlos_base_db, nlos_floor_db)
        obstruction_db = _compute_deepshadow_nlos_obstruction_db(
            d2d_m,
            topology_tensor,
            los_tensor,
            h_tx=h_tx,
            h_rx=h_rx,
            city_type=city_type,
            meters_per_pixel=meters_per_pixel,
            non_ground_threshold=non_ground_threshold,
            a2g_params=a2g_params,
        )
        nlos_path_db = nlos_base_db + obstruction_db
        if los_tensor is None:
            path_db = 0.5 * (los_path_db + nlos_path_db)
        else:
            los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
            path_db = los_prob * los_path_db + (1.0 - los_prob) * nlos_path_db
    elif mode == 'hybrid_shadowed_ripple_two_ray_vinogradov_nlos':
        coherent_db, _ = _coherent_two_ray_components_db(
            d2d_m,
            h_tx,
            h_rx,
            freq_ghz,
            eps_r=float(a2g_params.get("ground_eps_r", 5.0)),
            roughness_m=float(a2g_params.get("ground_roughness_m", 0.01)),
        )
        smooth_los_db = _damped_coherent_two_ray_path_loss_db(
            d2d_m,
            h_tx,
            h_rx,
            freq_ghz,
            eps_r=float(a2g_params.get("ground_eps_r", 5.0)),
            roughness_m=float(a2g_params.get("ground_roughness_m", 0.01)),
            excess_limit_db=float(a2g_params.get("interference_excess_limit_db", 10.5)),
            interference_decay_m=float(a2g_params.get("interference_decay_m", 900.0)),
            min_interference_blend=float(a2g_params.get("min_interference_blend", 0.35)),
        )
        nlos_base_db = _vinogradov_height_nlos_mean_path_loss_db_relaxed(d3d_m, h_tx, freq_ghz, city_type)
        nlos_floor_db = torch.maximum(
            fspl_db + float(a2g_params.get("nlos_min_margin_db", 2.5)),
            smooth_los_db + float(a2g_params.get("nlos_los_margin_db", 3.5)),
        )
        nlos_base_db = torch.maximum(nlos_base_db, nlos_floor_db)
        obstruction_db, shadow_depth_m, near_occ = _compute_directional_ripple_nlos_obstruction_db(
            d2d_m,
            topology_tensor,
            los_tensor,
            h_tx=h_tx,
            h_rx=h_rx,
            city_type=city_type,
            meters_per_pixel=meters_per_pixel,
            non_ground_threshold=non_ground_threshold,
            a2g_params=a2g_params,
        )
        ripple_db = _compute_shadowed_ripple_db(
            coherent_db,
            smooth_los_db,
            los_tensor,
            shadow_depth_m,
            near_occ,
            a2g_params=a2g_params,
        )
        nlos_path_db = nlos_base_db + obstruction_db
        if los_tensor is None:
            path_db = 0.5 * (smooth_los_db + nlos_path_db) + ripple_db
        else:
            los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
            path_db = los_prob * smooth_los_db + (1.0 - los_prob) * nlos_path_db + ripple_db
    elif mode == 'hybrid_fresnel_two_ray_cost231_a2g_nlos':
        log_f = math.log10(freq_mhz)
        a_hm = (1.1 * log_f - 0.7) * h_rx - (1.56 * log_f - 0.8)
        c_m = 3.0
        d_km = (d2d_m / 1000.0).clamp(min=0.001)
        hb_log = math.log10(max(h_tx, 1.0))
        cost231_db = (
            46.3
            + 33.9 * log_f
            - 13.82 * hb_log
            - a_hm
            + (44.9 - 6.55 * hb_log) * torch.log10(d_km)
            + c_m
        )
        los_path_db = _coherent_two_ray_path_loss_db(
            d2d_m,
            h_tx,
            h_rx,
            freq_ghz,
            eps_r=float(a2g_params.get("ground_eps_r", 5.0)),
            roughness_m=float(a2g_params.get("ground_roughness_m", 0.02)),
        )
        lambda0_db = 20.0 * math.log10((4.0 * math.pi * 1.0 * freq_ghz * 1.0e9) / 299792458.0)
        theta_deg = torch.rad2deg(torch.atan2((h_tx - h_rx) * torch.ones_like(d2d_m), d2d_m.clamp(min=1.0)))
        sin_theta = torch.sin(torch.deg2rad(theta_deg)).clamp(min=1e-4)
        los_log_coeff = float(a2g_params.get("los_log_coeff", -20.0))
        los_bias = float(a2g_params.get("los_bias", 0.0))
        nlos_bias = float(a2g_params.get("nlos_bias", -16.0))
        nlos_amp = float(a2g_params.get("nlos_amp", 12.0))
        nlos_tau = max(float(a2g_params.get("nlos_tau", 8.0)), 1e-3)
        a2g_los_db = lambda0_db + los_bias + los_log_coeff * torch.log10(sin_theta)
        a2g_nlos_db = lambda0_db + (nlos_bias + nlos_amp * torch.exp(-(90.0 - theta_deg) / nlos_tau))
        nlos_path_db = torch.maximum(cost231_db, a2g_nlos_db)
        if los_tensor is None:
            path_db = 0.5 * (torch.minimum(los_path_db, a2g_los_db) + nlos_path_db)
        else:
            los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
            los_blend = 0.7 * los_path_db + 0.3 * torch.minimum(los_path_db, a2g_los_db)
            path_db = los_prob * los_blend + (1.0 - los_prob) * nlos_path_db
    else:
        path_db = fspl_db

    return path_db.clamp(min=float(clip_min), max=float(clip_max))


def _resolve_try_relative_path(raw_path: str) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


def _load_formula_regime_calibration(calibration_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not calibration_json:
        return None
    path = _resolve_try_relative_path(calibration_json)
    if not path.exists():
        raise FileNotFoundError(f"Formula calibration JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _city_type_from_thresholds(density: float, height: float, thresholds: Dict[str, float]) -> str:
    density_q1 = float(thresholds.get("density_q1", 0.0))
    density_q2 = float(thresholds.get("density_q2", 1.0))
    height_q1 = float(thresholds.get("height_q1", 0.0))
    height_q2 = float(thresholds.get("height_q2", 1.0))
    if density >= density_q2 or height >= height_q2:
        return "dense_highrise"
    if density <= density_q1 and height <= height_q1:
        return "open_lowrise"
    return "mixed_midrise"


def _antenna_height_bin(antenna_height_m: float, thresholds: Dict[str, float]) -> str:
    q1 = float(thresholds.get("q1", 0.0))
    q2 = float(thresholds.get("q2", q1))
    if antenna_height_m <= q1:
        return "low_ant"
    if antenna_height_m <= q2:
        return "mid_ant"
    return "high_ant"


TRY54_TOPOLOGY_CLASSES = [
    "open_sparse_lowrise",
    "open_sparse_vertical",
    "mixed_compact_lowrise",
    "mixed_compact_midrise",
    "dense_block_midrise",
    "dense_block_highrise",
]


def _resolve_try54_topology_thresholds(
    data_cfg: Dict[str, Any],
    formula_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    raw = dict(data_cfg.get("topology_partitioning", {}))
    if raw:
        return {
            "density_q1": float(raw.get("density_q1", 0.12)),
            "density_q2": float(raw.get("density_q2", 0.28)),
            "height_q1": float(raw.get("height_q1", 12.0)),
            "height_q2": float(raw.get("height_q2", 28.0)),
        }
    formula_cfg = dict(formula_cfg or {})
    calibration_path = formula_cfg.get("regime_calibration_json")
    if calibration_path:
        try:
            calibration = _load_formula_regime_calibration(calibration_path)
            thresholds = dict(calibration.get("city_type_thresholds", {}))
            if thresholds:
                return {
                    "density_q1": float(thresholds.get("density_q1", 0.12)),
                    "density_q2": float(thresholds.get("density_q2", 0.28)),
                    "height_q1": float(thresholds.get("height_q1", 12.0)),
                    "height_q2": float(thresholds.get("height_q2", 28.0)),
                }
        except Exception:
            pass
    return {
        "density_q1": 0.12,
        "density_q2": 0.28,
        "height_q1": 12.0,
        "height_q2": 28.0,
    }


def _infer_try54_topology_class(
    density: float,
    height: float,
    thresholds: Dict[str, float],
) -> str:
    density_q1 = float(thresholds.get("density_q1", 0.12))
    density_q2 = float(thresholds.get("density_q2", 0.28))
    height_q1 = float(thresholds.get("height_q1", 12.0))
    height_q2 = float(thresholds.get("height_q2", 28.0))

    if density <= density_q1:
        if height <= height_q1:
            return "open_sparse_lowrise"
        return "open_sparse_vertical"
    if density >= density_q2:
        if height <= height_q2:
            return "dense_block_midrise"
        return "dense_block_highrise"
    if height <= height_q1:
        return "mixed_compact_lowrise"
    return "mixed_compact_midrise"


def _compute_try54_partition_metadata(
    raw_topology: np.ndarray,
    antenna_height_m: float,
    thresholds: Dict[str, float],
    antenna_thresholds: Dict[str, float],
    *,
    non_ground_threshold: float = 0.0,
) -> Dict[str, str | float]:
    non_ground = raw_topology != float(non_ground_threshold)
    density = float(np.mean(non_ground))
    non_zero = raw_topology[non_ground]
    mean_height = float(np.mean(non_zero)) if non_zero.size else 0.0
    topology_class = _infer_try54_topology_class(density, mean_height, thresholds)
    city_type = _city_type_from_thresholds(density, mean_height, thresholds)
    antenna_bin = _antenna_height_bin(float(antenna_height_m), antenna_thresholds)
    return {
        "topology_class": topology_class,
        "city_type": city_type,
        "antenna_bin": antenna_bin,
        "building_density": density,
        "mean_height": mean_height,
    }


def _filter_hdf5_refs_by_partition(
    hdf5_path: str,
    sample_refs: Sequence[Tuple[str, str]],
    *,
    input_column: str,
    scalar_specs: Sequence[Dict[str, Any]],
    constant_scalar_features: Dict[str, float],
    antenna_thresholds: Dict[str, float],
    topology_thresholds: Dict[str, float],
    topology_class: Optional[str],
    antenna_bin: Optional[str],
    non_ground_threshold: float,
) -> List[Tuple[str, str]]:
    refs = list(sample_refs)
    if not topology_class and not antenna_bin:
        return refs
    out: List[Tuple[str, str]] = []
    with h5py.File(hdf5_path, "r") as handle:
        for city, sample in refs:
            grp = handle[city][sample]
            raw_topology = np.asarray(grp[input_column][...], dtype=np.float32)
            if "antenna_height_m" in constant_scalar_features:
                antenna_height_m = float(constant_scalar_features["antenna_height_m"])
            else:
                antenna_height_m = 1.0
                antenna_height_m = float(
                    _resolve_hdf5_scalar_value_static(
                        grp,
                        "antenna_height_m",
                        scalar_specs,
                    )
                )
            meta = _compute_try54_partition_metadata(
                raw_topology,
                antenna_height_m,
                topology_thresholds,
                antenna_thresholds,
                non_ground_threshold=non_ground_threshold,
            )
            if topology_class and meta["topology_class"] != topology_class:
                continue
            if antenna_bin and meta["antenna_bin"] != antenna_bin:
                continue
            out.append((city, sample))
    return out


def _apply_formula_regime_calibration(
    prior_db: torch.Tensor,
    calibration: Optional[Dict[str, Any]],
    *,
    city: str,
    density: float,
    height: float,
    antenna_height_m: float,
    los_tensor: Optional[torch.Tensor],
    topology_tensor: Optional[torch.Tensor] = None,
    distance_map_tensor: Optional[torch.Tensor] = None,
    non_ground_threshold: float = 0.0,
    clip_min: float = 0.0,
    clip_max: float = 180.0,
    prefer_threshold_city_type: bool = False,
) -> torch.Tensor:
    if calibration is None:
        return prior_db.clamp(min=float(clip_min), max=float(clip_max))

    city_type = None
    if not prefer_threshold_city_type:
        city_type_map = dict(calibration.get("city_type_by_city", {}))
        city_type = city_type_map.get(city)
    if city_type is None:
        thresholds = dict(calibration.get("city_type_thresholds", {}))
        if thresholds:
            city_type = _city_type_from_thresholds(density, height, thresholds)
        else:
            city_type = _infer_city_type_simple(density, height)
    ant_bin = _antenna_height_bin(float(antenna_height_m), dict(calibration.get("antenna_height_thresholds", {})))
    coeff_map = dict(calibration.get("coefficients", {}))
    model_type = str(calibration.get("model_type", "quadratic_regime"))
    if model_type in {"regime_obstruction_linear_v1", "regime_obstruction_multiscale_v1"}:
        kernel_sizes = calibration.get("local_kernel_sizes", [calibration.get("local_kernel_size", 25)])
        kernel_sizes = [int(k) for k in kernel_sizes]
        if topology_tensor is None:
            raise ValueError("Obstruction-aware prior calibration requires topology_tensor.")
        if distance_map_tensor is None:
            distance_map_tensor = _compute_distance_map_2d(prior_db.shape[-1])
        distance_map_tensor = distance_map_tensor.to(dtype=prior_db.dtype)
        local_features = []
        for kernel_size in kernel_sizes:
            local_density, local_height = _compute_local_obstruction_features(
                topology_tensor,
                non_ground_threshold=non_ground_threshold,
                kernel_size=kernel_size,
            )
            local_features.append((local_density.to(dtype=prior_db.dtype), local_height.to(dtype=prior_db.dtype)))
        los_features = []
        base_los_tensor = los_tensor if los_tensor is not None else torch.zeros_like(prior_db)
        for kernel_size in kernel_sizes:
            kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            nlos_support = F.avg_pool2d(
                (base_los_tensor <= 0.5).to(dtype=torch.float32).unsqueeze(0),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            ).squeeze(0).to(dtype=prior_db.dtype)
            los_features.append(nlos_support.clamp(0.0, 1.0))

        def obstruction_pred(los_label: str) -> torch.Tensor:
            candidates = [
                f"{city_type}|{los_label}|{ant_bin}",
                f"{city_type}|{los_label}|mid_ant",
                f"{city_type}|{los_label}|low_ant",
                f"{city_type}|{los_label}|high_ant",
            ]
            payload = None
            for key in candidates:
                if key in coeff_map:
                    payload = coeff_map[key]
                    break
            if payload is None:
                return prior_db
            weights = [float(v) for v in payload.get("weights", [])]
            bias = float(payload.get("bias", 0.0))
            distance_scale = max(float(payload.get("distance_scale_m", 1.0)), 1.0)
            height_scale = max(float(payload.get("height_scale", 1.0)), 1.0)
            meters_per_pixel = max(float(payload.get("meters_per_pixel", 1.0)), 1e-6)
            receiver_height_m = max(float(payload.get("receiver_height_m", 1.5)), 0.1)
            logd = torch.log1p(distance_map_tensor * distance_scale)
            density_small, height_small = local_features[0]
            density_large, height_large = local_features[-1]
            nlos_small = los_features[0]
            nlos_large = los_features[-1]
            sigma_feature = _compute_a2g_shadow_sigma_db(
                image_size=prior_db.shape[-1],
                antenna_height_m=float(antenna_height_m),
                receiver_height_m=receiver_height_m,
                meters_per_pixel=meters_per_pixel,
                los_tensor=los_tensor,
            ).to(dtype=prior_db.dtype)
            ground_distance_m = distance_map_tensor * distance_scale
            theta_deg = torch.rad2deg(
                torch.atan2(
                    (float(antenna_height_m) - receiver_height_m) * torch.ones_like(ground_distance_m),
                    ground_distance_m.clamp(min=1.0),
                )
            )
            theta_norm = (theta_deg / 90.0).clamp(0.0, 1.0).to(dtype=prior_db.dtype)
            features = [
                prior_db * prior_db,
                prior_db,
                logd,
                density_small,
                density_large,
                height_small / height_scale,
                height_large / height_scale,
                density_large * logd,
                nlos_small,
                nlos_large,
                nlos_large * logd,
                sigma_feature,
                theta_norm,
                nlos_large * theta_norm,
            ]
            out = torch.full_like(prior_db, bias)
            for w, feat in zip(weights, features):
                out = out + float(w) * feat
            return out

        los_pred = obstruction_pred("LoS")
        nlos_pred = obstruction_pred("NLoS")
        if los_tensor is None:
            calibrated = 0.5 * (los_pred + nlos_pred)
        else:
            los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=prior_db.dtype)
            calibrated = los_prob * los_pred + (1.0 - los_prob) * nlos_pred
        return calibrated.clamp(min=float(clip_min), max=float(clip_max))

    def coeffs_for(los_label: str) -> Tuple[float, float, float]:
        candidates = [
            f"{city_type}|{los_label}|{ant_bin}",
            f"{city_type}|{los_label}|mid_ant",
            f"{city_type}|{los_label}|low_ant",
            f"{city_type}|{los_label}|high_ant",
        ]
        for key in candidates:
            payload = coeff_map.get(key)
            if payload:
                poly = list(payload.get("poly2", [0.0, 1.0, 0.0]))
                if len(poly) == 3:
                    return float(poly[0]), float(poly[1]), float(poly[2])
        return 0.0, 1.0, 0.0

    los_a2, los_a1, los_a0 = coeffs_for("LoS")
    nlos_a2, nlos_a1, nlos_a0 = coeffs_for("NLoS")
    los_pred = los_a2 * prior_db * prior_db + los_a1 * prior_db + los_a0
    nlos_pred = nlos_a2 * prior_db * prior_db + nlos_a1 * prior_db + nlos_a0
    if los_tensor is None:
        calibrated = 0.5 * (los_pred + nlos_pred)
    else:
        los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=prior_db.dtype)
        calibrated = los_prob * los_pred + (1.0 - los_prob) * nlos_pred
    return calibrated.clamp(min=float(clip_min), max=float(clip_max))


def uses_scalar_film_conditioning(cfg: Dict[str, Any]) -> bool:
    """When True, scalars are passed as a vector + FiLM (no extra input channels)."""
    m = cfg.get('model', {})
    return bool(m.get('use_scalar_film', False)) and bool(m.get('use_scalar_channels', False))


def compute_scalar_cond_dim(cfg: Dict[str, Any]) -> int:
    data_cfg = cfg['data']
    return len(list(data_cfg.get('scalar_feature_columns', []))) + len(dict(data_cfg.get('constant_scalar_features', {})))


def add_scalar_channels_from_config(cfg: Dict[str, Any]) -> bool:
    """Stack scalar features as constant spatial channels (legacy path)."""
    return bool(cfg['model'].get('use_scalar_channels', False)) and not uses_scalar_film_conditioning(cfg)


def return_scalar_cond_from_config(cfg: Dict[str, Any]) -> bool:
    """Dataset returns a 4th tensor [B, D] for FiLM when enabled and D > 0."""
    return uses_scalar_film_conditioning(cfg) and compute_scalar_cond_dim(cfg) > 0


def compute_input_channels(cfg: Dict[str, Any]) -> int:
    in_channels = 1
    if cfg['data'].get('los_input_column'):
        in_channels += 1
    if cfg['data'].get('distance_map_channel', False):
        in_channels += 1
    if bool(cfg['data'].get('path_loss_formula_input', {}).get('enabled', False)):
        in_channels += 1
        if bool(cfg['data'].get('path_loss_formula_input', {}).get('include_confidence_channel', False)):
            in_channels += 1
    if bool(cfg['data'].get('path_loss_obstruction_features', {}).get('enabled', False)):
        obstruction_cfg = dict(cfg['data'].get('path_loss_obstruction_features', {}))
        if bool(obstruction_cfg.get('include_shadow_depth', True)):
            in_channels += 1
        if bool(obstruction_cfg.get('include_distance_since_los_break', True)):
            in_channels += 1
        if bool(obstruction_cfg.get('include_max_blocker_height', True)):
            in_channels += 1
        if bool(obstruction_cfg.get('include_blocker_count', True)):
            in_channels += 1
    if bool(cfg['data'].get('tx_depth_map_channel', False)):
        in_channels += 1
    if bool(cfg['data'].get('elevation_angle_map_channel', False)):
        in_channels += 1
    if bool(cfg['data'].get('building_mask_channel', False)):
        in_channels += 1
    if bool(cfg['model'].get('use_scalar_channels', False)):
        if not bool(cfg['model'].get('use_scalar_film', False)):
            in_channels += len(list(cfg['data'].get('scalar_feature_columns', [])))
            in_channels += len(dict(cfg['data'].get('constant_scalar_features', {})))
    return in_channels


def unpack_cgan_batch(
    batch: Tuple[Any, ...],
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if len(batch) == 4:
        x, y, m, sc = batch
        return x.to(device), y.to(device), m.to(device), sc.to(device)
    x, y, m = batch
    return x.to(device), y.to(device), m.to(device), None


def forward_cgan_generator(
    generator: torch.nn.Module,
    x: torch.Tensor,
    scalar_cond: Optional[torch.Tensor],
) -> torch.Tensor:
    if scalar_cond is not None:
        return generator(x, scalar_cond)
    return generator(x)


def log_scalar_data_report(train_dataset: Dataset, cfg: Dict[str, Any], sample_limit: int = 1000) -> None:
    """Log min/max/mean/%zeros for configured scalar columns (rank-0 sanity check)."""
    if not bool(cfg['model'].get('use_scalar_channels', False)):
        return
    cols = list(cfg['data'].get('scalar_feature_columns', []))
    const_names = list(dict(cfg['data'].get('constant_scalar_features', {})).keys())
    names = cols + const_names
    if not names:
        return
    data_cfg = cfg['data']
    n_prefix = 1
    if data_cfg.get('los_input_column'):
        n_prefix += 1
    if bool(data_cfg.get('distance_map_channel', False)):
        n_prefix += 1
    if bool(data_cfg.get('path_loss_formula_input', {}).get('enabled', False)):
        n_prefix += 1
    n_sc = len(names)
    stats: Dict[str, List[float]] = defaultdict(list)
    n = min(int(sample_limit), len(train_dataset))
    for i in range(n):
        item = train_dataset[i]
        if len(item) == 4:
            sc = item[3]
            vals = [float(sc[j].item()) for j in range(min(n_sc, int(sc.shape[0])))]
        else:
            x = item[0]
            if x.shape[0] < n_prefix + n_sc:
                continue
            vals = [float(x[n_prefix + j, 0, 0].item()) for j in range(n_sc)]
        for name, v in zip(names, vals):
            stats[name].append(v)
    report: Dict[str, Any] = {}
    for name, vs in stats.items():
        if not vs:
            continue
        arr = np.asarray(vs, dtype=np.float64)
        report[name] = {
            'n': int(len(vs)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'frac_zero': float(np.mean(np.abs(arr) < 1e-12)),
        }
    if report:
        print(json.dumps({'scalar_data_report': report}, indent=2))


class CKMDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        root_dir: str,
        target_columns: List[str],
        image_size: int = 128,
        augment: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.4,
        add_scalar_channels: bool = True,
        return_scalar_cond: bool = False,
        scalar_feature_columns: Optional[List[str]] = None,
        constant_scalar_features: Optional[Dict[str, float]] = None,
        scalar_feature_norms: Optional[Dict[str, float]] = None,
        los_input_column: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(manifest_csv)
        self.target_columns = target_columns
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.add_scalar_channels = add_scalar_channels
        self.return_scalar_cond = bool(return_scalar_cond)
        self.scalar_feature_columns = scalar_feature_columns or []
        self.constant_scalar_features = constant_scalar_features or {}
        self.scalar_feature_norms_cfg = scalar_feature_norms or {}
        self.los_input_column = los_input_column
        self.target_availability: Dict[str, Dict[str, int]] = {}

        self.scalar_norms = _compute_scalar_norms(
            self.scalar_feature_columns,
            self.constant_scalar_features,
            self.scalar_feature_norms_cfg,
            df=self.df,
        )

        for col in self.target_columns:
            if col not in self.df.columns:
                self.target_availability[col] = {'present_rows': 0, 'total_rows': len(self.df)}
                print(f"[WARNING] Target column '{col}' is missing from manifest {manifest_csv}.")
                continue

            present_rows = int(self.df[col].notna().sum())
            self.target_availability[col] = {'present_rows': present_rows, 'total_rows': len(self.df)}
            if present_rows == 0:
                print(f"[WARNING] Target column '{col}' exists but has 0 labeled rows in {manifest_csv}.")
            elif present_rows < len(self.df):
                print(
                    f"[INFO] Target column '{col}' available for {present_rows}/{len(self.df)} rows in {manifest_csv}."
                )

    def __len__(self) -> int:
        return len(self.df)

    def _apply_sync_aug(self, images: List[Image.Image]) -> List[Image.Image]:
        if random.random() < self.hflip_prob:
            images = [TF.hflip(img) for img in images]
        if random.random() < self.vflip_prob:
            images = [TF.vflip(img) for img in images]
        if random.random() < self.rot90_prob:
            angle = random.choice([90, 180, 270])
            images = [TF.rotate(img, angle) for img in images]
        return images

    def _load_gray_or_none(self, rel_path: Optional[str]) -> Optional[Image.Image]:
        if rel_path is None or (isinstance(rel_path, float) and pd.isna(rel_path)):
            return None
        path = _resolve_path(self.root_dir, str(rel_path))
        if not path.exists():
            return None
        try:
            return Image.open(path).convert('L').resize((self.image_size, self.image_size), Image.BILINEAR)
        except (OSError, FileNotFoundError):
            return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        input_img = self._load_gray_or_none(row['input_path'])
        if input_img is None:
            raise FileNotFoundError(f"Missing input image at row {idx}: {row['input_path']}")

        los_input_img = None
        if self.los_input_column and self.los_input_column in self.df.columns:
            los_input_img = self._load_gray_or_none(row[self.los_input_column])

        target_imgs: List[Optional[Image.Image]] = [self._load_gray_or_none(row.get(col, None)) for col in self.target_columns]

        if self.augment:
            stack = [input_img]
            if los_input_img is not None:
                stack.append(los_input_img)
            stack.extend([img for img in target_imgs if img is not None])
            aug = self._apply_sync_aug(stack)
            input_img = aug[0]
            cursor = 1
            if los_input_img is not None:
                los_input_img = aug[cursor]
                cursor += 1
            rebuilt = []
            for img in target_imgs:
                if img is None:
                    rebuilt.append(None)
                else:
                    rebuilt.append(aug[cursor])
                    cursor += 1
            target_imgs = rebuilt

        model_input_channels = [TF.to_tensor(input_img)]
        if los_input_img is not None:
            model_input_channels.append(TF.to_tensor(los_input_img))

        scalar_values: List[float] = []
        if self.add_scalar_channels or self.return_scalar_cond:
            for col in self.scalar_feature_columns:
                raw_value = pd.to_numeric(row.get(col, 0.0), errors='coerce')
                value = 0.0 if pd.isna(raw_value) else float(raw_value)
                norm = self.scalar_norms.get(col, 1.0)
                scalar_values.append(value / max(norm, 1e-12))

            for col, value in self.constant_scalar_features.items():
                norm = self.scalar_norms.get(col, 1.0)
                scalar_values.append(float(value) / max(norm, 1e-12))

            if self.add_scalar_channels and scalar_values:
                h, w = model_input_channels[0].shape[1:]
                scalar_tensor = torch.tensor(scalar_values, dtype=torch.float32).view(len(scalar_values), 1, 1).expand(
                    len(scalar_values), h, w
                )
                model_input_channels.append(scalar_tensor)

        model_input = torch.cat(model_input_channels, dim=0)

        target_tensors = []
        mask_tensors = []
        for img in target_imgs:
            if img is None:
                target_tensors.append(torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32))
                mask_tensors.append(torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32))
            else:
                target_tensors.append(TF.to_tensor(img))
                mask_tensors.append(torch.ones((1, self.image_size, self.image_size), dtype=torch.float32))

        target_tensor = torch.cat(target_tensors, dim=0)
        mask_tensor = torch.cat(mask_tensors, dim=0)

        if self.return_scalar_cond and scalar_values:
            cond = torch.tensor(scalar_values, dtype=torch.float32)
            return model_input, target_tensor, mask_tensor, cond
        return model_input, target_tensor, mask_tensor


class CKMHDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        sample_refs: Sequence[Tuple[str, str]],
        target_columns: List[str],
        image_size: int = 128,
        augment: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.4,
        add_scalar_channels: bool = True,
        return_scalar_cond: bool = False,
        scalar_feature_columns: Optional[List[str]] = None,
        constant_scalar_features: Optional[Dict[str, float]] = None,
        scalar_feature_norms: Optional[Dict[str, float]] = None,
        los_input_column: Optional[str] = None,
        input_column: str = 'topology_map',
        input_metadata: Optional[Dict[str, Any]] = None,
        target_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        target_field_map: Optional[Dict[str, str]] = None,
        distance_map_channel: bool = False,
        path_loss_saturation_db: Optional[float] = None,
        scalar_table_csv: Optional[str] = None,
        hdf5_scalar_specs: Optional[List[Dict[str, Any]]] = None,
        path_loss_ignore_nonfinite: bool = True,
        exclude_non_ground_targets: bool = False,
        non_ground_threshold: float = 0.0,
        path_loss_no_data_mask_column: Optional[str] = None,
        derive_no_data_from_non_ground: bool = False,
        append_no_data_target: bool = False,
        path_loss_formula_input: Optional[Dict[str, Any]] = None,
        path_loss_obstruction_features: Optional[Dict[str, Any]] = None,
        tx_depth_map_channel: bool = False,
        elevation_angle_map_channel: bool = False,
        building_mask_channel: bool = False,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.sample_refs = list(sample_refs)
        self.target_columns = target_columns
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.add_scalar_channels = add_scalar_channels
        self.return_scalar_cond = bool(return_scalar_cond)
        self.scalar_feature_columns = scalar_feature_columns or []
        self.constant_scalar_features = constant_scalar_features or {}
        self.scalar_feature_norms_cfg = scalar_feature_norms or {}
        self.los_input_column = los_input_column
        self.input_column = input_column
        self.input_metadata = input_metadata or {}
        self.target_metadata = target_metadata or {}
        self.target_field_map = target_field_map or {}
        self.distance_map_channel = distance_map_channel
        self.tx_depth_map_channel = bool(tx_depth_map_channel)
        self.elevation_angle_map_channel = bool(elevation_angle_map_channel)
        self.building_mask_channel = bool(building_mask_channel)
        self.path_loss_saturation_db = path_loss_saturation_db
        self.path_loss_ignore_nonfinite = bool(path_loss_ignore_nonfinite)
        self.exclude_non_ground_targets = bool(exclude_non_ground_targets)
        self.non_ground_threshold = float(non_ground_threshold)
        self.path_loss_no_data_mask_column = str(path_loss_no_data_mask_column or "").strip()
        self.derive_no_data_from_non_ground = bool(derive_no_data_from_non_ground)
        self.append_no_data_target = bool(append_no_data_target)
        self.path_loss_formula_input = dict(path_loss_formula_input or {})
        self.path_loss_obstruction_features = dict(path_loss_obstruction_features or {})
        obstruction_cache_raw = str(self.path_loss_obstruction_features.get("precomputed_hdf5", "")).strip()
        self.obstruction_precomputed_hdf5 = _resolve_try_relative_path(obstruction_cache_raw) if obstruction_cache_raw else None
        formula_precomputed_raw = str(self.path_loss_formula_input.get("precomputed_hdf5", "")).strip()
        self.formula_precomputed_hdf5 = _resolve_try_relative_path(formula_precomputed_raw) if formula_precomputed_raw else None
        self.formula_regime_calibration = _load_formula_regime_calibration(
            self.path_loss_formula_input.get("regime_calibration_json")
        )
        self.formula_cache_enabled = bool(self.path_loss_formula_input.get("cache_enabled", False))
        cache_dir_raw = self.path_loss_formula_input.get("cache_dir")
        self.formula_cache_dir = Path(cache_dir_raw) if cache_dir_raw else None
        if self.formula_cache_enabled and self.formula_cache_dir is not None:
            self.formula_cache_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_scalar_specs_map = _normalize_hdf5_scalar_specs(hdf5_scalar_specs or [])
        self.scalar_table: Dict[Tuple[str, str], Dict[str, float]] = {}
        if scalar_table_csv:
            self.scalar_table = _load_hdf5_scalar_csv(Path(scalar_table_csv))

        table_df: Optional[pd.DataFrame] = None
        if self.scalar_table and self.scalar_feature_columns:
            rows = []
            for (c, s), payload in self.scalar_table.items():
                row = {'city': c, 'sample': s}
                row.update({k: payload.get(k, 0.0) for k in self.scalar_feature_columns})
                rows.append(row)
            if rows:
                table_df = pd.DataFrame(rows)

        self.scalar_norms = _compute_scalar_norms(
            self.scalar_feature_columns,
            self.constant_scalar_features,
            self.scalar_feature_norms_cfg,
            df=table_df,
        )
        self._handle: Optional[h5py.File] = None
        self._obstruction_handle: Optional[h5py.File] = None
        self._formula_precomputed_handle: Optional[h5py.File] = None

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        if bool(self.path_loss_formula_input.get('enabled', False)) and self.formula_precomputed_hdf5 is not None and not self.formula_precomputed_hdf5.exists():
            warnings.warn(
                f"Precomputed formula HDF5 not found, falling back to on-the-fly prior: {self.formula_precomputed_hdf5}",
                RuntimeWarning,
            )
            self.formula_precomputed_hdf5 = None
        if bool(self.path_loss_obstruction_features.get('enabled', False)) and self.obstruction_precomputed_hdf5 is not None and not self.obstruction_precomputed_hdf5.exists():
            warnings.warn(
                f"Precomputed obstruction HDF5 not found, falling back to on-the-fly features: {self.obstruction_precomputed_hdf5}",
                RuntimeWarning,
            )
            self.obstruction_precomputed_hdf5 = None
        self._resolved_no_data_target_source = self._resolve_no_data_target_source()

    def _resolve_no_data_target_source(self) -> str:
        if not self.append_no_data_target:
            return "disabled"
        if self.path_loss_no_data_mask_column and self.sample_refs:
            try:
                with h5py.File(self.hdf5_path, 'r') as handle:
                    for city, sample in self.sample_refs[: min(len(self.sample_refs), 8)]:
                        if self.path_loss_no_data_mask_column in handle[city][sample]:
                            return f"hdf5:{self.path_loss_no_data_mask_column}"
            except (OSError, KeyError):
                pass
        if self.derive_no_data_from_non_ground and self.exclude_non_ground_targets:
            return "fallback:non_ground_mask"
        if self.derive_no_data_from_non_ground:
            return "fallback:non_ground_mask_requested"
        return "disabled"

    def describe_no_data_target_source(self) -> str:
        return self._resolved_no_data_target_source

        if self.scalar_feature_columns and not self.scalar_table and not self.hdf5_scalar_specs_map:
            print(
                "[INFO] HDF5 scalars: no scalar_table_csv or hdf5_scalar_specs; "
                "scalar_feature_columns will use 0.0 unless attrs/datasets match column names in each sample group."
            )

    def __len__(self) -> int:
        return len(self.sample_refs)

    def _get_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_path, 'r')
        return self._handle

    def _get_obstruction_handle(self) -> Optional[h5py.File]:
        if self.obstruction_precomputed_hdf5 is None:
            return None
        if self._obstruction_handle is None:
            self._obstruction_handle = h5py.File(self.obstruction_precomputed_hdf5, 'r')
        return self._obstruction_handle

    def _get_formula_precomputed_handle(self) -> Optional[h5py.File]:
        if self.formula_precomputed_hdf5 is None:
            return None
        if self._formula_precomputed_handle is None:
            self._formula_precomputed_handle = h5py.File(self.formula_precomputed_hdf5, 'r')
        return self._formula_precomputed_handle

    def _resolve_hdf5_scalar_value(self, city: str, sample: str, col: str) -> float:
        key = (city, sample)
        if key in self.scalar_table and col in self.scalar_table[key]:
            v = float(self.scalar_table[key][col])
            return float(v) if np.isfinite(v) else 0.0
        spec = self.hdf5_scalar_specs_map.get(col)
        handle = self._get_handle()
        grp = handle[city][sample]
        if spec:
            got = _read_scalar_from_h5_group(grp, spec)
            if got is not None:
                return got
        if col in grp.attrs:
            try:
                raw = grp.attrs[col]
                v = float(np.asarray(raw).reshape(-1)[0])
                if np.isfinite(v):
                    return v
            except (TypeError, ValueError):
                pass
        if col in grp and isinstance(grp[col], h5py.Dataset):
            try:
                arr = np.asarray(grp[col][...], dtype=np.float64).reshape(-1)
                if arr.size:
                    v = float(np.nanmean(arr))
                    if np.isfinite(v):
                        return v
            except (TypeError, ValueError, OSError):
                pass
        # CKM_Dataset_*_antenna_height.h5 uses uav_height for the scalar channel antenna_height_m
        if col == "antenna_height_m" and "uav_height" in grp and isinstance(grp["uav_height"], h5py.Dataset):
            try:
                arr = np.asarray(grp["uav_height"][...], dtype=np.float64).reshape(-1)
                if arr.size:
                    v = float(np.nanmean(arr))
                    if np.isfinite(v):
                        return v
            except (TypeError, ValueError, OSError):
                pass
        return 0.0

    def _load_precomputed_obstruction_features(
        self, city: str, sample: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        handle = self._get_obstruction_handle()
        if handle is None:
            return None
        try:
            grp = handle[city][sample]
            ds = grp["features_u8"]
        except KeyError:
            return None
        arr = np.asarray(ds[...], dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[0] < 4:
            return None
        feats = torch.from_numpy(arr.astype(np.float32) / 255.0)
        return feats[0:1], feats[1:2], feats[2:3], feats[3:4]

    def _load_precomputed_formula_input(self, city: str, sample: str) -> Optional[torch.Tensor]:
        handle = self._get_formula_precomputed_handle()
        if handle is None:
            return None
        try:
            grp = handle[city][sample]
        except KeyError:
            return None
        for dataset_name in ("formula_norm_u8", "formula_norm_f16", "formula_norm_f32", "formula_norm"):
            if dataset_name not in grp:
                continue
            ds = grp[dataset_name]
            cache_dtype = ds.attrs.get("cache_dtype")
            arr = np.asarray(ds[...])
            tensor = _formula_hdf5_unpack(arr, dataset_name, cache_dtype)
            if tensor.ndim != 3 or tensor.shape[0] < 1:
                continue
            return tensor.to(dtype=torch.float32)
        return None

    def _read_field(self, city: str, sample: str, field_name: str, metadata: Optional[Dict[str, Any]]) -> torch.Tensor:
        handle = self._get_handle()
        if field_name not in handle[city][sample]:
            raise KeyError(f"Field '{field_name}' not found in {city}/{sample}")
        return _resize_array(handle[city][sample][field_name][...], self.image_size, metadata)

    def _formula_cache_path(self, city: str, sample: str, formula_cfg: Dict[str, Any], path_meta: Dict[str, Any]) -> Optional[Path]:
        if not self.formula_cache_enabled or self.formula_cache_dir is None:
            return None
        signature = _formula_cache_signature(self.image_size, formula_cfg, path_meta, self.non_ground_threshold)
        safe_city = city.replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_sample = sample.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return self.formula_cache_dir / f"{safe_city}__{safe_sample}__{signature}.pt"

    def _apply_sync_aug(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        if random.random() < self.hflip_prob:
            images = [TF.hflip(img) for img in images]
        if random.random() < self.vflip_prob:
            images = [TF.vflip(img) for img in images]
        if random.random() < self.rot90_prob:
            angle = random.choice([90, 180, 270])
            images = [TF.rotate(img, angle) for img in images]
        return images

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        city, sample = self.sample_refs[idx]
        handle = self._get_handle()
        raw_input = np.asarray(handle[city][sample][self.input_column][...], dtype=np.float32)
        input_tensor = _resize_array(raw_input, self.image_size, self.input_metadata)
        raw_topology_tensor = _resize_array(raw_input, self.image_size, None)
        non_ground_mask: Optional[torch.Tensor] = None
        if self.exclude_non_ground_targets:
            non_ground_mask = _resize_mask_nearest(raw_input != self.non_ground_threshold, self.image_size)

        los_input_tensor = None
        los_formula_tensor = None
        los_raw_tensor = None
        if self.los_input_column:
            los_metadata = self.target_metadata.get(self.los_input_column, {})
            raw_los = np.asarray(handle[city][sample][self.los_input_column][...], dtype=np.float32)
            los_input_tensor = _resize_array(raw_los, self.image_size, los_metadata)
            los_formula_tensor = los_input_tensor
            los_raw_tensor = torch.from_numpy(raw_los.astype(np.float32)).unsqueeze(0)

        target_tensors = []
        raw_path_loss_tensor: Optional[torch.Tensor] = None
        path_loss_invalid_mask: Optional[torch.Tensor] = None
        no_data_target_tensor: Optional[torch.Tensor] = None
        for col in self.target_columns:
            field_name = self.target_field_map.get(col, col)
            meta = self.target_metadata.get(col, {})
            if col == 'path_loss':
                raw = np.asarray(handle[city][sample][field_name][...], dtype=np.float32)
                invalid = ~np.isfinite(raw)
                finite_vals = raw[np.isfinite(raw)]
                fill_val = float(np.max(finite_vals)) if finite_vals.size > 0 else 0.0
                raw_fixed = np.where(np.isfinite(raw), raw, fill_val).astype(np.float32)
                if self.path_loss_ignore_nonfinite and invalid.any():
                    path_loss_invalid_mask = _resize_mask_nearest(invalid, self.image_size)
                if meta.get('predict_linear', False):
                    target_tensors.append(_path_loss_db_to_linear_normalized(raw_fixed, self.image_size))
                else:
                    target_tensors.append(_resize_array(raw_fixed, self.image_size, meta))
                if self.path_loss_saturation_db is not None:
                    raw_path_loss_tensor = _resize_array(raw_fixed, self.image_size, None)
            else:
                target_tensors.append(self._read_field(city, sample, field_name, meta))

        if self.append_no_data_target:
            explicit_no_data: Optional[torch.Tensor] = None
            if self.path_loss_no_data_mask_column:
                if self.path_loss_no_data_mask_column in handle[city][sample]:
                    raw_no_data = np.asarray(handle[city][sample][self.path_loss_no_data_mask_column][...])
                    explicit_no_data = _resize_mask_nearest(raw_no_data > 0, self.image_size).float()
            if explicit_no_data is not None:
                no_data_target_tensor = explicit_no_data
            elif self.derive_no_data_from_non_ground and non_ground_mask is not None:
                no_data_target_tensor = non_ground_mask.float()
            else:
                no_data_target_tensor = torch.zeros_like(target_tensors[0])
            target_tensors.append(no_data_target_tensor)

        distance_map_tensor = None
        if self.distance_map_channel:
            distance_map_tensor = _compute_distance_map_2d(self.image_size)

        formula_input_tensor = None
        prior_confidence_tensor = None
        obstruction_feature_tensors: List[torch.Tensor] = []
        formula_cfg = dict(self.path_loss_formula_input)
        if bool(formula_cfg.get('enabled', False)):
            antenna_height_m = 1.0
            if 'antenna_height_m' in self.scalar_feature_columns:
                antenna_height_m = self._resolve_hdf5_scalar_value(city, sample, 'antenna_height_m')
            elif 'antenna_height_m' in self.constant_scalar_features:
                antenna_height_m = float(self.constant_scalar_features['antenna_height_m'])
            path_meta = self.target_metadata.get('path_loss', {'scale': 180.0, 'offset': 0.0, 'clip_min': 0.0, 'clip_max': 180.0})
            precomputed_formula = self._load_precomputed_formula_input(city, sample)
            cache_path = None if precomputed_formula is not None else self._formula_cache_path(city, sample, formula_cfg, path_meta)
            if precomputed_formula is not None:
                formula_input_tensor = precomputed_formula
            elif cache_path is not None and cache_path.exists():
                try:
                    formula_input_tensor = _formula_cache_unpack(torch.load(cache_path, map_location='cpu', weights_only=False))
                except TypeError:
                    formula_input_tensor = _formula_cache_unpack(torch.load(cache_path, map_location='cpu'))
            else:
                building_density = float(np.mean(raw_input != self.non_ground_threshold))
                non_zero = raw_input[raw_input != self.non_ground_threshold]
                building_height_proxy = float(np.mean(non_zero)) if non_zero.size else 0.0
                prior_db = _compute_formula_path_loss_db(
                    image_size=self.image_size,
                    antenna_height_m=antenna_height_m,
                    receiver_height_m=float(formula_cfg.get('receiver_height_m', 1.5)),
                    frequency_ghz=float(formula_cfg.get('frequency_ghz', 7.125)),
                    meters_per_pixel=float(formula_cfg.get('meters_per_pixel', 1.0)),
                    formula_mode=str(formula_cfg.get('formula', 'cost231_hata')),
                    los_tensor=los_formula_tensor,
                    topology_tensor=raw_topology_tensor,
                    a2g_params=dict(formula_cfg.get('a2g_params', {})),
                    building_density=building_density,
                    building_height_proxy=building_height_proxy,
                    non_ground_threshold=float(self.non_ground_threshold),
                    clip_min=float(path_meta.get('clip_min', path_meta.get('offset', 0.0))),
                    clip_max=float(path_meta.get('clip_max', float(path_meta.get('offset', 0.0)) + float(path_meta.get('scale', 180.0)))),
                )
                prior_db = _apply_formula_regime_calibration(
                    prior_db,
                    self.formula_regime_calibration,
                    city=city,
                    density=building_density,
                    height=building_height_proxy,
                    antenna_height_m=float(antenna_height_m),
                    los_tensor=los_formula_tensor,
                    topology_tensor=raw_topology_tensor,
                    distance_map_tensor=distance_map_tensor,
                    non_ground_threshold=float(self.non_ground_threshold),
                    clip_min=float(path_meta.get('clip_min', path_meta.get('offset', 0.0))),
                    clip_max=float(path_meta.get('clip_max', float(path_meta.get('offset', 0.0)) + float(path_meta.get('scale', 180.0)))),
                    prefer_threshold_city_type=bool(formula_cfg.get('prefer_threshold_city_type', False)),
                )
                formula_input_tensor = _normalize_channel(prior_db, path_meta)
                if non_ground_mask is not None:
                    formula_input_tensor = formula_input_tensor * (1.0 - non_ground_mask)
                if cache_path is not None:
                    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
                    try:
                        torch.save(
                            _formula_cache_pack(
                                formula_input_tensor,
                                str(formula_cfg.get('cache_dtype', 'float16')),
                            ),
                            tmp_path,
                        )
                        os.replace(tmp_path, cache_path)
                    except OSError:
                        if tmp_path.exists():
                            tmp_path.unlink(missing_ok=True)

            if formula_input_tensor is not None and bool(formula_cfg.get('include_confidence_channel', False)):
                if distance_map_tensor is None:
                    distance_map_tensor = _compute_distance_map_2d(self.image_size)
                prior_confidence_tensor = _build_prior_confidence_channel(
                    los_formula_tensor,
                    distance_map_tensor,
                    raw_topology_tensor,
                    non_ground_threshold=float(self.non_ground_threshold),
                    kernel_size=int(formula_cfg.get('confidence_kernel_size', 31)),
                )
                if non_ground_mask is not None:
                    prior_confidence_tensor = prior_confidence_tensor * (1.0 - non_ground_mask)

        obstruction_cfg = dict(self.path_loss_obstruction_features)
        if bool(obstruction_cfg.get('enabled', False)):
            precomputed = self._load_precomputed_obstruction_features(city, sample)
            if precomputed is not None:
                (
                    shadow_depth_tensor,
                    distance_since_los_tensor,
                    blocker_height_tensor,
                    blocker_count_tensor,
                ) = precomputed
            else:
                (
                    shadow_depth_tensor,
                    distance_since_los_tensor,
                    blocker_height_tensor,
                    blocker_count_tensor,
                ) = _compute_ray_obstruction_proxy_features(
                    raw_input,
                    los_raw_tensor,
                    non_ground_threshold=self.non_ground_threshold,
                    meters_per_pixel=float(obstruction_cfg.get('meters_per_pixel', formula_cfg.get('meters_per_pixel', 1.0))),
                    angle_bins=int(obstruction_cfg.get('angle_bins', 720)),
                )
                shadow_depth_tensor = TF.resize(
                    shadow_depth_tensor,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                distance_since_los_tensor = TF.resize(
                    distance_since_los_tensor,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                blocker_height_tensor = TF.resize(
                    blocker_height_tensor,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                blocker_count_tensor = TF.resize(
                    blocker_count_tensor,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                if non_ground_mask is not None:
                    valid_ground = (1.0 - non_ground_mask).clamp(0.0, 1.0)
                    shadow_depth_tensor = shadow_depth_tensor * valid_ground
                    distance_since_los_tensor = distance_since_los_tensor * valid_ground
                    blocker_height_tensor = blocker_height_tensor * valid_ground
                    blocker_count_tensor = blocker_count_tensor * valid_ground
            if bool(obstruction_cfg.get('include_shadow_depth', True)):
                obstruction_feature_tensors.append(shadow_depth_tensor)
            if bool(obstruction_cfg.get('include_distance_since_los_break', True)):
                obstruction_feature_tensors.append(distance_since_los_tensor)
            if bool(obstruction_cfg.get('include_max_blocker_height', True)):
                obstruction_feature_tensors.append(blocker_height_tensor)
            if bool(obstruction_cfg.get('include_blocker_count', True)):
                obstruction_feature_tensors.append(blocker_count_tensor)

        if self.augment:
            stack = [input_tensor]
            if los_input_tensor is not None:
                stack.append(los_input_tensor)
            if distance_map_tensor is not None:
                stack.append(distance_map_tensor)
            if formula_input_tensor is not None:
                stack.append(formula_input_tensor)
            if prior_confidence_tensor is not None:
                stack.append(prior_confidence_tensor)
            stack.extend(obstruction_feature_tensors)
            if raw_path_loss_tensor is not None:
                stack.append(raw_path_loss_tensor)
            if path_loss_invalid_mask is not None:
                stack.append(path_loss_invalid_mask)
            if non_ground_mask is not None:
                stack.append(non_ground_mask)
            stack.extend(target_tensors)
            aug = self._apply_sync_aug(stack)
            input_tensor = aug[0]
            cursor = 1
            if los_input_tensor is not None:
                los_input_tensor = aug[cursor]
                cursor += 1
            if distance_map_tensor is not None:
                distance_map_tensor = aug[cursor]
                cursor += 1
            if formula_input_tensor is not None:
                formula_input_tensor = aug[cursor]
                cursor += 1
            if prior_confidence_tensor is not None:
                prior_confidence_tensor = aug[cursor]
                cursor += 1
            if obstruction_feature_tensors:
                new_obstruction: List[torch.Tensor] = []
                for _ in range(len(obstruction_feature_tensors)):
                    new_obstruction.append(aug[cursor])
                    cursor += 1
                obstruction_feature_tensors = new_obstruction
            if raw_path_loss_tensor is not None:
                raw_path_loss_tensor = aug[cursor]
                cursor += 1
            if path_loss_invalid_mask is not None:
                path_loss_invalid_mask = aug[cursor]
                cursor += 1
            if non_ground_mask is not None:
                non_ground_mask = aug[cursor]
                cursor += 1
            target_tensors = aug[cursor:]

        model_input_channels = [input_tensor]
        if los_input_tensor is not None:
            model_input_channels.append(los_input_tensor)
        if distance_map_tensor is not None:
            model_input_channels.append(distance_map_tensor)
        if formula_input_tensor is not None:
            model_input_channels.append(formula_input_tensor)
        if prior_confidence_tensor is not None:
            model_input_channels.append(prior_confidence_tensor)
        model_input_channels.extend(obstruction_feature_tensors)

        # --- Height-aware spatial channels (Gao et al. arXiv:2601.08436) ---
        if self.tx_depth_map_channel or self.elevation_angle_map_channel:
            ah_m = 1.0
            if 'antenna_height_m' in self.scalar_feature_columns:
                ah_m = self._resolve_hdf5_scalar_value(city, sample, 'antenna_height_m')
            elif 'antenna_height_m' in self.constant_scalar_features:
                ah_m = float(self.constant_scalar_features['antenna_height_m'])
            input_scale = float(self.input_metadata.get('scale', 255.0))
            topo_meters = input_tensor.float() * input_scale

        if self.tx_depth_map_channel:
            depth = topo_meters - ah_m
            depth_norm = (depth / max(input_scale, 1.0)).clamp(-1.0, 1.0)
            model_input_channels.append(depth_norm)

        if self.elevation_angle_map_channel:
            dist_map = _compute_distance_map_2d(self.image_size)
            meters_per_pixel = float(self.path_loss_formula_input.get('meters_per_pixel', 1.0))
            half = (self.image_size - 1) / 2.0
            max_dist_px = max(half * (2.0 ** 0.5), 1.0)
            ground_d_m = dist_map.squeeze(0) * max_dist_px * meters_per_pixel
            h_rx = float(self.path_loss_formula_input.get('receiver_height_m', 1.5))
            theta = torch.atan2(
                torch.full_like(ground_d_m, ah_m - h_rx),
                ground_d_m.clamp(min=1.0),
            )
            theta_norm = (torch.rad2deg(theta) / 90.0).clamp(0.0, 1.0).unsqueeze(0)
            model_input_channels.append(theta_norm)

        if self.building_mask_channel:
            input_scale = float(self.input_metadata.get('scale', 255.0))
            bldg_mask = (input_tensor.float() * input_scale > float(self.non_ground_threshold)).float()
            model_input_channels.append(bldg_mask)

        scalar_values: List[float] = []
        if self.add_scalar_channels or self.return_scalar_cond:
            for col in self.scalar_feature_columns:
                norm = self.scalar_norms.get(col, 1.0)
                val = self._resolve_hdf5_scalar_value(city, sample, col)
                scalar_values.append(float(val) / max(norm, 1e-12))
            for col, value in self.constant_scalar_features.items():
                norm = self.scalar_norms.get(col, 1.0)
                scalar_values.append(float(value) / max(norm, 1e-12))

            if self.add_scalar_channels and scalar_values:
                h, w = model_input_channels[0].shape[1:]
                scalar_tensor = torch.tensor(scalar_values, dtype=torch.float32).view(len(scalar_values), 1, 1).expand(
                    len(scalar_values), h, w
                )
                model_input_channels.append(scalar_tensor)

        model_input = torch.cat(model_input_channels, dim=0)
        target_tensor = torch.cat(target_tensors, dim=0)
        mask_tensor = torch.ones_like(target_tensor, dtype=torch.float32)
        if 'path_loss' in self.target_columns:
            path_loss_idx = self.target_columns.index('path_loss')
            if non_ground_mask is not None:
                mask_tensor[path_loss_idx] = mask_tensor[path_loss_idx] * (1.0 - non_ground_mask).clamp(0.0, 1.0)
            if raw_path_loss_tensor is not None and self.path_loss_saturation_db is not None:
                saturated = (raw_path_loss_tensor >= float(self.path_loss_saturation_db)).squeeze(0)
                mask_tensor[path_loss_idx] = mask_tensor[path_loss_idx] * (~saturated).float()
            if self.path_loss_ignore_nonfinite and path_loss_invalid_mask is not None:
                inv = path_loss_invalid_mask.squeeze(0)
                mask_tensor[path_loss_idx] = mask_tensor[path_loss_idx] * (1.0 - inv)
        if self.return_scalar_cond and scalar_values:
            cond = torch.tensor(scalar_values, dtype=torch.float32)
            return model_input, target_tensor, mask_tensor, cond
        return model_input, target_tensor, mask_tensor


def _list_hdf5_samples(hdf5_path: str) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    with h5py.File(hdf5_path, 'r') as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                refs.append((city, sample))
    return refs


def _split_hdf5_samples(
    sample_refs: Sequence[Tuple[str, str]],
    val_ratio: float,
    split_seed: int,
    test_ratio: float = 0.0,
    split_mode: str = "random",
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    refs = list(sample_refs)
    if len(refs) < 2:
        return refs, refs, []

    split_mode = str(split_mode or "random").lower()
    test_ratio = max(0.0, float(test_ratio))
    val_ratio = max(0.0, float(val_ratio))
    if val_ratio + test_ratio >= 1.0:
        raise ValueError('data.val_ratio + data.test_ratio must be < 1.0')

    total = len(refs)
    test_size = int(round(total * test_ratio))
    val_size = int(round(total * val_ratio))

    if test_ratio > 0.0:
        test_size = max(test_size, 1)
    if val_ratio > 0.0:
        val_size = max(val_size, 1)

    max_held_out = max(total - 1, 0)
    if test_size + val_size > max_held_out:
        overflow = test_size + val_size - max_held_out
        reducible_test = max(test_size - (1 if test_ratio > 0.0 else 0), 0)
        reduce_test = min(overflow, reducible_test)
        test_size -= reduce_test
        overflow -= reduce_test
        reducible_val = max(val_size - (1 if val_ratio > 0.0 else 0), 0)
        reduce_val = min(overflow, reducible_val)
        val_size -= reduce_val
        overflow -= reduce_val
        if overflow > 0:
            raise ValueError('Not enough samples to create train/val/test split with the requested ratios.')

    rng = random.Random(split_seed)

    if split_mode in {"city_holdout", "group_by_city", "city"}:
        by_city: Dict[str, List[Tuple[str, str]]] = {}
        for city, sample in refs:
            by_city.setdefault(city, []).append((city, sample))
        city_names = list(by_city.keys())
        if len(city_names) < 3:
            split_mode = "random"
        else:
            rng.shuffle(city_names)
            test_refs: List[Tuple[str, str]] = []
            val_refs: List[Tuple[str, str]] = []
            train_refs: List[Tuple[str, str]] = []
            test_city_count = 0
            val_city_count = 0
            for city in city_names:
                city_refs = by_city[city]
                remaining_cities = len(city_names) - test_city_count - val_city_count
                if len(test_refs) < test_size and remaining_cities > 2:
                    test_refs.extend(city_refs)
                    test_city_count += 1
                    continue
                if len(val_refs) < val_size and remaining_cities > 1:
                    val_refs.extend(city_refs)
                    val_city_count += 1
                    continue
                train_refs.extend(city_refs)
            if not train_refs:
                split_mode = "random"
            else:
                return train_refs, val_refs, test_refs

    if split_mode not in {"random", "sample_random"}:
        raise ValueError(f"Unsupported data.split_mode {split_mode!r}. Expected 'random' or 'city_holdout'.")

    rng.shuffle(refs)
    test_refs = refs[:test_size]
    val_start = test_size
    val_end = val_start + val_size
    val_refs = refs[val_start:val_end]
    train_refs = refs[val_end:]
    return train_refs, val_refs, test_refs


def _resize_mask_nearest(mask_2d: np.ndarray, image_size: int) -> torch.Tensor:
    """Resize a boolean/float mask with nearest-neighbor (preserve sharp invalid regions)."""
    t = torch.from_numpy(np.asarray(mask_2d, dtype=np.float32)).unsqueeze(0)
    out = TF.resize(
        t,
        [image_size, image_size],
        interpolation=InterpolationMode.NEAREST,
    )
    return (out > 0.5).float()


def _load_hdf5_scalar_csv(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    if not path.is_file():
        warnings.warn(f"scalar_table_csv not found ({path}); scalar channels default to 0 until the file exists.")
        return {}
    df = pd.read_csv(path)
    required = {'city', 'sample'}
    cols = set(df.columns.str.strip())
    if not required.issubset(cols):
        raise ValueError(f"scalar_table_csv {path} must contain columns {required}, got {sorted(cols)}")
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    value_cols = [c for c in df.columns if c not in ('city', 'sample')]
    for _, row in df.iterrows():
        key = (str(row['city']).strip(), str(row['sample']).strip())
        payload: Dict[str, float] = {}
        for vc in value_cols:
            v = pd.to_numeric(row[vc], errors='coerce')
            if pd.isna(v):
                payload[vc] = 0.0
            else:
                payload[vc] = float(v)
        out[key] = payload
    return out


def _normalize_hdf5_scalar_specs(raw: Any) -> Dict[str, Dict[str, Optional[str]]]:
    """Map scalar column name -> {from_attr, from_dataset}."""
    specs: Dict[str, Dict[str, Optional[str]]] = {}
    if not raw:
        return specs
    entries = raw if isinstance(raw, list) else []
    for item in entries:
        if not isinstance(item, dict):
            continue
        name = str(item.get('name', '')).strip()
        if not name:
            continue
        specs[name] = {
            'from_attr': item.get('from_attr') or item.get('attr') or None,
            'from_dataset': item.get('from_dataset') or item.get('dataset') or None,
        }
        if specs[name]['from_attr'] is None and specs[name]['from_dataset'] is None:
            specs[name]['from_attr'] = name
            specs[name]['from_dataset'] = name
    return specs


def _read_scalar_from_h5_group(group: h5py.Group, spec: Dict[str, Optional[str]]) -> Optional[float]:
    attr_name = spec.get('from_attr')
    if attr_name and attr_name in group.attrs:
        raw = group.attrs[attr_name]
        try:
            v = float(np.asarray(raw).reshape(-1)[0])
            if np.isfinite(v):
                return v
        except (TypeError, ValueError):
            return None
    ds_name = spec.get('from_dataset')
    if ds_name and ds_name in group:
        ds = group[ds_name]
        try:
            arr = np.asarray(ds[...], dtype=np.float64).reshape(-1)
            if arr.size == 0:
                return None
            v = float(np.nanmean(arr))
            if np.isfinite(v):
                return v
        except (TypeError, ValueError, OSError):
            return None
    return None


def _resolve_hdf5_scalar_value_static(
    group: h5py.Group,
    column: str,
    scalar_specs: Sequence[Dict[str, Any]],
) -> float:
    normalized_specs = _normalize_hdf5_scalar_specs(list(scalar_specs))
    if column in normalized_specs:
        got = _read_scalar_from_h5_group(group, normalized_specs[column])
        if got is not None:
            return float(got)
    if column in group.attrs:
        try:
            raw = group.attrs[column]
            value = float(np.asarray(raw).reshape(-1)[0])
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    if column in group and isinstance(group[column], h5py.Dataset):
        try:
            arr = np.asarray(group[column][...], dtype=np.float64).reshape(-1)
            if arr.size:
                value = float(np.nanmean(arr))
                if np.isfinite(value):
                    return value
        except (TypeError, ValueError, OSError):
            pass
    if column == "antenna_height_m" and "uav_height" in group and isinstance(group["uav_height"], h5py.Dataset):
        try:
            arr = np.asarray(group["uav_height"][...], dtype=np.float64).reshape(-1)
            if arr.size:
                value = float(np.nanmean(arr))
                if np.isfinite(value):
                    return value
        except (TypeError, ValueError, OSError):
            pass
    return 0.0


def _filter_hdf5_refs_by_los_dominance(
    hdf5_path: str,
    refs: Sequence[Tuple[str, str]],
    mode: Optional[str],
    field_name: str,
    threshold: float,
) -> List[Tuple[str, str]]:
    if not mode or str(mode).lower() in ('none', 'null', ''):
        return list(refs)
    mode_l = str(mode).lower()
    if mode_l not in ('los_only', 'nlos_only'):
        raise ValueError(f"data.los_sample_filter must be null, 'los_only', or 'nlos_only', got {mode!r}")
    out: List[Tuple[str, str]] = []
    with h5py.File(hdf5_path, 'r') as handle:
        for city, sample in refs:
            if city not in handle or sample not in handle[city]:
                continue
            grp = handle[city][sample]
            if field_name not in grp:
                continue
            arr = np.asarray(grp[field_name][...], dtype=np.float32)
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            frac_los = float(np.mean(arr > 0.5))
            is_los_dom = frac_los >= float(threshold)
            if mode_l == 'los_only' and is_los_dom:
                out.append((city, sample))
            elif mode_l == 'nlos_only' and not is_los_dom:
                out.append((city, sample))
    return out


def build_dataset_splits_from_config(cfg: Dict[str, Any]) -> Dict[str, Dataset]:
    data_cfg = cfg['data']
    target_columns = list(cfg['target_columns'])
    dataset_format = str(data_cfg.get('format', 'manifest')).lower()
    train_aug = _augmentation_kwargs(cfg)

    if dataset_format == 'manifest':
        common = dict(
            root_dir=data_cfg['root_dir'],
            target_columns=target_columns,
            image_size=int(data_cfg['image_size']),
            add_scalar_channels=add_scalar_channels_from_config(cfg),
            return_scalar_cond=return_scalar_cond_from_config(cfg),
            scalar_feature_columns=list(data_cfg.get('scalar_feature_columns', [])),
            constant_scalar_features=dict(data_cfg.get('constant_scalar_features', {})),
            scalar_feature_norms=dict(data_cfg.get('scalar_feature_norms', {})),
            los_input_column=data_cfg.get('los_input_column'),
        )
        splits: Dict[str, Dataset] = {
            'train': CKMDataset(
                manifest_csv=data_cfg['train_manifest'],
                **train_aug,
                **common,
            ),
            'val': CKMDataset(
                manifest_csv=data_cfg['val_manifest'],
                augment=False,
                **common,
            ),
        }
        test_manifest = data_cfg.get('test_manifest')
        if test_manifest:
            splits['test'] = CKMDataset(
                manifest_csv=test_manifest,
                augment=False,
                **common,
            )
        return splits

    if dataset_format != 'hdf5':
        raise ValueError(f"Unsupported data.format '{dataset_format}'. Expected 'manifest' or 'hdf5'.")

    sample_refs = _list_hdf5_samples(data_cfg['hdf5_path'])
    partition_filter = dict(data_cfg.get("partition_filter", {}))
    topology_class_filter = str(partition_filter.get("topology_class", "")).strip() or None
    antenna_bin_filter = str(partition_filter.get("antenna_bin", "")).strip() or None
    if topology_class_filter or antenna_bin_filter:
        formula_cfg = dict(data_cfg.get("path_loss_formula_input", {}))
        calibration = None
        calibration_path = formula_cfg.get("regime_calibration_json")
        if calibration_path:
            try:
                calibration = _load_formula_regime_calibration(str(calibration_path))
            except FileNotFoundError:
                calibration = None
        antenna_thresholds = dict((calibration or {}).get("antenna_height_thresholds", {}))
        topology_thresholds = _resolve_try54_topology_thresholds(data_cfg, formula_cfg)
        sample_refs = _filter_hdf5_refs_by_partition(
            str(data_cfg["hdf5_path"]),
            sample_refs,
            input_column=str(data_cfg.get("input_column", "topology_map")),
            scalar_specs=list(data_cfg.get("hdf5_scalar_specs", [])),
            constant_scalar_features=dict(data_cfg.get("constant_scalar_features", {})),
            antenna_thresholds=antenna_thresholds,
            topology_thresholds=topology_thresholds,
            topology_class=topology_class_filter,
            antenna_bin=antenna_bin_filter,
            non_ground_threshold=float(data_cfg.get("non_ground_threshold", 0.0)),
        )
        print(
            f"[INFO] partition_filter topology_class={topology_class_filter!r} antenna_bin={antenna_bin_filter!r} "
            f"-> samples={len(sample_refs)}"
        )
    train_refs, val_refs, test_refs = _split_hdf5_samples(
        sample_refs,
        float(data_cfg.get('val_ratio', 0.1)),
        int(data_cfg.get('split_seed', cfg.get('seed', 42))),
        float(data_cfg.get('test_ratio', 0.0)),
        str(data_cfg.get('split_mode', 'random')),
    )
    h5_path = str(data_cfg['hdf5_path'])
    los_mode = data_cfg.get('los_sample_filter')
    if los_mode:
        los_field = str(data_cfg.get('los_classify_field', 'los_mask'))
        los_th = float(data_cfg.get('los_classify_threshold', 0.5))
        train_refs = _filter_hdf5_refs_by_los_dominance(h5_path, train_refs, los_mode, los_field, los_th)
        val_refs = _filter_hdf5_refs_by_los_dominance(h5_path, val_refs, los_mode, los_field, los_th)
        if test_refs:
            test_refs = _filter_hdf5_refs_by_los_dominance(h5_path, test_refs, los_mode, los_field, los_th)
        print(
            f"[INFO] los_sample_filter={los_mode!r} field={los_field} threshold={los_th} "
            f"-> train={len(train_refs)} val={len(val_refs)} test={len(test_refs)}"
        )
    if not train_refs:
        raise ValueError(
            'No HDF5 training samples after split/LoS filter. Check data.los_sample_filter, ratios, and HDF5 path.'
        )
    common_hdf5 = dict(
        hdf5_path=data_cfg['hdf5_path'],
        target_columns=target_columns,
        image_size=int(data_cfg['image_size']),
        add_scalar_channels=add_scalar_channels_from_config(cfg),
        return_scalar_cond=return_scalar_cond_from_config(cfg),
        scalar_feature_columns=list(data_cfg.get('scalar_feature_columns', [])),
        constant_scalar_features=dict(data_cfg.get('constant_scalar_features', {})),
        scalar_feature_norms=dict(data_cfg.get('scalar_feature_norms', {})),
        los_input_column=data_cfg.get('los_input_column'),
        input_column=str(data_cfg.get('input_column', 'topology_map')),
        input_metadata=dict(data_cfg.get('input_metadata', {})),
        target_metadata=dict(cfg.get('target_metadata', {})),
        target_field_map=dict(data_cfg.get('target_field_map', {})),
        distance_map_channel=bool(data_cfg.get('distance_map_channel', False)),
        path_loss_saturation_db=data_cfg.get('path_loss_saturation_db'),
        scalar_table_csv=data_cfg.get('scalar_table_csv'),
        hdf5_scalar_specs=list(data_cfg.get('hdf5_scalar_specs', [])),
        path_loss_ignore_nonfinite=bool(data_cfg.get('path_loss_ignore_nonfinite', True)),
        exclude_non_ground_targets=bool(data_cfg.get('exclude_non_ground_targets', False)),
        non_ground_threshold=float(data_cfg.get('non_ground_threshold', 0.0)),
        path_loss_no_data_mask_column=data_cfg.get('path_loss_no_data_mask_column'),
        derive_no_data_from_non_ground=bool(data_cfg.get('derive_no_data_from_non_ground', False)),
        append_no_data_target=bool(cfg.get('no_data_auxiliary', {}).get('enabled', False)),
        path_loss_formula_input=dict(data_cfg.get('path_loss_formula_input', {})),
        path_loss_obstruction_features=dict(data_cfg.get('path_loss_obstruction_features', {})),
        tx_depth_map_channel=bool(data_cfg.get('tx_depth_map_channel', False)),
        elevation_angle_map_channel=bool(data_cfg.get('elevation_angle_map_channel', False)),
        building_mask_channel=bool(data_cfg.get('building_mask_channel', False)),
    )
    splits = {
        'train': CKMHDF5Dataset(
            sample_refs=train_refs,
            **train_aug,
            **common_hdf5,
        ),
        'val': CKMHDF5Dataset(
            sample_refs=val_refs,
            augment=False,
            **common_hdf5,
        ),
    }
    if test_refs:
        splits['test'] = CKMHDF5Dataset(
            sample_refs=test_refs,
            augment=False,
            **common_hdf5,
        )
    return splits


def build_datasets_from_config(cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    splits = build_dataset_splits_from_config(cfg)
    return splits['train'], splits['val']


def build_cross_validation_datasets_from_config(cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    data_cfg = cfg['data']
    target_columns = list(cfg['target_columns'])
    dataset_format = str(data_cfg.get('format', 'manifest')).lower()
    train_aug = _augmentation_kwargs(cfg)

    if dataset_format == 'manifest':
        common = dict(
            root_dir=data_cfg['root_dir'],
            target_columns=target_columns,
            image_size=int(data_cfg['image_size']),
            add_scalar_channels=add_scalar_channels_from_config(cfg),
            return_scalar_cond=return_scalar_cond_from_config(cfg),
            scalar_feature_columns=list(data_cfg.get('scalar_feature_columns', [])),
            constant_scalar_features=dict(data_cfg.get('constant_scalar_features', {})),
            scalar_feature_norms=dict(data_cfg.get('scalar_feature_norms', {})),
            los_input_column=data_cfg.get('los_input_column'),
        )
        train_manifest = data_cfg['train_manifest']
        val_manifest = data_cfg['val_manifest']
        dev_train = torch.utils.data.ConcatDataset(
            [
                CKMDataset(
                    manifest_csv=train_manifest,
                    **train_aug,
                    **common,
                ),
                CKMDataset(
                    manifest_csv=val_manifest,
                    **train_aug,
                    **common,
                ),
            ]
        )
        dev_eval = torch.utils.data.ConcatDataset(
            [
                CKMDataset(manifest_csv=train_manifest, augment=False, **common),
                CKMDataset(manifest_csv=val_manifest, augment=False, **common),
            ]
        )
        test_manifest = data_cfg.get('test_manifest')
        test_dataset: Optional[Dataset] = None
        if test_manifest:
            test_dataset = CKMDataset(manifest_csv=test_manifest, augment=False, **common)
        return dev_train, dev_eval, test_dataset

    if dataset_format != 'hdf5':
        raise ValueError(f"Unsupported data.format '{dataset_format}'. Expected 'manifest' or 'hdf5'.")

    splits = build_dataset_splits_from_config(cfg)
    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits.get('test')
    if not isinstance(train_dataset, CKMHDF5Dataset) or not isinstance(val_dataset, CKMHDF5Dataset):
        raise TypeError('Expected CKMHDF5Dataset instances for HDF5 cross-validation.')

    dev_refs = list(train_dataset.sample_refs) + list(val_dataset.sample_refs)
    common_hdf5 = dict(
        hdf5_path=data_cfg['hdf5_path'],
        target_columns=target_columns,
        image_size=int(data_cfg['image_size']),
        add_scalar_channels=add_scalar_channels_from_config(cfg),
        return_scalar_cond=return_scalar_cond_from_config(cfg),
        scalar_feature_columns=list(data_cfg.get('scalar_feature_columns', [])),
        constant_scalar_features=dict(data_cfg.get('constant_scalar_features', {})),
        scalar_feature_norms=dict(data_cfg.get('scalar_feature_norms', {})),
        los_input_column=data_cfg.get('los_input_column'),
        input_column=str(data_cfg.get('input_column', 'topology_map')),
        input_metadata=dict(data_cfg.get('input_metadata', {})),
        target_metadata=dict(cfg.get('target_metadata', {})),
        target_field_map=dict(data_cfg.get('target_field_map', {})),
        distance_map_channel=bool(data_cfg.get('distance_map_channel', False)),
        path_loss_saturation_db=data_cfg.get('path_loss_saturation_db'),
        scalar_table_csv=data_cfg.get('scalar_table_csv'),
        hdf5_scalar_specs=list(data_cfg.get('hdf5_scalar_specs', [])),
        path_loss_ignore_nonfinite=bool(data_cfg.get('path_loss_ignore_nonfinite', True)),
        exclude_non_ground_targets=bool(data_cfg.get('exclude_non_ground_targets', False)),
        non_ground_threshold=float(data_cfg.get('non_ground_threshold', 0.0)),
        path_loss_no_data_mask_column=data_cfg.get('path_loss_no_data_mask_column'),
        derive_no_data_from_non_ground=bool(data_cfg.get('derive_no_data_from_non_ground', False)),
        append_no_data_target=bool(cfg.get('no_data_auxiliary', {}).get('enabled', False)),
        path_loss_formula_input=dict(data_cfg.get('path_loss_formula_input', {})),
        path_loss_obstruction_features=dict(data_cfg.get('path_loss_obstruction_features', {})),
        tx_depth_map_channel=bool(data_cfg.get('tx_depth_map_channel', False)),
        elevation_angle_map_channel=bool(data_cfg.get('elevation_angle_map_channel', False)),
        building_mask_channel=bool(data_cfg.get('building_mask_channel', False)),
    )
    dev_train = CKMHDF5Dataset(
        sample_refs=dev_refs,
        **train_aug,
        **common_hdf5,
    )
    dev_eval = CKMHDF5Dataset(
        sample_refs=dev_refs,
        augment=False,
        **common_hdf5,
    )
    return dev_train, dev_eval, test_dataset
