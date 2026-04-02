from __future__ import annotations

import json
import math
import random
import warnings
from collections import defaultdict
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


def _compute_formula_path_loss_db(
    image_size: int,
    antenna_height_m: float,
    receiver_height_m: float,
    frequency_ghz: float,
    meters_per_pixel: float,
    formula_mode: str,
    los_tensor: Optional[torch.Tensor] = None,
    a2g_params: Optional[Dict[str, float]] = None,
    clip_min: float = 0.0,
    clip_max: float = 180.0,
) -> torch.Tensor:
    distance_norm = _compute_distance_map_2d(image_size)
    if los_tensor is not None:
        distance_norm = distance_norm.to(device=los_tensor.device, dtype=los_tensor.dtype)
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

    fspl_db = 32.45 + 20.0 * torch.log10(d3d_m / 1000.0) + 20.0 * math.log10(freq_mhz)

    if mode == 'two_ray_ground':
        wavelength_m = 0.299792458 / freq_ghz
        crossover_m = max((4.0 * math.pi * h_tx * h_rx) / wavelength_m, 1.0)
        two_ray_db = 40.0 * torch.log10(d3d_m) - 20.0 * math.log10(h_tx) - 20.0 * math.log10(h_rx)
        path_db = torch.where(d3d_m <= crossover_m, fspl_db, two_ray_db)
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
        lambda0_db = 20.0 * math.log10((4.0 * math.pi * h_tx * freq_ghz * 1e9) / 299792458.0)
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
    elif mode == 'a2g_paper_eq9_eq10_eq11':
        theta_deg = torch.rad2deg(torch.atan2((h_tx - h_rx) * torch.ones_like(d2d_m), d2d_m.clamp(min=1.0)))
        sin_theta = torch.sin(torch.deg2rad(theta_deg)).clamp(min=1e-4)
        lambda0_db = 20.0 * math.log10((4.0 * math.pi * h_tx * freq_ghz * 1e9) / 299792458.0)
        los_log_coeff = float(a2g_params.get("los_log_coeff", -20.0))
        los_bias = float(a2g_params.get("los_bias", 0.0))
        nlos_bias = float(a2g_params.get("nlos_bias", -16.16))
        nlos_amp = float(a2g_params.get("nlos_amp", 12.0436))
        nlos_tau = max(float(a2g_params.get("nlos_tau", 7.52)), 1e-3)
        los_path_db = lambda0_db + los_bias + los_log_coeff * torch.log10(sin_theta)
        nlos_path_db = lambda0_db + (nlos_bias + nlos_amp * torch.exp(-(90.0 - theta_deg) / nlos_tau))
        if los_tensor is None:
            path_db = 0.5 * (los_path_db + nlos_path_db)
        else:
            los_prob = los_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
            path_db = los_prob * los_path_db + (1.0 - los_prob) * nlos_path_db
    else:
        path_db = fspl_db

    return path_db.clamp(min=float(clip_min), max=float(clip_max))


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


def _apply_formula_regime_calibration(
    prior_db: torch.Tensor,
    calibration: Optional[Dict[str, Any]],
    *,
    city: str,
    density: float,
    height: float,
    antenna_height_m: float,
    los_tensor: Optional[torch.Tensor],
    distance_tensor: Optional[torch.Tensor],
    topology_tensor: Optional[torch.Tensor],
    non_ground_threshold: float,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    if calibration is None:
        return prior_db.clamp(min=float(clip_min), max=float(clip_max))

    city_type_map = dict(calibration.get("city_type_by_city", {}))
    city_type = city_type_map.get(city)
    if city_type is None:
        city_type = _city_type_from_thresholds(density, height, dict(calibration.get("city_type_thresholds", {})))
    ant_bin = _antenna_height_bin(float(antenna_height_m), dict(calibration.get("antenna_height_thresholds", {})))
    coeff_map = dict(calibration.get("coefficients", {}))
    model_type = str(calibration.get("model_type", "quadratic_regime"))
    local_density = None
    local_height = None
    if model_type in {"regime_obstruction_linear_v1", "regime_obstruction_multiscale_v1"}:
        kernel_sizes = calibration.get("local_kernel_sizes", [calibration.get("local_kernel_size", 25)])
        kernel_sizes = [int(k) for k in kernel_sizes]
        if topology_tensor is None:
            raise ValueError("Obstruction-aware prior calibration requires topology_tensor.")
        if distance_tensor is None:
            distance_tensor = _compute_distance_map_2d(prior_db.shape[-1])
        distance_tensor = distance_tensor.to(dtype=prior_db.dtype)
        local_features = []
        for kernel_size in kernel_sizes:
            local_density, local_height = _compute_local_obstruction_features(
                topology_tensor,
                non_ground_threshold=non_ground_threshold,
                kernel_size=kernel_size,
            )
            local_features.append(
                (
                    local_density.to(dtype=prior_db.dtype),
                    local_height.to(dtype=prior_db.dtype),
                )
            )
        los_features = []
        base_los_tensor = los_tensor if los_tensor is not None else torch.zeros_like(prior_db)
        for kernel_size in kernel_sizes:
            nlos_support = F.avg_pool2d(
                (base_los_tensor <= 0.5).to(dtype=torch.float32).unsqueeze(0),
                kernel_size=kernel_size if kernel_size % 2 == 1 else kernel_size + 1,
                stride=1,
                padding=(kernel_size if kernel_size % 2 == 1 else kernel_size + 1) // 2,
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
            logd = torch.log1p(distance_tensor * distance_scale)
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
            ground_distance_m = distance_tensor * distance_scale
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
        if bool(cfg['data'].get('path_loss_formula_input', {}).get('include_shadow_sigma_channel', False)):
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
        path_loss_formula_input: Optional[Dict[str, Any]] = None,
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
        self.path_loss_saturation_db = path_loss_saturation_db
        self.path_loss_ignore_nonfinite = bool(path_loss_ignore_nonfinite)
        self.exclude_non_ground_targets = bool(exclude_non_ground_targets)
        self.non_ground_threshold = float(non_ground_threshold)
        self.path_loss_formula_input = dict(path_loss_formula_input or {})
        self.formula_regime_calibration = _load_formula_regime_calibration(
            self.path_loss_formula_input.get("regime_calibration_json")
        )
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

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

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

    def _read_field(self, city: str, sample: str, field_name: str, metadata: Optional[Dict[str, Any]]) -> torch.Tensor:
        handle = self._get_handle()
        if field_name not in handle[city][sample]:
            raise KeyError(f"Field '{field_name}' not found in {city}/{sample}")
        return _resize_array(handle[city][sample][field_name][...], self.image_size, metadata)

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
        non_ground_mask: Optional[torch.Tensor] = None
        if self.exclude_non_ground_targets:
            non_ground_mask = _resize_mask_nearest(raw_input != self.non_ground_threshold, self.image_size)

        los_input_tensor = None
        los_formula_tensor = None
        if self.los_input_column:
            los_metadata = self.target_metadata.get(self.los_input_column, {})
            raw_los = np.asarray(handle[city][sample][self.los_input_column][...], dtype=np.float32)
            los_input_tensor = _resize_array(raw_los, self.image_size, los_metadata)
            los_formula_tensor = los_input_tensor

        target_tensors = []
        raw_path_loss_tensor: Optional[torch.Tensor] = None
        path_loss_invalid_mask: Optional[torch.Tensor] = None
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

        distance_map_tensor = None
        if self.distance_map_channel:
            distance_map_tensor = _compute_distance_map_2d(self.image_size)

        formula_input_tensor = None
        sigma_input_tensor = None
        formula_cfg = dict(self.path_loss_formula_input)
        if bool(formula_cfg.get('enabled', False)):
            antenna_height_m = 1.0
            if 'antenna_height_m' in self.scalar_feature_columns:
                antenna_height_m = self._resolve_hdf5_scalar_value(city, sample, 'antenna_height_m')
            elif 'antenna_height_m' in self.constant_scalar_features:
                antenna_height_m = float(self.constant_scalar_features['antenna_height_m'])
            path_meta = self.target_metadata.get('path_loss', {'scale': 180.0, 'offset': 0.0, 'clip_min': 0.0, 'clip_max': 180.0})
            prior_db = _compute_formula_path_loss_db(
                image_size=self.image_size,
                antenna_height_m=antenna_height_m,
                receiver_height_m=float(formula_cfg.get('receiver_height_m', 1.5)),
                frequency_ghz=float(formula_cfg.get('frequency_ghz', 7.125)),
                meters_per_pixel=float(formula_cfg.get('meters_per_pixel', 1.0)),
                formula_mode=str(formula_cfg.get('formula', 'cost231_hata')),
                los_tensor=los_formula_tensor,
                a2g_params=dict(formula_cfg.get('a2g_params', {})),
                clip_min=float(path_meta.get('clip_min', path_meta.get('offset', 0.0))),
                clip_max=float(path_meta.get('clip_max', float(path_meta.get('offset', 0.0)) + float(path_meta.get('scale', 180.0)))),
            )
            building_density = float(np.mean(raw_input != self.non_ground_threshold))
            non_zero = raw_input[raw_input != self.non_ground_threshold]
            building_height_proxy = float(np.mean(non_zero)) if non_zero.size else 0.0
            prior_db = _apply_formula_regime_calibration(
                prior_db,
                self.formula_regime_calibration,
                city=city,
                density=building_density,
            height=building_height_proxy,
            antenna_height_m=float(antenna_height_m),
            los_tensor=los_formula_tensor,
            distance_tensor=distance_map_tensor,
            topology_tensor=input_tensor,
            non_ground_threshold=self.non_ground_threshold,
            clip_min=float(path_meta.get('clip_min', path_meta.get('offset', 0.0))),
            clip_max=float(path_meta.get('clip_max', float(path_meta.get('offset', 0.0)) + float(path_meta.get('scale', 180.0)))),
        )
            # prior_db already has channel dimension [1, H, W]; do not add a second one
            formula_input_tensor = _normalize_channel(prior_db, path_meta)
            if non_ground_mask is not None:
                formula_input_tensor = formula_input_tensor * (1.0 - non_ground_mask)
            if bool(formula_cfg.get('include_shadow_sigma_channel', False)):
                sigma_db = _compute_a2g_shadow_sigma_db(
                    image_size=self.image_size,
                    antenna_height_m=float(antenna_height_m),
                    receiver_height_m=float(formula_cfg.get('receiver_height_m', 1.5)),
                    meters_per_pixel=float(formula_cfg.get('meters_per_pixel', 1.0)),
                    los_tensor=los_formula_tensor,
                    a2g_params=dict(formula_cfg.get('a2g_params', {})),
                )
                sigma_meta = {'scale': float(formula_cfg.get('shadow_sigma_scale', 12.0)), 'offset': 0.0}
                sigma_input_tensor = _normalize_channel(sigma_db, sigma_meta)
                if non_ground_mask is not None:
                    sigma_input_tensor = sigma_input_tensor * (1.0 - non_ground_mask)

        if self.augment:
            stack = [input_tensor]
            if los_input_tensor is not None:
                stack.append(los_input_tensor)
            if distance_map_tensor is not None:
                stack.append(distance_map_tensor)
            if formula_input_tensor is not None:
                stack.append(formula_input_tensor)
            if sigma_input_tensor is not None:
                stack.append(sigma_input_tensor)
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
            if sigma_input_tensor is not None:
                sigma_input_tensor = aug[cursor]
                cursor += 1
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
        if sigma_input_tensor is not None:
            model_input_channels.append(sigma_input_tensor)

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
        if non_ground_mask is not None:
            mask_tensor = mask_tensor * (1.0 - non_ground_mask).clamp(0.0, 1.0)
        if 'path_loss' in self.target_columns:
            path_loss_idx = self.target_columns.index('path_loss')
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
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    refs = list(sample_refs)
    if len(refs) < 2:
        return refs, refs, []

    rng = random.Random(split_seed)
    rng.shuffle(refs)
    total = len(refs)

    test_ratio = max(0.0, float(test_ratio))
    val_ratio = max(0.0, float(val_ratio))
    if val_ratio + test_ratio >= 1.0:
        raise ValueError('data.val_ratio + data.test_ratio must be < 1.0')

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
                augment=bool(cfg['augmentation']['enable']),
                hflip_prob=float(cfg['augmentation']['hflip_prob']),
                vflip_prob=float(cfg['augmentation']['vflip_prob']),
                rot90_prob=float(cfg['augmentation']['rot90_prob']),
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
    train_refs, val_refs, test_refs = _split_hdf5_samples(
        sample_refs,
        float(data_cfg.get('val_ratio', 0.1)),
        int(data_cfg.get('split_seed', cfg.get('seed', 42))),
        float(data_cfg.get('test_ratio', 0.0)),
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
        path_loss_formula_input=dict(data_cfg.get('path_loss_formula_input', {})),
    )
    splits = {
        'train': CKMHDF5Dataset(
            sample_refs=train_refs,
            augment=bool(cfg['augmentation']['enable']),
            hflip_prob=float(cfg['augmentation']['hflip_prob']),
            vflip_prob=float(cfg['augmentation']['vflip_prob']),
            rot90_prob=float(cfg['augmentation']['rot90_prob']),
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
                    augment=bool(cfg['augmentation']['enable']),
                    hflip_prob=float(cfg['augmentation']['hflip_prob']),
                    vflip_prob=float(cfg['augmentation']['vflip_prob']),
                    rot90_prob=float(cfg['augmentation']['rot90_prob']),
                    **common,
                ),
                CKMDataset(
                    manifest_csv=val_manifest,
                    augment=bool(cfg['augmentation']['enable']),
                    hflip_prob=float(cfg['augmentation']['hflip_prob']),
                    vflip_prob=float(cfg['augmentation']['vflip_prob']),
                    rot90_prob=float(cfg['augmentation']['rot90_prob']),
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
        path_loss_formula_input=dict(data_cfg.get('path_loss_formula_input', {})),
    )
    dev_train = CKMHDF5Dataset(
        sample_refs=dev_refs,
        augment=bool(cfg['augmentation']['enable']),
        hflip_prob=float(cfg['augmentation']['hflip_prob']),
        vflip_prob=float(cfg['augmentation']['vflip_prob']),
        rot90_prob=float(cfg['augmentation']['rot90_prob']),
        **common_hdf5,
    )
    dev_eval = CKMHDF5Dataset(
        sample_refs=dev_refs,
        augment=False,
        **common_hdf5,
    )
    return dev_train, dev_eval, test_dataset
