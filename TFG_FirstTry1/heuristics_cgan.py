from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def denormalize_array(arr: np.ndarray, metadata: Dict[str, object]) -> np.ndarray:
    scale = float(metadata.get('scale', 1.0))
    offset = float(metadata.get('offset', 0.0))
    return arr * scale + offset


def median_filter_2d(arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1:
        return arr
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd')

    pad = kernel_size // 2
    padded = np.pad(arr, pad_width=pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    return np.median(windows, axis=(-1, -2))


def apply_regression_heuristics(
    arr_physical: np.ndarray,
    metadata: Dict[str, object],
    kernel_size: int = 3,
) -> np.ndarray:
    out = np.array(arr_physical, copy=True)

    clip_min = metadata.get('clip_min')
    clip_max = metadata.get('clip_max')
    if clip_min is not None or clip_max is not None:
        out = np.clip(
            out,
            clip_min if clip_min is not None else -np.inf,
            clip_max if clip_max is not None else np.inf,
        )

    if kernel_size > 1:
        out = median_filter_2d(out, kernel_size=kernel_size)

    return out


def apply_augmented_los_heuristics(
    soft_map: np.ndarray,
    binary_los_input: Optional[np.ndarray] = None,
    kernel_size: int = 3,
    threshold: float = 0.5,
    export_binary: bool = False,
    enforce_binary_los_consistency: bool = False,
    binary_los_consistency_floor: float = 0.5,
) -> Dict[str, np.ndarray]:
    probs = np.clip(soft_map, 0.0, 1.0)

    if kernel_size > 1:
        probs = median_filter_2d(probs, kernel_size=kernel_size)

    if binary_los_input is not None and enforce_binary_los_consistency:
        probs = np.maximum(probs, binary_los_input * float(binary_los_consistency_floor))

    outputs = {
        'probabilities': probs,
    }
    if export_binary:
        outputs['binary'] = (probs >= threshold).astype(np.float32)
    return outputs


def apply_binary_mask_heuristics(
    mask_map: np.ndarray,
    threshold: float = 0.5,
    export_binary: bool = False,
) -> Dict[str, np.ndarray]:
    probs = np.clip(mask_map, 0.0, 1.0)

    outputs = {
        'probabilities': probs,
    }
    if export_binary:
        outputs['binary'] = (probs >= threshold).astype(np.float32)
    return outputs


def derive_channel_power_from_path_loss(
    path_loss_db: np.ndarray,
    tx_power_dbm: float,
    tx_gain_dbi: float = 0.0,
    rx_gain_dbi: float = 0.0,
    other_losses_db: float = 0.0,
) -> np.ndarray:
    return (
        float(tx_power_dbm)
        + float(tx_gain_dbi)
        + float(rx_gain_dbi)
        - float(other_losses_db)
        - np.asarray(path_loss_db, dtype=np.float32)
    ).astype(np.float32)


def derive_snr_maps(
    rx_power_dbm: np.ndarray,
    bandwidth_hz: float,
    noise_figure_db: float = 0.0,
) -> Dict[str, np.ndarray]:
    bandwidth_hz = max(float(bandwidth_hz), 1.0)
    noise_floor_dbm = -174.0 + 10.0 * np.log10(bandwidth_hz) + float(noise_figure_db)
    snr_db = np.asarray(rx_power_dbm, dtype=np.float32) - noise_floor_dbm
    snr_linear = np.power(10.0, snr_db / 10.0, dtype=np.float32)
    return {
        'noise_floor_dbm': np.full_like(snr_db, noise_floor_dbm, dtype=np.float32),
        'snr_db': snr_db.astype(np.float32),
        'snr_linear': snr_linear.astype(np.float32),
    }


def derive_link_availability(
    rx_power_dbm: np.ndarray,
    reception_threshold_dbm: float,
) -> np.ndarray:
    return (np.asarray(rx_power_dbm, dtype=np.float32) >= float(reception_threshold_dbm)).astype(np.float32)
