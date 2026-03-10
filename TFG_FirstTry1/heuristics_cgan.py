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
