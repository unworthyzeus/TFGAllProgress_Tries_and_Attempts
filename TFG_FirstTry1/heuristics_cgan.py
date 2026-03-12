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


def find_best_binary_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    probs = np.asarray(probabilities, dtype=np.float32)
    truth = np.asarray(targets, dtype=np.float32)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19, dtype=np.float32)

    best = {
        'best_threshold': 0.5,
        'best_f1': -1.0,
        'best_accuracy': 0.0,
        'best_iou': 0.0,
        'best_precision': 0.0,
        'best_recall': 0.0,
    }

    truth_binary = truth >= 0.5
    total = max(int(truth_binary.size), 1)
    for threshold in thresholds:
        pred_binary = probs >= float(threshold)
        tp = int(np.logical_and(pred_binary, truth_binary).sum())
        tn = int(np.logical_and(~pred_binary, ~truth_binary).sum())
        fp = int(np.logical_and(pred_binary, ~truth_binary).sum())
        fn = int(np.logical_and(~pred_binary, truth_binary).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        accuracy = (tp + tn) / total
        iou = tp / max(tp + fp + fn, 1)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)

        if f1 > best['best_f1'] or (abs(f1 - best['best_f1']) < 1e-12 and accuracy > best['best_accuracy']):
            best = {
                'best_threshold': float(threshold),
                'best_f1': float(f1),
                'best_accuracy': float(accuracy),
                'best_iou': float(iou),
                'best_precision': float(precision),
                'best_recall': float(recall),
            }
    return best


def evaluate_binary_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    probs = np.asarray(probabilities, dtype=np.float32)
    truth = np.asarray(targets, dtype=np.float32)
    truth_binary = truth >= 0.5
    pred_binary = probs >= float(threshold)

    tp = int(np.logical_and(pred_binary, truth_binary).sum())
    tn = int(np.logical_and(~pred_binary, ~truth_binary).sum())
    fp = int(np.logical_and(pred_binary, ~truth_binary).sum())
    fn = int(np.logical_and(~pred_binary, truth_binary).sum())

    total = max(int(truth_binary.size), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / total
    iou = tp / max(tp + fp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    return {
        'threshold': float(threshold),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
    }


def derive_augmented_los_heuristic(
    los_probabilities: np.ndarray,
    path_loss_db: Optional[np.ndarray] = None,
    delay_spread_ns: Optional[np.ndarray] = None,
    angular_spread_deg: Optional[np.ndarray] = None,
    binary_los_input: Optional[np.ndarray] = None,
    kernel_size: int = 3,
    threshold: float = 0.5,
    export_binary: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    default_weights = {
        'los_mask': 0.55,
        'path_loss': 0.25,
        'delay_spread': 0.10,
        'angular_spread': 0.10,
        'binary_los_floor': 0.75,
    }
    merged_weights = {**default_weights, **(weights or {})}

    probs = np.clip(np.asarray(los_probabilities, dtype=np.float32), 0.0, 1.0)
    weighted_sum = merged_weights['los_mask'] * probs
    total_weight = merged_weights['los_mask']

    if path_loss_db is not None:
        path_loss_arr = np.asarray(path_loss_db, dtype=np.float32)
        path_loss_support = 1.0 - np.clip(path_loss_arr / 180.0, 0.0, 1.0)
        weighted_sum += merged_weights['path_loss'] * path_loss_support
        total_weight += merged_weights['path_loss']

    if delay_spread_ns is not None:
        delay_arr = np.asarray(delay_spread_ns, dtype=np.float32)
        delay_support = 1.0 - np.clip(delay_arr / 1000.0, 0.0, 1.0)
        weighted_sum += merged_weights['delay_spread'] * delay_support
        total_weight += merged_weights['delay_spread']

    if angular_spread_deg is not None:
        angular_arr = np.asarray(angular_spread_deg, dtype=np.float32)
        angular_support = 1.0 - np.clip(angular_arr / 180.0, 0.0, 1.0)
        weighted_sum += merged_weights['angular_spread'] * angular_support
        total_weight += merged_weights['angular_spread']

    augmented = weighted_sum / max(total_weight, 1e-12)
    augmented = np.clip(augmented, 0.0, 1.0)

    if binary_los_input is not None:
        augmented = np.maximum(augmented, np.asarray(binary_los_input, dtype=np.float32) * float(merged_weights['binary_los_floor']))

    if kernel_size > 1:
        augmented = median_filter_2d(augmented, kernel_size=kernel_size)

    outputs = {'probabilities': augmented.astype(np.float32)}
    if export_binary:
        outputs['binary'] = (augmented >= float(threshold)).astype(np.float32)
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
