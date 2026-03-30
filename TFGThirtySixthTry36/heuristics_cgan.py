from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def denormalize_array(arr: np.ndarray, metadata: Dict[str, object]) -> np.ndarray:
    if metadata.get('predict_linear', False):
        arr = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
        log_linear = arr * 18.0 - 18.0
        linear = np.power(10.0, log_linear)
        linear = np.clip(linear, 1e-18, 1.0)
        return (-10.0 * np.log10(linear)).astype(np.float32)
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


def apply_path_loss_los_correction(
    pred_pl_db: np.ndarray,
    los_mask: np.ndarray,
    distance_map_normalized: np.ndarray,
    frequency_ghz: float = 7.125,
    blend_weight: float = 0.3,
    max_distance_m: float = 362.0,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    In LoS regions, blend predicted path loss with free-space prior.
    PL_fs = 20*log10(d) + 20*log10(f) + 92.45 (d in m, f in GHz).
    distance_map_normalized: [0,1] from _compute_distance_map_2d.
    """
    los = np.asarray(los_mask, dtype=np.float32)
    d = np.asarray(distance_map_normalized, dtype=np.float32) * max_distance_m
    d = np.clip(d, 1.0, np.inf)
    pl_fs = 20.0 * np.log10(d) + 20.0 * np.log10(frequency_ghz) + 92.45
    blend = los * blend_weight
    if valid_mask is not None:
        blend = blend * np.asarray(valid_mask, dtype=np.float32)
    return ((1.0 - blend) * pred_pl_db + blend * pl_fs).astype(np.float32)


def compute_distance_map_normalized(shape: tuple[int, int], max_distance_m: float = 362.0) -> np.ndarray:
    height, width = shape
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing='ij')
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist_pixels = float(np.sqrt(cx ** 2 + cy ** 2))
    if max_dist_pixels <= 0.0:
        return np.zeros((height, width), dtype=np.float32)
    normalized = dist / max_dist_pixels
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def build_path_loss_heuristic_prior(
    pred_pl_db: np.ndarray,
    metadata: Dict[str, object],
    kernel_size: int = 3,
    los_mask: Optional[np.ndarray] = None,
    los_correction_enabled: bool = False,
    frequency_ghz: float = 7.125,
    blend_weight: float = 0.3,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    prior = apply_regression_heuristics(pred_pl_db, metadata, kernel_size=kernel_size)
    if los_correction_enabled and los_mask is not None:
        distance_map = compute_distance_map_normalized(prior.shape)
        prior = apply_path_loss_los_correction(
            prior,
            los_mask,
            distance_map,
            frequency_ghz=frequency_ghz,
            blend_weight=blend_weight,
            valid_mask=valid_mask,
        )
    return prior.astype(np.float32)


def apply_path_loss_confidence_fallback(
    pred_pl_db: np.ndarray,
    confidence_map: np.ndarray,
    metadata: Dict[str, object],
    confidence_threshold: float = 0.5,
    fallback_mode: str = 'replace',
    kernel_size: int = 3,
    los_mask: Optional[np.ndarray] = None,
    los_correction_enabled: bool = False,
    frequency_ghz: float = 7.125,
    blend_weight: float = 0.3,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    pred = np.asarray(pred_pl_db, dtype=np.float32)
    conf = np.clip(np.asarray(confidence_map, dtype=np.float32), 0.0, 1.0)
    heuristic = build_path_loss_heuristic_prior(
        pred,
        metadata,
        kernel_size=kernel_size,
        los_mask=los_mask,
        los_correction_enabled=los_correction_enabled,
        frequency_ghz=frequency_ghz,
        blend_weight=blend_weight,
        valid_mask=valid_mask,
    )
    if fallback_mode == 'blend':
        final = (conf * pred + (1.0 - conf) * heuristic).astype(np.float32)
    else:
        final = np.where(conf >= float(confidence_threshold), pred, heuristic).astype(np.float32)
    return {
        'final_path_loss_db': final,
        'heuristic_path_loss_db': heuristic,
        'confidence_map': conf,
        'low_confidence_mask': (conf < float(confidence_threshold)).astype(np.float32),
    }


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
