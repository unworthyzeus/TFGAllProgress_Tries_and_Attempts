from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from config_utils import (
    anchor_data_paths_to_config_file,
    ensure_output_dir,
    is_cuda_device,
    load_config,
    load_torch_checkpoint,
    move_optimizer_state_to_device,
    resolve_device,
)
from data_utils import (
    build_datasets_from_config,
    compute_input_channels,
    compute_scalar_cond_dim,
    forward_expert_model,
    log_scalar_data_report,
    path_loss_linear_normalized_to_db,
    unpack_expert_batch,
    uses_scalar_film_conditioning,
)
from evaluate_topology_expert import (
    build_loader_for_split,
    denormalize_channel,
    finalize_metric_totals,
    init_metric_totals,
    load_saved_heuristic_calibration,
    summarize_loader,
    update_metric_totals,
)
from topology_expert_heuristics import apply_path_loss_confidence_fallback
from model_topology_expert import UNetGenerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_context(device_cfg: str, cli_local_rank: int | None = None) -> Dict[str, Any]:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        device = resolve_device(device_cfg)
        return {
            'distributed': False,
            'rank': 0,
            'local_rank': 0,
            'world_size': 1,
            'device': device,
            'is_main_process': True,
        }

    if not torch.cuda.is_available():
        raise RuntimeError('DistributedDataParallel requires CUDA GPUs, but torch.cuda.is_available() is false.')

    local_rank = int(os.environ.get('LOCAL_RANK', cli_local_rank if cli_local_rank is not None else 0))
    rank = int(os.environ.get('RANK', '0'))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda', local_rank)
    return {
        'distributed': True,
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'device': device,
        'is_main_process': rank == 0,
    }


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def create_ema_model(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def update_ema_model(ema_model: nn.Module, source_model: nn.Module, decay: float) -> None:
    if decay <= 0.0:
        return
    ema_state = ema_model.state_dict()
    source_state = source_model.state_dict()
    with torch.no_grad():
        for name, source_value in source_state.items():
            ema_value = ema_state[name]
            if torch.is_floating_point(source_value):
                ema_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(source_value)


def reduce_training_loss(
    g_running: float,
    batch_count: int,
    device: Any,
    distributed: bool,
) -> float:
    if not distributed:
        denom = max(batch_count, 1)
        return g_running / denom

    stats = torch.tensor([g_running, float(batch_count)], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    denom = max(float(stats[1].item()), 1.0)
    return float(stats[0].item() / denom)


def resolve_resume_checkpoint(out_dir: Path, configured_resume: str | None) -> Path | None:
    if configured_resume:
        resume_path = Path(configured_resume)
        return resume_path if resume_path.exists() else None

    epoch_candidates = sorted(out_dir.glob('epoch_*_model.pt'))
    if epoch_candidates:
        def extract_epoch(path: Path) -> int:
            stem = path.stem
            try:
                return int(stem.split('_')[1])
            except Exception:
                return -1

        return max(epoch_candidates, key=extract_epoch)

    best_path = out_dir / 'best_model.pt'
    if best_path.exists():
        return best_path
    legacy_best_candidates = sorted(out_dir.glob('best*.pt'))
    for legacy_path in legacy_best_candidates:
        if legacy_path.name != 'best_model.pt':
            return legacy_path
    return None


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    lr = float(learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if isinstance(getattr(optimizer, 'defaults', None), dict):
        optimizer.defaults['lr'] = lr


def set_optimizer_weight_decay(optimizer: torch.optim.Optimizer, weight_decay: float) -> None:
    decay = float(weight_decay)
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = decay
    if isinstance(getattr(optimizer, 'defaults', None), dict):
        optimizer.defaults['weight_decay'] = decay


def _rewind_training_state_from_checkpoint(
    checkpoint_path: Path,
    device: object,
    generator_model: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    scaler_g: amp.GradScaler,
    scheduler_g: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    ema_generator: Optional[nn.Module],
) -> None:
    state = load_torch_checkpoint(checkpoint_path, device)
    target_generator = unwrap_model(generator_model)
    if 'generator' in state:
        target_generator.load_state_dict(state['generator'])
    elif 'model' in state:
        target_generator.load_state_dict(state['model'])
    else:
        raise KeyError('Checkpoint missing generator/model state')
    if ema_generator is not None:
        if 'generator_ema' in state and state['generator_ema'] is not None:
            ema_generator.load_state_dict(state['generator_ema'])
        else:
            ema_generator.load_state_dict(target_generator.state_dict())
    if 'optimizer_g' in state and state['optimizer_g'] is not None:
        optimizer_g.load_state_dict(state['optimizer_g'])
        move_optimizer_state_to_device(optimizer_g, device)
    elif 'optimizer' in state and state['optimizer'] is not None:
        optimizer_g.load_state_dict(state['optimizer'])
        move_optimizer_state_to_device(optimizer_g, device)
    if 'scaler_g' in state and state['scaler_g'] is not None:
        scaler_g.load_state_dict(state['scaler_g'])
    elif 'scaler' in state and state['scaler'] is not None:
        scaler_g.load_state_dict(state['scaler'])
    if scheduler_g is not None:
        if 'scheduler_g' in state and state['scheduler_g'] is not None:
            scheduler_g.load_state_dict(state['scheduler_g'])
        elif 'scheduler' in state and state['scheduler'] is not None:
            scheduler_g.load_state_dict(state['scheduler'])


def resolve_selection_metric(cfg: Dict, target_columns: List[str]) -> str:
    configured = str(cfg.get('training', {}).get('selection_metric', '')).strip()
    if configured:
        return configured
    if 'path_loss' in target_columns:
        return 'path_loss.rmse_physical'
    return 'val_recon_loss'


def get_path_loss_hybrid_cfg(cfg: Dict) -> Dict[str, object]:
    return dict(cfg.get('path_loss_hybrid', {}))


def is_path_loss_hybrid_enabled(cfg: Dict) -> bool:
    return bool(get_path_loss_hybrid_cfg(cfg).get('enabled', False))


def expected_model_out_channels(cfg: Dict, target_columns: List[str]) -> int:
    extra_channels = 1 if is_path_loss_hybrid_enabled(cfg) else 0
    return len(target_columns) + extra_channels


def extract_primary_outputs(outputs: torch.Tensor, target_columns: List[str]) -> torch.Tensor:
    return outputs[:, : len(target_columns)]


def extract_confidence_logits(outputs: torch.Tensor, target_columns: List[str], cfg: Dict) -> torch.Tensor | None:
    if not is_path_loss_hybrid_enabled(cfg):
        return None
    confidence_index = len(target_columns)
    if outputs.size(1) <= confidence_index:
        return None
    return outputs[:, confidence_index : confidence_index + 1]


def parse_selection_metrics(cfg: Dict) -> Dict[str, float]:
    configured = cfg.get('training', {}).get('selection_metrics', {})
    if not isinstance(configured, dict):
        return {}
    parsed: Dict[str, float] = {}
    for metric_name, weight in configured.items():
        weight_value = float(weight)
        if abs(weight_value) < 1e-12:
            continue
        parsed[str(metric_name)] = weight_value
    return parsed


def extract_summary_metric(summary: Dict[str, Dict[str, float]], metric_name: str) -> float | None:
    current: object = summary
    for part in metric_name.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, (int, float)):
        value = float(current)
        if np.isfinite(value):
            return value
    return None


def resolve_selection_value(
    selection_metric: str,
    selection_metrics: Dict[str, float],
    selection_metric_baselines: Dict[str, float],
    val_recon: float,
    val_summary: Dict[str, Dict[str, float]],
) -> Tuple[float, str, Dict[str, float], Dict[str, float], Dict[str, float]]:
    if selection_metrics:
        raw_values: Dict[str, float] = {}
        normalized_values: Dict[str, float] = {}
        weighted_values: Dict[str, float] = {}
        total_weight = 0.0
        for metric_name, weight in selection_metrics.items():
            metric_value = extract_summary_metric(val_summary, metric_name)
            if metric_value is None:
                continue
            raw_values[metric_name] = float(metric_value)
            baseline = float(selection_metric_baselines.get(metric_name, metric_value))
            if (not np.isfinite(baseline)) or abs(baseline) < 1e-12:
                baseline = 1.0
            normalized_value = float(metric_value) / baseline
            normalized_values[metric_name] = normalized_value
            weighted_values[metric_name] = float(weight) * normalized_value
            total_weight += abs(float(weight))
        if raw_values and total_weight > 0.0:
            combined_value = float(sum(weighted_values.values()) / total_weight)
            return combined_value, 'weighted_selection_metrics', raw_values, normalized_values, weighted_values
    if selection_metric == 'val_recon_loss':
        return float(val_recon), selection_metric, {'val_recon_loss': float(val_recon)}, {}, {'val_recon_loss': 1.0}
    metric_value = extract_summary_metric(val_summary, selection_metric)
    if metric_value is None:
        return float(val_recon), 'val_recon_loss', {'val_recon_loss': float(val_recon)}, {}, {'val_recon_loss': 1.0}
    metric_value = float(metric_value)
    return metric_value, selection_metric, {selection_metric: metric_value}, {}, {selection_metric: 1.0}


def compute_confidence_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    target_metadata: Dict[str, Dict[str, object]],
    cfg: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    zero = torch.tensor(0.0, device=outputs.device)
    if not is_path_loss_hybrid_enabled(cfg) or 'path_loss' not in target_columns:
        return zero, {}

    confidence_logits = extract_confidence_logits(outputs, target_columns, cfg)
    if confidence_logits is None:
        return zero, {}

    path_loss_idx = target_columns.index('path_loss')
    pred_path_loss = outputs[:, path_loss_idx : path_loss_idx + 1]
    target_path_loss = targets[:, path_loss_idx : path_loss_idx + 1]
    path_loss_mask = masks[:, path_loss_idx : path_loss_idx + 1]
    denom = path_loss_mask.sum().clamp_min(1.0)

    path_loss_metadata = target_metadata.get('path_loss', {})
    pred_db = denormalize_channel(pred_path_loss, path_loss_metadata)
    target_db = denormalize_channel(target_path_loss, path_loss_metadata)
    abs_error_db = torch.abs(pred_db - target_db)

    hybrid_cfg = get_path_loss_hybrid_cfg(cfg)
    error_threshold_db = float(hybrid_cfg.get('confidence_error_threshold_db', 8.0))
    confidence_target = (abs_error_db <= error_threshold_db).float()

    confidence_probs = torch.sigmoid(confidence_logits)
    raw_loss = (confidence_probs - confidence_target) ** 2
    confidence_loss = (raw_loss * path_loss_mask).sum() / denom
    confidence_loss_weight = float(hybrid_cfg.get('confidence_loss_weight', 0.3))
    weighted_loss = confidence_loss_weight * confidence_loss

    confidence_pred = (confidence_probs >= 0.5).float()
    confidence_accuracy = (((confidence_pred == confidence_target).float()) * path_loss_mask).sum() / denom

    return weighted_loss, {
        'confidence_mse': float(confidence_loss.item()),
        'confidence_accuracy': float(confidence_accuracy.item()),
        'confidence_target_mean': float(((confidence_target * path_loss_mask).sum() / denom).item()),
    }


def _denormalize_path_loss_to_db(values: torch.Tensor, metadata: Dict[str, object]) -> torch.Tensor:
    """Differentiable denormalization of path loss to dB matching evaluation."""
    if metadata.get('predict_linear', False):
        return path_loss_linear_normalized_to_db(values)
    scale = float(metadata.get('scale', 1.0))
    offset = float(metadata.get('offset', 0.0))
    return values * scale + offset


def _build_heuristic_prior_differentiable(
    pred_db: torch.Tensor,
    metadata: Dict[str, object],
    kernel_size: int,
    los_mask: Optional[torch.Tensor],
    los_correction_enabled: bool,
    frequency_ghz: float,
    blend_weight: float,
    max_distance_m: float = 362.0,
) -> torch.Tensor:
    """Differentiable heuristic prior: clip + avg_pool (instead of median) + optional LoS correction."""
    clip_min = metadata.get('clip_min')
    clip_max = metadata.get('clip_max')
    out = pred_db
    if clip_min is not None or clip_max is not None:
        out = out.clamp(
            clip_min if clip_min is not None else float('-inf'),
            clip_max if clip_max is not None else float('inf'),
        )
    if kernel_size > 1 and kernel_size % 2 == 1:
        pad = kernel_size // 2
        out = F.avg_pool2d(out, kernel_size, stride=1, padding=pad, count_include_pad=False)
    if los_correction_enabled and los_mask is not None:
        b, _, h, w = out.shape
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        yy = torch.arange(h, device=out.device, dtype=out.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.arange(w, device=out.device, dtype=out.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_dist_pixels = float((cx ** 2 + cy ** 2) ** 0.5)
        if max_dist_pixels > 0:
            dist_norm = (dist / max_dist_pixels).clamp(0.0, 1.0)
        else:
            dist_norm = torch.zeros_like(dist)
        d_m = dist_norm * max_distance_m
        d_m = d_m.clamp(min=1.0)
        pl_fs = 20.0 * torch.log10(d_m) + 20.0 * torch.log10(torch.tensor(frequency_ghz, device=out.device, dtype=out.dtype)) + 92.45
        los = los_mask.float()
        blend = los * blend_weight
        out = (1.0 - blend) * out + blend * pl_fs
    return out


def compute_path_loss_mse_db_direct(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    target_metadata: Dict[str, Dict[str, object]],
    cfg: Dict,
) -> torch.Tensor:
    """
    FifthTry5: MSE(pred_db, target_db) directly in dB. No fusion during training.
    Fusion only at inference.
    """
    # Direct MSE in dB should be available even when hybrid/confidence is disabled.
    # Hybrid/confidence affects fusion/metrics, but not the raw path-loss regression loss.
    if 'path_loss' not in target_columns:
        return torch.tensor(0.0, device=outputs.device)

    path_loss_idx = target_columns.index('path_loss')
    primary = outputs[:, : len(target_columns)]
    pred_path = primary[:, path_loss_idx : path_loss_idx + 1]
    target_path = targets[:, path_loss_idx : path_loss_idx + 1]
    path_mask = masks[:, path_loss_idx : path_loss_idx + 1]

    path_meta = target_metadata.get('path_loss', {})
    pred_db = _denormalize_path_loss_to_db(pred_path, path_meta)
    target_db = _denormalize_path_loss_to_db(target_path, path_meta)

    diff = pred_db - target_db
    squared = diff ** 2
    masked = squared * path_mask
    denom = path_mask.sum().clamp_min(1.0)
    mse_db = masked.sum() / denom
    scale = float(path_meta.get('scale', 180.0))
    scale = max(scale, 1.0)
    return mse_db / (scale ** 2)


def compute_multiscale_regression_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    target_losses: Dict[str, str],
    cfg: Dict,
) -> torch.Tensor:
    ms_cfg = dict(cfg.get('multiscale_targets', {}))
    if not bool(ms_cfg.get('enabled', False)):
        return torch.tensor(0.0, device=outputs.device)

    scales = [int(scale) for scale in ms_cfg.get('scales', [2, 4]) if int(scale) > 1]
    if not scales:
        return torch.tensor(0.0, device=outputs.device)

    raw_weights = list(ms_cfg.get('weights', []))
    if raw_weights and len(raw_weights) != len(scales):
        raw_weights = raw_weights[: len(scales)]
    if not raw_weights:
        raw_weights = [1.0] * len(scales)

    primary = outputs[:, : len(target_columns)]
    configured_targets = [str(name) for name in ms_cfg.get('targets', target_columns)]
    target_weights = {str(name): float(weight) for name, weight in dict(ms_cfg.get('target_weights', {})).items()}
    min_valid_ratio = float(ms_cfg.get('min_valid_ratio', 0.5))
    loss_weight = float(ms_cfg.get('loss_weight', 0.0))
    if loss_weight <= 0.0:
        return torch.tensor(0.0, device=outputs.device)

    total = torch.tensor(0.0, device=outputs.device)
    total_weight = 0.0
    for name in configured_targets:
        if name not in target_columns:
            continue
        if str(target_losses.get(name, 'mse')).lower() == 'bce':
            continue
        channel_weight = float(target_weights.get(name, 1.0))
        if channel_weight <= 0.0:
            continue
        idx = target_columns.index(name)
        pred_channel = primary[:, idx : idx + 1]
        target_channel = targets[:, idx : idx + 1]
        mask_channel = masks[:, idx : idx + 1]
        for factor, scale_weight in zip(scales, raw_weights):
            pred_ds = F.avg_pool2d(pred_channel, kernel_size=factor, stride=factor)
            target_ds = F.avg_pool2d(target_channel, kernel_size=factor, stride=factor)
            mask_ds = F.avg_pool2d(mask_channel, kernel_size=factor, stride=factor)
            valid = (mask_ds >= min_valid_ratio).float()
            denom = valid.sum().clamp_min(1.0)
            mse_ds = (((pred_ds - target_ds) ** 2) * valid).sum() / denom
            combined_weight = float(scale_weight) * channel_weight
            total = total + combined_weight * mse_ds
            total_weight += combined_weight

    if total_weight <= 0.0:
        return torch.tensor(0.0, device=outputs.device)
    return loss_weight * (total / total_weight)


def compute_gradient_regression_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    target_losses: Dict[str, str],
    cfg: Dict,
) -> torch.Tensor:
    grad_cfg = dict(cfg.get('gradient_targets', {}))
    if not bool(grad_cfg.get('enabled', False)):
        return torch.tensor(0.0, device=outputs.device)

    configured_targets = [str(name) for name in grad_cfg.get('targets', target_columns)]
    target_weights = {str(name): float(weight) for name, weight in dict(grad_cfg.get('target_weights', {})).items()}
    loss_weight = float(grad_cfg.get('loss_weight', 0.0))
    min_valid_ratio = float(grad_cfg.get('min_valid_ratio', 0.5))
    if loss_weight <= 0.0:
        return torch.tensor(0.0, device=outputs.device)

    primary = outputs[:, : len(target_columns)]
    total = torch.tensor(0.0, device=outputs.device)
    total_weight = 0.0

    for name in configured_targets:
        if name not in target_columns:
            continue
        if str(target_losses.get(name, 'mse')).lower() == 'bce':
            continue
        channel_weight = float(target_weights.get(name, 1.0))
        if channel_weight <= 0.0:
            continue
        idx = target_columns.index(name)
        pred = primary[:, idx : idx + 1]
        tgt = targets[:, idx : idx + 1]
        msk = masks[:, idx : idx + 1]

        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        tgt_dx = tgt[:, :, :, 1:] - tgt[:, :, :, :-1]
        mask_dx = msk[:, :, :, 1:] * msk[:, :, :, :-1]

        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dy = tgt[:, :, 1:, :] - tgt[:, :, :-1, :]
        mask_dy = msk[:, :, 1:, :] * msk[:, :, :-1, :]

        valid_dx = (mask_dx >= min_valid_ratio).float()
        valid_dy = (mask_dy >= min_valid_ratio).float()
        denom_dx = valid_dx.sum().clamp_min(1.0)
        denom_dy = valid_dy.sum().clamp_min(1.0)

        loss_dx = (((pred_dx - tgt_dx) ** 2) * valid_dx).sum() / denom_dx
        loss_dy = (((pred_dy - tgt_dy) ** 2) * valid_dy).sum() / denom_dy
        total = total + channel_weight * 0.5 * (loss_dx + loss_dy)
        total_weight += channel_weight

    if total_weight <= 0.0:
        return torch.tensor(0.0, device=outputs.device)
    return loss_weight * (total / total_weight)


def overwrite_hybrid_path_loss_metrics(
    summary: Dict[str, Dict[str, float]],
    pred_maps: List[np.ndarray],
    target_maps: List[np.ndarray],
) -> None:
    if not pred_maps or not target_maps:
        return

    mse_values: List[float] = []
    mae_values: List[float] = []
    mse_linear_values: List[float] = []
    mae_linear_values: List[float] = []
    invalid_pixel_count = 0
    invalid_map_count = 0
    skipped_map_count = 0

    for pred_map, target_map in zip(pred_maps, target_maps):
        pred_arr = np.asarray(pred_map, dtype=np.float32)
        tgt_arr = np.asarray(target_map, dtype=np.float32)
        finite_mask = np.isfinite(pred_arr) & np.isfinite(tgt_arr)
        invalid_pixels = int(finite_mask.size - int(np.count_nonzero(finite_mask)))
        if invalid_pixels > 0:
            invalid_map_count += 1
            invalid_pixel_count += invalid_pixels
        if not np.any(finite_mask):
            skipped_map_count += 1
            continue

        diff = pred_arr[finite_mask] - tgt_arr[finite_mask]
        mse_values.append(float(np.mean(diff ** 2)))
        mae_values.append(float(np.mean(np.abs(diff))))

        pred_linear = np.clip(np.power(10.0, -pred_arr[finite_mask] / 10.0), 1e-18, 1.0)
        target_linear = np.clip(np.power(10.0, -tgt_arr[finite_mask] / 10.0), 1e-18, 1.0)
        linear_diff = pred_linear - target_linear
        mse_linear_values.append(float(np.mean(linear_diff ** 2)))
        mae_linear_values.append(float(np.mean(np.abs(linear_diff))))

    summary.setdefault('path_loss', {})
    summary.setdefault('_hybrid', {})
    summary['_hybrid']['invalid_path_loss_pixel_count'] = int(invalid_pixel_count)
    summary['_hybrid']['invalid_path_loss_map_count'] = int(invalid_map_count)
    summary['_hybrid']['skipped_path_loss_map_count'] = int(skipped_map_count)

    if mse_values:
        summary['path_loss']['mse_physical'] = float(np.mean(mse_values))
        summary['path_loss']['rmse_physical'] = float(np.sqrt(summary['path_loss']['mse_physical']))
        summary['path_loss']['mae_physical'] = float(np.mean(mae_values))
    else:
        summary['path_loss']['mse_physical'] = float('nan')
        summary['path_loss']['rmse_physical'] = float('nan')
        summary['path_loss']['mae_physical'] = float('nan')

    if mse_linear_values:
        summary['path_loss']['mse_linear'] = float(np.mean(mse_linear_values))
        summary['path_loss']['rmse_linear'] = float(np.sqrt(summary['path_loss']['mse_linear']))
        summary['path_loss']['mae_linear'] = float(np.mean(mae_linear_values))
    else:
        summary['path_loss']['mse_linear'] = float('nan')
        summary['path_loss']['rmse_linear'] = float('nan')
        summary['path_loss']['mae_linear'] = float('nan')

    summary['path_loss']['linear_quantity'] = 'received_to_transmitted_power_ratio'
    summary['path_loss']['unit_linear'] = 'unitless'
    summary['path_loss']['hybrid_fused_metrics'] = True


def build_loss_map(target_columns: List[str], target_losses: Dict[str, str]):
    loss_map = {}
    for name in target_columns:
        mode = target_losses.get(name, 'mse').lower()
        if mode == 'mse':
            loss_map[name] = nn.MSELoss(reduction='none')
        elif mode == 'l1':
            loss_map[name] = nn.L1Loss(reduction='none')
        elif mode == 'bce':
            loss_map[name] = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type '{mode}' for target '{name}'")
    return loss_map


def build_optimizer(
    optimizer_name: str,
    params,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    momentum: float,
    device: object,
    foreach_enabled: bool = False,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == 'adamw':
        return torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            foreach=bool(foreach_enabled),
        )
    if name == 'adam':
        return torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            foreach=bool(foreach_enabled),
        )
    if name == 'rmsprop':
        return torch.optim.RMSprop(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=bool(foreach_enabled),
        )
    if name == 'sgd':
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=bool(foreach_enabled),
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def compute_reconstruction_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    loss_map: Dict[str, nn.Module],
    mse_weight: float,
    l1_weight: float,
    target_loss_weights: Dict[str, float],
    exclude_columns: Optional[List[str]] = None,
) -> torch.Tensor:
    exclude = set(exclude_columns or [])
    total = torch.tensor(0.0, device=outputs.device)
    valid_count = torch.tensor(0.0, device=outputs.device)
    for i, name in enumerate(target_columns):
        if name in exclude:
            continue
        pred = outputs[:, i : i + 1]
        tgt = targets[:, i : i + 1]
        msk = masks[:, i : i + 1]
        raw = loss_map[name](pred, tgt)
        masked = raw * msk
        denom = msk.sum().clamp_min(1.0)
        loss = masked.sum() / denom
        if isinstance(loss_map[name], nn.L1Loss):
            loss = l1_weight * loss
        elif isinstance(loss_map[name], nn.MSELoss):
            loss = mse_weight * loss
        loss = float(target_loss_weights.get(name, 1.0)) * loss
        total = total + loss
        valid_count = valid_count + (msk.sum() > 0).float()
    return total / valid_count.clamp_min(1.0)


def _los_mask_from_inputs(cfg: Dict, inputs: torch.Tensor) -> Optional[torch.Tensor]:
    if not cfg.get('data', {}).get('los_input_column'):
        return None
    if inputs.shape[1] < 2:
        return None
    return (inputs[:, 1:2] > 0.5).float()


def _support_target_index(target_columns: List[str], target_losses: Dict[str, str]) -> int:
    for idx, name in enumerate(target_columns):
        if target_losses.get(name, 'mse').lower() != 'bce':
            return idx
    return 0


def _init_support_totals() -> Dict[str, float]:
    return {
        'sample_count': 0.0,
        'valid_pixel_count': 0.0,
        'los_valid_pixel_count': 0.0,
        'nlos_valid_pixel_count': 0.0,
        'no_data_positive_count': 0.0,
        'no_data_count': 0.0,
    }


def _update_support_totals(
    support: Dict[str, float],
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    target_losses: Dict[str, str],
    los_mask: Optional[torch.Tensor],
) -> None:
    support['sample_count'] += float(targets.shape[0])
    metric_idx = _support_target_index(target_columns, target_losses)
    valid = (masks[:, metric_idx : metric_idx + 1] > 0).float()
    valid_count = float(valid.sum().item())
    support['valid_pixel_count'] += valid_count
    if los_mask is not None:
        los_valid = valid * los_mask
        support['los_valid_pixel_count'] += float(los_valid.sum().item())
        support['nlos_valid_pixel_count'] += float((valid * (1.0 - los_mask)).sum().item())
    if 'no_data' in target_columns:
        no_data_idx = target_columns.index('no_data')
        no_data_mask = (masks[:, no_data_idx : no_data_idx + 1] > 0).float()
        no_data_target = targets[:, no_data_idx : no_data_idx + 1]
        support['no_data_count'] += float(no_data_mask.sum().item())
        support['no_data_positive_count'] += float(((no_data_target > 0.5).float() * no_data_mask).sum().item())


def _finalize_support_totals(support: Dict[str, float]) -> Dict[str, float | int]:
    valid = max(float(support.get('valid_pixel_count', 0.0)), 0.0)
    los_valid = max(float(support.get('los_valid_pixel_count', 0.0)), 0.0)
    nlos_valid = max(float(support.get('nlos_valid_pixel_count', 0.0)), 0.0)
    no_data_count = max(float(support.get('no_data_count', 0.0)), 0.0)
    return {
        'sample_count': int(support.get('sample_count', 0.0)),
        'valid_pixel_count': int(valid),
        'los_valid_pixel_count': int(los_valid),
        'nlos_valid_pixel_count': int(nlos_valid),
        'los_fraction': float(los_valid / valid) if valid > 0 else float('nan'),
        'nlos_fraction': float(nlos_valid / valid) if valid > 0 else float('nan'),
        'no_data_positive_fraction': float(support.get('no_data_positive_count', 0.0) / no_data_count) if no_data_count > 0 else float('nan'),
    }


def _init_binary_totals() -> Dict[str, float]:
    return {
        'count': 0.0,
        'tp': 0.0,
        'tn': 0.0,
        'fp': 0.0,
        'fn': 0.0,
        'sum_bce': 0.0,
        'sum_squared_error': 0.0,
        'sum_absolute_error': 0.0,
    }


def _update_binary_totals(
    totals: Dict[str, float],
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    if logits.numel() == 0 or target.numel() == 0:
        return
    if mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = mask > 0
    if not bool(valid.any().item()):
        return
    logits_valid = logits[valid]
    target_valid = target[valid].float()
    probs_valid = torch.sigmoid(logits_valid)
    pred = probs_valid >= 0.5
    tgt = target_valid >= 0.5
    count = float(target_valid.numel())
    totals['count'] += count
    totals['tp'] += float((pred & tgt).sum().item())
    totals['tn'] += float((~pred & ~tgt).sum().item())
    totals['fp'] += float((pred & ~tgt).sum().item())
    totals['fn'] += float((~pred & tgt).sum().item())
    bce = F.binary_cross_entropy_with_logits(logits_valid, target_valid, reduction='sum')
    diff = probs_valid - target_valid
    totals['sum_bce'] += float(bce.item())
    totals['sum_squared_error'] += float(torch.sum(diff ** 2).item())
    totals['sum_absolute_error'] += float(torch.sum(torch.abs(diff)).item())


def _finalize_binary_totals(totals: Dict[str, float]) -> Dict[str, float | int]:
    count = max(float(totals.get('count', 0.0)), 0.0)
    if count <= 0.0:
        return {
            'count': 0,
            'bce': float('nan'),
            'mse': float('nan'),
            'rmse': float('nan'),
            'mae': float('nan'),
            'accuracy': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'iou': float('nan'),
            'positive_fraction_target': float('nan'),
            'positive_fraction_pred': float('nan'),
        }
    tp = float(totals.get('tp', 0.0))
    tn = float(totals.get('tn', 0.0))
    fp = float(totals.get('fp', 0.0))
    fn = float(totals.get('fn', 0.0))
    mse = float(totals.get('sum_squared_error', 0.0) / count)
    return {
        'count': int(round(count)),
        'bce': float(totals.get('sum_bce', 0.0) / count),
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': float(totals.get('sum_absolute_error', 0.0) / count),
        'accuracy': float((tp + tn) / count),
        'precision': float(tp / max(tp + fp, 1.0)),
        'recall': float(tp / max(tp + fn, 1.0)),
        'iou': float(tp / max(tp + fp + fn, 1.0)),
        'positive_fraction_target': float((tp + fn) / count),
        'positive_fraction_pred': float((tp + fp) / count),
    }


def _update_no_data_binary_if_present(
    totals: Dict[str, float],
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
) -> None:
    if 'no_data' not in target_columns:
        return
    idx = target_columns.index('no_data')
    _update_binary_totals(
        totals,
        outputs[:, idx : idx + 1].detach(),
        targets[:, idx : idx + 1].detach(),
        masks[:, idx : idx + 1].detach(),
    )


def _update_quantized_u8_totals(
    totals: Dict[str, Dict[str, float]],
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    target_losses: Dict[str, str],
    target_metadata: Dict[str, Dict[str, object]],
) -> None:
    for idx, name in enumerate(target_columns):
        if target_losses.get(name, 'mse').lower() == 'bce':
            continue
        metadata = target_metadata.get(name, {})
        if not metadata:
            continue
        pred = outputs[:, idx : idx + 1].detach().clamp(0.0, 1.0)
        tgt = targets[:, idx : idx + 1].detach().clamp(0.0, 1.0)
        valid = masks[:, idx : idx + 1] > 0
        count = float(valid.sum().item())
        if count == 0.0:
            continue
        pred_q = torch.round(pred * 255.0) / 255.0
        tgt_q = torch.round(tgt * 255.0) / 255.0
        pred_phys = denormalize_channel(pred_q, metadata)
        tgt_phys = denormalize_channel(tgt_q, metadata)
        diff = (pred_phys - tgt_phys)[valid]
        totals[name]['sum_squared_error_physical_quantized_u8'] = totals[name].get('sum_squared_error_physical_quantized_u8', 0.0) + float(torch.sum(diff ** 2).item())
        totals[name]['sum_absolute_error_physical_quantized_u8'] = totals[name].get('sum_absolute_error_physical_quantized_u8', 0.0) + float(torch.sum(torch.abs(diff)).item())


def _attach_metric_extras(
    summary: Dict[str, Dict[str, float]],
    totals: Dict[str, Dict[str, float]],
    target_columns: List[str],
    sample_count: int,
    image_size: int,
) -> None:
    total_pixels = float(max(sample_count, 0) * max(image_size, 1) * max(image_size, 1))
    for name in target_columns:
        metrics = summary.get(name)
        if not isinstance(metrics, dict):
            continue
        count = float(totals.get(name, {}).get('count', 0.0))
        metrics['count'] = int(count)
        metrics['fraction_of_valid_pixels'] = float(count / total_pixels) if total_pixels > 0 else float('nan')
        q_mse = totals.get(name, {}).get('sum_squared_error_physical_quantized_u8')
        q_mae = totals.get(name, {}).get('sum_absolute_error_physical_quantized_u8')
        if q_mse is not None and q_mae is not None and count > 0:
            mse_q = float(q_mse) / count
            metrics['mse_physical_quantized_u8'] = mse_q
            metrics['rmse_physical_quantized_u8'] = float(np.sqrt(mse_q))
            metrics['mae_physical_quantized_u8'] = float(q_mae) / count


def _prune_old_checkpoints(out_dir: Path, keep_epoch: int) -> None:
    """Remove epoch_*_model.pt except keep_epoch. Always keep best_model.pt."""
    for p in out_dir.glob('epoch_*_model.pt'):
        try:
            suffix = p.stem.replace('epoch_', '').replace('_model', '')
            if suffix.isdigit() and int(suffix) != keep_epoch:
                p.unlink()
        except (OSError, ValueError):
            pass


def build_dataloaders(
    cfg: Dict,
    pin_memory: bool,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    is_main_process: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], int, Optional[DistributedSampler]]:
    train_set, val_set = build_datasets_from_config(cfg)
    subset_size = cfg['training'].get('subset_size')
    if subset_size:
        subset_size = min(int(subset_size), len(train_set))
        train_set = Subset(train_set, range(subset_size))
    train_num_workers = int(cfg['data']['num_workers'])
    if distributed and world_size > 1:
        train_num_workers = max(1, train_num_workers // world_size)
    val_num_workers = max(1, train_num_workers // 2) if train_num_workers > 0 else 0
    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        persistent_workers=train_num_workers > 0,
    )
    val_loader: Optional[DataLoader] = None
    if is_main_process:
        val_loader = DataLoader(
            val_set,
            batch_size=int(cfg['training']['batch_size']),
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=pin_memory,
            persistent_workers=val_num_workers > 0,
        )
    in_channels = compute_input_channels(cfg)
    return train_loader, val_loader, in_channels, train_sampler


def validate_generator(
    generator: nn.Module,
    loader: DataLoader,
    device: str,
    cfg: Dict,
    target_columns: List[str],
    loss_map: Dict[str, nn.Module],
    amp_enabled: bool,
    mse_weight: float,
    l1_weight: float,
    target_loss_weights: Dict[str, float],
    target_losses: Dict[str, str],
    target_metadata: Dict[str, Dict[str, object]],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    generator.eval()
    total = 0.0
    totals = init_metric_totals(target_columns)
    confidence_mse_values: List[float] = []
    confidence_acc_values: List[float] = []
    confidence_target_means: List[float] = []
    fused_path_loss_preds: List[np.ndarray] = []
    fused_path_loss_targets: List[np.ndarray] = []
    post_cfg = dict(cfg.get('postprocess', {}))
    los_totals = init_metric_totals(target_columns)
    nlos_totals = init_metric_totals(target_columns)
    support_totals = _init_support_totals()
    no_data_binary_totals = _init_binary_totals()
    with torch.no_grad():
        for batch in tqdm(loader, desc='val', leave=False):
            x, y, m, sc = unpack_expert_batch(batch, device)
            with amp.autocast(device_type='cuda', enabled=amp_enabled):
                pred = forward_expert_model(generator, x, sc)
                primary_pred = extract_primary_outputs(pred, target_columns)
                recon = compute_reconstruction_loss(primary_pred, y, m, target_columns, loss_map, mse_weight, l1_weight, target_loss_weights)
                confidence_loss, confidence_stats = compute_confidence_loss(pred, y, m, target_columns, target_metadata, cfg)
            total += recon.item() + confidence_loss.item()
            update_metric_totals(totals, primary_pred, y, m, target_columns, target_losses, target_metadata)
            _update_quantized_u8_totals(totals, primary_pred, y, m, target_columns, target_losses, target_metadata)
            _update_no_data_binary_if_present(no_data_binary_totals, primary_pred, y, m, target_columns)
            los_mask = _los_mask_from_inputs(cfg, x)
            if los_mask is not None:
                los_masks = m * los_mask
                nlos_masks = m * (1.0 - los_mask)
                update_metric_totals(los_totals, primary_pred, y, los_masks, target_columns, target_losses, target_metadata)
                update_metric_totals(nlos_totals, primary_pred, y, nlos_masks, target_columns, target_losses, target_metadata)
                _update_quantized_u8_totals(los_totals, primary_pred, y, los_masks, target_columns, target_losses, target_metadata)
                _update_quantized_u8_totals(nlos_totals, primary_pred, y, nlos_masks, target_columns, target_losses, target_metadata)
            _update_support_totals(support_totals, y, m, target_columns, target_losses, los_mask)

            if confidence_stats:
                confidence_mse_values.append(float(confidence_stats['confidence_mse']))
                confidence_acc_values.append(float(confidence_stats['confidence_accuracy']))
                confidence_target_means.append(float(confidence_stats['confidence_target_mean']))

            if is_path_loss_hybrid_enabled(cfg) and 'path_loss' in target_columns:
                confidence_logits = extract_confidence_logits(pred, target_columns, cfg)
                if confidence_logits is not None:
                    path_loss_idx = target_columns.index('path_loss')
                    path_loss_metadata = target_metadata.get('path_loss', {})
                    pred_db = denormalize_channel(primary_pred[:, path_loss_idx : path_loss_idx + 1], path_loss_metadata).detach().cpu().numpy()
                    tgt_db = denormalize_channel(y[:, path_loss_idx : path_loss_idx + 1], path_loss_metadata).detach().cpu().numpy()
                    confidence_probs = torch.sigmoid(confidence_logits).detach().cpu().numpy()
                    los_mask_batch = None
                    if cfg.get('data', {}).get('los_input_column'):
                        los_mask_batch = x[:, 1:2].detach().cpu().numpy()
                    for batch_idx in range(pred_db.shape[0]):
                        fallback_outputs = apply_path_loss_confidence_fallback(
                            pred_db[batch_idx, 0],
                            confidence_probs[batch_idx, 0],
                            path_loss_metadata,
                            confidence_threshold=float(get_path_loss_hybrid_cfg(cfg).get('fallback_threshold', 0.5)),
                            fallback_mode=str(get_path_loss_hybrid_cfg(cfg).get('fallback_mode', 'replace')),
                            kernel_size=int(post_cfg.get('path_loss_median_kernel', post_cfg.get('regression_median_kernel', 3))),
                            los_mask=los_mask_batch[batch_idx, 0] if los_mask_batch is not None else None,
                            los_correction_enabled=bool(post_cfg.get('path_loss_los_correction', False)),
                            frequency_ghz=float(post_cfg.get('path_loss_los_frequency_ghz', 7.125)),
                            blend_weight=float(post_cfg.get('path_loss_los_blend_weight', 0.3)),
                        )
                        fused_path_loss_preds.append(np.asarray(fallback_outputs['final_path_loss_db'], dtype=np.float32))
                        fused_path_loss_targets.append(np.asarray(tgt_db[batch_idx, 0], dtype=np.float32))

    summary = finalize_metric_totals(totals, target_columns, target_losses, target_metadata)
    if 'no_data' in target_columns:
        summary['no_data'] = _finalize_binary_totals(no_data_binary_totals)
    sample_count = int(support_totals.get('sample_count', 0.0))
    image_size = int(cfg.get('data', {}).get('image_size', 0) or 0)
    _attach_metric_extras(summary, totals, target_columns, sample_count, image_size)
    los_summary = finalize_metric_totals(los_totals, target_columns, target_losses, target_metadata)
    nlos_summary = finalize_metric_totals(nlos_totals, target_columns, target_losses, target_metadata)
    _attach_metric_extras(los_summary, los_totals, target_columns, sample_count, image_size)
    _attach_metric_extras(nlos_summary, nlos_totals, target_columns, sample_count, image_size)
    summary['_los'] = los_summary
    summary['_nlos'] = nlos_summary
    summary['_support'] = _finalize_support_totals(support_totals)
    if confidence_mse_values:
        summary['_hybrid'] = {
            'confidence_mse': float(np.mean(confidence_mse_values)),
            'confidence_accuracy': float(np.mean(confidence_acc_values)),
            'confidence_target_mean': float(np.mean(confidence_target_means)),
        }
    overwrite_hybrid_path_loss_metrics(summary, fused_path_loss_preds, fused_path_loss_targets)
    loss_components: Dict[str, float] = {}
    weighted_loss_components: Dict[str, float] = {}
    loss_kinds: Dict[str, str] = {}
    for name in target_columns:
        metrics = summary.get(name, {})
        loss_kind = str(target_losses.get(name, 'mse')).lower()
        loss_kinds[name] = loss_kind
        target_weight = float(target_loss_weights.get(name, 1.0))
        if loss_kind == 'mse':
            raw_loss = float(metrics.get('mse', float('nan')))
            weighted_loss = raw_loss * float(mse_weight) * target_weight
        elif loss_kind == 'l1':
            raw_loss = float(metrics.get('mae', float('nan')))
            weighted_loss = raw_loss * float(l1_weight) * target_weight
        else:
            raw_loss = float(metrics.get('bce', metrics.get('mse', float('nan'))))
            weighted_loss = raw_loss * target_weight
        loss_components[name] = raw_loss
        weighted_loss_components[name] = weighted_loss
    summary['_loss'] = {
        'val_recon_loss': float(total / max(len(loader), 1)),
        'target_losses': loss_components,
        'weighted_target_losses': weighted_loss_components,
        'target_loss_kinds': loss_kinds,
        'space': 'normalized_model_space',
    }

    return total / max(len(loader), 1), summary


def save_validation_summary(
    cfg: Dict,
    summary: Dict[str, Dict[str, float]],
    epoch: int,
    out_dir: Path,
    is_best: bool,
    selection_metric: str,
    selection_metric_value: float,
    selection_raw_values: Dict[str, float],
    selection_normalized_values: Dict[str, float],
    selection_weighted_values: Dict[str, float],
) -> None:
    summary['_checkpoint'] = {'epoch': int(epoch)}
    summary['_evaluation'] = {'split': 'val', 'source': 'train_topology_expert.py'}
    summary['_selection'] = {
        'metric': str(selection_metric),
        'value': float(selection_metric_value),
        'is_best_checkpoint': bool(is_best),
    }
    if selection_raw_values:
        summary['_selection']['raw_values'] = dict(selection_raw_values)
    if selection_normalized_values:
        summary['_selection']['normalized_values'] = dict(selection_normalized_values)
    if selection_weighted_values:
        summary['_selection']['weighted_values'] = dict(selection_weighted_values)

    latest_path = out_dir / 'validate_metrics_latest.json'
    latest_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    epoch_path = out_dir / f'validate_metrics_epoch_{epoch}.json'
    epoch_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    if is_best:
        best_metrics_path = out_dir / 'validate_metrics_best.json'
        best_metrics_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')


def _build_metrics_payload(
    val_summary: Dict[str, Dict[str, float]],
    train_summary: Optional[Dict[str, Dict[str, float]]],
    target_columns: List[str],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    los_summary = val_summary.get('_los', {})
    nlos_summary = val_summary.get('_nlos', {})
    train_los_summary = (train_summary or {}).get('_los', {}) if isinstance(train_summary, dict) else {}
    train_nlos_summary = (train_summary or {}).get('_nlos', {}) if isinstance(train_summary, dict) else {}
    for name in target_columns:
        if isinstance(val_summary.get(name), dict):
            metrics[name] = dict(val_summary[name])
        if isinstance(train_summary, dict) and isinstance(train_summary.get(name), dict):
            metrics[f'train_{name}'] = dict(train_summary[name])
        if isinstance(los_summary, dict) and isinstance(los_summary.get(name), dict):
            metrics[f'{name}_LoS'] = dict(los_summary[name])
        if isinstance(nlos_summary, dict) and isinstance(nlos_summary.get(name), dict):
            metrics[f'{name}_NLoS'] = dict(nlos_summary[name])
        if isinstance(train_los_summary, dict) and isinstance(train_los_summary.get(name), dict):
            metrics[f'train_{name}_LoS'] = dict(train_los_summary[name])
        if isinstance(train_nlos_summary, dict) and isinstance(train_nlos_summary.get(name), dict):
            metrics[f'train_{name}_NLoS'] = dict(train_nlos_summary[name])
    return metrics


def _build_experiment_payload(cfg: Dict, val_support: Dict[str, float | int]) -> Dict[str, object]:
    data_cfg = dict(cfg.get('data', {}))
    model_cfg = dict(cfg.get('model', {}))
    train_cfg = dict(cfg.get('training', {}))
    exp_cfg = dict(cfg.get('experiment', {}))
    return {
        'topology_class': exp_cfg.get('topology_class', data_cfg.get('partition_filter', {}).get('topology_class')),
        'split_mode': data_cfg.get('split_mode'),
        'sample_count': int(val_support.get('sample_count', 0)),
        'image_size': int(data_cfg.get('image_size', 0)),
        'batch_size': int(train_cfg.get('batch_size', 0)),
        'model_arch': 'topology_unet',
        'base_channels': int(model_cfg.get('base_channels', 0)),
        'out_channels': int(model_cfg.get('out_channels', 0)),
        'use_scalar_film': bool(model_cfg.get('use_scalar_film', False)),
        'learning_rate': float(train_cfg.get('generator_lr', train_cfg.get('learning_rate', 0.0))),
        'save_every': int(train_cfg.get('save_every', 1)),
        'no_data_aux_enabled': 'no_data' in list(cfg.get('target_columns', [])),
        'target_columns': list(cfg.get('target_columns', [])),
    }


def _add_structured_validation_payload(
    cfg: Dict,
    val_summary: Dict[str, Dict[str, float]],
    train_summary: Optional[Dict[str, Dict[str, float]]],
    target_columns: List[str],
    epoch: int,
    best_epoch: int,
    best_score: float,
    current_score: float,
    selection_metric: str,
    is_best: bool,
    runtime_payload: Dict[str, object],
) -> None:
    val_support = val_summary.get('_support', {})
    train_support = (train_summary or {}).get('_support', {}) if isinstance(train_summary, dict) else {}
    val_summary['metrics'] = _build_metrics_payload(val_summary, train_summary, target_columns)
    val_summary['experiment'] = _build_experiment_payload(cfg, val_support if isinstance(val_support, dict) else {})
    val_summary['support'] = dict(val_support) if isinstance(val_support, dict) else {}
    if isinstance(train_support, dict):
        val_summary['train_support'] = dict(train_support)
    val_summary['runtime'] = dict(runtime_payload)
    val_summary['checkpoint'] = {
        'epoch': int(epoch),
        'best_epoch': int(best_epoch),
        'best_score': float(best_score),
    }
    val_summary['selection'] = {
        'metric': str(selection_metric),
        'current_score': float(current_score),
        'is_best_epoch': bool(is_best),
    }


def save_final_test_summary(
    cfg: Dict,
    generator: nn.Module,
    device: object,
    target_columns: List[str],
    target_losses: Dict[str, str],
    target_metadata: Dict[str, Dict[str, object]],
    amp_enabled: bool,
    out_dir: Path,
) -> bool:
    try:
        test_loader = build_loader_for_split(cfg, 'test', device)
    except ValueError:
        print("Skipping final test evaluation because this config has no 'test' split.")
        return False

    saved_calibration = load_saved_heuristic_calibration(cfg)
    if saved_calibration is None:
        print("Skipping final test evaluation because no validation calibration file was found. Run validate_topology_expert.py first.")
        return False

    fixed_path_loss_kernel = None
    if 'path_loss' in saved_calibration:
        fixed_path_loss_kernel = int(saved_calibration['path_loss'].get('best_median_kernel', cfg.get('postprocess', {}).get('path_loss_median_kernel', 5)))

    summary = summarize_loader(
        generator,
        test_loader,
        device,
        cfg,
        target_columns,
        target_losses,
        target_metadata,
        amp_enabled,
        fixed_path_loss_kernel=fixed_path_loss_kernel,
    )
    summary['_evaluation'] = {
        'split': 'test',
        'source': 'train_topology_expert.py',
        'used_saved_heuristic_calibration': bool(saved_calibration is not None),
    }
    if fixed_path_loss_kernel is not None:
        summary['_evaluation']['loaded_path_loss_median_kernel'] = int(fixed_path_loss_kernel)
    (out_dir / 'eval_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a Try 56 topology expert')
    parser.add_argument('--config', type=str, default='experiments/fiftysixthtry56_topology_experts/fiftysixthtry56_expert_open_sparse_lowrise.yaml')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    ddp_context = init_distributed_context(str(cfg['runtime']['device']), args.local_rank)
    device = ddp_context['device']
    if ddp_context['is_main_process']:
        out_dir = ensure_output_dir(cfg['runtime']['output_dir'])
    else:
        out_dir = Path(cfg['runtime']['output_dir'])
    if ddp_context['distributed']:
        dist.barrier()
    set_seed(int(cfg['seed']) + int(ddp_context['rank']))

    target_columns = list(cfg['target_columns'])
    target_losses = dict(cfg.get('target_losses', {}))
    expected_out_channels = expected_model_out_channels(cfg, target_columns)
    if int(cfg['model']['out_channels']) != expected_out_channels:
        raise ValueError(f"model.out_channels must match expected output channels ({expected_out_channels})")

    train_loader, val_loader, in_channels, train_sampler = build_dataloaders(
        cfg,
        pin_memory=is_cuda_device(device),
        distributed=bool(ddp_context['distributed']),
        rank=int(ddp_context['rank']),
        world_size=int(ddp_context['world_size']),
        is_main_process=bool(ddp_context['is_main_process']),
    )

    if ddp_context['is_main_process']:
        log_scalar_data_report(train_loader.dataset, cfg, sample_limit=int(cfg['training'].get('scalar_report_samples', 1000)))

    scalar_cond_dim = int(compute_scalar_cond_dim(cfg)) if uses_scalar_film_conditioning(cfg) else 0
    film_hidden = int(cfg['model'].get('scalar_film_hidden', 128))
    upsample_mode = str(cfg['model'].get('upsample_mode', 'transpose'))

    generator = UNetGenerator(
        in_channels=in_channels,
        out_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['base_channels']),
        gradient_checkpointing=bool(cfg['model'].get('gradient_checkpointing', False)),
        path_loss_hybrid=is_path_loss_hybrid_enabled(cfg),
        norm_type=str(cfg['model'].get('norm_type', 'batch')),
        scalar_cond_dim=scalar_cond_dim,
        scalar_film_hidden=film_hidden,
        upsample_mode=upsample_mode,
        dropout_down3=float(cfg['model'].get('dropout_down3', 0.1)),
        dropout_down4=float(cfg['model'].get('dropout_down4', 0.2)),
        dropout_up1=float(cfg['model'].get('dropout_up1', 0.1)),
    ).to(device)
    generator_model: nn.Module = generator

    ema_decay = float(cfg['training'].get('ema_decay', 0.99))
    if ema_decay == 1.0:
        ema_decay = 0.0
    if ema_decay < 0.0 or ema_decay > 1.0:
        raise ValueError('training.ema_decay must be in [0, 1], with 1 meaning disabled')
    ema_generator: Optional[nn.Module] = create_ema_model(generator) if ema_decay > 0.0 else None

    generator_optimizer_name = str(cfg['training'].get('generator_optimizer', 'adam'))
    optimizer_foreach_default = bool(cfg['training'].get('optimizer_foreach', False))
    generator_optimizer_foreach = bool(cfg['training'].get('generator_optimizer_foreach', optimizer_foreach_default))
    momentum = float(cfg['training'].get('momentum', 0.0))

    opt_g = build_optimizer(
        generator_optimizer_name,
        generator.parameters(),
        float(cfg['training']['generator_lr']),
        float(cfg['training']['weight_decay']),
        float(cfg['training']['beta1']),
        float(cfg['training']['beta2']),
        momentum,
        device,
        generator_optimizer_foreach,
    )
    scheduler_g: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
    lr_scheduler_cfg = cfg['training'].get('lr_scheduler')
    if lr_scheduler_cfg and str(lr_scheduler_cfg).lower() == 'reduce_on_plateau':
        factor = float(cfg['training'].get('lr_scheduler_factor', 0.5))
        patience = int(cfg['training'].get('lr_scheduler_patience', 5))
        min_lr = float(cfg['training'].get('lr_scheduler_min_lr', 1e-6))
        scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_g, mode='min', factor=factor, patience=patience, min_lr=min_lr,
        )
    loss_map = build_loss_map(target_columns, target_losses)
    amp_enabled = bool(cfg['training']['amp']) and is_cuda_device(device)
    scaler_g = amp.GradScaler('cuda', enabled=amp_enabled)
    lambda_recon = float(cfg['loss']['lambda_recon'])
    mse_weight = float(cfg['loss']['mse_weight'])
    l1_weight = float(cfg['loss']['l1_weight'])
    target_loss_weights = {str(k): float(v) for k, v in dict(cfg['loss'].get('target_loss_weights', {})).items()}
    clip_grad = float(cfg['training']['clip_grad_norm'])
    save_validation_json_each_epoch = bool(cfg['training'].get('save_validation_json_each_epoch', True))
    run_final_test_after_training = bool(cfg['training'].get('run_final_test_after_training', True))
    selection_metric = resolve_selection_metric(cfg, target_columns)
    selection_metrics = parse_selection_metrics(cfg)
    selection_metric_baselines: Dict[str, float] = {}
    best_selection_value = float('inf')
    best_epoch = 0
    best_val_recon = float('inf')
    history = []
    start_epoch = 1
    es_cfg = cfg['training'].get('early_stopping') or {}
    es_enabled = bool(es_cfg.get('enabled', False))
    es_patience = int(es_cfg.get('patience', 10))
    es_min_delta = float(es_cfg.get('min_delta', 0.0))
    es_rewind_to_best_model = bool(es_cfg.get('rewind_to_best_model', False))
    epochs_without_improvement = 0
    target_metadata = dict(cfg.get('target_metadata', {}))

    resume_path = resolve_resume_checkpoint(out_dir, cfg['runtime'].get('resume_checkpoint'))
    if resume_path is not None:
        state = load_torch_checkpoint(resume_path, device)
        if 'generator' in state:
            generator.load_state_dict(state['generator'])
        if ema_generator is not None:
            if 'generator_ema' in state:
                ema_generator.load_state_dict(state['generator_ema'])
            else:
                ema_generator.load_state_dict(generator.state_dict())
        if 'optimizer_g' in state:
            opt_g.load_state_dict(state['optimizer_g'])
            move_optimizer_state_to_device(opt_g, device)
        if 'scaler_g' in state:
            scaler_g.load_state_dict(state['scaler_g'])
        if scheduler_g is not None and 'scheduler_g' in state:
            scheduler_g.load_state_dict(state['scheduler_g'])
        resume_generator_lr = float(cfg['training']['generator_lr'])
        resume_weight_decay = float(cfg['training']['weight_decay'])
        set_optimizer_learning_rate(opt_g, resume_generator_lr)
        set_optimizer_weight_decay(opt_g, resume_weight_decay)
        if 'best_selection_metric_value' in state:
            best_selection_value = float(state['best_selection_metric_value'])
        elif 'best_val_recon_loss' in state and selection_metric == 'val_recon_loss':
            best_selection_value = float(state['best_val_recon_loss'])
        elif 'selection_metric_value' in state:
            best_selection_value = float(state['selection_metric_value'])
        best_epoch = int(state.get('best_epoch', state.get('epoch', 0)))
        saved_selection_metric_baselines = state.get('selection_metric_baselines')
        if isinstance(saved_selection_metric_baselines, dict):
            selection_metric_baselines = {
                str(name): float(value)
                for name, value in saved_selection_metric_baselines.items()
                if isinstance(value, (int, float))
            }
        if 'best_val_recon_loss' in state:
            best_val_recon = float(state['best_val_recon_loss'])
        elif 'val_recon_loss' in state:
            best_val_recon = float(state['val_recon_loss'])
        if 'history' in state and isinstance(state['history'], list):
            history = list(state['history'])
        start_epoch = int(state.get('epoch', 0)) + 1
        if ddp_context['is_main_process']:
            print(f"Resuming from {resume_path} at epoch {start_epoch}")

    if ddp_context['distributed']:
        ddp_find_unused = bool(cfg['training'].get('ddp_find_unused_parameters', False))
        generator_model = DistributedDataParallel(
            generator,
            device_ids=[int(ddp_context['local_rank'])],
            output_device=int(ddp_context['local_rank']),
            find_unused_parameters=ddp_find_unused,
        )

    for epoch in range(start_epoch, int(cfg['training']['epochs']) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        generator_model.train()
        g_running = 0.0
        batch_count = 0
        source_generator = unwrap_model(generator_model)
        train_metric_totals = init_metric_totals(target_columns)
        train_los_totals = init_metric_totals(target_columns)
        train_nlos_totals = init_metric_totals(target_columns)
        train_support_totals = _init_support_totals()
        train_no_data_binary_totals = _init_binary_totals()

        for batch in tqdm(
            train_loader,
            desc=f'epoch {epoch}',
            leave=False,
            disable=not ddp_context['is_main_process'],
        ):
            x, y, m, sc = unpack_expert_batch(batch, device)
            with amp.autocast(device_type='cuda', enabled=amp_enabled):
                fake = forward_expert_model(generator_model, x, sc)
                fake_primary = extract_primary_outputs(fake, target_columns)
                if 'path_loss' in target_columns:
                    recon_other = compute_reconstruction_loss(
                        fake_primary, y, m, target_columns, loss_map, mse_weight, l1_weight, target_loss_weights,
                        exclude_columns=['path_loss'],
                    )
                    path_loss_mse_db = compute_path_loss_mse_db_direct(
                        fake, y, m, target_columns, target_metadata, cfg,
                    )
                    multiscale_regression = compute_multiscale_regression_loss(
                        fake, y, m, target_columns, target_losses, cfg,
                    )
                    gradient_regression = compute_gradient_regression_loss(
                        fake, y, m, target_columns, target_losses, cfg,
                    )
                    n_total = len(target_columns)
                    n_other = n_total - 1
                    if n_other <= 0:
                        recon_loss = path_loss_mse_db + multiscale_regression + gradient_regression
                    else:
                        recon_loss = (recon_other * n_other + path_loss_mse_db) / n_total + multiscale_regression + gradient_regression
                else:
                    recon_loss = compute_reconstruction_loss(
                        fake_primary, y, m, target_columns, loss_map, mse_weight, l1_weight, target_loss_weights,
                    )
                    recon_loss = recon_loss + compute_multiscale_regression_loss(
                        fake, y, m, target_columns, target_losses, cfg,
                    )
                    recon_loss = recon_loss + compute_gradient_regression_loss(
                        fake, y, m, target_columns, target_losses, cfg,
                    )
                confidence_loss, _confidence_stats = compute_confidence_loss(fake, y, m, target_columns, target_metadata, cfg)
                g_loss = lambda_recon * recon_loss + confidence_loss

            with torch.no_grad():
                metric_pred = fake_primary.detach()
                update_metric_totals(train_metric_totals, metric_pred, y, m, target_columns, target_losses, target_metadata)
                _update_quantized_u8_totals(train_metric_totals, metric_pred, y, m, target_columns, target_losses, target_metadata)
                _update_no_data_binary_if_present(train_no_data_binary_totals, metric_pred, y, m, target_columns)
                los_mask = _los_mask_from_inputs(cfg, x)
                if los_mask is not None:
                    los_masks = m * los_mask
                    nlos_masks = m * (1.0 - los_mask)
                    update_metric_totals(train_los_totals, metric_pred, y, los_masks, target_columns, target_losses, target_metadata)
                    update_metric_totals(train_nlos_totals, metric_pred, y, nlos_masks, target_columns, target_losses, target_metadata)
                    _update_quantized_u8_totals(train_los_totals, metric_pred, y, los_masks, target_columns, target_losses, target_metadata)
                    _update_quantized_u8_totals(train_nlos_totals, metric_pred, y, nlos_masks, target_columns, target_losses, target_metadata)
                _update_support_totals(train_support_totals, y, m, target_columns, target_losses, los_mask)

            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(g_loss).backward()
            if clip_grad > 0:
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad)
            scaler_g.step(opt_g)
            scaler_g.update()

            if ema_generator is not None and ema_decay > 0.0:
                update_ema_model(ema_generator, source_generator, ema_decay)

            g_running += g_loss.item()
            batch_count += 1

        train_g_loss = reduce_training_loss(
            g_running,
            batch_count,
            device,
            bool(ddp_context['distributed']),
        )
        train_summary = finalize_metric_totals(train_metric_totals, target_columns, target_losses, target_metadata)
        if 'no_data' in target_columns:
            train_summary['no_data'] = _finalize_binary_totals(train_no_data_binary_totals)
        train_sample_count = int(train_support_totals.get('sample_count', 0.0))
        image_size = int(cfg.get('data', {}).get('image_size', 0) or 0)
        _attach_metric_extras(train_summary, train_metric_totals, target_columns, train_sample_count, image_size)
        train_los_summary = finalize_metric_totals(train_los_totals, target_columns, target_losses, target_metadata)
        train_nlos_summary = finalize_metric_totals(train_nlos_totals, target_columns, target_losses, target_metadata)
        _attach_metric_extras(train_los_summary, train_los_totals, target_columns, train_sample_count, image_size)
        _attach_metric_extras(train_nlos_summary, train_nlos_totals, target_columns, train_sample_count, image_size)
        train_summary['_los'] = train_los_summary
        train_summary['_nlos'] = train_nlos_summary
        train_summary['_support'] = _finalize_support_totals(train_support_totals)
        if ddp_context['distributed']:
            dist.barrier()

        if ddp_context['is_main_process']:
            prev_best_selection = float(best_selection_value)
            if val_loader is None:
                raise RuntimeError('Validation loader is required on rank 0.')
            validation_generator = ema_generator if ema_generator is not None else generator
            val_recon, val_summary = validate_generator(
                validation_generator,
                val_loader,
                device,
                cfg,
                target_columns,
                loss_map,
                amp_enabled,
                mse_weight,
                l1_weight,
                target_loss_weights,
                target_losses,
                target_metadata,
            )
            row = {
                'epoch': epoch,
                'generator_loss': train_g_loss,
                'val_recon_loss': val_recon,
                'world_size': int(ddp_context['world_size']),
            }
            path_loss_summary = val_summary.get('path_loss', {})
            if 'rmse_physical' in path_loss_summary:
                row['path_loss_rmse_physical'] = float(path_loss_summary['rmse_physical'])
            selection_value, selection_metric_used, selection_raw_values, selection_normalized_values, selection_weighted_values = resolve_selection_value(
                selection_metric,
                selection_metrics,
                selection_metric_baselines,
                val_recon,
                val_summary,
            )
            for metric_name, metric_value in selection_raw_values.items():
                if metric_name not in selection_metric_baselines:
                    baseline = float(metric_value)
                    if (not np.isfinite(baseline)) or abs(baseline) < 1e-12:
                        baseline = 1.0
                    selection_metric_baselines[metric_name] = baseline
            row['selection_metric'] = selection_metric_used
            row['selection_metric_value'] = float(selection_value)
            if selection_metric_used == 'weighted_selection_metrics':
                row['selection_metric_components'] = dict(selection_raw_values)
                row['selection_metric_normalized_components'] = dict(selection_normalized_values)
            history.append(row)
            print(json.dumps(row))
            current_lr_g = float(opt_g.param_groups[0]['lr'])
            runtime_payload = {
                'generator_loss': float(train_g_loss),
                'val_recon_loss': float(val_recon),
                'learning_rate': current_lr_g,
                'world_size': int(ddp_context['world_size']),
                'selection_metric': selection_metric_used,
                'selection_metric_value': float(selection_value),
            }
            val_summary['_train'] = dict(runtime_payload)
            if selection_raw_values:
                val_summary['_train']['selection_metric_components'] = dict(selection_raw_values)

            if es_enabled and es_patience > 0:
                if selection_value < (prev_best_selection - es_min_delta):
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if scheduler_g is not None:
                scheduler_g.step(float(selection_value))
            if ddp_context['is_main_process']:
                current_lr_g = opt_g.param_groups[0]['lr']
                if current_lr_g < float(cfg['training']['generator_lr']) * 0.99:
                    print(json.dumps({'lr_generator': current_lr_g}))

            best_val_recon = min(best_val_recon, float(val_recon))
            is_best = selection_value < best_selection_value
            if is_best:
                best_selection_value = float(selection_value)
                best_epoch = int(epoch)

            state = {
                'generator': unwrap_model(generator_model).state_dict(),
                'optimizer_g': opt_g.state_dict(),
                'scaler_g': scaler_g.state_dict(),
                'epoch': epoch,
                'val_recon_loss': val_recon,
                'best_val_recon_loss': best_val_recon,
                'selection_metric': selection_metric_used,
                'selection_metric_value': float(selection_value),
                'selection_metric_baselines': dict(selection_metric_baselines),
                'best_selection_metric_value': best_selection_value,
                'best_epoch': int(best_epoch),
                'history': history,
                'config': cfg,
                'world_size': int(ddp_context['world_size']),
            }
            if ema_generator is not None:
                state['generator_ema'] = ema_generator.state_dict()
                state['ema_decay'] = ema_decay
            if scheduler_g is not None:
                state['scheduler_g'] = scheduler_g.state_dict()
            if is_best:
                torch.save(state, out_dir / 'best_model.pt')
            if epoch % int(cfg['training']['save_every']) == 0:
                torch.save(state, out_dir / f'epoch_{epoch}_model.pt')
                _prune_old_checkpoints(out_dir, keep_epoch=epoch)

            if save_validation_json_each_epoch:
                _add_structured_validation_payload(
                    cfg,
                    val_summary,
                    train_summary,
                    target_columns,
                    epoch,
                    best_epoch=best_epoch,
                    best_score=float(best_selection_value),
                    current_score=float(selection_value),
                    selection_metric=selection_metric_used,
                    is_best=is_best,
                    runtime_payload=runtime_payload,
                )
                save_validation_summary(
                    cfg,
                    val_summary,
                    epoch,
                    out_dir,
                    is_best=is_best,
                    selection_metric=selection_metric_used,
                    selection_metric_value=float(selection_value),
                    selection_raw_values=selection_raw_values,
                    selection_normalized_values=selection_normalized_values,
                    selection_weighted_values=selection_weighted_values,
                )
        control_tensor = torch.tensor([0], device=device, dtype=torch.int32)
        if ddp_context['is_main_process'] and es_enabled and es_patience > 0 and epochs_without_improvement >= es_patience:
            if es_rewind_to_best_model:
                control_tensor[0] = 2
                print(
                    json.dumps(
                        {
                            'rewind_to_best_model': True,
                            'epochs_without_improvement': int(epochs_without_improvement),
                            'patience': int(es_patience),
                        }
                    )
                )
            else:
                control_tensor[0] = 1
                print(
                    json.dumps(
                        {
                            'early_stopping': True,
                            'epochs_without_improvement': int(epochs_without_improvement),
                            'patience': int(es_patience),
                        }
                    )
                )
        if ddp_context['distributed']:
            dist.broadcast(control_tensor, src=0)
        control_value = int(control_tensor.item())
        if control_value == 2:
            best_path = out_dir / 'best_model.pt'
            if not best_path.exists():
                raise FileNotFoundError(f'Best checkpoint not found for rewind: {best_path}')
            _rewind_training_state_from_checkpoint(
                best_path,
                device,
                generator_model,
                opt_g,
                scaler_g,
                scheduler_g,
                ema_generator,
            )
            epochs_without_improvement = 0
            if is_cuda_device(device):
                torch.cuda.empty_cache()
            if ddp_context['distributed']:
                dist.barrier()
            continue
        if control_value == 1:
            if is_cuda_device(device):
                torch.cuda.empty_cache()
            if ddp_context['distributed']:
                dist.barrier()
            break

    if ddp_context['is_main_process']:
        with (out_dir / 'history.json').open('w', encoding='utf-8') as handle:
            json.dump(history, handle, indent=2)

    if ddp_context['distributed']:
        dist.barrier()
    if run_final_test_after_training and ddp_context['is_main_process']:
        best_path = out_dir / 'best_model.pt'
        if best_path.exists():
            best_state = load_torch_checkpoint(best_path, device)
            if 'generator_ema' in best_state and best_state['generator_ema'] is not None:
                generator.load_state_dict(best_state['generator_ema'])
            elif 'generator' in best_state:
                generator.load_state_dict(best_state['generator'])
            save_final_test_summary(
                cfg,
                generator,
                device,
                target_columns,
                target_losses,
                target_metadata,
                amp_enabled,
                out_dir,
            )
    if ddp_context['distributed']:
        dist.barrier()
    cleanup_distributed()


if __name__ == '__main__':
    main()
