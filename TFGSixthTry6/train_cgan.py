from __future__ import annotations

import argparse
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

from config_utils import ensure_output_dir, is_cuda_device, anchor_data_paths_to_config_file, load_config, load_torch_checkpoint, move_optimizer_state_to_device, resolve_device
from data_utils import build_datasets_from_config, compute_input_channels, path_loss_linear_normalized_to_db
from evaluate_cgan import (
    build_loader_for_split,
    denormalize_channel,
    finalize_metric_totals,
    init_metric_totals,
    load_saved_heuristic_calibration,
    summarize_loader,
    update_metric_totals,
)
from heuristics_cgan import apply_path_loss_confidence_fallback
from model_cgan import PatchDiscriminator, UNetGenerator


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


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def reduce_training_losses(
    g_running: float,
    d_running: float,
    batch_count: int,
    device: Any,
    distributed: bool,
) -> Tuple[float, float]:
    if not distributed:
        denom = max(batch_count, 1)
        return g_running / denom, d_running / denom

    stats = torch.tensor([g_running, d_running, float(batch_count)], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    denom = max(float(stats[2].item()), 1.0)
    return float(stats[0].item() / denom), float(stats[1].item() / denom)


def resolve_resume_checkpoint(out_dir: Path, configured_resume: str | None) -> Path | None:
    if configured_resume:
        resume_path = Path(configured_resume)
        return resume_path if resume_path.exists() else None

    epoch_candidates = sorted(out_dir.glob('epoch_*_cgan.pt'))
    if epoch_candidates:
        def extract_epoch(path: Path) -> int:
            stem = path.stem
            try:
                return int(stem.split('_')[1])
            except Exception:
                return -1

        return max(epoch_candidates, key=extract_epoch)

    best_path = out_dir / 'best_cgan.pt'
    if best_path.exists():
        return best_path
    return None


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
    """Differentiable denormalization of path loss to dB (matches evaluate_cgan.denormalize_channel)."""
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
    if not is_path_loss_hybrid_enabled(cfg) or 'path_loss' not in target_columns:
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


def build_adversarial_loss(loss_name: str) -> nn.Module:
    mode = loss_name.lower()
    if mode == 'bce':
        return nn.BCEWithLogitsLoss()
    if mode == 'mse':
        return nn.MSELoss()
    raise ValueError(f"Unsupported adversarial loss '{loss_name}'. Expected 'bce' or 'mse'.")


def resolve_adversarial_loss_name(cfg: Dict, device: object) -> str:
    loss_cfg = dict(cfg.get('loss', {}))
    configured = loss_cfg.get('adversarial_loss')
    if configured:
        return str(configured)
    return 'bce' if is_cuda_device(device) else 'mse'


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


def _prune_old_checkpoints(out_dir: Path, keep_epoch: int) -> None:
    """Remove epoch_*_cgan.pt except keep_epoch. Always keep best_cgan.pt."""
    for p in out_dir.glob('epoch_*_cgan.pt'):
        try:
            suffix = p.stem.replace('epoch_', '').replace('_cgan', '')
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
    with torch.no_grad():
        for x, y, m in tqdm(loader, desc='val', leave=False):
            x, y, m = x.to(device), y.to(device), m.to(device)
            with amp.autocast(device_type='cuda', enabled=amp_enabled):
                pred = generator(x)
                primary_pred = extract_primary_outputs(pred, target_columns)
                recon = compute_reconstruction_loss(primary_pred, y, m, target_columns, loss_map, mse_weight, l1_weight, target_loss_weights)
                confidence_loss, confidence_stats = compute_confidence_loss(pred, y, m, target_columns, target_metadata, cfg)
            total += recon.item() + confidence_loss.item()
            update_metric_totals(totals, primary_pred, y, m, target_columns, target_losses, target_metadata)

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
    if confidence_mse_values:
        summary['_hybrid'] = {
            'confidence_mse': float(np.mean(confidence_mse_values)),
            'confidence_accuracy': float(np.mean(confidence_acc_values)),
            'confidence_target_mean': float(np.mean(confidence_target_means)),
        }
    overwrite_hybrid_path_loss_metrics(summary, fused_path_loss_preds, fused_path_loss_targets)

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
    summary['_evaluation'] = {'split': 'val', 'source': 'train_cgan.py'}
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

    latest_path = out_dir / 'validate_metrics_cgan_latest.json'
    latest_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    epoch_path = out_dir / f'validate_metrics_epoch_{epoch}_cgan.json'
    epoch_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    if is_best:
        best_metrics_path = out_dir / 'validate_metrics_cgan_best.json'
        best_metrics_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')


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
        print("Skipping final test evaluation because no validation calibration file was found. Run validate_cgan.py first.")
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
        'source': 'train_cgan.py',
        'used_saved_heuristic_calibration': bool(saved_calibration is not None),
    }
    if fixed_path_loss_kernel is not None:
        summary['_evaluation']['loaded_path_loss_median_kernel'] = int(fixed_path_loss_kernel)
    (out_dir / 'eval_metrics_cgan.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description='Train cGAN + U-Net CKM predictor')
    parser.add_argument('--config', type=str, default='configs/cgan_unet.yaml')
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

    generator = UNetGenerator(
        in_channels=in_channels,
        out_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['base_channels']),
        gradient_checkpointing=bool(cfg['model'].get('gradient_checkpointing', False)),
        path_loss_hybrid=is_path_loss_hybrid_enabled(cfg),
        norm_type=str(cfg['model'].get('norm_type', 'batch')),
    ).to(device)
    discriminator = PatchDiscriminator(
        in_channels=in_channels,
        target_channels=len(target_columns),
        base_channels=int(cfg['model']['disc_base_channels']),
        norm_type=str(cfg['model'].get('disc_norm_type', cfg['model'].get('norm_type', 'batch'))),
        input_downsample_factor=int(cfg['model'].get('disc_input_downsample_factor', 1)),
    ).to(device)
    generator_model: nn.Module = generator
    discriminator_model: nn.Module = discriminator

    generator_optimizer_name = str(cfg['training'].get('generator_optimizer', 'adam'))
    discriminator_optimizer_name = str(cfg['training'].get('discriminator_optimizer', 'adam'))
    optimizer_foreach_default = bool(cfg['training'].get('optimizer_foreach', False))
    generator_optimizer_foreach = bool(cfg['training'].get('generator_optimizer_foreach', optimizer_foreach_default))
    discriminator_optimizer_foreach = bool(cfg['training'].get('discriminator_optimizer_foreach', optimizer_foreach_default))
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
    opt_d = build_optimizer(
        discriminator_optimizer_name,
        discriminator.parameters(),
        float(cfg['training']['discriminator_lr']),
        float(cfg['training']['weight_decay']),
        float(cfg['training']['beta1']),
        float(cfg['training']['beta2']),
        momentum,
        device,
        discriminator_optimizer_foreach,
    )

    scheduler_g: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
    scheduler_d: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
    lr_scheduler_cfg = cfg['training'].get('lr_scheduler')
    if lr_scheduler_cfg and str(lr_scheduler_cfg).lower() == 'reduce_on_plateau':
        factor = float(cfg['training'].get('lr_scheduler_factor', 0.5))
        patience = int(cfg['training'].get('lr_scheduler_patience', 5))
        min_lr = float(cfg['training'].get('lr_scheduler_min_lr', 1e-6))
        scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_g, mode='min', factor=factor, patience=patience, min_lr=min_lr,
        )
        scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_d, mode='min', factor=factor, patience=patience, min_lr=min_lr,
        )

    adversarial_loss_name = resolve_adversarial_loss_name(cfg, device)
    adv_criterion = build_adversarial_loss(adversarial_loss_name)
    loss_map = build_loss_map(target_columns, target_losses)
    amp_enabled = bool(cfg['training']['amp']) and is_cuda_device(device)
    scaler_g = amp.GradScaler('cuda', enabled=amp_enabled)
    scaler_d = amp.GradScaler('cuda', enabled=amp_enabled)
    lambda_gan = float(cfg['loss']['lambda_gan'])
    gan_enabled = lambda_gan > 0.0
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
    best_val_recon = float('inf')
    history = []
    start_epoch = 1
    target_metadata = dict(cfg.get('target_metadata', {}))

    resume_path = resolve_resume_checkpoint(out_dir, cfg['runtime'].get('resume_checkpoint'))
    if resume_path is not None:
        state = load_torch_checkpoint(resume_path, device)
        if 'generator' in state:
            generator.load_state_dict(state['generator'])
        if 'discriminator' in state:
            discriminator.load_state_dict(state['discriminator'])
        if 'optimizer_g' in state:
            opt_g.load_state_dict(state['optimizer_g'])
            move_optimizer_state_to_device(opt_g, device)
        if 'optimizer_d' in state:
            opt_d.load_state_dict(state['optimizer_d'])
            move_optimizer_state_to_device(opt_d, device)
        if 'scaler_g' in state:
            scaler_g.load_state_dict(state['scaler_g'])
        if 'scaler_d' in state:
            scaler_d.load_state_dict(state['scaler_d'])
        if scheduler_g is not None and 'scheduler_g' in state:
            scheduler_g.load_state_dict(state['scheduler_g'])
        if scheduler_d is not None and 'scheduler_d' in state:
            scheduler_d.load_state_dict(state['scheduler_d'])
        if 'best_selection_metric_value' in state:
            best_selection_value = float(state['best_selection_metric_value'])
        elif 'best_val_recon_loss' in state and selection_metric == 'val_recon_loss':
            best_selection_value = float(state['best_val_recon_loss'])
        elif 'selection_metric_value' in state:
            best_selection_value = float(state['selection_metric_value'])
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
        discriminator_model = DistributedDataParallel(
            discriminator,
            device_ids=[int(ddp_context['local_rank'])],
            output_device=int(ddp_context['local_rank']),
            find_unused_parameters=ddp_find_unused,
        )

    for epoch in range(start_epoch, int(cfg['training']['epochs']) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        generator_model.train()
        discriminator_model.train()
        g_running = 0.0
        d_running = 0.0
        batch_count = 0

        for x, y, m in tqdm(
            train_loader,
            desc=f'epoch {epoch}',
            leave=False,
            disable=not ddp_context['is_main_process'],
        ):
            x, y, m = x.to(device), y.to(device), m.to(device)

            if gan_enabled:
                discriminator_model.train()
                set_requires_grad(discriminator, True)
                # The discriminator step does not backpropagate through the generator,
                # so avoid building a generator graph here to reduce peak VRAM.
                with torch.no_grad():
                    with amp.autocast(device_type='cuda', enabled=amp_enabled):
                        fake = generator_model(x)
                        fake_primary = extract_primary_outputs(fake, target_columns)

                opt_d.zero_grad(set_to_none=True)
                with amp.autocast(device_type='cuda', enabled=amp_enabled):
                    real_logits = discriminator_model(x, y)
                    real_labels = torch.ones_like(real_logits)
                    d_loss_real = 0.5 * adv_criterion(real_logits, real_labels)
                scaler_d.scale(d_loss_real).backward()

                with amp.autocast(device_type='cuda', enabled=amp_enabled):
                    fake_logits = discriminator_model(x, fake_primary.detach())
                    fake_labels = torch.zeros_like(fake_logits)
                    d_loss_fake = 0.5 * adv_criterion(fake_logits, fake_labels)
                scaler_d.scale(d_loss_fake).backward()

                d_loss = d_loss_real.detach() + d_loss_fake.detach()
                if clip_grad > 0:
                    scaler_d.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                d_loss = torch.tensor(0.0, device=device)

            if gan_enabled:
                # The generator loss needs gradients through the discriminator output
                # but not through discriminator parameters themselves.
                set_requires_grad(discriminator, False)
                discriminator_model.eval()
            with amp.autocast(device_type='cuda', enabled=amp_enabled):
                fake = generator_model(x)
                fake_primary = extract_primary_outputs(fake, target_columns)
                if gan_enabled:
                    fake_logits_for_g = discriminator_model(x, fake_primary)
                    gan_loss = adv_criterion(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
                else:
                    gan_loss = torch.tensor(0.0, device=device)
                if is_path_loss_hybrid_enabled(cfg) and 'path_loss' in target_columns:
                    recon_other = compute_reconstruction_loss(
                        fake_primary, y, m, target_columns, loss_map, mse_weight, l1_weight, target_loss_weights,
                        exclude_columns=['path_loss'],
                    )
                    path_loss_mse_db = compute_path_loss_mse_db_direct(
                        fake, y, m, target_columns, target_metadata, cfg,
                    )
                    n_total = len(target_columns)
                    n_other = n_total - 1
                    if n_other <= 0:
                        recon_loss = path_loss_mse_db
                    else:
                        recon_loss = (recon_other * n_other + path_loss_mse_db) / n_total
                else:
                    recon_loss = compute_reconstruction_loss(
                        fake_primary, y, m, target_columns, loss_map, mse_weight, l1_weight, target_loss_weights,
                    )
                confidence_loss, _confidence_stats = compute_confidence_loss(fake, y, m, target_columns, target_metadata, cfg)
                g_loss = lambda_gan * gan_loss + lambda_recon * recon_loss + confidence_loss

            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(g_loss).backward()
            if clip_grad > 0:
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad)
            scaler_g.step(opt_g)
            scaler_g.update()

            g_running += g_loss.item()
            d_running += d_loss.item()
            batch_count += 1

        train_g_loss, train_d_loss = reduce_training_losses(
            g_running,
            d_running,
            batch_count,
            device,
            bool(ddp_context['distributed']),
        )
        if ddp_context['distributed']:
            dist.barrier()

        if ddp_context['is_main_process']:
            if val_loader is None:
                raise RuntimeError('Validation loader is required on rank 0.')
            val_recon, val_summary = validate_generator(
                generator,
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
                'adversarial_loss': adversarial_loss_name,
                'generator_loss': train_g_loss,
                'discriminator_loss': train_d_loss,
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

            if scheduler_g is not None:
                scheduler_g.step(float(selection_value))
            if scheduler_d is not None:
                scheduler_d.step(float(selection_value))
                if ddp_context['is_main_process']:
                    current_lr_g = opt_g.param_groups[0]['lr']
                    if current_lr_g < float(cfg['training']['generator_lr']) * 0.99:
                        print(json.dumps({'lr_generator': current_lr_g}))
                    if scheduler_d is not None:
                        current_lr_d = opt_d.param_groups[0]['lr']
                        if current_lr_d < float(cfg['training']['discriminator_lr']) * 0.99:
                            print(json.dumps({'lr_discriminator': current_lr_d}))

            best_val_recon = min(best_val_recon, float(val_recon))

            state = {
                'generator': unwrap_model(generator_model).state_dict(),
                'discriminator': unwrap_model(discriminator_model).state_dict(),
                'optimizer_g': opt_g.state_dict(),
                'optimizer_d': opt_d.state_dict(),
                'scaler_g': scaler_g.state_dict(),
                'scaler_d': scaler_d.state_dict(),
                'epoch': epoch,
                'val_recon_loss': val_recon,
                'best_val_recon_loss': best_val_recon,
                'selection_metric': selection_metric_used,
                'selection_metric_value': float(selection_value),
                'selection_metric_baselines': dict(selection_metric_baselines),
                'best_selection_metric_value': best_selection_value,
                'history': history,
                'config': cfg,
                'world_size': int(ddp_context['world_size']),
            }
            if scheduler_g is not None:
                state['scheduler_g'] = scheduler_g.state_dict()
            if scheduler_d is not None:
                state['scheduler_d'] = scheduler_d.state_dict()
            is_best = selection_value < best_selection_value
            if selection_value < best_selection_value:
                best_selection_value = float(selection_value)
                state['best_selection_metric_value'] = best_selection_value
                torch.save(state, out_dir / 'best_cgan.pt')
            if epoch % int(cfg['training']['save_every']) == 0:
                torch.save(state, out_dir / f'epoch_{epoch}_cgan.pt')
                _prune_old_checkpoints(out_dir, keep_epoch=epoch)

            if save_validation_json_each_epoch:
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
        if ddp_context['distributed']:
            dist.barrier()

    if ddp_context['is_main_process']:
        with (out_dir / 'history_cgan.json').open('w', encoding='utf-8') as handle:
            json.dump(history, handle, indent=2)

    if ddp_context['distributed']:
        dist.barrier()
    if run_final_test_after_training and ddp_context['is_main_process']:
        best_path = out_dir / 'best_cgan.pt'
        if best_path.exists():
            best_state = load_torch_checkpoint(best_path, device)
            if 'generator' in best_state:
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
