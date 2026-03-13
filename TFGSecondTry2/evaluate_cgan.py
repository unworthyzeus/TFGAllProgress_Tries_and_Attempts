from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_utils import is_cuda_device, load_config, load_torch_checkpoint, resolve_device
from data_utils import build_dataset_splits_from_config, compute_input_channels, path_loss_linear_normalized_to_db
from heuristics_cgan import apply_regression_heuristics
from model_cgan import UNetGenerator


def calibration_file_path(cfg: Dict) -> Path:
    return Path(cfg['runtime']['output_dir']) / 'heuristic_calibration.json'


def load_saved_heuristic_calibration(cfg: Dict) -> Dict[str, object] | None:
    path = calibration_file_path(cfg)
    if not path.exists():
        legacy_path = Path(cfg['runtime']['output_dir']) / 'los_mask_calibration.json'
        if not legacy_path.exists():
            return None
        path = legacy_path
    return json.loads(path.read_text(encoding='utf-8'))


def compute_path_loss_kernel_metrics(
    pred_maps: List[np.ndarray],
    target_maps: List[np.ndarray],
    metadata: Dict[str, object],
    kernel_size: int,
) -> Dict[str, float]:
    mse_values: List[float] = []
    mae_values: List[float] = []
    for pred_map, target_map in zip(pred_maps, target_maps):
        processed = apply_regression_heuristics(pred_map, metadata, kernel_size=kernel_size)
        diff = processed - target_map
        mse_values.append(float(np.mean(diff ** 2)))
        mae_values.append(float(np.mean(np.abs(diff))))
    return {
        'kernel_size': float(kernel_size),
        'mse_physical': float(np.mean(mse_values)) if mse_values else float('nan'),
        'rmse_physical': float(np.sqrt(np.mean(mse_values))) if mse_values else float('nan'),
        'mae_physical': float(np.mean(mae_values)) if mae_values else float('nan'),
    }


def find_best_path_loss_kernel(
    pred_maps: List[np.ndarray],
    target_maps: List[np.ndarray],
    metadata: Dict[str, object],
    candidate_kernels: List[int],
) -> Dict[str, float]:
    best: Dict[str, float] | None = None
    for kernel in candidate_kernels:
        metrics = compute_path_loss_kernel_metrics(pred_maps, target_maps, metadata, int(kernel))
        if best is None or metrics['mse_physical'] < best['best_kernel_mse_physical']:
            best = {
                'best_median_kernel': int(kernel),
                'best_kernel_mse_physical': float(metrics['mse_physical']),
                'best_kernel_rmse_physical': float(metrics['rmse_physical']),
                'best_kernel_mae_physical': float(metrics['mae_physical']),
            }
    return best or {
        'best_median_kernel': 1,
        'best_kernel_mse_physical': float('nan'),
        'best_kernel_rmse_physical': float('nan'),
        'best_kernel_mae_physical': float('nan'),
    }


def denormalize_channel(values: torch.Tensor, metadata: Dict[str, object]) -> torch.Tensor:
    if metadata.get('predict_linear', False):
        return path_loss_linear_normalized_to_db(values)
    scale = float(metadata.get('scale', 1.0))
    offset = float(metadata.get('offset', 0.0))
    return values * scale + offset


def _linear_normalized_to_power_ratio(normalized: torch.Tensor) -> torch.Tensor:
    """Convert normalized [0,1] linear path loss to power ratio (linear scale)."""
    n = normalized.clamp(0.0, 1.0)
    log_linear = n * 18.0 - 18.0
    return torch.pow(10.0, log_linear).clamp(min=1e-18)


def init_metric_totals(target_columns: List[str]) -> Dict[str, Dict[str, float]]:
    totals = {
        name: {
            'count': 0.0,
            'sum_squared_error': 0.0,
            'sum_absolute_error': 0.0,
            'sum_correct': 0.0,
            'sum_squared_error_physical': 0.0,
            'sum_absolute_error_physical': 0.0,
        }
        for name in target_columns
    }
    for name in target_columns:
        totals[name]['sum_squared_error_linear'] = 0.0
        totals[name]['sum_absolute_error_linear'] = 0.0
    return totals


def update_metric_totals(
    totals: Dict[str, Dict[str, float]],
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    target_losses: Dict[str, str],
    target_metadata: Dict[str, Dict[str, object]],
) -> None:
    for i, name in enumerate(target_columns):
        pred = outputs[:, i : i + 1]
        tgt = targets[:, i : i + 1]
        msk = masks[:, i : i + 1]

        valid = msk > 0
        valid_count = float(valid.sum().item())
        if valid_count == 0.0:
            continue

        loss_name = target_losses.get(name, 'mse').lower()
        if loss_name == 'bce':
            probs = torch.sigmoid(pred)
            diff = (probs - tgt)[valid]
            preds = (probs > 0.5).float()
            totals[name]['sum_correct'] += float((preds[valid] == tgt[valid]).float().sum().item())
        else:
            diff = (pred - tgt)[valid]

        totals[name]['count'] += valid_count
        totals[name]['sum_squared_error'] += float(torch.sum(diff ** 2).item())
        totals[name]['sum_absolute_error'] += float(torch.sum(torch.abs(diff)).item())

        metadata = target_metadata.get(name, {})
        if metadata and loss_name != 'bce':
            pred_phys = denormalize_channel(pred, metadata)
            tgt_phys = denormalize_channel(tgt, metadata)
            diff_phys = (pred_phys - tgt_phys)[valid]
            totals[name]['sum_squared_error_physical'] += float(torch.sum(diff_phys ** 2).item())
            totals[name]['sum_absolute_error_physical'] += float(torch.sum(torch.abs(diff_phys)).item())
            if name == 'path_loss' and metadata.get('predict_linear', False):
                pred_lin = _linear_normalized_to_power_ratio(pred)
                tgt_lin = _linear_normalized_to_power_ratio(tgt)
                diff_lin = (pred_lin - tgt_lin)[valid]
                totals[name]['sum_squared_error_linear'] += float(torch.sum(diff_lin ** 2).item())
                totals[name]['sum_absolute_error_linear'] += float(torch.sum(torch.abs(diff_lin)).item())


def finalize_metric_totals(
    totals: Dict[str, Dict[str, float]],
    target_columns: List[str],
    target_losses: Dict[str, str],
    target_metadata: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}

    for name in target_columns:
        summary[name] = {}
        count = totals[name]['count']
        if count > 0.0:
            mse = totals[name]['sum_squared_error'] / count
            summary[name]['mse'] = float(mse)
            summary[name]['rmse'] = float(np.sqrt(mse))
            summary[name]['mae'] = float(totals[name]['sum_absolute_error'] / count)
            summary[name]['metric_space'] = 'normalized_model_space'
            summary[name]['unit_normalized'] = 'normalized_0_1'

            if target_losses.get(name, 'mse').lower() == 'bce':
                summary[name]['accuracy'] = float(totals[name]['sum_correct'] / count)

            metadata = target_metadata.get(name, {})
            if metadata:
                mse_physical = totals[name]['sum_squared_error_physical'] / count
                summary[name]['mse_physical'] = float(mse_physical)
                summary[name]['rmse_physical'] = float(np.sqrt(mse_physical))
                summary[name]['mae_physical'] = float(totals[name]['sum_absolute_error_physical'] / count)
                unit = metadata.get('unit')
                if unit:
                    summary[name]['unit_physical'] = str(unit)
                if name == 'path_loss' and metadata.get('predict_linear', False):
                    mse_linear = totals[name]['sum_squared_error_linear'] / count
                    summary[name]['mse_linear'] = float(mse_linear)
                    summary[name]['rmse_linear'] = float(np.sqrt(mse_linear))
                    summary[name]['mae_linear'] = float(totals[name]['sum_absolute_error_linear'] / count)
                    summary[name]['linear_quantity'] = 'received_to_transmitted_power_ratio'
                    summary[name]['unit_linear'] = 'unitless'
                    summary[name]['linear_unit_legacy'] = 'power_ratio'

        unit = target_metadata.get(name, {}).get('unit')
        if unit:
            summary[name]['unit'] = str(unit)

    return summary


def aggregate_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    target_losses: Dict[str, str],
    target_metadata: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}

    for i, name in enumerate(target_columns):
        pred = outputs[:, i : i + 1]
        tgt = targets[:, i : i + 1]
        msk = masks[:, i : i + 1]

        valid = msk > 0
        if valid.sum().item() == 0:
            metrics[name] = {'mse': float('nan'), 'rmse': float('nan'), 'mae': float('nan')}
            continue

        loss_name = target_losses.get(name, 'mse').lower()
        if loss_name == 'bce':
            probs = torch.sigmoid(pred)
            diff = (probs - tgt)[valid]
            mse = torch.mean(diff ** 2).item()
            rmse = float(np.sqrt(mse))
            mae = torch.mean(torch.abs(diff)).item()
            preds = (probs > 0.5).float()
            acc = (preds[valid] == tgt[valid]).float().mean().item()
            metrics[name] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'accuracy': acc}
            continue

        diff = (pred - tgt)[valid]
        mse = torch.mean(diff ** 2).item()
        rmse = float(np.sqrt(mse))
        mae = torch.mean(torch.abs(diff)).item()
        record: Dict[str, float] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'metric_space': 'normalized_model_space',
            'unit_normalized': 'normalized_0_1',
        }

        metadata = target_metadata.get(name, {})
        if metadata:
            pred_phys = denormalize_channel(pred, metadata)
            tgt_phys = denormalize_channel(tgt, metadata)
            diff_phys = (pred_phys - tgt_phys)[valid]
            record['mse_physical'] = torch.mean(diff_phys ** 2).item()
            record['rmse_physical'] = float(np.sqrt(record['mse_physical']))
            record['mae_physical'] = torch.mean(torch.abs(diff_phys)).item()
            unit = metadata.get('unit')
            if unit:
                record['unit'] = str(unit)
                record['unit_physical'] = str(unit)
            if name == 'path_loss' and metadata.get('predict_linear', False):
                pred_lin = _linear_normalized_to_power_ratio(pred)
                tgt_lin = _linear_normalized_to_power_ratio(tgt)
                diff_lin = (pred_lin - tgt_lin)[valid]
                mse_linear = torch.mean(diff_lin ** 2).item()
                record['mse_linear'] = mse_linear
                record['rmse_linear'] = float(np.sqrt(mse_linear))
                record['mae_linear'] = torch.mean(torch.abs(diff_lin)).item()
                record['linear_quantity'] = 'received_to_transmitted_power_ratio'
                record['unit_linear'] = 'unitless'
                record['linear_unit_legacy'] = 'power_ratio'

        metrics[name] = record

    return metrics


def build_loader_for_split(cfg: Dict, split_name: str, device: object) -> DataLoader:
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.setdefault('augmentation', {})['enable'] = False
    splits = build_dataset_splits_from_config(eval_cfg)
    if split_name not in splits:
        raise ValueError(f"Split '{split_name}' is not available for this config.")
    dataset = splits[split_name]
    return DataLoader(
        dataset,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=False,
        num_workers=max(1, int(cfg['data']['num_workers']) // 2),
        pin_memory=is_cuda_device(device),
        persistent_workers=max(1, int(cfg['data']['num_workers']) // 2) > 0,
    )


def summarize_loader(
    generator: UNetGenerator,
    loader: DataLoader,
    device: object,
    target_columns: List[str],
    target_losses: Dict[str, str],
    target_metadata: Dict[str, Dict[str, object]],
    amp_enabled: bool,
    tune_path_loss_kernel: bool = False,
    fixed_path_loss_kernel: int | None = None,
    path_loss_kernel_candidates: List[int] | None = None,
) -> Dict[str, Dict[str, float]]:
    totals = init_metric_totals(target_columns)
    path_loss_preds: List[np.ndarray] = []
    path_loss_targets: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets, masks in tqdm(loader, desc='eval_cgan', leave=False):
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            with amp.autocast(device_type='cuda', enabled=amp_enabled):
                outputs = generator(inputs)

            if 'path_loss' in target_columns:
                path_loss_index = target_columns.index('path_loss')
                pred = outputs[:, path_loss_index : path_loss_index + 1]
                tgt = targets[:, path_loss_index : path_loss_index + 1]
                metadata = target_metadata.get('path_loss', {})
                pred_phys = denormalize_channel(pred, metadata).detach().cpu().numpy()
                tgt_phys = denormalize_channel(tgt, metadata).detach().cpu().numpy()
                for batch_idx in range(pred_phys.shape[0]):
                    path_loss_preds.append(np.asarray(pred_phys[batch_idx, 0], dtype=np.float32))
                    path_loss_targets.append(np.asarray(tgt_phys[batch_idx, 0], dtype=np.float32))

            update_metric_totals(totals, outputs, targets, masks, target_columns, target_losses, target_metadata)

    summary = finalize_metric_totals(totals, target_columns, target_losses, target_metadata)

    if path_loss_preds and path_loss_targets:
        path_loss_metadata = target_metadata.get('path_loss', {})
        if tune_path_loss_kernel:
            candidate_kernels = path_loss_kernel_candidates or [1, 3, 5, 7]
            kernel_metrics = find_best_path_loss_kernel(path_loss_preds, path_loss_targets, path_loss_metadata, candidate_kernels)
            summary.setdefault('path_loss', {}).update(kernel_metrics)
        if fixed_path_loss_kernel is not None:
            calibrated_metrics = compute_path_loss_kernel_metrics(path_loss_preds, path_loss_targets, path_loss_metadata, int(fixed_path_loss_kernel))
            summary.setdefault('path_loss', {}).update({
                'calibrated_median_kernel': int(fixed_path_loss_kernel),
                'calibrated_mse_physical': float(calibrated_metrics['mse_physical']),
                'calibrated_rmse_physical': float(calibrated_metrics['rmse_physical']),
                'calibrated_mae_physical': float(calibrated_metrics['mae_physical']),
            })
    return summary


def compute_generalization_gap(
    train_summary: Dict[str, Dict[str, float]],
    val_summary: Dict[str, Dict[str, float]],
    target_columns: List[str],
) -> Dict[str, Dict[str, float]]:
    gap: Dict[str, Dict[str, float]] = {}
    for name in target_columns:
        gap[name] = {}
        train_metrics = train_summary.get(name, {})
        val_metrics = val_summary.get(name, {})
        for key, val_value in val_metrics.items():
            if key == 'unit' or key not in train_metrics:
                continue
            train_value = train_metrics[key]
            if isinstance(train_value, (int, float)) and isinstance(val_value, (int, float)):
                gap[f"{name}"] [f"{key}_gap"] = float(val_value - train_value)
                if abs(float(train_value)) > 1e-12:
                    gap[f"{name}"] [f"{key}_ratio"] = float(val_value / train_value)
        unit = val_metrics.get('unit') or train_metrics.get('unit')
        if unit:
            gap[name]['unit'] = str(unit)
    return gap


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate cGAN generator on validation data')
    parser.add_argument('--config', type=str, default='configs/cgan_unet.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test', 'both'], default='test')
    parser.add_argument('--save-heuristic-calibration', action='store_true')
    parser.add_argument('--use-saved-heuristic-calibration', action='store_true')
    parser.add_argument('--save-los-calibration', action='store_true')
    parser.add_argument('--use-saved-los-calibration', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg['runtime']['device'])

    target_columns = list(cfg['target_columns'])
    target_losses = dict(cfg.get('target_losses', {}))
    target_metadata = dict(cfg.get('target_metadata', {}))
    if int(cfg['model']['out_channels']) != len(target_columns):
        raise ValueError('model.out_channels must match len(target_columns)')

    generator = UNetGenerator(
        in_channels=compute_input_channels(cfg),
        out_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['base_channels']),
        gradient_checkpointing=bool(cfg['model'].get('gradient_checkpointing', False)),
    ).to(device)

    state = load_torch_checkpoint(args.checkpoint, device)
    generator.load_state_dict(state['generator'] if 'generator' in state else state)
    generator.eval()

    amp_enabled = bool(cfg['training']['amp']) and is_cuda_device(device)
    los_calibration_path = calibration_file_path(cfg)
    fixed_path_loss_kernel = None
    saved_calibration = None
    should_use_saved_calibration = (
        args.use_saved_heuristic_calibration
        or args.use_saved_los_calibration
        or args.split == 'test'
    )
    if should_use_saved_calibration:
        saved_calibration = load_saved_heuristic_calibration(cfg)
        if saved_calibration is not None:
            if 'path_loss' in saved_calibration:
                fixed_path_loss_kernel = int(saved_calibration['path_loss']['best_median_kernel'])

    path_loss_kernel_candidates = list(cfg.get('postprocess', {}).get('path_loss_median_kernel_candidates', [1, 3, 5, 7]))

    if args.split == 'both':
        train_loader = build_loader_for_split(cfg, 'train', device)
        val_loader = build_loader_for_split(cfg, 'val', device)
        train_summary = summarize_loader(
            generator,
            train_loader,
            device,
            target_columns,
            target_losses,
            target_metadata,
            amp_enabled,
            fixed_path_loss_kernel=fixed_path_loss_kernel,
            path_loss_kernel_candidates=path_loss_kernel_candidates,
        )
        val_summary = summarize_loader(
            generator,
            val_loader,
            device,
            target_columns,
            target_losses,
            target_metadata,
            amp_enabled,
            tune_path_loss_kernel=True,
            fixed_path_loss_kernel=fixed_path_loss_kernel,
            path_loss_kernel_candidates=path_loss_kernel_candidates,
        )
        summary: Dict[str, Dict[str, float]] = {
            'train': train_summary,
            'val': val_summary,
            '_generalization_gap': compute_generalization_gap(train_summary, val_summary, target_columns),
        }
        try:
            test_loader = build_loader_for_split(cfg, 'test', device)
            summary['test'] = summarize_loader(generator, test_loader, device, target_columns, target_losses, target_metadata, amp_enabled)
        except ValueError:
            pass
    else:
        loader = build_loader_for_split(cfg, args.split, device)
        summary = summarize_loader(
            generator,
            loader,
            device,
            target_columns,
            target_losses,
            target_metadata,
            amp_enabled,
            tune_path_loss_kernel=args.split == 'val',
            fixed_path_loss_kernel=fixed_path_loss_kernel,
            path_loss_kernel_candidates=path_loss_kernel_candidates,
        )

    should_save_calibration = args.save_heuristic_calibration or args.save_los_calibration
    if should_save_calibration:
        calibration_source = None
        if args.split == 'val':
            calibration_source = {
                'path_loss': summary.get('path_loss', {}),
            }
        elif args.split == 'both':
            calibration_source = {
                'path_loss': summary.get('val', {}).get('path_loss', {}),
            }
        if calibration_source and 'best_median_kernel' in calibration_source.get('path_loss', {}):
            calibration_payload = {
                'path_loss': {
                    'best_median_kernel': int(calibration_source['path_loss'].get('best_median_kernel', cfg.get('postprocess', {}).get('path_loss_median_kernel', 5))),
                    'best_kernel_mse_physical': float(calibration_source['path_loss'].get('best_kernel_mse_physical', 0.0)),
                    'best_kernel_rmse_physical': float(calibration_source['path_loss'].get('best_kernel_rmse_physical', 0.0)),
                    'best_kernel_mae_physical': float(calibration_source['path_loss'].get('best_kernel_mae_physical', 0.0)),
                },
                'source_split': 'val',
                'checkpoint': args.checkpoint,
            }
            los_calibration_path.parent.mkdir(parents=True, exist_ok=True)
            los_calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding='utf-8')

    if 'epoch' in state:
        summary['_checkpoint'] = {'epoch': int(state['epoch'])}
    if 'val_recon_loss' in state:
        summary.setdefault('_checkpoint', {})['val_recon_loss'] = float(state['val_recon_loss'])
    summary.setdefault('_evaluation', {})['split'] = args.split
    if fixed_path_loss_kernel is not None:
        summary['_evaluation']['loaded_path_loss_median_kernel'] = int(fixed_path_loss_kernel)
    summary['_evaluation']['used_saved_heuristic_calibration'] = bool(should_use_saved_calibration and saved_calibration is not None)
    if should_save_calibration:
        summary['_evaluation']['saved_heuristic_calibration'] = str(los_calibration_path)

    print(json.dumps(summary, indent=2))

    file_name = 'eval_metrics_cgan.json' if args.split != 'both' else 'eval_metrics_cgan_both.json'
    out_path = Path(cfg['runtime']['output_dir']) / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()