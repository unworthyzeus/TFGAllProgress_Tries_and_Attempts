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
from data_utils import build_dataset_splits_from_config, compute_input_channels
from model_unet import CKMUNet


def denormalize_channel(values: torch.Tensor, metadata: Dict[str, object]) -> torch.Tensor:
    scale = float(metadata.get('scale', 1.0))
    offset = float(metadata.get('offset', 0.0))
    return values * scale + offset


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

        diff = (pred - tgt)[valid]
        mse = torch.mean(diff ** 2).item()
        rmse = float(np.sqrt(mse))
        mae = torch.mean(torch.abs(diff)).item()

        record = {'mse': mse, 'rmse': rmse, 'mae': mae}

        metadata = target_metadata.get(name, {})
        if metadata and target_losses.get(name, 'mse').lower() != 'bce':
            pred_phys = denormalize_channel(pred, metadata)
            tgt_phys = denormalize_channel(tgt, metadata)
            diff_phys = (pred_phys - tgt_phys)[valid]
            record['mse_physical'] = torch.mean(diff_phys ** 2).item()
            record['rmse_physical'] = float(np.sqrt(record['mse_physical']))
            record['mae_physical'] = torch.mean(torch.abs(diff_phys)).item()
            unit = metadata.get('unit')
            if unit:
                record['unit'] = str(unit)

        if target_losses.get(name, 'mse').lower() == 'bce':
            probs = torch.sigmoid(pred)
            preds = (probs > 0.5).float()
            acc = (preds[valid] == tgt[valid]).float().mean().item()
            record['accuracy'] = acc

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
    model: CKMUNet,
    loader: DataLoader,
    device: object,
    target_columns: List[str],
    target_losses: Dict[str, str],
    target_metadata: Dict[str, Dict[str, object]],
    amp_enabled: bool,
) -> Dict[str, Dict[str, float]]:
    running = {name: {'mse': [], 'rmse': [], 'mae': [], 'accuracy': [], 'mse_physical': [], 'rmse_physical': [], 'mae_physical': []} for name in target_columns}

    with torch.no_grad():
        for inputs, targets, masks in tqdm(loader, desc='eval', leave=False):
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            with amp.autocast(device_type='cuda', enabled=amp_enabled):
                outputs = model(inputs)

            batch_metrics = aggregate_metrics(outputs, targets, masks, target_columns, target_losses, target_metadata)
            for name in target_columns:
                for k, v in batch_metrics[name].items():
                    if k == 'unit' or np.isnan(v):
                        continue
                    running[name][k].append(v)

    summary: Dict[str, Dict[str, float]] = {}
    for name in target_columns:
        summary[name] = {}
        for k, values in running[name].items():
            if values:
                summary[name][k] = float(np.mean(values))
        unit = target_metadata.get(name, {}).get('unit')
        if unit:
            summary[name]['unit'] = str(unit)
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
                gap[name][f'{key}_gap'] = float(val_value - train_value)
                if abs(float(train_value)) > 1e-12:
                    gap[name][f'{key}_ratio'] = float(val_value / train_value)
        unit = val_metrics.get('unit') or train_metrics.get('unit')
        if unit:
            gap[name]['unit'] = str(unit)
    return gap


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate CKM proposal prototype model')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test', 'both'], default='test')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg['runtime']['device'])

    target_columns = list(cfg['target_columns'])
    target_losses = dict(cfg.get('target_losses', {}))
    target_metadata = dict(cfg.get('target_metadata', {}))
    if int(cfg['model']['out_channels']) != len(target_columns):
        raise ValueError("model.out_channels must match len(target_columns)")

    in_channels = compute_input_channels(cfg)
    model = CKMUNet(
        in_channels=in_channels,
        out_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['base_channels']),
    ).to(device)

    state = load_torch_checkpoint(args.checkpoint, device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()
    amp_enabled = bool(cfg['training']['amp']) and is_cuda_device(device)
    if args.split == 'both':
        train_loader = build_loader_for_split(cfg, 'train', device)
        val_loader = build_loader_for_split(cfg, 'val', device)
        train_summary = summarize_loader(model, train_loader, device, target_columns, target_losses, target_metadata, amp_enabled)
        val_summary = summarize_loader(model, val_loader, device, target_columns, target_losses, target_metadata, amp_enabled)
        summary: Dict[str, Dict[str, float]] = {
            'train': train_summary,
            'val': val_summary,
            '_generalization_gap': compute_generalization_gap(train_summary, val_summary, target_columns),
        }
        try:
            test_loader = build_loader_for_split(cfg, 'test', device)
            summary['test'] = summarize_loader(model, test_loader, device, target_columns, target_losses, target_metadata, amp_enabled)
        except ValueError:
            pass
    else:
        loader = build_loader_for_split(cfg, args.split, device)
        summary = summarize_loader(model, loader, device, target_columns, target_losses, target_metadata, amp_enabled)

    summary.setdefault('_evaluation', {})['split'] = args.split

    print(json.dumps(summary, indent=2))

    file_name = 'eval_metrics.json' if args.split != 'both' else 'eval_metrics_both.json'
    out_path = Path(cfg['runtime']['output_dir']) / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
