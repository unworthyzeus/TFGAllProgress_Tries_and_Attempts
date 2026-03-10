from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_utils import load_config, resolve_device
from data_utils import CKMDataset
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
            metrics[name] = {'mse': float('nan'), 'mae': float('nan')}
            continue

        diff = (pred - tgt)[valid]
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()

        record = {'mse': mse, 'mae': mae}

        metadata = target_metadata.get(name, {})
        if metadata and target_losses.get(name, 'mse').lower() != 'bce':
            pred_phys = denormalize_channel(pred, metadata)
            tgt_phys = denormalize_channel(tgt, metadata)
            diff_phys = (pred_phys - tgt_phys)[valid]
            record['mse_physical'] = torch.mean(diff_phys ** 2).item()
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate CKM proposal prototype model')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg['runtime']['device'])

    target_columns = list(cfg['target_columns'])
    target_losses = dict(cfg.get('target_losses', {}))
    target_metadata = dict(cfg.get('target_metadata', {}))
    if int(cfg['model']['out_channels']) != len(target_columns):
        raise ValueError("model.out_channels must match len(target_columns)")

    dataset = CKMDataset(
        manifest_csv=cfg['data']['val_manifest'],
        root_dir=cfg['data']['root_dir'],
        target_columns=target_columns,
        image_size=int(cfg['data']['image_size']),
        augment=False,
        add_scalar_channels=bool(cfg['model']['use_scalar_channels']),
        scalar_feature_columns=list(cfg['data'].get('scalar_feature_columns', [])),
        constant_scalar_features=dict(cfg['data'].get('constant_scalar_features', {})),
        scalar_feature_norms=dict(cfg['data'].get('scalar_feature_norms', {})),
        los_input_column=cfg['data'].get('los_input_column'),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=False,
        num_workers=max(1, int(cfg['data']['num_workers']) // 2),
        pin_memory=True,
        persistent_workers=max(1, int(cfg['data']['num_workers']) // 2) > 0,
    )

    in_channels = 1
    if cfg['data'].get('los_input_column'):
        in_channels += 1
    if bool(cfg['model']['use_scalar_channels']):
        in_channels += len(list(cfg['data'].get('scalar_feature_columns', [])))
        in_channels += len(dict(cfg['data'].get('constant_scalar_features', {})))
    model = CKMUNet(
        in_channels=in_channels,
        out_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['base_channels']),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()

    running = {name: {'mse': [], 'mae': [], 'accuracy': []} for name in target_columns}

    with torch.no_grad():
        for inputs, targets, masks in tqdm(loader, desc='eval'):
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            with autocast(enabled=bool(cfg['training']['amp']) and device == 'cuda'):
                outputs = model(inputs)

            batch_metrics = aggregate_metrics(outputs, targets, masks, target_columns, target_losses, target_metadata)
            for name in target_columns:
                for k, v in batch_metrics[name].items():
                    if not np.isnan(v):
                        running[name][k].append(v)

    summary: Dict[str, Dict[str, float]] = {}
    for name in target_columns:
        summary[name] = {}
        for k, values in running[name].items():
            if values:
                summary[name][k] = float(np.mean(values))

    print(json.dumps(summary, indent=2))

    out_path = Path(cfg['runtime']['output_dir']) / 'eval_metrics.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
