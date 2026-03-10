from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config_utils import ensure_output_dir, load_config, resolve_device
from data_utils import CKMDataset
from model_unet import CKMUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def compute_weighted_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    loss_map: Dict[str, nn.Module],
    mse_weight: float,
    l1_weight: float,
) -> torch.Tensor:
    total = torch.tensor(0.0, device=outputs.device)
    valid_count = torch.tensor(0.0, device=outputs.device)

    for i, name in enumerate(target_columns):
        pred_ch = outputs[:, i : i + 1]
        tgt_ch = targets[:, i : i + 1]
        msk_ch = masks[:, i : i + 1]

        raw = loss_map[name](pred_ch, tgt_ch)
        masked = raw * msk_ch
        denom = msk_ch.sum().clamp_min(1.0)
        channel_loss = masked.sum() / denom

        if isinstance(loss_map[name], nn.L1Loss):
            channel_loss = l1_weight * channel_loss
        elif isinstance(loss_map[name], nn.MSELoss):
            channel_loss = mse_weight * channel_loss

        total = total + channel_loss
        valid_count = valid_count + (msk_ch.sum() > 0).float()

    return total / valid_count.clamp_min(1.0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    target_columns: List[str],
    loss_map: Dict[str, nn.Module],
    amp_enabled: bool,
    mse_weight: float,
    l1_weight: float,
    clip_grad_norm: float,
) -> float:
    model.train()
    running = 0.0

    for inputs, targets, masks in tqdm(loader, desc='train', leave=False):
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            outputs = model(inputs)
            loss = compute_weighted_loss(
                outputs,
                targets,
                masks,
                target_columns,
                loss_map,
                mse_weight,
                l1_weight,
            )

        scaler.scale(loss).backward()
        if clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()

    return running / max(len(loader), 1)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_columns: List[str],
    loss_map: Dict[str, nn.Module],
    amp_enabled: bool,
    mse_weight: float,
    l1_weight: float,
) -> float:
    model.eval()
    running = 0.0

    with torch.no_grad():
        for inputs, targets, masks in tqdm(loader, desc='val', leave=False):
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            with autocast(enabled=amp_enabled):
                outputs = model(inputs)
                loss = compute_weighted_loss(
                    outputs,
                    targets,
                    masks,
                    target_columns,
                    loss_map,
                    mse_weight,
                    l1_weight,
                )
            running += loss.item()

    return running / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train CKM proposal prototype model')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg['seed']))

    device = resolve_device(cfg['runtime']['device'])
    out_dir = ensure_output_dir(cfg['runtime']['output_dir'])

    target_columns = list(cfg['target_columns'])
    target_losses = dict(cfg.get('target_losses', {}))
    if int(cfg['model']['out_channels']) != len(target_columns):
        raise ValueError("model.out_channels must match len(target_columns)")

    train_set = CKMDataset(
        manifest_csv=cfg['data']['train_manifest'],
        root_dir=cfg['data']['root_dir'],
        target_columns=target_columns,
        image_size=int(cfg['data']['image_size']),
        augment=bool(cfg['augmentation']['enable']),
        hflip_prob=float(cfg['augmentation']['hflip_prob']),
        vflip_prob=float(cfg['augmentation']['vflip_prob']),
        rot90_prob=float(cfg['augmentation']['rot90_prob']),
        add_scalar_channels=bool(cfg['model']['use_scalar_channels']),
        scalar_feature_columns=list(cfg['data'].get('scalar_feature_columns', [])),
        constant_scalar_features=dict(cfg['data'].get('constant_scalar_features', {})),
        scalar_feature_norms=dict(cfg['data'].get('scalar_feature_norms', {})),
        los_input_column=cfg['data'].get('los_input_column'),
    )

    val_set = CKMDataset(
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

    subset_size = cfg['training'].get('subset_size')
    if subset_size:
        subset_size = min(int(subset_size), len(train_set))
        train_set = Subset(train_set, range(subset_size))

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=True,
        num_workers=int(cfg['data']['num_workers']),
        pin_memory=True,
        persistent_workers=int(cfg['data']['num_workers']) > 0,
    )

    val_loader = DataLoader(
        val_set,
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

    resume = cfg['runtime'].get('resume_checkpoint')
    if resume:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state['model'] if 'model' in state else state)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg['training']['learning_rate']),
        weight_decay=float(cfg['training']['weight_decay']),
    )

    scaler = GradScaler(enabled=bool(cfg['training']['amp']) and device == 'cuda')
    loss_map = build_loss_map(target_columns, target_losses)

    best_val = float('inf')
    history = []

    for epoch in range(1, int(cfg['training']['epochs']) + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            target_columns,
            loss_map,
            bool(cfg['training']['amp']) and device == 'cuda',
            float(cfg['loss']['mse_weight']),
            float(cfg['loss']['l1_weight']),
            float(cfg['training']['clip_grad_norm']),
        )

        val_loss = eval_epoch(
            model,
            val_loader,
            device,
            target_columns,
            loss_map,
            bool(cfg['training']['amp']) and device == 'cuda',
            float(cfg['loss']['mse_weight']),
            float(cfg['loss']['l1_weight']),
        )

        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        history.append(row)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, out_dir / 'best.pt')

        if epoch % int(cfg['training']['save_every']) == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, out_dir / f'epoch_{epoch}.pt')

    with (out_dir / 'history.json').open('w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)


if __name__ == '__main__':
    main()
