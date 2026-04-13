from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config_utils import ensure_output_dir, is_cuda_device, anchor_data_paths_to_config_file, load_config, load_torch_checkpoint, move_optimizer_state_to_device, resolve_device
from data_utils import build_datasets_from_config, compute_input_channels
from model_unet import CKMUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_resume_checkpoint(out_dir: Path, configured_resume: str | None) -> Path | None:
    if configured_resume:
        resume_path = Path(configured_resume)
        return resume_path if resume_path.exists() else None

    epoch_candidates = sorted(out_dir.glob('epoch_*.pt'))
    if epoch_candidates:
        def extract_epoch(path: Path) -> int:
            stem = path.stem
            try:
                return int(stem.split('_')[1])
            except Exception:
                return -1

        return max(epoch_candidates, key=extract_epoch)

    best_path = out_dir / 'best.pt'
    if best_path.exists():
        return best_path
    return None


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


def build_optimizer(cfg: Dict, model: nn.Module, device: object) -> torch.optim.Optimizer:
    optimizer_name = str(cfg['training'].get('optimizer', 'adamw')).lower()
    learning_rate = float(cfg['training']['learning_rate'])
    weight_decay = float(cfg['training']['weight_decay'])
    momentum = float(cfg['training'].get('momentum', 0.0))

    if optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            foreach=is_cuda_device(device),
        )
    if optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=is_cuda_device(device),
        )
    if optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=is_cuda_device(device),
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def apply_optimizer_hparams_from_cfg(
    optimizer: torch.optim.Optimizer,
    *,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
) -> None:
    lr = float(learning_rate)
    decay = float(weight_decay)
    mom = float(momentum)

    for group in optimizer.param_groups:
        group['lr'] = lr
        group['weight_decay'] = decay
        if 'momentum' in group:
            group['momentum'] = mom

    defaults = getattr(optimizer, 'defaults', None)
    if isinstance(defaults, dict):
        defaults['lr'] = lr
        defaults['weight_decay'] = decay
        if 'momentum' in defaults:
            defaults['momentum'] = mom


def compute_weighted_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_columns: List[str],
    loss_map: Dict[str, nn.Module],
    mse_weight: float,
    l1_weight: float,
    target_loss_weights: Dict[str, float],
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
        channel_loss = float(target_loss_weights.get(name, 1.0)) * channel_loss

        total = total + channel_loss
        valid_count = valid_count + (msk_ch.sum() > 0).float()

    return total / valid_count.clamp_min(1.0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: str,
    target_columns: List[str],
    loss_map: Dict[str, nn.Module],
    amp_enabled: bool,
    mse_weight: float,
    l1_weight: float,
    target_loss_weights: Dict[str, float],
    clip_grad_norm: float,
) -> float:
    model.train()
    running = 0.0

    for inputs, targets, masks in tqdm(loader, desc='train', leave=False):
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type='cuda', enabled=amp_enabled):
            outputs = model(inputs)
            loss = compute_weighted_loss(
                outputs,
                targets,
                masks,
                target_columns,
                loss_map,
                mse_weight,
                l1_weight,
                target_loss_weights,
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
    target_loss_weights: Dict[str, float],
) -> float:
    model.eval()
    running = 0.0

    with torch.no_grad():
        for inputs, targets, masks in tqdm(loader, desc='val', leave=False):
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            with amp.autocast(device_type='cuda', enabled=amp_enabled):
                outputs = model(inputs)
                loss = compute_weighted_loss(
                    outputs,
                    targets,
                    masks,
                    target_columns,
                    loss_map,
                    mse_weight,
                    l1_weight,
                    target_loss_weights,
                )
            running += loss.item()

    return running / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train CKM proposal prototype model')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    set_seed(int(cfg['seed']))

    device = resolve_device(cfg['runtime']['device'])
    out_dir = ensure_output_dir(cfg['runtime']['output_dir'])

    target_columns = list(cfg['target_columns'])
    target_losses = dict(cfg.get('target_losses', {}))
    if int(cfg['model']['out_channels']) != len(target_columns):
        raise ValueError("model.out_channels must match len(target_columns)")

    train_set, val_set = build_datasets_from_config(cfg)

    subset_size = cfg['training'].get('subset_size')
    if subset_size:
        subset_size = min(int(subset_size), len(train_set))
        train_set = Subset(train_set, range(subset_size))

    pin_memory = is_cuda_device(device)

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=True,
        num_workers=int(cfg['data']['num_workers']),
        pin_memory=pin_memory,
        persistent_workers=int(cfg['data']['num_workers']) > 0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=False,
        num_workers=max(1, int(cfg['data']['num_workers']) // 2),
        pin_memory=pin_memory,
        persistent_workers=max(1, int(cfg['data']['num_workers']) // 2) > 0,
    )

    in_channels = compute_input_channels(cfg)
    model = CKMUNet(
        in_channels=in_channels,
        out_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['base_channels']),
        gradient_checkpointing=bool(cfg['model'].get('gradient_checkpointing', False)),
    ).to(device)

    optimizer = build_optimizer(cfg, model, device)

    amp_enabled = bool(cfg['training']['amp']) and is_cuda_device(device)
    scaler = amp.GradScaler('cuda', enabled=amp_enabled)
    loss_map = build_loss_map(target_columns, target_losses)
    target_loss_weights = {str(k): float(v) for k, v in dict(cfg['loss'].get('target_loss_weights', {})).items()}

    best_val = float('inf')
    history = []
    start_epoch = 1

    resume_path = resolve_resume_checkpoint(out_dir, cfg['runtime'].get('resume_checkpoint'))
    if resume_path is not None:
        state = load_torch_checkpoint(resume_path, device)
        model.load_state_dict(state['model'] if 'model' in state else state)
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
            move_optimizer_state_to_device(optimizer, device)
            apply_optimizer_hparams_from_cfg(
                optimizer,
                learning_rate=float(cfg['training']['learning_rate']),
                weight_decay=float(cfg['training']['weight_decay']),
                momentum=float(cfg['training'].get('momentum', 0.0)),
            )
        if 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        if 'best_val_loss' in state:
            best_val = float(state['best_val_loss'])
        elif 'val_loss' in state:
            best_val = float(state['val_loss'])
        if 'history' in state and isinstance(state['history'], list):
            history = list(state['history'])
        start_epoch = int(state.get('epoch', 0)) + 1
        print(f"Resuming from {resume_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, int(cfg['training']['epochs']) + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            target_columns,
            loss_map,
            amp_enabled,
            float(cfg['loss']['mse_weight']),
            float(cfg['loss']['l1_weight']),
            target_loss_weights,
            float(cfg['training']['clip_grad_norm']),
        )

        val_loss = eval_epoch(
            model,
            val_loader,
            device,
            target_columns,
            loss_map,
            amp_enabled,
            float(cfg['loss']['mse_weight']),
            float(cfg['loss']['l1_weight']),
            target_loss_weights,
        )

        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        history.append(row)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'best_val_loss': best_val,
                    'history': history,
                },
                out_dir / 'best.pt',
            )

        if epoch % int(cfg['training']['save_every']) == 0:
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'best_val_loss': best_val,
                    'history': history,
                },
                out_dir / f'epoch_{epoch}.pt',
            )

    with (out_dir / 'history.json').open('w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)


if __name__ == '__main__':
    main()
