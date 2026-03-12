from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config_utils import ensure_output_dir, is_cuda_device, load_config, load_torch_checkpoint, move_optimizer_state_to_device, resolve_device
from data_utils import build_datasets_from_config, compute_input_channels
from model_cgan import PatchDiscriminator, UNetGenerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == 'adam':
        return torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            foreach=is_cuda_device(device),
        )
    if name == 'rmsprop':
        return torch.optim.RMSprop(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=is_cuda_device(device),
        )
    if name == 'sgd':
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=is_cuda_device(device),
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
) -> torch.Tensor:
    total = torch.tensor(0.0, device=outputs.device)
    valid_count = torch.tensor(0.0, device=outputs.device)
    for i, name in enumerate(target_columns):
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


def build_dataloaders(cfg: Dict, pin_memory: bool) -> Tuple[DataLoader, DataLoader, int]:
    train_set, val_set = build_datasets_from_config(cfg)
    subset_size = cfg['training'].get('subset_size')
    if subset_size:
        subset_size = min(int(subset_size), len(train_set))
        train_set = Subset(train_set, range(subset_size))
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
    return train_loader, val_loader, in_channels


def validate_generator(
    generator: nn.Module,
    loader: DataLoader,
    device: str,
    target_columns: List[str],
    loss_map: Dict[str, nn.Module],
    amp_enabled: bool,
    mse_weight: float,
    l1_weight: float,
    target_loss_weights: Dict[str, float],
) -> float:
    generator.eval()
    total = 0.0
    with torch.no_grad():
        for x, y, m in tqdm(loader, desc='val', leave=False):
            x, y, m = x.to(device), y.to(device), m.to(device)
            with amp.autocast(device_type='cuda', enabled=amp_enabled):
                pred = generator(x)
                recon = compute_reconstruction_loss(pred, y, m, target_columns, loss_map, mse_weight, l1_weight, target_loss_weights)
            total += recon.item()
    return total / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train cGAN + U-Net CKM predictor')
    parser.add_argument('--config', type=str, default='configs/cgan_unet.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg['seed']))
    device = resolve_device(cfg['runtime']['device'])
    out_dir = ensure_output_dir(cfg['runtime']['output_dir'])

    target_columns = list(cfg['target_columns'])
    target_losses = dict(cfg.get('target_losses', {}))
    if int(cfg['model']['out_channels']) != len(target_columns):
        raise ValueError('model.out_channels must match len(target_columns)')

    train_loader, val_loader, in_channels = build_dataloaders(cfg, pin_memory=is_cuda_device(device))

    generator = UNetGenerator(
        in_channels=in_channels,
        out_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['base_channels']),
        gradient_checkpointing=bool(cfg['model'].get('gradient_checkpointing', False)),
    ).to(device)
    discriminator = PatchDiscriminator(
        in_channels=in_channels,
        target_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['disc_base_channels']),
    ).to(device)

    generator_optimizer_name = str(cfg['training'].get('generator_optimizer', 'adam'))
    discriminator_optimizer_name = str(cfg['training'].get('discriminator_optimizer', 'adam'))
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
    )

    adversarial_loss_name = resolve_adversarial_loss_name(cfg, device)
    adv_criterion = build_adversarial_loss(adversarial_loss_name)
    loss_map = build_loss_map(target_columns, target_losses)
    amp_enabled = bool(cfg['training']['amp']) and is_cuda_device(device)
    scaler_g = amp.GradScaler('cuda', enabled=amp_enabled)
    scaler_d = amp.GradScaler('cuda', enabled=amp_enabled)
    lambda_gan = float(cfg['loss']['lambda_gan'])
    lambda_recon = float(cfg['loss']['lambda_recon'])
    mse_weight = float(cfg['loss']['mse_weight'])
    l1_weight = float(cfg['loss']['l1_weight'])
    target_loss_weights = {str(k): float(v) for k, v in dict(cfg['loss'].get('target_loss_weights', {})).items()}
    clip_grad = float(cfg['training']['clip_grad_norm'])
    best_val = float('inf')
    history = []
    start_epoch = 1

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
        if 'best_val_recon_loss' in state:
            best_val = float(state['best_val_recon_loss'])
        elif 'val_recon_loss' in state:
            best_val = float(state['val_recon_loss'])
        if 'history' in state and isinstance(state['history'], list):
            history = list(state['history'])
        start_epoch = int(state.get('epoch', 0)) + 1
        print(f"Resuming from {resume_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, int(cfg['training']['epochs']) + 1):
        generator.train()
        discriminator.train()
        g_running = 0.0
        d_running = 0.0

        for x, y, m in tqdm(train_loader, desc=f'epoch {epoch}', leave=False):
            x, y, m = x.to(device), y.to(device), m.to(device)

            with amp.autocast(device_type='cuda', enabled=amp_enabled):
                fake = generator(x)
                real_logits = discriminator(x, y)
                fake_logits = discriminator(x, fake.detach())
                real_labels = torch.ones_like(real_logits)
                fake_labels = torch.zeros_like(fake_logits)
                d_loss = 0.5 * (adv_criterion(real_logits, real_labels) + adv_criterion(fake_logits, fake_labels))

            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            if clip_grad > 0:
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
            scaler_d.step(opt_d)
            scaler_d.update()

            with amp.autocast(device_type='cuda', enabled=amp_enabled):
                fake = generator(x)
                fake_logits_for_g = discriminator(x, fake)
                gan_loss = adv_criterion(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
                recon_loss = compute_reconstruction_loss(fake, y, m, target_columns, loss_map, mse_weight, l1_weight, target_loss_weights)
                g_loss = lambda_gan * gan_loss + lambda_recon * recon_loss

            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(g_loss).backward()
            if clip_grad > 0:
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad)
            scaler_g.step(opt_g)
            scaler_g.update()

            g_running += g_loss.item()
            d_running += d_loss.item()

        val_recon = validate_generator(generator, val_loader, device, target_columns, loss_map, amp_enabled, mse_weight, l1_weight, target_loss_weights)
        row = {
            'epoch': epoch,
            'adversarial_loss': adversarial_loss_name,
            'generator_loss': g_running / max(len(train_loader), 1),
            'discriminator_loss': d_running / max(len(train_loader), 1),
            'val_recon_loss': val_recon,
        }
        history.append(row)
        print(json.dumps(row))

        state = {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': opt_g.state_dict(),
            'optimizer_d': opt_d.state_dict(),
            'scaler_g': scaler_g.state_dict(),
            'scaler_d': scaler_d.state_dict(),
            'epoch': epoch,
            'val_recon_loss': val_recon,
            'best_val_recon_loss': best_val,
            'history': history,
            'config': cfg,
        }
        if val_recon < best_val:
            best_val = val_recon
            state['best_val_recon_loss'] = best_val
            torch.save(state, out_dir / 'best_cgan.pt')
        if epoch % int(cfg['training']['save_every']) == 0:
            torch.save(state, out_dir / f'epoch_{epoch}_cgan.pt')

    with (out_dir / 'history_cgan.json').open('w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)


if __name__ == '__main__':
    main()
