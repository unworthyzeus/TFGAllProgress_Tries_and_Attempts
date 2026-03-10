from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config_utils import ensure_output_dir, load_config, resolve_device
from data_utils import CKMDataset
from model_cgan import PatchDiscriminator, UNetGenerator


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


def compute_reconstruction_loss(
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
        total = total + loss
        valid_count = valid_count + (msk.sum() > 0).float()
    return total / valid_count.clamp_min(1.0)


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, int]:
    target_columns = list(cfg['target_columns'])
    common = dict(
        root_dir=cfg['data']['root_dir'],
        target_columns=target_columns,
        image_size=int(cfg['data']['image_size']),
        add_scalar_channels=bool(cfg['model']['use_scalar_channels']),
        scalar_feature_columns=list(cfg['data'].get('scalar_feature_columns', [])),
        constant_scalar_features=dict(cfg['data'].get('constant_scalar_features', {})),
        scalar_feature_norms=dict(cfg['data'].get('scalar_feature_norms', {})),
        los_input_column=cfg['data'].get('los_input_column'),
    )
    train_set = CKMDataset(
        manifest_csv=cfg['data']['train_manifest'],
        augment=bool(cfg['augmentation']['enable']),
        hflip_prob=float(cfg['augmentation']['hflip_prob']),
        vflip_prob=float(cfg['augmentation']['vflip_prob']),
        rot90_prob=float(cfg['augmentation']['rot90_prob']),
        **common,
    )
    val_set = CKMDataset(
        manifest_csv=cfg['data']['val_manifest'],
        augment=False,
        **common,
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
) -> float:
    generator.eval()
    total = 0.0
    with torch.no_grad():
        for x, y, m in tqdm(loader, desc='val', leave=False):
            x, y, m = x.to(device), y.to(device), m.to(device)
            with autocast(enabled=amp_enabled):
                pred = generator(x)
                recon = compute_reconstruction_loss(pred, y, m, target_columns, loss_map, mse_weight, l1_weight)
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

    train_loader, val_loader, in_channels = build_dataloaders(cfg)

    generator = UNetGenerator(
        in_channels=in_channels,
        out_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['base_channels']),
    ).to(device)
    discriminator = PatchDiscriminator(
        in_channels=in_channels,
        target_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['disc_base_channels']),
    ).to(device)

    resume = cfg['runtime'].get('resume_checkpoint')
    if resume:
        state = torch.load(resume, map_location=device)
        if 'generator' in state:
            generator.load_state_dict(state['generator'])
        if 'discriminator' in state:
            discriminator.load_state_dict(state['discriminator'])

    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(cfg['training']['generator_lr']),
        betas=(float(cfg['training']['beta1']), float(cfg['training']['beta2'])),
        weight_decay=float(cfg['training']['weight_decay']),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(cfg['training']['discriminator_lr']),
        betas=(float(cfg['training']['beta1']), float(cfg['training']['beta2'])),
        weight_decay=float(cfg['training']['weight_decay']),
    )

    adv_criterion = nn.BCEWithLogitsLoss()
    loss_map = build_loss_map(target_columns, target_losses)
    scaler_g = GradScaler(enabled=bool(cfg['training']['amp']) and device == 'cuda')
    scaler_d = GradScaler(enabled=bool(cfg['training']['amp']) and device == 'cuda')
    lambda_gan = float(cfg['loss']['lambda_gan'])
    lambda_recon = float(cfg['loss']['lambda_recon'])
    mse_weight = float(cfg['loss']['mse_weight'])
    l1_weight = float(cfg['loss']['l1_weight'])
    clip_grad = float(cfg['training']['clip_grad_norm'])
    amp_enabled = bool(cfg['training']['amp']) and device == 'cuda'

    best_val = float('inf')
    history = []

    for epoch in range(1, int(cfg['training']['epochs']) + 1):
        generator.train()
        discriminator.train()
        g_running = 0.0
        d_running = 0.0

        for x, y, m in tqdm(train_loader, desc=f'epoch {epoch}', leave=False):
            x, y, m = x.to(device), y.to(device), m.to(device)

            with autocast(enabled=amp_enabled):
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

            with autocast(enabled=amp_enabled):
                fake = generator(x)
                fake_logits_for_g = discriminator(x, fake)
                gan_loss = adv_criterion(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
                recon_loss = compute_reconstruction_loss(fake, y, m, target_columns, loss_map, mse_weight, l1_weight)
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

        val_recon = validate_generator(generator, val_loader, device, target_columns, loss_map, amp_enabled, mse_weight, l1_weight)
        row = {
            'epoch': epoch,
            'generator_loss': g_running / max(len(train_loader), 1),
            'discriminator_loss': d_running / max(len(train_loader), 1),
            'val_recon_loss': val_recon,
        }
        history.append(row)
        print(json.dumps(row))

        state = {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'epoch': epoch,
            'val_recon_loss': val_recon,
            'config': cfg,
        }
        if val_recon < best_val:
            best_val = val_recon
            torch.save(state, out_dir / 'best_cgan.pt')
        if epoch % int(cfg['training']['save_every']) == 0:
            torch.save(state, out_dir / f'epoch_{epoch}_cgan.pt')

    with (out_dir / 'history_cgan.json').open('w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)


if __name__ == '__main__':
    main()
