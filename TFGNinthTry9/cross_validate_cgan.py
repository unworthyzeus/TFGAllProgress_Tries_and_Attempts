from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import torch
from torch import amp
from torch.utils.data import DataLoader, Subset

from config_utils import anchor_data_paths_to_config_file, ensure_output_dir, is_cuda_device, load_config, resolve_device
from data_utils import (
    build_cross_validation_datasets_from_config,
    compute_input_channels,
    compute_scalar_cond_dim,
    forward_cgan_generator,
    unpack_cgan_batch,
    uses_scalar_film_conditioning,
)
from evaluate_cgan import summarize_loader
from model_cgan import PatchDiscriminator, UNetGenerator
from train_cgan import (
    build_adversarial_loss,
    build_loss_map,
    build_optimizer,
    compute_reconstruction_loss,
    is_path_loss_hybrid_enabled,
    resolve_adversarial_loss_name,
    set_seed,
    validate_generator,
)


def build_fold_indices(total_size: int, folds: int, seed: int) -> List[List[int]]:
    if folds < 2:
        raise ValueError('folds must be >= 2')
    if total_size < folds:
        raise ValueError('Number of folds cannot exceed development dataset size')

    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_sizes = [total_size // folds] * folds
    for i in range(total_size % folds):
        fold_sizes[i] += 1

    result: List[List[int]] = []
    cursor = 0
    for fold_size in fold_sizes:
        result.append(indices[cursor:cursor + fold_size])
        cursor += fold_size
    return result


def summarize_scalar_metrics(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for record in records for key in record.keys()})
    summary: Dict[str, float] = {}
    for key in keys:
        values = [float(record[key]) for record in records if key in record]
        if not values:
            continue
        summary[f'{key}_mean'] = float(mean(values))
        summary[f'{key}_std'] = float(pstdev(values)) if len(values) > 1 else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='K-fold cross-validation for CKM cGAN while keeping test held out')
    parser.add_argument('--config', type=str, default='configs/cgan_unet_hdf5.yaml')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    cv_seed = int(args.seed if args.seed is not None else cfg['data'].get('split_seed', cfg.get('seed', 42)))
    set_seed(int(cfg['seed']))
    device = resolve_device(cfg['runtime']['device'])
    pin_memory = is_cuda_device(device)
    amp_enabled = bool(cfg['training']['amp']) and is_cuda_device(device)

    dev_train_dataset, dev_eval_dataset, test_dataset = build_cross_validation_datasets_from_config(cfg)
    folds = build_fold_indices(len(dev_eval_dataset), int(args.folds), cv_seed)
    base_out_dir = ensure_output_dir(Path(cfg['runtime']['output_dir']) / f'crossval_{args.folds}fold')

    target_columns = list(cfg['target_columns'])
    target_losses = dict(cfg.get('target_losses', {}))
    target_metadata = dict(cfg.get('target_metadata', {}))
    target_loss_weights = {str(k): float(v) for k, v in dict(cfg['loss'].get('target_loss_weights', {})).items()}
    target_metadata = dict(cfg.get('target_metadata', {}))
    in_channels = compute_input_channels(cfg)

    fold_results = []

    for fold_index, val_indices in enumerate(folds, start=1):
        train_indices = [idx for current_fold, current_indices in enumerate(folds, start=1) if current_fold != fold_index for idx in current_indices]
        fold_out_dir = ensure_output_dir(base_out_dir / f'fold_{fold_index}')

        train_subset = Subset(dev_train_dataset, train_indices)
        val_subset = Subset(dev_eval_dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=int(cfg['training']['batch_size']),
            shuffle=True,
            num_workers=int(cfg['data']['num_workers']),
            pin_memory=pin_memory,
            persistent_workers=int(cfg['data']['num_workers']) > 0,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=int(cfg['training']['batch_size']),
            shuffle=False,
            num_workers=max(1, int(cfg['data']['num_workers']) // 2),
            pin_memory=pin_memory,
            persistent_workers=max(1, int(cfg['data']['num_workers']) // 2) > 0,
        )

        sc_dim = int(compute_scalar_cond_dim(cfg)) if uses_scalar_film_conditioning(cfg) else 0
        film_h = int(cfg['model'].get('scalar_film_hidden', 128))
        generator = UNetGenerator(
            in_channels=in_channels,
            out_channels=int(cfg['model']['out_channels']),
            base_channels=int(cfg['model']['base_channels']),
            gradient_checkpointing=bool(cfg['model'].get('gradient_checkpointing', False)),
            path_loss_hybrid=is_path_loss_hybrid_enabled(cfg),
            norm_type=str(cfg['model'].get('norm_type', 'batch')),
            scalar_cond_dim=sc_dim,
            scalar_film_hidden=film_h,
        ).to(device)
        discriminator = PatchDiscriminator(
            in_channels=in_channels,
            target_channels=int(cfg['model']['out_channels']),
            base_channels=int(cfg['model']['disc_base_channels']),
            norm_type=str(cfg['model'].get('disc_norm_type', cfg['model'].get('norm_type', 'batch'))),
            input_downsample_factor=int(cfg['model'].get('disc_input_downsample_factor', 1)),
        ).to(device)

        momentum = float(cfg['training'].get('momentum', 0.0))
        opt_g = build_optimizer(
            str(cfg['training'].get('generator_optimizer', 'adam')),
            generator.parameters(),
            float(cfg['training']['generator_lr']),
            float(cfg['training']['weight_decay']),
            float(cfg['training']['beta1']),
            float(cfg['training']['beta2']),
            momentum,
            device,
        )
        opt_d = build_optimizer(
            str(cfg['training'].get('discriminator_optimizer', 'adam')),
            discriminator.parameters(),
            float(cfg['training']['discriminator_lr']),
            float(cfg['training']['weight_decay']),
            float(cfg['training']['beta1']),
            float(cfg['training']['beta2']),
            momentum,
            device,
        )
        scaler_g = amp.GradScaler('cuda', enabled=amp_enabled)
        scaler_d = amp.GradScaler('cuda', enabled=amp_enabled)
        loss_map = build_loss_map(target_columns, target_losses)
        adversarial_loss_name = resolve_adversarial_loss_name(cfg, device)
        adv_criterion = build_adversarial_loss(adversarial_loss_name)
        lambda_gan = float(cfg['loss']['lambda_gan'])
        lambda_recon = float(cfg['loss']['lambda_recon'])
        mse_weight = float(cfg['loss']['mse_weight'])
        l1_weight = float(cfg['loss']['l1_weight'])
        clip_grad = float(cfg['training']['clip_grad_norm'])

        best_val = float('inf')
        history = []
        best_state = None

        for epoch in range(1, int(cfg['training']['epochs']) + 1):
            generator.train()
            discriminator.train()
            g_running = 0.0
            d_running = 0.0

            for batch in train_loader:
                x, y, m, sc = unpack_cgan_batch(batch, device)

                with amp.autocast(device_type='cuda', enabled=amp_enabled):
                    fake = forward_cgan_generator(generator, x, sc)
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
                    fake = forward_cgan_generator(generator, x, sc)
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

            val_recon, _val_summary = validate_generator(
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
                'generator_loss': g_running / max(len(train_loader), 1),
                'discriminator_loss': d_running / max(len(train_loader), 1),
                'val_recon_loss': val_recon,
            }
            history.append(row)
            print(json.dumps({'fold': fold_index, **row}))

            if val_recon < best_val:
                best_val = val_recon
                best_state = {
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
                    'fold': fold_index,
                    'folds': int(args.folds),
                    'config': cfg,
                }
                torch.save(best_state, fold_out_dir / 'best_cgan.pt')

        with (fold_out_dir / 'history_cgan.json').open('w', encoding='utf-8') as handle:
            json.dump(history, handle, indent=2)

        if best_state is None:
            raise RuntimeError(f'No best state generated for fold {fold_index}')

        generator.load_state_dict(best_state['generator'])
        val_metrics = summarize_loader(generator, val_loader, device, target_columns, target_losses, target_metadata, amp_enabled)
        fold_record: Dict[str, object] = {
            'fold': fold_index,
            'best_epoch': int(best_state['epoch']),
            'best_val_recon_loss': float(best_state['best_val_recon_loss']),
            'val_metrics': val_metrics,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
        }

        if args.evaluate_test and test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=int(cfg['training']['batch_size']),
                shuffle=False,
                num_workers=max(1, int(cfg['data']['num_workers']) // 2),
                pin_memory=pin_memory,
                persistent_workers=max(1, int(cfg['data']['num_workers']) // 2) > 0,
            )
            fold_record['test_metrics'] = summarize_loader(generator, test_loader, device, target_columns, target_losses, target_metadata, amp_enabled)

        fold_results.append(fold_record)
        (fold_out_dir / 'crossval_metrics_cgan.json').write_text(json.dumps(fold_record, indent=2), encoding='utf-8')

    aggregate = {
        'folds': int(args.folds),
        'seed': cv_seed,
        'evaluate_test': bool(args.evaluate_test),
        'test_held_out': test_dataset is not None,
        'summary': summarize_scalar_metrics([
            {'best_val_recon_loss': float(record['best_val_recon_loss'])} for record in fold_results
        ]),
        'fold_results': fold_results,
    }
    aggregate_path = base_out_dir / 'crossval_summary_cgan.json'
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()