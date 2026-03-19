from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import amp
from torch.utils.data import DataLoader, Subset

from config_utils import ensure_output_dir, is_cuda_device, load_config, resolve_device
from data_utils import build_cross_validation_datasets_from_config, compute_input_channels
from evaluate import summarize_loader
from model_unet import CKMUNet
from train import build_loss_map, build_optimizer, eval_epoch, run_epoch, set_seed


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
    parser = argparse.ArgumentParser(description='K-fold cross-validation for CKM U-Net while keeping test held out')
    parser.add_argument('--config', type=str, default='configs/baseline_hdf5.yaml')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cv_seed = int(args.seed if args.seed is not None else cfg['data'].get('split_seed', cfg.get('seed', 42)))
    set_seed(int(cfg['seed']))
    device = resolve_device(cfg['runtime']['device'])

    dev_train_dataset, dev_eval_dataset, test_dataset = build_cross_validation_datasets_from_config(cfg)
    folds = build_fold_indices(len(dev_eval_dataset), int(args.folds), cv_seed)

    base_out_dir = ensure_output_dir(Path(cfg['runtime']['output_dir']) / f'crossval_{args.folds}fold')
    pin_memory = is_cuda_device(device)
    amp_enabled = bool(cfg['training']['amp']) and is_cuda_device(device)
    target_columns = list(cfg['target_columns'])
    target_losses = dict(cfg.get('target_losses', {}))
    target_metadata = dict(cfg.get('target_metadata', {}))
    target_loss_weights = {str(k): float(v) for k, v in dict(cfg['loss'].get('target_loss_weights', {})).items()}
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

        model = CKMUNet(
            in_channels=in_channels,
            out_channels=int(cfg['model']['out_channels']),
            base_channels=int(cfg['model']['base_channels']),
            gradient_checkpointing=bool(cfg['model'].get('gradient_checkpointing', False)),
        ).to(device)
        optimizer = build_optimizer(cfg, model, device)
        scaler = amp.GradScaler('cuda', enabled=amp_enabled)
        loss_map = build_loss_map(target_columns, target_losses)

        best_val = float('inf')
        history = []
        best_state = None

        for epoch in range(1, int(cfg['training']['epochs']) + 1):
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
            print(f'[fold {fold_index}/{args.folds}] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}')

            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'best_val_loss': best_val,
                    'history': history,
                    'fold': fold_index,
                    'folds': int(args.folds),
                }
                torch.save(best_state, fold_out_dir / 'best.pt')

        with (fold_out_dir / 'history.json').open('w', encoding='utf-8') as handle:
            json.dump(history, handle, indent=2)

        if best_state is None:
            raise RuntimeError(f'No best state generated for fold {fold_index}')

        model.load_state_dict(best_state['model'])
        val_metrics = summarize_loader(model, val_loader, device, target_columns, target_losses, target_metadata, amp_enabled)
        fold_record: Dict[str, object] = {
            'fold': fold_index,
            'best_epoch': int(best_state['epoch']),
            'best_val_loss': float(best_state['best_val_loss']),
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
            fold_record['test_metrics'] = summarize_loader(model, test_loader, device, target_columns, target_losses, target_metadata, amp_enabled)

        fold_results.append(fold_record)
        (fold_out_dir / 'crossval_metrics.json').write_text(json.dumps(fold_record, indent=2), encoding='utf-8')

    aggregate = {
        'folds': int(args.folds),
        'seed': cv_seed,
        'evaluate_test': bool(args.evaluate_test),
        'test_held_out': test_dataset is not None,
        'summary': summarize_scalar_metrics([
            {'best_val_loss': float(record['best_val_loss'])} for record in fold_results
        ]),
    }
    aggregate['fold_results'] = fold_results

    aggregate_path = base_out_dir / 'crossval_summary.json'
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()