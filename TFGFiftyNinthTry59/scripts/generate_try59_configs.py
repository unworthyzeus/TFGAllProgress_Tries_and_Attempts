from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRY56_TOPOLOGY_CLASSES = [
    "open_sparse_lowrise",
    "open_sparse_vertical",
    "mixed_compact_lowrise",
    "mixed_compact_midrise",
    "dense_block_midrise",
    "dense_block_highrise",
]

BASE_CONFIG = ROOT.parent / 'TFGTwentySixthTry26' / 'experiments' / 'twentysixthtry26_delay_angular_gradient' / 'twentysixthtry26_delay_angular_gradient.yaml'
OUTPUT_DIR = ROOT / 'experiments' / 'fiftyninthtry59_topology_experts'


def _load_base() -> Dict:
    with BASE_CONFIG.open('r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError('Base config must be a YAML mapping.')
    return cfg


def _make_expert_cfg(base_cfg: Dict, topology_class: str) -> Dict:
    cfg = deepcopy(base_cfg)
    cfg['seed'] = 56

    data_cfg = cfg.setdefault('data', {})
    data_cfg['hdf5_path'] = '../../../Datasets/CKM_Dataset_270326.h5'
    data_cfg['split_mode'] = 'city_holdout'
    data_cfg['topology_mask_channel'] = True
    data_cfg['topology_mask_threshold'] = 0.0
    data_cfg['apply_topology_mask_to_targets'] = True
    data_cfg['append_no_data_target'] = True
    data_cfg['partition_filter'] = {'topology_class': topology_class}
    data_cfg['topology_partitioning'] = {
        'density_q1': 0.12,
        'density_q2': 0.28,
        'height_q1': 12.0,
        'height_q2': 28.0,
    }
    data_cfg['num_workers'] = 2

    model_cfg = cfg.setdefault('model', {})
    model_cfg['base_channels'] = 10
    model_cfg['gradient_checkpointing'] = False
    model_cfg['out_channels'] = 2
    model_cfg['scalar_film_hidden'] = 32
    model_cfg['dropout_down3'] = 0.18
    model_cfg['dropout_down4'] = 0.30
    model_cfg['dropout_up1'] = 0.16
    model_cfg.pop('disc_base_channels', None)
    model_cfg.pop('disc_norm_type', None)
    model_cfg.pop('disc_input_downsample_factor', None)

    training_cfg = cfg.setdefault('training', {})
    training_cfg['epochs'] = 500
    training_cfg['batch_size'] = 6
    training_cfg['save_every'] = 1
    training_cfg['generator_optimizer'] = 'adamw'
    training_cfg['generator_lr'] = 2.0e-3
    training_cfg['weight_decay'] = 0.10
    training_cfg['ema_decay'] = 1.0
    training_cfg['lr_scheduler'] = 'none'
    training_cfg['early_stopping'] = {
        'enabled': True,
        'patience': 5,
        'min_delta': 0.0,
        'rewind_to_best_model': True,
    }
    training_cfg.pop('discriminator_optimizer', None)
    training_cfg.pop('discriminator_lr', None)
    training_cfg.pop('discriminator_optimizer_foreach', None)
    training_cfg['auto_batch_by_vram'] = {
        'enabled': False,
        'reference_vram_gb': 12.0,
        'reference_batch_size': 6,
        'min_batch_size': 1,
        'max_batch_size': 6,
        'safety_factor': 0.92,
    }

    cfg['target_columns'] = ['angular_spread', 'no_data']
    cfg['target_losses'] = {
        'angular_spread': 'mse',
        'no_data': 'bce',
    }
    cfg['target_metadata'] = {
        'angular_spread': cfg['target_metadata']['angular_spread'],
    }
    loss_cfg = cfg.setdefault('loss', {})
    loss_cfg['lambda_recon'] = 1.0
    loss_cfg['target_loss_weights'] = {
        'angular_spread': 1.0,
        'no_data': 0.05,
    }
    loss_cfg.pop('lambda_gan', None)
    loss_cfg.pop('adversarial_loss', None)

    multiscale_cfg = cfg.setdefault('multiscale_targets', {})
    multiscale_cfg['enabled'] = False

    gradient_cfg = cfg.setdefault('gradient_targets', {})
    gradient_cfg['enabled'] = True
    gradient_cfg['targets'] = ['angular_spread']
    gradient_cfg['target_weights'] = {'angular_spread': 1.0}
    gradient_cfg['min_valid_ratio'] = 0.5
    gradient_cfg['loss_weight'] = 0.1

    runtime_cfg = cfg.setdefault('runtime', {})
    runtime_cfg['output_dir'] = f'outputs/fiftyninthtry59_{topology_class}'
    runtime_cfg['resume_checkpoint'] = None

    training_cfg['selection_metrics'] = {'angular_spread.rmse_physical': 1.0}

    cfg.setdefault('experiment', {})
    cfg['experiment'] = {
        'try': '59',
        'family': 'try59_angular_topology_experts',
        'topology_class': topology_class,
        'focused_target': 'angular_spread',
        'uses_topology_mask': True,
        'uses_no_data_aux': True,
        'speed_profile': 'throughput_optimized',
    }
    return cfg


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = _load_base()
    registry = []
    for topology_class in TRY56_TOPOLOGY_CLASSES:
        cfg = _make_expert_cfg(base_cfg, topology_class)
        out_path = OUTPUT_DIR / f'fiftyninthtry59_expert_{topology_class}.yaml'
        with out_path.open('w', encoding='utf-8') as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        registry.append(
            {
                'topology_class': topology_class,
                'config': str(out_path.relative_to(ROOT)).replace('\\', '/'),
                'output_dir': cfg['runtime']['output_dir'],
            }
        )

    registry_path = OUTPUT_DIR / 'fiftyninthtry59_expert_registry.yaml'
    with registry_path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump({'experts': registry}, handle, sort_keys=False)

    print(f'Generated {len(registry)} Try 59 expert configs in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
