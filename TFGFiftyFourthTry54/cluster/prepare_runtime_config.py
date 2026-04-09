from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f'Config at {path} must be a YAML mapping.')
    return data


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def detect_vram_gb() -> Optional[float]:
    env_value = os.environ.get('GPU_VRAM_GB')
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            pass

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if lines:
            return float(lines[0]) / 1024.0
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return float(total_memory) / (1024.0 ** 3)
    except Exception:
        pass

    return None


def apply_dynamic_batch_size(cfg: Dict[str, Any], detected_vram_gb: Optional[float]) -> Dict[str, Any]:
    training_cfg = dict(cfg.get('training', {}))
    auto_cfg = dict(training_cfg.get('auto_batch_by_vram', {}))
    if not auto_cfg.get('enabled', False) or detected_vram_gb is None:
        cfg['training'] = training_cfg
        return cfg

    reference_vram = max(float(auto_cfg.get('reference_vram_gb', 16.0)), 1.0)
    reference_batch = max(int(auto_cfg.get('reference_batch_size', training_cfg.get('batch_size', 1))), 1)
    min_batch = max(int(auto_cfg.get('min_batch_size', 1)), 1)
    max_batch = max(int(auto_cfg.get('max_batch_size', reference_batch)), min_batch)
    safety_factor = float(auto_cfg.get('safety_factor', 0.9))
    effective_vram = max(detected_vram_gb * safety_factor, 0.1)

    scaled_batch = int(round(reference_batch * (effective_vram / reference_vram)))
    resolved_batch = min(max(scaled_batch, min_batch), max_batch)

    training_cfg['batch_size'] = resolved_batch
    training_cfg['resolved_batch_size'] = resolved_batch
    training_cfg['detected_vram_gb'] = round(detected_vram_gb, 2)
    cfg['training'] = training_cfg
    return cfg


def apply_runtime_overrides(
    cfg: Dict[str, Any],
    output_suffix: Optional[str],
    seed_offset: int,
    hdf5_path: Optional[str] = None,
    scalar_table_csv: Optional[str] = None,
    formula_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if seed_offset:
        cfg['seed'] = int(cfg.get('seed', 42)) + int(seed_offset)

    if output_suffix:
        runtime_cfg = dict(cfg.get('runtime', {}))
        base_output_dir = str(runtime_cfg.get('output_dir', 'outputs/run'))
        runtime_cfg['output_dir'] = f"{base_output_dir}_{output_suffix}"
        cfg['runtime'] = runtime_cfg

    if hdf5_path or scalar_table_csv:
        data_cfg = dict(cfg.get('data', {}))
        if hdf5_path:
            data_cfg['hdf5_path'] = hdf5_path
        if scalar_table_csv:
            data_cfg['scalar_table_csv'] = scalar_table_csv
        cfg['data'] = data_cfg

    if formula_cache_dir:
        data_cfg = dict(cfg.get('data', {}))
        formula_cfg = dict(data_cfg.get('path_loss_formula_input', {}))
        formula_cfg['cache_dir'] = formula_cache_dir
        formula_cfg['cache_enabled'] = True
        data_cfg['path_loss_formula_input'] = formula_cfg
        cfg['data'] = data_cfg

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a runtime config with batch size adapted to GPU VRAM.')
    parser.add_argument('--input-config', required=True)
    parser.add_argument('--output-config', required=True)
    parser.add_argument('--output-suffix', default='')
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--hdf5-path', default='', help='Override data.hdf5_path (e.g. shared cluster path)')
    parser.add_argument(
        '--scalar-csv-path',
        default='',
        help='Override data.scalar_table_csv (per-sample antenna height CSV on cluster)',
    )
    parser.add_argument(
        '--formula-cache-dir',
        default='',
        help='Override data.path_loss_formula_input.cache_dir, typically to node-local scratch.',
    )
    args = parser.parse_args()

    input_path = Path(args.input_config)
    output_path = Path(args.output_config)

    cfg = load_yaml(input_path)
    detected_vram_gb = detect_vram_gb()
    cfg = apply_dynamic_batch_size(cfg, detected_vram_gb)
    cfg = apply_runtime_overrides(
        cfg, args.output_suffix or None, args.seed_offset,
        args.hdf5_path or None,
        args.scalar_csv_path or None,
        args.formula_cache_dir or None,
    )
    save_yaml(output_path, cfg)

    batch_size = cfg.get('training', {}).get('batch_size')
    print(f'input_config={input_path}')
    print(f'output_config={output_path}')
    print(f'detected_vram_gb={detected_vram_gb}')
    print(f'resolved_batch_size={batch_size}')
    print(f'output_suffix={args.output_suffix}')
    print(f'seed={cfg.get("seed")}')


if __name__ == '__main__':
    main()
