from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from config_utils import load_config, resolve_device
from heuristics_cgan import (
    apply_augmented_los_heuristics,
    apply_regression_heuristics,
    denormalize_array,
)
from model_cgan import UNetGenerator


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr)
    arr_min, arr_max = arr.min(), arr.max()
    if abs(arr_max - arr_min) < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - arr_min) / (arr_max - arr_min)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def parse_scalar_values(raw: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not raw:
        return result
    for part in [chunk.strip() for chunk in raw.split(',') if chunk.strip()]:
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        result[key.strip()] = float(value.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Run cGAN+U-Net inference on one image')
    parser.add_argument('--config', type=str, default='configs/cgan_unet.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--los-input', type=str, default=None)
    parser.add_argument('--scalar-values', type=str, default='')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg['runtime']['device'])
    image_size = int(cfg['data']['image_size'])
    target_columns = list(cfg['target_columns'])
    target_metadata = dict(cfg.get('target_metadata', {}))
    post_cfg = dict(cfg.get('postprocess', {}))

    in_channels = 1
    if cfg['data'].get('los_input_column'):
        in_channels += 1
    if bool(cfg['model']['use_scalar_channels']):
        in_channels += len(list(cfg['data'].get('scalar_feature_columns', [])))
        in_channels += len(dict(cfg['data'].get('constant_scalar_features', {})))

    generator = UNetGenerator(
        in_channels=in_channels,
        out_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['base_channels']),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(state['generator'] if 'generator' in state else state)
    generator.eval()

    image = Image.open(args.input).convert('L').resize((image_size, image_size), Image.BILINEAR)
    channels = [TF.to_tensor(image)]
    binary_los_np = None
    if cfg['data'].get('los_input_column'):
        if not args.los_input:
            raise ValueError('Config expects LoS input channel; provide --los-input path.')
        los_image = Image.open(args.los_input).convert('L').resize((image_size, image_size), Image.BILINEAR)
        channels.append(TF.to_tensor(los_image))
        binary_los_np = np.asarray(los_image, dtype=np.float32) / 255.0

    if bool(cfg['model']['use_scalar_channels']):
        scalar_columns = list(cfg['data'].get('scalar_feature_columns', []))
        constant_scalars = dict(cfg['data'].get('constant_scalar_features', {}))
        scalar_norms = dict(cfg['data'].get('scalar_feature_norms', {}))
        user_scalars = parse_scalar_values(args.scalar_values)
        values = []
        for col in scalar_columns:
            raw_value = float(user_scalars.get(col, 0.0))
            norm = float(scalar_norms.get(col, 1.0)) or 1.0
            values.append(raw_value / norm)
        for col, raw_value in constant_scalars.items():
            norm = float(scalar_norms.get(col, 1.0)) or 1.0
            values.append(float(raw_value) / norm)
        if values:
            scalar_tensor = torch.tensor(values, dtype=torch.float32).view(len(values), 1, 1).expand(len(values), image_size, image_size)
            channels.append(scalar_tensor)

    model_input = torch.cat(channels, dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = generator(model_input).squeeze(0).cpu().numpy()

    out_dir = Path(cfg['runtime']['output_dir']) / 'predict_cgan_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'predictions_raw.npy', pred)

    for i, name in enumerate(target_columns):
        arr = pred[i]
        if name == 'augmented_los':
            arr_physical = denormalize_array(arr, target_metadata.get(name, {}))
            if bool(post_cfg.get('enable', True)):
                los_outputs = apply_augmented_los_heuristics(
                    arr_physical,
                    binary_los_input=binary_los_np,
                    kernel_size=int(post_cfg.get('augmented_los_median_kernel', 3)),
                    threshold=float(post_cfg.get('augmented_los_threshold', 0.5)),
                    export_binary=bool(post_cfg.get('export_augmented_los_binary', False)),
                    enforce_binary_los_consistency=bool(post_cfg.get('enforce_binary_los_consistency', False)),
                    binary_los_consistency_floor=float(post_cfg.get('binary_los_consistency_floor', 0.5)),
                )
                arr_physical = los_outputs['probabilities']
                if 'binary' in los_outputs:
                    np.save(out_dir / f'{name}_binary.npy', los_outputs['binary'])
            np.save(out_dir / f'{name}_soft.npy', arr_physical)
            arr = arr_physical
        elif cfg.get('target_losses', {}).get(name, 'mse').lower() == 'bce':
            arr = 1.0 / (1.0 + np.exp(-arr))
            np.save(out_dir / f'{name}_probabilities.npy', arr)
        else:
            arr_physical = denormalize_array(arr, target_metadata.get(name, {}))
            if bool(post_cfg.get('enable', True)):
                arr_physical = apply_regression_heuristics(
                    arr_physical,
                    target_metadata.get(name, {}),
                    kernel_size=int(post_cfg.get('regression_median_kernel', 3)),
                )
            np.save(out_dir / f'{name}_physical.npy', arr_physical)
        Image.fromarray(to_uint8(arr), mode='L').save(out_dir / f'{name}.png')

    print(f'cGAN predictions saved to: {out_dir}')


if __name__ == '__main__':
    main()
