from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from config_utils import load_config, resolve_device
from model_unet import CKMUNet


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
    parts = [chunk.strip() for chunk in raw.split(',') if chunk.strip()]
    for part in parts:
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        result[key.strip()] = float(value.strip())
    return result


def denormalize_array(arr: np.ndarray, metadata: Dict[str, object]) -> np.ndarray:
    scale = float(metadata.get('scale', 1.0))
    offset = float(metadata.get('offset', 0.0))
    return arr * scale + offset


def main() -> None:
    parser = argparse.ArgumentParser(description='Run CKM inference on one image')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
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
    if int(cfg['model']['out_channels']) != len(target_columns):
        raise ValueError("model.out_channels must match len(target_columns)")

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

    image = Image.open(args.input).convert('L').resize((image_size, image_size), Image.BILINEAR)
    channels = [TF.to_tensor(image)]

    if cfg['data'].get('los_input_column'):
        if not args.los_input:
            raise ValueError('Config expects LoS input channel; provide --los-input path.')
        los_image = Image.open(args.los_input).convert('L').resize((image_size, image_size), Image.BILINEAR)
        channels.append(TF.to_tensor(los_image))

    if bool(cfg['model']['use_scalar_channels']):
        scalar_columns = list(cfg['data'].get('scalar_feature_columns', []))
        constant_scalars = dict(cfg['data'].get('constant_scalar_features', {}))
        scalar_norms = dict(cfg['data'].get('scalar_feature_norms', {}))
        user_scalars = parse_scalar_values(args.scalar_values)

        values = []
        for col in scalar_columns:
            raw_value = float(user_scalars.get(col, 0.0))
            norm = float(scalar_norms.get(col, 1.0))
            norm = norm if abs(norm) > 1e-12 else 1.0
            values.append(raw_value / norm)

        for col, raw_value in constant_scalars.items():
            norm = float(scalar_norms.get(col, 1.0))
            norm = norm if abs(norm) > 1e-12 else 1.0
            values.append(float(raw_value) / norm)

        if values:
            scalar_tensor = torch.tensor(values, dtype=torch.float32).view(len(values), 1, 1).expand(len(values), image_size, image_size)
            channels.append(scalar_tensor)

    model_input = torch.cat(channels, dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(model_input).squeeze(0).cpu().numpy()

    out_dir = Path(cfg['runtime']['output_dir']) / 'predict_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / 'predictions_raw.npy', pred)

    for i, name in enumerate(target_columns):
        arr = pred[i]
        if cfg.get('target_losses', {}).get(name, 'mse').lower() == 'bce':
            arr = 1.0 / (1.0 + np.exp(-arr))
            np.save(out_dir / f'{name}_probabilities.npy', arr)
        else:
            metadata = target_metadata.get(name, {})
            arr_physical = denormalize_array(arr, metadata) if metadata else arr
            np.save(out_dir / f'{name}_physical.npy', arr_physical)
        img = Image.fromarray(to_uint8(arr), mode='L')
        img.save(out_dir / f'{name}.png')

    print(f'Prediction maps saved to: {out_dir}')


if __name__ == '__main__':
    main()
