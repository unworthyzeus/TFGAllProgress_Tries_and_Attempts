from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from config_utils import anchor_data_paths_to_config_file, load_config, load_torch_checkpoint, resolve_device
from data_utils import (
    _compute_distance_map_2d,
    compute_input_channels,
    compute_scalar_cond_dim,
    uses_scalar_film_conditioning,
)
from heuristics_cgan import (
    apply_augmented_los_heuristics,
    apply_binary_mask_heuristics,
    apply_path_loss_confidence_fallback,
    apply_path_loss_los_correction,
    apply_regression_heuristics,
    denormalize_array,
    derive_augmented_los_heuristic,
    derive_channel_power_from_path_loss,
    derive_link_availability,
    derive_snr_maps,
)
from model_cgan import UNetGenerator


def load_saved_heuristic_calibration(output_dir: Path) -> Dict[str, object] | None:
    calibration_path = output_dir / 'heuristic_calibration.json'
    if not calibration_path.exists():
        legacy_path = output_dir / 'los_mask_calibration.json'
        if not legacy_path.exists():
            return None
        calibration_path = legacy_path
    return json.loads(calibration_path.read_text(encoding='utf-8'))


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
    anchor_data_paths_to_config_file(cfg, args.config)
    device = resolve_device(cfg['runtime']['device'])
    image_size = int(cfg['data']['image_size'])
    target_columns = list(cfg['target_columns'])
    target_metadata = dict(cfg.get('target_metadata', {}))
    post_cfg = dict(cfg.get('postprocess', {}))
    hybrid_cfg = dict(cfg.get('path_loss_hybrid', {}))
    hybrid_enabled = bool(hybrid_cfg.get('enabled', False))
    if bool(post_cfg.get('use_saved_heuristic_calibration', True)):
        saved_calibration = load_saved_heuristic_calibration(Path(cfg['runtime']['output_dir']))
        if saved_calibration is not None:
            if 'path_loss' in saved_calibration:
                post_cfg['path_loss_median_kernel'] = int(saved_calibration['path_loss'].get('best_median_kernel', post_cfg.get('path_loss_median_kernel', 5)))

    in_channels = compute_input_channels(cfg)

    sc_dim = int(compute_scalar_cond_dim(cfg)) if uses_scalar_film_conditioning(cfg) else 0
    film_h = int(cfg['model'].get('scalar_film_hidden', 128))
    generator = UNetGenerator(
        in_channels=in_channels,
        out_channels=int(cfg['model']['out_channels']),
        base_channels=int(cfg['model']['base_channels']),
        gradient_checkpointing=bool(cfg['model'].get('gradient_checkpointing', False)),
        path_loss_hybrid=hybrid_enabled,
        norm_type=str(cfg['model'].get('norm_type', 'batch')),
        scalar_cond_dim=sc_dim,
        scalar_film_hidden=film_h,
    ).to(device)

    state = load_torch_checkpoint(args.checkpoint, device)
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

    if cfg['data'].get('distance_map_channel', False):
        channels.append(_compute_distance_map_2d(image_size))

    scalar_cond_batch: torch.Tensor | None = None
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
            if uses_scalar_film_conditioning(cfg):
                scalar_cond_batch = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                scalar_tensor = torch.tensor(values, dtype=torch.float32).view(len(values), 1, 1).expand(
                    len(values), image_size, image_size
                )
                channels.append(scalar_tensor)

    model_input = torch.cat(channels, dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        if scalar_cond_batch is not None:
            pred = generator(model_input, scalar_cond_batch).squeeze(0).cpu().numpy()
        else:
            pred = generator(model_input).squeeze(0).cpu().numpy()

    out_dir = Path(cfg['runtime']['output_dir']) / 'predict_cgan_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'predictions_raw.npy', pred)
    physical_outputs: Dict[str, np.ndarray] = {}
    confidence_map = None
    if hybrid_enabled and pred.shape[0] > len(target_columns):
        confidence_map = 1.0 / (1.0 + np.exp(-pred[len(target_columns)]))
        np.save(out_dir / 'path_loss_confidence.npy', confidence_map)
        Image.fromarray(to_uint8(confidence_map), mode='L').save(out_dir / 'path_loss_confidence.png')

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
        elif name == 'los_mask':
            if cfg.get('target_losses', {}).get(name, 'mse').lower() == 'bce':
                arr_probs = 1.0 / (1.0 + np.exp(-arr))
            else:
                arr_probs = denormalize_array(arr, target_metadata.get(name, {}))

            if bool(post_cfg.get('enable', True)):
                los_mask_outputs = apply_binary_mask_heuristics(
                    arr_probs,
                    threshold=float(post_cfg.get('los_mask_threshold', 0.5)),
                    export_binary=bool(post_cfg.get('export_los_mask_binary', False)),
                )
                arr_probs = los_mask_outputs['probabilities']
                if 'binary' in los_mask_outputs:
                    np.save(out_dir / f'{name}_binary.npy', los_mask_outputs['binary'])

            np.save(out_dir / f'{name}_probabilities.npy', arr_probs)
            arr = arr_probs
            physical_outputs[name] = arr_probs
        elif cfg.get('target_losses', {}).get(name, 'mse').lower() == 'bce':
            arr = 1.0 / (1.0 + np.exp(-arr))
            np.save(out_dir / f'{name}_probabilities.npy', arr)
            physical_outputs[name] = arr
        else:
            arr_physical = denormalize_array(arr, target_metadata.get(name, {}))
            if name == 'path_loss' and hybrid_enabled and confidence_map is not None:
                fallback_outputs = apply_path_loss_confidence_fallback(
                    arr_physical,
                    confidence_map,
                    target_metadata.get(name, {}),
                    confidence_threshold=float(hybrid_cfg.get('fallback_threshold', 0.5)),
                    fallback_mode=str(hybrid_cfg.get('fallback_mode', 'replace')),
                    kernel_size=int(post_cfg.get('path_loss_median_kernel', post_cfg.get('regression_median_kernel', 3))),
                    los_mask=binary_los_np,
                    los_correction_enabled=bool(post_cfg.get('path_loss_los_correction', False)) and binary_los_np is not None,
                    frequency_ghz=float(post_cfg.get('path_loss_los_frequency_ghz', 7.125)),
                    blend_weight=float(post_cfg.get('path_loss_los_blend_weight', 0.3)),
                )
                np.save(out_dir / 'path_loss_coarse_physical.npy', arr_physical)
                np.save(out_dir / 'path_loss_heuristic_prior.npy', fallback_outputs['heuristic_path_loss_db'])
                np.save(out_dir / 'path_loss_low_confidence_mask.npy', fallback_outputs['low_confidence_mask'])
                Image.fromarray(to_uint8(fallback_outputs['heuristic_path_loss_db']), mode='L').save(out_dir / 'path_loss_heuristic_prior.png')
                Image.fromarray((fallback_outputs['low_confidence_mask'] * 255.0).astype(np.uint8), mode='L').save(out_dir / 'path_loss_low_confidence_mask.png')
                arr_physical = fallback_outputs['final_path_loss_db']
            elif bool(post_cfg.get('enable', True)):
                regression_kernel = int(post_cfg.get('regression_median_kernel', 3))
                if name == 'path_loss':
                    regression_kernel = int(post_cfg.get('path_loss_median_kernel', regression_kernel))
                arr_physical = apply_regression_heuristics(
                    arr_physical,
                    target_metadata.get(name, {}),
                    kernel_size=regression_kernel,
                )
                if name == 'path_loss' and bool(post_cfg.get('path_loss_los_correction', False)) and binary_los_np is not None:
                    dist_map = _compute_distance_map_2d(image_size).numpy()[0]
                    arr_physical = apply_path_loss_los_correction(
                        arr_physical,
                        binary_los_np,
                        dist_map,
                        frequency_ghz=float(post_cfg.get('path_loss_los_frequency_ghz', 7.125)),
                        blend_weight=float(post_cfg.get('path_loss_los_blend_weight', 0.3)),
                    )
            np.save(out_dir / f'{name}_physical.npy', arr_physical)
            physical_outputs[name] = arr_physical
        Image.fromarray(to_uint8(arr), mode='L').save(out_dir / f'{name}.png')

    if bool(post_cfg.get('export_augmented_los_heuristic', False)) and 'los_mask' in physical_outputs:
        augmented_outputs = derive_augmented_los_heuristic(
            physical_outputs['los_mask'],
            path_loss_db=physical_outputs.get('path_loss'),
            delay_spread_ns=physical_outputs.get('delay_spread'),
            angular_spread_deg=physical_outputs.get('angular_spread'),
            binary_los_input=binary_los_np,
            kernel_size=int(post_cfg.get('augmented_los_heuristic_median_kernel', 3)),
            threshold=float(post_cfg.get('augmented_los_heuristic_threshold', 0.5)),
            export_binary=bool(post_cfg.get('export_augmented_los_heuristic_binary', True)),
            weights=dict(post_cfg.get('augmented_los_heuristic_weights', {})),
        )
        np.save(out_dir / 'augmented_los_heuristic.npy', augmented_outputs['probabilities'])
        Image.fromarray(to_uint8(augmented_outputs['probabilities']), mode='L').save(out_dir / 'augmented_los_heuristic.png')
        if 'binary' in augmented_outputs:
            np.save(out_dir / 'augmented_los_heuristic_binary.npy', augmented_outputs['binary'])
            Image.fromarray((augmented_outputs['binary'] * 255.0).astype(np.uint8), mode='L').save(out_dir / 'augmented_los_heuristic_binary.png')

    link_budget_cfg = dict(post_cfg.get('link_budget', {}))
    if bool(post_cfg.get('export_derived_link_budget_maps', False)) and 'path_loss' in physical_outputs:
        channel_power_dbm = derive_channel_power_from_path_loss(
            physical_outputs['path_loss'],
            tx_power_dbm=float(link_budget_cfg.get('tx_power_dbm', 46.0)),
            tx_gain_dbi=float(link_budget_cfg.get('tx_gain_dbi', 0.0)),
            rx_gain_dbi=float(link_budget_cfg.get('rx_gain_dbi', 0.0)),
            other_losses_db=float(link_budget_cfg.get('other_losses_db', 0.0)),
        )
        np.save(out_dir / 'channel_power_derived_dbm.npy', channel_power_dbm)
        Image.fromarray(to_uint8(channel_power_dbm), mode='L').save(out_dir / 'channel_power_derived_dbm.png')

        bandwidth_hz = link_budget_cfg.get('bandwidth_hz')
        if bandwidth_hz is not None:
            snr_outputs = derive_snr_maps(
                channel_power_dbm,
                bandwidth_hz=float(bandwidth_hz),
                noise_figure_db=float(link_budget_cfg.get('noise_figure_db', 0.0)),
            )
            np.save(out_dir / 'noise_floor_derived_dbm.npy', snr_outputs['noise_floor_dbm'])
            np.save(out_dir / 'snr_derived_db.npy', snr_outputs['snr_db'])
            np.save(out_dir / 'snr_derived_linear.npy', snr_outputs['snr_linear'])
            Image.fromarray(to_uint8(snr_outputs['snr_db']), mode='L').save(out_dir / 'snr_derived_db.png')

        reception_threshold_dbm = link_budget_cfg.get('reception_threshold_dbm')
        if reception_threshold_dbm is not None:
            link_available = derive_link_availability(channel_power_dbm, float(reception_threshold_dbm))
            np.save(out_dir / 'link_available_binary.npy', link_available)
            Image.fromarray((link_available * 255.0).astype(np.uint8), mode='L').save(out_dir / 'link_available_binary.png')

    print(f'cGAN predictions saved to: {out_dir}')


if __name__ == '__main__':
    main()
