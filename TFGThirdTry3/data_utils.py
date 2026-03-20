from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _resolve_path(root_dir: Path, rel_path: str) -> Path:
    return root_dir / rel_path.replace('\\', '/').replace('\r', '').replace('\n', '')


def _normalize_array(arr: np.ndarray, metadata: Optional[Dict[str, Any]]) -> np.ndarray:
    if metadata is None:
        return arr.astype(np.float32, copy=False)

    scale = float(metadata.get('scale', 1.0))
    offset = float(metadata.get('offset', 0.0))
    if abs(scale) < 1e-12:
        scale = 1.0
    return ((arr.astype(np.float32, copy=False) - offset) / scale).astype(np.float32, copy=False)


def _resize_array(arr: np.ndarray, image_size: int, metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    normalized = _normalize_array(np.asarray(arr), metadata)
    if normalized.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {normalized.shape}")
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    return TF.resize(
        tensor,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def _path_loss_db_to_linear_normalized(arr: np.ndarray, image_size: int) -> torch.Tensor:
    """Convert path loss dB to linear scale, normalize to [0,1] via log scale."""
    arr = np.asarray(arr, dtype=np.float32)
    linear = np.power(10.0, -arr / 10.0)
    linear = np.clip(linear, 1e-18, 1.0)
    log_linear = np.log10(linear)
    normalized = (log_linear + 18.0) / 18.0
    tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
    return TF.resize(
        tensor,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def path_loss_linear_normalized_to_db(tensor: torch.Tensor) -> torch.Tensor:
    """Convert normalized linear path loss back to dB."""
    # Metrics and confidence targets must avoid fp16 underflow near 1e-18.
    working = tensor.to(dtype=torch.float32)
    normalized = working.clamp(0.0, 1.0)
    log_linear = normalized * 18.0 - 18.0
    linear = torch.pow(10.0, log_linear).clamp(min=1e-18)
    return -10.0 * torch.log10(linear)


def _compute_scalar_norms(
    scalar_feature_columns: Sequence[str],
    constant_scalar_features: Dict[str, float],
    scalar_feature_norms: Dict[str, float],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    for col in scalar_feature_columns:
        if col in scalar_feature_norms:
            norms[col] = max(float(scalar_feature_norms[col]), 1.0)
        elif df is not None and col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            max_abs = float(values.abs().max()) if not values.empty else 1.0
            norms[col] = max(max_abs, 1.0)
        else:
            norms[col] = 1.0

    for col, value in constant_scalar_features.items():
        if col in scalar_feature_norms:
            norms[col] = max(float(scalar_feature_norms[col]), 1.0)
        else:
            norms[col] = max(abs(float(value)), 1.0)

    return norms


def _compute_distance_map_2d(image_size: int) -> torch.Tensor:
    """2D horizontal distance from map center (antenna at center). Normalized to [0, 1]."""
    half = (image_size - 1) / 2.0
    y = torch.arange(image_size, dtype=torch.float32) - half
    x = torch.arange(image_size, dtype=torch.float32) - half
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt(xx ** 2 + yy ** 2)
    max_dist = 256.0 * (2.0 ** 0.5)
    normalized = (dist / max_dist).clamp(0.0, 1.0)
    return normalized.unsqueeze(0)


def compute_input_channels(cfg: Dict[str, Any]) -> int:
    in_channels = 1
    if cfg['data'].get('los_input_column'):
        in_channels += 1
    if cfg['data'].get('distance_map_channel', False):
        in_channels += 1
    if bool(cfg['model']['use_scalar_channels']):
        in_channels += len(list(cfg['data'].get('scalar_feature_columns', [])))
        in_channels += len(dict(cfg['data'].get('constant_scalar_features', {})))
    return in_channels


class CKMDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        root_dir: str,
        target_columns: List[str],
        image_size: int = 128,
        augment: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.4,
        add_scalar_channels: bool = True,
        scalar_feature_columns: Optional[List[str]] = None,
        constant_scalar_features: Optional[Dict[str, float]] = None,
        scalar_feature_norms: Optional[Dict[str, float]] = None,
        los_input_column: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(manifest_csv)
        self.target_columns = target_columns
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.add_scalar_channels = add_scalar_channels
        self.scalar_feature_columns = scalar_feature_columns or []
        self.constant_scalar_features = constant_scalar_features or {}
        self.scalar_feature_norms_cfg = scalar_feature_norms or {}
        self.los_input_column = los_input_column
        self.target_availability: Dict[str, Dict[str, int]] = {}

        self.scalar_norms = _compute_scalar_norms(
            self.scalar_feature_columns,
            self.constant_scalar_features,
            self.scalar_feature_norms_cfg,
            df=self.df,
        )

        for col in self.target_columns:
            if col not in self.df.columns:
                self.target_availability[col] = {'present_rows': 0, 'total_rows': len(self.df)}
                print(f"[WARNING] Target column '{col}' is missing from manifest {manifest_csv}.")
                continue

            present_rows = int(self.df[col].notna().sum())
            self.target_availability[col] = {'present_rows': present_rows, 'total_rows': len(self.df)}
            if present_rows == 0:
                print(f"[WARNING] Target column '{col}' exists but has 0 labeled rows in {manifest_csv}.")
            elif present_rows < len(self.df):
                print(
                    f"[INFO] Target column '{col}' available for {present_rows}/{len(self.df)} rows in {manifest_csv}."
                )

    def __len__(self) -> int:
        return len(self.df)

    def _apply_sync_aug(self, images: List[Image.Image]) -> List[Image.Image]:
        if random.random() < self.hflip_prob:
            images = [TF.hflip(img) for img in images]
        if random.random() < self.vflip_prob:
            images = [TF.vflip(img) for img in images]
        if random.random() < self.rot90_prob:
            angle = random.choice([90, 180, 270])
            images = [TF.rotate(img, angle) for img in images]
        return images

    def _load_gray_or_none(self, rel_path: Optional[str]) -> Optional[Image.Image]:
        if rel_path is None or (isinstance(rel_path, float) and pd.isna(rel_path)):
            return None
        path = _resolve_path(self.root_dir, str(rel_path))
        if not path.exists():
            return None
        try:
            return Image.open(path).convert('L').resize((self.image_size, self.image_size), Image.BILINEAR)
        except (OSError, FileNotFoundError):
            return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        input_img = self._load_gray_or_none(row['input_path'])
        if input_img is None:
            raise FileNotFoundError(f"Missing input image at row {idx}: {row['input_path']}")

        los_input_img = None
        if self.los_input_column and self.los_input_column in self.df.columns:
            los_input_img = self._load_gray_or_none(row[self.los_input_column])

        target_imgs: List[Optional[Image.Image]] = [self._load_gray_or_none(row.get(col, None)) for col in self.target_columns]

        if self.augment:
            stack = [input_img]
            if los_input_img is not None:
                stack.append(los_input_img)
            stack.extend([img for img in target_imgs if img is not None])
            aug = self._apply_sync_aug(stack)
            input_img = aug[0]
            cursor = 1
            if los_input_img is not None:
                los_input_img = aug[cursor]
                cursor += 1
            rebuilt = []
            for img in target_imgs:
                if img is None:
                    rebuilt.append(None)
                else:
                    rebuilt.append(aug[cursor])
                    cursor += 1
            target_imgs = rebuilt

        model_input_channels = [TF.to_tensor(input_img)]
        if los_input_img is not None:
            model_input_channels.append(TF.to_tensor(los_input_img))

        if self.add_scalar_channels:
            scalar_values = []
            for col in self.scalar_feature_columns:
                raw_value = pd.to_numeric(row.get(col, 0.0), errors='coerce')
                value = 0.0 if pd.isna(raw_value) else float(raw_value)
                norm = self.scalar_norms.get(col, 1.0)
                scalar_values.append(value / norm)

            for col, value in self.constant_scalar_features.items():
                norm = self.scalar_norms.get(col, 1.0)
                scalar_values.append(float(value) / norm)

            if scalar_values:
                h, w = model_input_channels[0].shape[1:]
                scalar_tensor = torch.tensor(scalar_values, dtype=torch.float32).view(len(scalar_values), 1, 1).expand(len(scalar_values), h, w)
                model_input_channels.append(scalar_tensor)

        model_input = torch.cat(model_input_channels, dim=0)

        target_tensors = []
        mask_tensors = []
        for img in target_imgs:
            if img is None:
                target_tensors.append(torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32))
                mask_tensors.append(torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32))
            else:
                target_tensors.append(TF.to_tensor(img))
                mask_tensors.append(torch.ones((1, self.image_size, self.image_size), dtype=torch.float32))

        target_tensor = torch.cat(target_tensors, dim=0)
        mask_tensor = torch.cat(mask_tensors, dim=0)

        return model_input, target_tensor, mask_tensor


class CKMHDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        sample_refs: Sequence[Tuple[str, str]],
        target_columns: List[str],
        image_size: int = 128,
        augment: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.4,
        add_scalar_channels: bool = True,
        scalar_feature_columns: Optional[List[str]] = None,
        constant_scalar_features: Optional[Dict[str, float]] = None,
        scalar_feature_norms: Optional[Dict[str, float]] = None,
        los_input_column: Optional[str] = None,
        input_column: str = 'topology_map',
        input_metadata: Optional[Dict[str, Any]] = None,
        target_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        target_field_map: Optional[Dict[str, str]] = None,
        distance_map_channel: bool = False,
        path_loss_saturation_db: Optional[float] = None,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.sample_refs = list(sample_refs)
        self.target_columns = target_columns
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.add_scalar_channels = add_scalar_channels
        self.scalar_feature_columns = scalar_feature_columns or []
        self.constant_scalar_features = constant_scalar_features or {}
        self.scalar_feature_norms_cfg = scalar_feature_norms or {}
        self.los_input_column = los_input_column
        self.input_column = input_column
        self.input_metadata = input_metadata or {}
        self.target_metadata = target_metadata or {}
        self.target_field_map = target_field_map or {}
        self.distance_map_channel = distance_map_channel
        self.path_loss_saturation_db = path_loss_saturation_db
        self.scalar_norms = _compute_scalar_norms(
            self.scalar_feature_columns,
            self.constant_scalar_features,
            self.scalar_feature_norms_cfg,
        )
        self._handle: Optional[h5py.File] = None

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        if self.scalar_feature_columns:
            print(
                "[INFO] HDF5 dataset has no per-sample scalar columns; requested scalar_feature_columns will be filled with 0.0."
            )

    def __len__(self) -> int:
        return len(self.sample_refs)

    def _get_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_path, 'r')
        return self._handle

    def _read_field(self, city: str, sample: str, field_name: str, metadata: Optional[Dict[str, Any]]) -> torch.Tensor:
        handle = self._get_handle()
        if field_name not in handle[city][sample]:
            raise KeyError(f"Field '{field_name}' not found in {city}/{sample}")
        return _resize_array(handle[city][sample][field_name][...], self.image_size, metadata)

    def _apply_sync_aug(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        if random.random() < self.hflip_prob:
            images = [TF.hflip(img) for img in images]
        if random.random() < self.vflip_prob:
            images = [TF.vflip(img) for img in images]
        if random.random() < self.rot90_prob:
            angle = random.choice([90, 180, 270])
            images = [TF.rotate(img, angle) for img in images]
        return images

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        city, sample = self.sample_refs[idx]
        input_tensor = self._read_field(city, sample, self.input_column, self.input_metadata)

        los_input_tensor = None
        if self.los_input_column:
            los_metadata = self.target_metadata.get(self.los_input_column, {})
            los_input_tensor = self._read_field(city, sample, self.los_input_column, los_metadata)

        target_tensors = []
        raw_path_loss_tensor: Optional[torch.Tensor] = None
        for col in self.target_columns:
            field_name = self.target_field_map.get(col, col)
            meta = self.target_metadata.get(col, {})
            if col == 'path_loss' and meta.get('predict_linear', False):
                handle = self._get_handle()
                raw = np.asarray(handle[city][sample][field_name][...], dtype=np.float32)
                target_tensors.append(_path_loss_db_to_linear_normalized(raw, self.image_size))
            else:
                target_tensors.append(self._read_field(city, sample, field_name, meta))
            if col == 'path_loss' and self.path_loss_saturation_db is not None:
                handle = self._get_handle()
                raw = np.asarray(handle[city][sample][field_name][...], dtype=np.float32)
                raw_path_loss_tensor = _resize_array(raw, self.image_size, None)

        distance_map_tensor = None
        if self.distance_map_channel:
            distance_map_tensor = _compute_distance_map_2d(self.image_size)

        if self.augment:
            stack = [input_tensor]
            if los_input_tensor is not None:
                stack.append(los_input_tensor)
            if distance_map_tensor is not None:
                stack.append(distance_map_tensor)
            if raw_path_loss_tensor is not None:
                stack.append(raw_path_loss_tensor)
            stack.extend(target_tensors)
            aug = self._apply_sync_aug(stack)
            input_tensor = aug[0]
            cursor = 1
            if los_input_tensor is not None:
                los_input_tensor = aug[cursor]
                cursor += 1
            if distance_map_tensor is not None:
                distance_map_tensor = aug[cursor]
                cursor += 1
            if raw_path_loss_tensor is not None:
                raw_path_loss_tensor = aug[cursor]
                cursor += 1
            target_tensors = aug[cursor:]

        model_input_channels = [input_tensor]
        if los_input_tensor is not None:
            model_input_channels.append(los_input_tensor)
        if distance_map_tensor is not None:
            model_input_channels.append(distance_map_tensor)

        if self.add_scalar_channels:
            scalar_values = []
            for col in self.scalar_feature_columns:
                norm = self.scalar_norms.get(col, 1.0)
                scalar_values.append(0.0 / norm)
            for col, value in self.constant_scalar_features.items():
                norm = self.scalar_norms.get(col, 1.0)
                scalar_values.append(float(value) / norm)

            if scalar_values:
                h, w = model_input_channels[0].shape[1:]
                scalar_tensor = torch.tensor(scalar_values, dtype=torch.float32).view(len(scalar_values), 1, 1).expand(len(scalar_values), h, w)
                model_input_channels.append(scalar_tensor)

        model_input = torch.cat(model_input_channels, dim=0)
        target_tensor = torch.cat(target_tensors, dim=0)
        mask_tensor = torch.ones_like(target_tensor, dtype=torch.float32)
        if raw_path_loss_tensor is not None and self.path_loss_saturation_db is not None:
            path_loss_idx = self.target_columns.index('path_loss')
            saturated = (raw_path_loss_tensor >= float(self.path_loss_saturation_db)).squeeze(0)
            mask_tensor[path_loss_idx] = (~saturated).float()
        return model_input, target_tensor, mask_tensor


def _list_hdf5_samples(hdf5_path: str) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    with h5py.File(hdf5_path, 'r') as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                refs.append((city, sample))
    return refs


def _split_hdf5_samples(
    sample_refs: Sequence[Tuple[str, str]],
    val_ratio: float,
    split_seed: int,
    test_ratio: float = 0.0,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    refs = list(sample_refs)
    if len(refs) < 2:
        return refs, refs, []

    rng = random.Random(split_seed)
    rng.shuffle(refs)
    total = len(refs)

    test_ratio = max(0.0, float(test_ratio))
    val_ratio = max(0.0, float(val_ratio))
    if val_ratio + test_ratio >= 1.0:
        raise ValueError('data.val_ratio + data.test_ratio must be < 1.0')

    test_size = int(round(total * test_ratio))
    val_size = int(round(total * val_ratio))

    if test_ratio > 0.0:
        test_size = max(test_size, 1)
    if val_ratio > 0.0:
        val_size = max(val_size, 1)

    max_held_out = max(total - 1, 0)
    if test_size + val_size > max_held_out:
        overflow = test_size + val_size - max_held_out
        reducible_test = max(test_size - (1 if test_ratio > 0.0 else 0), 0)
        reduce_test = min(overflow, reducible_test)
        test_size -= reduce_test
        overflow -= reduce_test
        reducible_val = max(val_size - (1 if val_ratio > 0.0 else 0), 0)
        reduce_val = min(overflow, reducible_val)
        val_size -= reduce_val
        overflow -= reduce_val
        if overflow > 0:
            raise ValueError('Not enough samples to create train/val/test split with the requested ratios.')

    test_refs = refs[:test_size]
    val_start = test_size
    val_end = val_start + val_size
    val_refs = refs[val_start:val_end]
    train_refs = refs[val_end:]
    return train_refs, val_refs, test_refs


def build_dataset_splits_from_config(cfg: Dict[str, Any]) -> Dict[str, Dataset]:
    data_cfg = cfg['data']
    target_columns = list(cfg['target_columns'])
    dataset_format = str(data_cfg.get('format', 'manifest')).lower()

    if dataset_format == 'manifest':
        common = dict(
            root_dir=data_cfg['root_dir'],
            target_columns=target_columns,
            image_size=int(data_cfg['image_size']),
            add_scalar_channels=bool(cfg['model']['use_scalar_channels']),
            scalar_feature_columns=list(data_cfg.get('scalar_feature_columns', [])),
            constant_scalar_features=dict(data_cfg.get('constant_scalar_features', {})),
            scalar_feature_norms=dict(data_cfg.get('scalar_feature_norms', {})),
            los_input_column=data_cfg.get('los_input_column'),
        )
        splits: Dict[str, Dataset] = {
            'train': CKMDataset(
                manifest_csv=data_cfg['train_manifest'],
                augment=bool(cfg['augmentation']['enable']),
                hflip_prob=float(cfg['augmentation']['hflip_prob']),
                vflip_prob=float(cfg['augmentation']['vflip_prob']),
                rot90_prob=float(cfg['augmentation']['rot90_prob']),
                **common,
            ),
            'val': CKMDataset(
                manifest_csv=data_cfg['val_manifest'],
                augment=False,
                **common,
            ),
        }
        test_manifest = data_cfg.get('test_manifest')
        if test_manifest:
            splits['test'] = CKMDataset(
                manifest_csv=test_manifest,
                augment=False,
                **common,
            )
        return splits

    if dataset_format != 'hdf5':
        raise ValueError(f"Unsupported data.format '{dataset_format}'. Expected 'manifest' or 'hdf5'.")

    sample_refs = _list_hdf5_samples(data_cfg['hdf5_path'])
    train_refs, val_refs, test_refs = _split_hdf5_samples(
        sample_refs,
        float(data_cfg.get('val_ratio', 0.1)),
        int(data_cfg.get('split_seed', cfg.get('seed', 42))),
        float(data_cfg.get('test_ratio', 0.0)),
    )
    common_hdf5 = dict(
        hdf5_path=data_cfg['hdf5_path'],
        target_columns=target_columns,
        image_size=int(data_cfg['image_size']),
        add_scalar_channels=bool(cfg['model']['use_scalar_channels']),
        scalar_feature_columns=list(data_cfg.get('scalar_feature_columns', [])),
        constant_scalar_features=dict(data_cfg.get('constant_scalar_features', {})),
        scalar_feature_norms=dict(data_cfg.get('scalar_feature_norms', {})),
        los_input_column=data_cfg.get('los_input_column'),
        input_column=str(data_cfg.get('input_column', 'topology_map')),
        input_metadata=dict(data_cfg.get('input_metadata', {})),
        target_metadata=dict(cfg.get('target_metadata', {})),
        target_field_map=dict(data_cfg.get('target_field_map', {})),
        distance_map_channel=bool(data_cfg.get('distance_map_channel', False)),
        path_loss_saturation_db=data_cfg.get('path_loss_saturation_db'),
    )
    splits = {
        'train': CKMHDF5Dataset(
            sample_refs=train_refs,
            augment=bool(cfg['augmentation']['enable']),
            hflip_prob=float(cfg['augmentation']['hflip_prob']),
            vflip_prob=float(cfg['augmentation']['vflip_prob']),
            rot90_prob=float(cfg['augmentation']['rot90_prob']),
            **common_hdf5,
        ),
        'val': CKMHDF5Dataset(
            sample_refs=val_refs,
            augment=False,
            **common_hdf5,
        ),
    }
    if test_refs:
        splits['test'] = CKMHDF5Dataset(
            sample_refs=test_refs,
            augment=False,
            **common_hdf5,
        )
    return splits


def merge_hdf5_splits_for_inference(cfg: Dict[str, Any]) -> 'CKMHDF5Dataset':
    """Train + val + test in one dataset, augment off (HDF5 only)."""
    data_cfg = cfg['data']
    if str(data_cfg.get('format', 'hdf5')).lower() != 'hdf5':
        raise ValueError("merge_hdf5_splits_for_inference requires data.format: hdf5")
    splits = build_dataset_splits_from_config(cfg)
    train_ds = splits['train']
    if not isinstance(train_ds, CKMHDF5Dataset):
        raise TypeError('merge_hdf5_splits_for_inference expected CKMHDF5Dataset splits.')
    refs: List[Tuple[str, str]] = []
    refs.extend(train_ds.sample_refs)
    refs.extend(splits['val'].sample_refs)
    if 'test' in splits:
        refs.extend(splits['test'].sample_refs)
    return CKMHDF5Dataset(
        hdf5_path=str(data_cfg['hdf5_path']),
        sample_refs=refs,
        target_columns=list(cfg['target_columns']),
        image_size=int(data_cfg['image_size']),
        augment=False,
        hflip_prob=0.0,
        vflip_prob=0.0,
        rot90_prob=0.0,
        add_scalar_channels=bool(cfg['model']['use_scalar_channels']),
        scalar_feature_columns=list(data_cfg.get('scalar_feature_columns', [])),
        constant_scalar_features=dict(data_cfg.get('constant_scalar_features', {})),
        scalar_feature_norms=dict(data_cfg.get('scalar_feature_norms', {})),
        los_input_column=data_cfg.get('los_input_column'),
        input_column=str(data_cfg.get('input_column', 'topology_map')),
        input_metadata=dict(data_cfg.get('input_metadata', {})),
        target_metadata=dict(cfg.get('target_metadata', {})),
        target_field_map=dict(data_cfg.get('target_field_map', {})),
        distance_map_channel=bool(data_cfg.get('distance_map_channel', False)),
        path_loss_saturation_db=data_cfg.get('path_loss_saturation_db'),
    )


def build_datasets_from_config(cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    splits = build_dataset_splits_from_config(cfg)
    return splits['train'], splits['val']


def build_cross_validation_datasets_from_config(cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    data_cfg = cfg['data']
    target_columns = list(cfg['target_columns'])
    dataset_format = str(data_cfg.get('format', 'manifest')).lower()

    if dataset_format == 'manifest':
        common = dict(
            root_dir=data_cfg['root_dir'],
            target_columns=target_columns,
            image_size=int(data_cfg['image_size']),
            add_scalar_channels=bool(cfg['model']['use_scalar_channels']),
            scalar_feature_columns=list(data_cfg.get('scalar_feature_columns', [])),
            constant_scalar_features=dict(data_cfg.get('constant_scalar_features', {})),
            scalar_feature_norms=dict(data_cfg.get('scalar_feature_norms', {})),
            los_input_column=data_cfg.get('los_input_column'),
        )
        train_manifest = data_cfg['train_manifest']
        val_manifest = data_cfg['val_manifest']
        dev_train = torch.utils.data.ConcatDataset(
            [
                CKMDataset(
                    manifest_csv=train_manifest,
                    augment=bool(cfg['augmentation']['enable']),
                    hflip_prob=float(cfg['augmentation']['hflip_prob']),
                    vflip_prob=float(cfg['augmentation']['vflip_prob']),
                    rot90_prob=float(cfg['augmentation']['rot90_prob']),
                    **common,
                ),
                CKMDataset(
                    manifest_csv=val_manifest,
                    augment=bool(cfg['augmentation']['enable']),
                    hflip_prob=float(cfg['augmentation']['hflip_prob']),
                    vflip_prob=float(cfg['augmentation']['vflip_prob']),
                    rot90_prob=float(cfg['augmentation']['rot90_prob']),
                    **common,
                ),
            ]
        )
        dev_eval = torch.utils.data.ConcatDataset(
            [
                CKMDataset(manifest_csv=train_manifest, augment=False, **common),
                CKMDataset(manifest_csv=val_manifest, augment=False, **common),
            ]
        )
        test_manifest = data_cfg.get('test_manifest')
        test_dataset: Optional[Dataset] = None
        if test_manifest:
            test_dataset = CKMDataset(manifest_csv=test_manifest, augment=False, **common)
        return dev_train, dev_eval, test_dataset

    if dataset_format != 'hdf5':
        raise ValueError(f"Unsupported data.format '{dataset_format}'. Expected 'manifest' or 'hdf5'.")

    splits = build_dataset_splits_from_config(cfg)
    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits.get('test')
    if not isinstance(train_dataset, CKMHDF5Dataset) or not isinstance(val_dataset, CKMHDF5Dataset):
        raise TypeError('Expected CKMHDF5Dataset instances for HDF5 cross-validation.')

    dev_refs = list(train_dataset.sample_refs) + list(val_dataset.sample_refs)
    common_hdf5 = dict(
        hdf5_path=data_cfg['hdf5_path'],
        target_columns=target_columns,
        image_size=int(data_cfg['image_size']),
        add_scalar_channels=bool(cfg['model']['use_scalar_channels']),
        scalar_feature_columns=list(data_cfg.get('scalar_feature_columns', [])),
        constant_scalar_features=dict(data_cfg.get('constant_scalar_features', {})),
        scalar_feature_norms=dict(data_cfg.get('scalar_feature_norms', {})),
        los_input_column=data_cfg.get('los_input_column'),
        input_column=str(data_cfg.get('input_column', 'topology_map')),
        input_metadata=dict(data_cfg.get('input_metadata', {})),
        target_metadata=dict(cfg.get('target_metadata', {})),
        target_field_map=dict(data_cfg.get('target_field_map', {})),
        distance_map_channel=bool(data_cfg.get('distance_map_channel', False)),
        path_loss_saturation_db=data_cfg.get('path_loss_saturation_db'),
    )
    dev_train = CKMHDF5Dataset(
        sample_refs=dev_refs,
        augment=bool(cfg['augmentation']['enable']),
        hflip_prob=float(cfg['augmentation']['hflip_prob']),
        vflip_prob=float(cfg['augmentation']['vflip_prob']),
        rot90_prob=float(cfg['augmentation']['rot90_prob']),
        **common_hdf5,
    )
    dev_eval = CKMHDF5Dataset(
        sample_refs=dev_refs,
        augment=False,
        **common_hdf5,
    )
    return dev_train, dev_eval, test_dataset
