from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def _resolve_path(root_dir: Path, rel_path: str) -> Path:
    return root_dir / rel_path.replace('\\', '/').replace('\r', '').replace('\n', '')


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

        self.scalar_norms: Dict[str, float] = {}
        for col in self.scalar_feature_columns:
            if col in self.scalar_feature_norms_cfg:
                self.scalar_norms[col] = max(float(self.scalar_feature_norms_cfg[col]), 1.0)
            elif col in self.df.columns:
                values = pd.to_numeric(self.df[col], errors='coerce').dropna()
                max_abs = float(values.abs().max()) if not values.empty else 1.0
                self.scalar_norms[col] = max(max_abs, 1.0)
            else:
                self.scalar_norms[col] = 1.0

        for col, value in self.constant_scalar_features.items():
            if col in self.scalar_feature_norms_cfg:
                self.scalar_norms[col] = max(float(self.scalar_feature_norms_cfg[col]), 1.0)
            else:
                self.scalar_norms[col] = max(abs(float(value)), 1.0)

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
