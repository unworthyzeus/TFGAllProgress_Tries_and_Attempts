"""Try 77 — dataset utilities for spread prediction (delay_spread, angular_spread).

Shares the 6-class topology partition + city-holdout split *semantics* with
Try 76 (re-implemented from scratch here as well — no imports across tries).

Experts: 12 = 6 topology classes × {delay_spread, angular_spread}. Each
expert picks one metric and masks by ``topology == 0`` (ground only).

Target ranges: delay_spread in nanoseconds (typically 0–400 ns), angular_spread
in degrees (typically 0–90 deg). Both are heavy-tailed non-negative — see
``docs/distribution_classes.md`` from Try 76 for evidence.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


TOPOLOGY_THRESHOLDS = {
    "density_q1": 0.12,
    "density_q2": 0.28,
    "height_q1": 12.0,
    "height_q2": 28.0,
}

TOPOLOGY_CLASSES: Tuple[str, ...] = (
    "open_sparse_lowrise",
    "open_sparse_vertical",
    "mixed_compact_lowrise",
    "mixed_compact_midrise",
    "dense_block_midrise",
    "dense_block_highrise",
)

METRIC_FIELDS: Dict[str, str] = {
    "delay_spread": "delay_spread",
    "angular_spread": "angular_spread",
}


def classify_topology(topo_m: np.ndarray, non_ground_threshold: float = 0.0) -> str:
    non_ground = topo_m != float(non_ground_threshold)
    density = float(np.mean(non_ground)) if non_ground.size else 0.0
    heights = topo_m[non_ground]
    mean_h = float(np.mean(heights)) if heights.size else 0.0
    d1 = TOPOLOGY_THRESHOLDS["density_q1"]
    d2 = TOPOLOGY_THRESHOLDS["density_q2"]
    h1 = TOPOLOGY_THRESHOLDS["height_q1"]
    h2 = TOPOLOGY_THRESHOLDS["height_q2"]
    if density <= d1:
        return "open_sparse_lowrise" if mean_h <= h1 else "open_sparse_vertical"
    if density >= d2:
        return "dense_block_midrise" if mean_h <= h2 else "dense_block_highrise"
    return "mixed_compact_lowrise" if mean_h <= h1 else "mixed_compact_midrise"


SampleRef = Tuple[str, str]


def list_hdf5_samples(hdf5_path: Path) -> List[SampleRef]:
    refs: List[SampleRef] = []
    with h5py.File(str(hdf5_path), "r") as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                refs.append((city, sample))
    return refs


def split_city_holdout(
    sample_refs: Sequence[SampleRef],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: int = 42,
) -> Tuple[List[SampleRef], List[SampleRef], List[SampleRef]]:
    """Identical to Try 75 / Try 76 city-holdout logic."""
    refs = list(sample_refs)
    if len(refs) < 2:
        return refs, list(refs), []
    val_ratio = max(0.0, float(val_ratio))
    test_ratio = max(0.0, float(test_ratio))
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")
    total = len(refs)
    test_size = int(round(total * test_ratio))
    val_size = int(round(total * val_ratio))
    if test_ratio > 0.0:
        test_size = max(test_size, 1)
    if val_ratio > 0.0:
        val_size = max(val_size, 1)
    max_held = max(total - 1, 0)
    if test_size + val_size > max_held:
        overflow = test_size + val_size - max_held
        reduce_test = min(overflow, max(test_size - (1 if test_ratio > 0.0 else 0), 0))
        test_size -= reduce_test
        overflow -= reduce_test
        reduce_val = min(overflow, max(val_size - (1 if val_ratio > 0.0 else 0), 0))
        val_size -= reduce_val
        overflow -= reduce_val
        if overflow > 0:
            raise ValueError("Not enough samples for the requested split ratios.")

    rng = random.Random(split_seed)
    by_city: Dict[str, List[SampleRef]] = {}
    for city, sample in refs:
        by_city.setdefault(city, []).append((city, sample))
    city_names = list(by_city.keys())
    if len(city_names) < 3:
        rng.shuffle(refs)
        test_refs = refs[:test_size]
        val_refs = refs[test_size:test_size + val_size]
        train_refs = refs[test_size + val_size:]
        return train_refs, val_refs, test_refs

    rng.shuffle(city_names)
    test_refs: List[SampleRef] = []
    val_refs: List[SampleRef] = []
    train_refs: List[SampleRef] = []
    t_cty = 0
    v_cty = 0
    for city in city_names:
        city_refs = by_city[city]
        remaining = len(city_names) - t_cty - v_cty
        if len(test_refs) < test_size and remaining > 2:
            test_refs.extend(city_refs)
            t_cty += 1
            continue
        if len(val_refs) < val_size and remaining > 1:
            val_refs.extend(city_refs)
            v_cty += 1
            continue
        train_refs.extend(city_refs)
    if not train_refs:
        rng.shuffle(refs)
        test_refs = refs[:test_size]
        val_refs = refs[test_size:test_size + val_size]
        train_refs = refs[test_size + val_size:]
    return train_refs, val_refs, test_refs


@dataclass
class HeightEmbedding:
    n_freq: int = 16
    min_height_m: float = 12.0
    max_height_m: float = 478.0

    def __call__(self, h_m: torch.Tensor) -> torch.Tensor:
        log_min = math.log(max(self.min_height_m, 1e-3))
        log_max = math.log(max(self.max_height_m, self.min_height_m + 1e-3))
        freqs = torch.exp(torch.linspace(log_min, log_max, self.n_freq, device=h_m.device))
        angles = h_m.unsqueeze(-1) / freqs.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


@dataclass
class Try77Config:
    hdf5_path: Path
    topology_class: str
    metric: str  # "delay_spread" or "angular_spread"
    image_size: int = 513
    topology_norm_m: float = 90.0
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42


def _read_field(grp, name: str) -> np.ndarray:
    """Lookup HDF5 field with fallbacks for naming inconsistencies."""
    if name in grp:
        return np.asarray(grp[name][...], dtype=np.float32)
    alts = [name, name + "_map", name.replace("_", "") + "_map"]
    for a in alts:
        if a in grp:
            return np.asarray(grp[a][...], dtype=np.float32)
    raise KeyError(f"No field matching {name!r} in {list(grp.keys())}")


class Try77ExpertDataset(Dataset):
    """Loads (topology, los, nlos, ground_mask, h_tx) -> spread map."""

    def __init__(self, cfg: Try77Config, sample_refs: Sequence[SampleRef], augment: bool = False) -> None:
        self.cfg = cfg
        self._refs = list(sample_refs)
        self.augment = bool(augment)
        if cfg.metric not in METRIC_FIELDS:
            raise ValueError(f"Unknown metric {cfg.metric!r}; expected one of {list(METRIC_FIELDS)}")

    def __len__(self) -> int:
        return len(self._refs)

    @staticmethod
    def _apply_aug(arrays: List[np.ndarray]) -> List[np.ndarray]:
        k = random.randint(0, 3)
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.5
        out: List[np.ndarray] = []
        for a in arrays:
            b = np.rot90(a, k=k, axes=(-2, -1)) if k else a
            if flip_h:
                b = b[..., :, ::-1]
            if flip_v:
                b = b[..., ::-1, :]
            out.append(np.ascontiguousarray(b))
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        city, sample = self._refs[idx]
        with h5py.File(str(self.cfg.hdf5_path), "r") as handle:
            grp = handle[city][sample]
            topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
            los_mask = np.asarray(grp["los_mask"][...], dtype=np.float32)
            target = _read_field(grp, METRIC_FIELDS[self.cfg.metric])
            uav_height = float(np.asarray(grp["uav_height"][...]).reshape(-1)[0])

        ground = (topology == 0.0).astype(np.float32)
        topology_input = topology * ground / max(self.cfg.topology_norm_m, 1e-3)
        los = los_mask * ground
        nlos = (1.0 - los_mask) * ground

        # Targets are non-negative. NaNs / sub-zeros become "no-data" (masked).
        valid_target = np.isfinite(target) & (target >= 0.0)
        loss_mask = ground * valid_target.astype(np.float32)
        target = np.where(valid_target, target, 0.0).astype(np.float32)

        channels = np.stack([topology_input, los, nlos, ground], axis=0)

        if self.augment:
            channels, target, loss_mask, ground = self._apply_aug(
                [channels, target, loss_mask, ground]
            )

        return {
            "city": city,
            "sample": sample,
            "inputs": torch.from_numpy(channels),
            "target": torch.from_numpy(target).unsqueeze(0),
            "loss_mask": torch.from_numpy(loss_mask).unsqueeze(0),
            "ground_mask": torch.from_numpy(ground).unsqueeze(0),
            "antenna_height_m": torch.tensor(uav_height, dtype=torch.float32),
        }


def build_expert_datasets(
    cfg: Try77Config,
    classify_cache: Optional[Dict[SampleRef, str]] = None,
) -> Tuple[Try77ExpertDataset, Try77ExpertDataset, Try77ExpertDataset]:
    refs = list_hdf5_samples(cfg.hdf5_path)
    train_refs, val_refs, test_refs = split_city_holdout(
        refs, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, split_seed=cfg.split_seed
    )
    cache: Dict[SampleRef, str] = dict(classify_cache or {})

    def _classify(ref: SampleRef) -> str:
        if ref in cache:
            return cache[ref]
        city, sample = ref
        with h5py.File(str(cfg.hdf5_path), "r") as handle:
            topo = np.asarray(handle[city][sample]["topology_map"][...], dtype=np.float32)
        label = classify_topology(topo)
        cache[ref] = label
        return label

    def _filter(refs_: Sequence[SampleRef], *, drop_small_valid: bool) -> List[SampleRef]:
        filtered: List[SampleRef] = []
        for ref in refs_:
            if _classify(ref) != cfg.topology_class:
                continue
            if not drop_small_valid:
                filtered.append(ref)
                continue
            city, sample = ref
            with h5py.File(str(cfg.hdf5_path), "r") as handle:
                grp = handle[city][sample]
                topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
                target = _read_field(grp, METRIC_FIELDS[cfg.metric])
            ground = topo == 0.0
            valid_target = np.isfinite(target) & (target >= 0.0)
            # Drop degenerate samples: <64 valid pixels risks GroupNorm numerics
            # (all-zero inputs) and provides negligible training signal.
            if int((ground & valid_target).sum()) < 64:
                continue
            filtered.append(ref)
        return filtered

    return (
        Try77ExpertDataset(cfg, _filter(train_refs, drop_small_valid=True), augment=True),
        Try77ExpertDataset(cfg, _filter(val_refs, drop_small_valid=False), augment=False),
        Try77ExpertDataset(cfg, _filter(test_refs, drop_small_valid=False), augment=False),
    )
