"""Try 76 — minimal dataset utilities.

Implements from scratch:
  - City-holdout sample split (bitwise identical to Try 75's
    ``_split_hdf5_samples`` — verified by ``tests/test_split_matches_try75.py``).
  - 12-expert routing (6 topology classes × {LoS, NLoS}).
  - Ground mask derivation (``topology_map == 0``).
  - Sinusoidal antenna-height embedding.

This module purposely does NOT import from any previous try.
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


# ---------------------------------------------------------------------------
# Topology classification (same thresholds as Try 54 / Try 67 "city_type_6")
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Split (reimplemented to match Try 75 behaviour exactly)
# ---------------------------------------------------------------------------

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
    """City-holdout split with the exact same semantics as Try 75.

    Algorithm: city_holdout → shuffle city names with Random(split_seed),
    greedily fill test bucket up to `test_ratio * N` samples while leaving
    at least 2 cities for {val, train}, then val up to `val_ratio * N` while
    leaving at least 1 city for train, then everything else to train.
    """
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

    max_held_out = max(total - 1, 0)
    if test_size + val_size > max_held_out:
        overflow = test_size + val_size - max_held_out
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
        # Fall back to random split; kept for parity with Try 75's edge case.
        rng.shuffle(refs)
        test_refs = refs[:test_size]
        val_refs = refs[test_size:test_size + val_size]
        train_refs = refs[test_size + val_size:]
        return train_refs, val_refs, test_refs

    rng.shuffle(city_names)
    test_refs: List[SampleRef] = []
    val_refs: List[SampleRef] = []
    train_refs: List[SampleRef] = []
    test_city_count = 0
    val_city_count = 0
    for city in city_names:
        city_refs = by_city[city]
        remaining = len(city_names) - test_city_count - val_city_count
        if len(test_refs) < test_size and remaining > 2:
            test_refs.extend(city_refs)
            test_city_count += 1
            continue
        if len(val_refs) < val_size and remaining > 1:
            val_refs.extend(city_refs)
            val_city_count += 1
            continue
        train_refs.extend(city_refs)
    if not train_refs:
        # Same safety net as Try 75.
        rng.shuffle(refs)
        test_refs = refs[:test_size]
        val_refs = refs[test_size:test_size + val_size]
        train_refs = refs[test_size + val_size:]
    return train_refs, val_refs, test_refs


# ---------------------------------------------------------------------------
# Sinusoidal height embedding
# ---------------------------------------------------------------------------

@dataclass
class HeightEmbedding:
    n_freq: int = 16
    min_height_m: float = 12.0
    max_height_m: float = 478.0

    def __call__(self, h_m: torch.Tensor) -> torch.Tensor:
        """h_m: (B,) tensor, returns (B, 2 * n_freq)."""
        log_min = math.log(max(self.min_height_m, 1e-3))
        log_max = math.log(max(self.max_height_m, self.min_height_m + 1e-3))
        freqs = torch.exp(torch.linspace(log_min, log_max, self.n_freq, device=h_m.device))
        angles = h_m.unsqueeze(-1) / freqs.unsqueeze(0)  # (B, n_freq)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Try76Config:
    hdf5_path: Path
    topology_class: str
    region_mode: str  # "los_only" or "nlos_only"
    image_size: int = 513
    topology_norm_m: float = 90.0
    path_loss_scale: float = 1.0  # path_loss is already in dB
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    path_loss_no_data_mask_column: Optional[str] = None
    derive_no_data_from_non_ground: bool = False


def _valid_target_mask(
    grp: h5py.Group,
    path_loss: np.ndarray,
    ground: np.ndarray,
    no_data_mask_column: Optional[str],
    derive_no_data_from_non_ground: bool,
) -> np.ndarray:
    # Path loss is strictly positive in valid pixels; zeros are used as no-data
    # placeholders in this dataset (especially visible in some NLoS slices).
    valid = np.isfinite(path_loss) & (path_loss > 0.0)
    if no_data_mask_column:
        key = str(no_data_mask_column).strip()
        if key and key in grp:
            no_data = np.asarray(grp[key][...], dtype=np.float32) > 0.5
            valid &= ~no_data
    if derive_no_data_from_non_ground:
        valid &= ground
    return valid


class Try76ExpertDataset(Dataset):
    """Loads (topology, los, nlos, ground_mask, h_tx) -> path_loss for one expert.

    The caller picks a sample subset (train/val/test, plus topology filter)
    and passes it to __init__. All HDF5 reads happen lazily per-item so we do
    not hold the file open across workers.
    """

    def __init__(
        self,
        cfg: Try76Config,
        sample_refs: Sequence[SampleRef],
    ) -> None:
        self.cfg = cfg
        self._refs = list(sample_refs)

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        city, sample = self._refs[idx]
        with h5py.File(str(self.cfg.hdf5_path), "r") as handle:
            grp = handle[city][sample]
            topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
            los_mask = np.asarray(grp["los_mask"][...], dtype=np.float32)
            path_loss = np.asarray(grp["path_loss"][...], dtype=np.float32)
            uav_height = float(np.asarray(grp["uav_height"][...]).reshape(-1)[0])
            ground_bool = topology == 0.0
            valid_target = _valid_target_mask(
                grp,
                path_loss,
                ground_bool,
                self.cfg.path_loss_no_data_mask_column,
                self.cfg.derive_no_data_from_non_ground,
            )

        ground = (topology == 0.0).astype(np.float32)
        topology_input = topology * ground / max(self.cfg.topology_norm_m, 1e-3)
        los = los_mask * ground
        nlos = (1.0 - los_mask) * ground

        if self.cfg.region_mode == "los_only":
            region = (los_mask > 0.5).astype(np.float32) * ground
        elif self.cfg.region_mode == "nlos_only":
            region = (los_mask <= 0.5).astype(np.float32) * ground
        else:
            region = ground
        loss_mask = region * valid_target.astype(np.float32)
        path_loss = np.where(valid_target, path_loss, 0.0).astype(np.float32)

        channels = np.stack([topology_input, los, nlos, ground], axis=0)  # (4, H, W)

        return {
            "city": city,
            "sample": sample,
            "inputs": torch.from_numpy(channels),
            "target": torch.from_numpy(path_loss).unsqueeze(0),
            "loss_mask": torch.from_numpy(loss_mask).unsqueeze(0),
            "ground_mask": torch.from_numpy(ground).unsqueeze(0),
            "antenna_height_m": torch.tensor(uav_height, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Helper: full pipeline
# ---------------------------------------------------------------------------

def build_expert_datasets(
    cfg: Try76Config,
    classify_cache: Optional[Dict[SampleRef, str]] = None,
) -> Tuple[Try76ExpertDataset, Try76ExpertDataset, Try76ExpertDataset]:
    """Enumerate HDF5 samples, split by city, filter by topology class, wrap in datasets."""
    refs = list_hdf5_samples(cfg.hdf5_path)
    train_refs, val_refs, test_refs = split_city_holdout(
        refs,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        split_seed=cfg.split_seed,
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

    def _filter(refs_: Sequence[SampleRef]) -> List[SampleRef]:
        filtered: List[SampleRef] = []
        for ref in refs_:
            if _classify(ref) != cfg.topology_class:
                continue
            city, sample = ref
            with h5py.File(str(cfg.hdf5_path), "r") as handle:
                grp = handle[city][sample]
                topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
                los_mask = np.asarray(grp["los_mask"][...], dtype=np.float32)
                path_loss = np.asarray(grp["path_loss"][...], dtype=np.float32)
            ground = topo == 0.0
            if cfg.region_mode == "los_only":
                region = (los_mask > 0.5) & ground
            elif cfg.region_mode == "nlos_only":
                region = (los_mask <= 0.5) & ground
            else:
                region = ground
            valid_target = _valid_target_mask(
                grp,
                path_loss,
                ground,
                cfg.path_loss_no_data_mask_column,
                cfg.derive_no_data_from_non_ground,
            )
            expert_mask = region & valid_target
            # Drop degenerate samples with no valid pixels for the selected expert.
            if int(expert_mask.sum()) == 0:
                continue
            filtered.append(ref)
        return filtered

    return (
        Try76ExpertDataset(cfg, _filter(train_refs)),
        Try76ExpertDataset(cfg, _filter(val_refs)),
        Try76ExpertDataset(cfg, _filter(test_refs)),
    )
