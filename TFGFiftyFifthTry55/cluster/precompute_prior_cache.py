#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Subset


TRY_DIR = Path(__file__).resolve().parent.parent
if str(TRY_DIR) not in sys.path:
    sys.path.insert(0, str(TRY_DIR))

import config_utils
import data_utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute shared HDF5 formula priors for Try 55.")
    p.add_argument("--config", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--limit", type=int, default=0, help="Optional max samples per split; 0 means all.")
    p.add_argument("--output-hdf5", required=True, help="Target HDF5 file for the shared compressed prior cache.")
    return p.parse_args()


def formula_channel_index(cfg: dict[str, Any]) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):
        idx += 1
    if cfg["data"].get("distance_map_channel", False):
        idx += 1
    return idx


def iter_split_entries(split_dataset: Any) -> Iterable[Tuple[int, str, str]]:
    if isinstance(split_dataset, Subset):
        base = split_dataset.dataset
        indices = list(split_dataset.indices)
        refs = getattr(base, "sample_refs", [])
        for local_idx, base_idx in enumerate(indices):
            city, sample = refs[int(base_idx)]
            yield local_idx, city, sample
        return
    refs = getattr(split_dataset, "sample_refs", [])
    for idx, (city, sample) in enumerate(refs):
        yield idx, city, sample


def ensure_writable_cache_file(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(output_path, "a"):
            return
    except Exception as exc:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup = output_path.with_name(f"{output_path.stem}.corrupt_{timestamp}{output_path.suffix}")
        output_path.rename(backup)
        print(
            f"[prior-hdf5] existing cache looked corrupted; preserved backup at {backup} ({type(exc).__name__}: {exc})",
            flush=True,
        )


def write_split(handle: h5py.File, dataset: Any, split_name: str, limit: int, channel_idx: int, started: float) -> None:
    entries = list(iter_split_entries(dataset))
    total = len(entries)
    limit = total if int(limit) <= 0 else min(int(limit), total)
    print(f"[prior-hdf5:{split_name}] {limit}/{total} samples", flush=True)
    written = 0
    skipped = 0
    for entry_idx, city, sample in entries[:limit]:
        group = handle.require_group(city).require_group(sample)
        if "formula_norm_u8" in group:
            skipped += 1
            continue
        item = dataset[entry_idx]
        model_input = item[0]
        prior = model_input[channel_idx : channel_idx + 1].detach().cpu().to(dtype=torch.float32)
        prior_u8 = np.clip(np.round(np.asarray(prior.squeeze(0)) * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)
        ds = group.create_dataset(
            "formula_norm_u8",
            data=prior_u8,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=prior_u8.shape,
        )
        ds.attrs["cache_dtype"] = "uint8"
        ds.attrs["split"] = split_name
        ds.attrs["normalized_range"] = "0_to_1"
        written += 1
        if written == 1 or (written + skipped) % 50 == 0 or written + skipped == limit:
            elapsed = time.time() - started
            print(
                f"[prior-hdf5:{split_name}] {written + skipped}/{limit} written={written} skipped={skipped} elapsed={elapsed/60.0:.1f} min",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    cfg = config_utils.load_config(args.config)
    config_utils.anchor_data_paths_to_config_file(cfg, args.config)
    cfg["augmentation"] = dict(cfg.get("augmentation", {}))
    cfg["augmentation"]["enable"] = False
    data_cfg = dict(cfg.get("data", {}))
    formula_cfg = dict(data_cfg.get("path_loss_formula_input", {}))
    formula_cfg["cache_enabled"] = False
    formula_cfg["cache_dir"] = ""
    formula_cfg["precomputed_hdf5"] = ""
    data_cfg["path_loss_formula_input"] = formula_cfg
    cfg["data"] = data_cfg

    splits = data_utils.build_dataset_splits_from_config(cfg)
    output_path = Path(args.output_hdf5)
    ensure_writable_cache_file(output_path)
    channel_idx = formula_channel_index(cfg)

    started = time.time()
    try:
        with h5py.File(output_path, "a") as handle:
            handle.attrs["format"] = "try55_formula_prior_cache"
            handle.attrs["cache_dtype"] = "uint8"
            handle.attrs["compression"] = "gzip"
            handle.attrs["compression_level"] = 4
            handle.attrs["image_size"] = int(cfg["data"]["image_size"])
            handle.attrs["normalized_range"] = "0_to_1"
            for split_name in args.splits:
                if split_name not in splits:
                    continue
                write_split(handle, splits[split_name], split_name, int(args.limit), channel_idx, started)
    except Exception as exc:
        if output_path.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup = output_path.with_name(f"{output_path.stem}.corrupt_runtime_{timestamp}{output_path.suffix}")
            output_path.rename(backup)
            print(
                f"[prior-hdf5] write failed; preserved corrupted cache at {backup} and retrying clean ({type(exc).__name__}: {exc})",
                flush=True,
            )
            with h5py.File(output_path, "w") as handle:
                handle.attrs["format"] = "try55_formula_prior_cache"
                handle.attrs["cache_dtype"] = "uint8"
                handle.attrs["compression"] = "gzip"
                handle.attrs["compression_level"] = 4
                handle.attrs["image_size"] = int(cfg["data"]["image_size"])
                handle.attrs["normalized_range"] = "0_to_1"
                for split_name in args.splits:
                    if split_name not in splits:
                        continue
                    write_split(handle, splits[split_name], split_name, int(args.limit), channel_idx, started)
        else:
            raise


if __name__ == "__main__":
    main()
