#!/usr/bin/env python3
"""
Export CKM HDF5 samples into per-field PNGs for inspection.

This is meant for debugging/metadata inspection (e.g., checking if antenna height is encoded
somewhere outside the HDF5). It does NOT change the dataset; it just writes images.

Example:
  python export_hdf5_to_images.py --hdf5 c:\\TFG\\TFGpractice\\Datasets\\CKM_Dataset_180326.h5 --out c:\\TFG\\TFGpractice\\Datasets\\New --limit 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np
from PIL import Image


FIELDS: Tuple[str, ...] = (
    "topology_map",
    "los_mask",
    "path_loss",
    "delay_spread",
    "angular_spread",
)


def iter_refs(hdf5_path: Path) -> Iterable[Tuple[str, str]]:
    with h5py.File(hdf5_path, "r") as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                yield city, sample


def normalize_to_uint8(field: str, arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if field == "los_mask":
        return (a > 0.5).astype(np.uint8) * 255
    if a.dtype == np.uint8 and field in {"path_loss", "angular_spread"}:
        return a
    if a.dtype == np.uint16 and field == "delay_spread":
        # Stretch to 8-bit for viewing.
        maxv = float(np.max(a)) if a.size else 0.0
        if maxv <= 0:
            return np.zeros_like(a, dtype=np.uint8)
        return (a.astype(np.float32) / maxv * 255.0).clip(0, 255).astype(np.uint8)
    # float32 topology_map
    af = a.astype(np.float32)
    mn = float(np.min(af)) if af.size else 0.0
    mx = float(np.max(af)) if af.size else 1.0
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(af, dtype=np.uint8)
    return ((af - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Export HDF5 samples to PNGs.")
    p.add_argument("--hdf5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=20, help="Max number of samples to export (across all cities).")
    args = p.parse_args()

    hdf5_path = Path(args.hdf5)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with h5py.File(hdf5_path, "r") as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                group = handle[city][sample]
                sample_out = out_dir / city / sample
                sample_out.mkdir(parents=True, exist_ok=True)
                for field in FIELDS:
                    if field not in group:
                        continue
                    arr = np.asarray(group[field][...])
                    png = normalize_to_uint8(field, arr)
                    Image.fromarray(png, mode="L").save(sample_out / f"{field}.png")
                count += 1
                if count >= max(args.limit, 1):
                    print(f"Exported {count} samples to {out_dir}")
                    return

    print(f"Exported {count} samples to {out_dir}")


if __name__ == "__main__":
    main()

