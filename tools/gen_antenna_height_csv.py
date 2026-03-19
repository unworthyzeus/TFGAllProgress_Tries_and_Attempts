#!/usr/bin/env python3
"""
Build Datasets/CKM_180326_antenna_height.csv with columns: city, sample, antenna_height_m.

For each HDF5 sample, reads optional group attribute or 0-d dataset `antenna_height_m`
(or `antenna_height`); if missing, writes 0.0. Replace zeros with real heights when your
HDF5 or spreadsheet provides them.

Usage:
  python tools/gen_antenna_height_csv.py --hdf5 Datasets/CKM_Dataset_180326.h5

Works from TFGpractice or from tools/; relative paths also check the TFGpractice parent folder.
For per-sample height, use the file that contains `uav_height` (e.g. CKM_Dataset_180326_antenna_height.h5).
The older CKM_Dataset_180326.h5 has no height field → CSV would be all zeros.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def read_height_m(group: h5py.Group) -> float:
    # uav_height: common in CKM_Dataset_*_antenna_height.h5 (often shape (1,1) float32)
    for key in (
        "uav_height",
        "antenna_height_m",
        "antenna_height",
        "height_m",
        "drone_height_m",
    ):
        if key in group.attrs:
            try:
                v = float(np.asarray(group.attrs[key]).reshape(-1)[0])
                if np.isfinite(v):
                    return v
            except (TypeError, ValueError):
                pass
        if key in group and isinstance(group[key], h5py.Dataset):
            try:
                arr = np.asarray(group[key][...], dtype=np.float64).reshape(-1)
                if arr.size:
                    v = float(np.nanmean(arr))
                    if np.isfinite(v):
                        return v
            except (TypeError, ValueError, OSError):
                pass
    return 0.0


# Parent of tools/ == TFGpractice project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_hdf5_path(p: Path) -> Path:
    p = Path(p)
    if p.is_file():
        return p.resolve()
    cwd_try = (Path.cwd() / p).resolve()
    if cwd_try.is_file():
        return cwd_try
    root_try = (_PROJECT_ROOT / p).resolve()
    if root_try.is_file():
        return root_try
    expected = _PROJECT_ROOT / "Datasets" / "CKM_Dataset_180326_antenna_height.h5"
    print(
        f"Error: HDF5 not found.\n"
        f"  Given: {p}\n"
        f"  Tried: {cwd_try}\n"
        f"  Tried: {root_try}\n"
        f"  Example: {expected}",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _resolve_out_path(p: Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    s = p.as_posix()
    if s.startswith("Datasets/") or s.startswith("Datasets\\") or s == "Datasets":
        return (_PROJECT_ROOT / p).resolve()
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.parent.is_dir():
        return cwd_p
    return (_PROJECT_ROOT / p).resolve()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True, type=Path, help="Path to CKM map HDF5 (e.g. Datasets/CKM_Dataset_180326.h5)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("Datasets/CKM_180326_antenna_height.csv"),
        help="Output CSV path (default: Datasets/CKM_180326_antenna_height.csv under TFGpractice)",
    )
    args = p.parse_args()

    hdf5_path = _resolve_hdf5_path(args.hdf5)
    out_path = _resolve_out_path(args.out)

    rows = []
    with h5py.File(hdf5_path, "r") as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                g = handle[city][sample]
                h_m = read_height_m(g)
                rows.append({"city": city, "sample": sample, "antenna_height_m": h_m})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    nonzero = int((df["antenna_height_m"] != 0.0).sum())
    print(f"Wrote {len(df)} rows to {out_path} ({nonzero} non-zero heights from HDF5).")


if __name__ == "__main__":
    main()
