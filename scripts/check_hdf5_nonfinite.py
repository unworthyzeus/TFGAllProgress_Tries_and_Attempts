#!/usr/bin/env python3
"""
Scan CKM HDF5: NaN/Inf counts, global finite min/max/mean per dataset, and optional
scan of *all* numeric datasets under each sample (not only the usual map names).

Training: path_loss non-finite pixels are filled with the **max finite value in that map**
(not 0); with path_loss_ignore_nonfinite: true they are still masked in the loss.

  python scripts/check_hdf5_nonfinite.py
  python scripts/check_hdf5_nonfinite.py --progress-every 1500
  python scripts/check_hdf5_nonfinite.py --all-datasets
  python scripts/check_hdf5_nonfinite.py --max-samples 500   # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent
DEFAULT_HDF5 = PRACTICE_ROOT / "Datasets" / "CKM_Dataset_180326_antenna_height.h5"

# Default CKM map-like fields (used when not --all-datasets)
FIELDS_DEFAULT: tuple[str, ...] = (
    "topology_map",
    "path_loss",
    "delay_spread",
    "angular_spread",
    "los_mask",
    "uav_height",
)


def new_agg() -> Dict[str, Any]:
    return {
        "nan_pixels": 0,
        "inf_pixels": 0,
        "samples_touched": 0,
        "samples_with_bad": 0,
        "finite_n": 0,
        "finite_sum": 0.0,
        "global_min": np.inf,
        "global_max": -np.inf,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HDF5 audit: NaN/Inf + global finite min/max/mean; scans all samples by default"
    )
    p.add_argument("--hdf5", type=str, default=str(DEFAULT_HDF5), help="Path to CKM HDF5")
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, stop after this many (city,sample) groups (default: 0 = all)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print progress to stderr every N samples (0 = off). Default 2000.",
    )
    p.add_argument(
        "--all-datasets",
        action="store_true",
        help="Include every numeric Dataset under each sample group (union of names), not only the default field list",
    )
    return p.parse_args()


def numeric_dataset_keys(grp: Any) -> List[str]:
    import h5py

    out: List[str] = []
    for k in grp.keys():
        ds = grp[k]
        if not isinstance(ds, h5py.Dataset):
            continue
        kind = getattr(ds.dtype, "kind", "")
        if kind in ("f", "i", "u"):
            out.append(str(k))
    return sorted(out)


def merge_stats(agg: Dict[str, Any], arr: np.ndarray) -> None:
    nan_m = np.isnan(arr)
    inf_m = np.isinf(arr)
    agg["nan_pixels"] += int(np.count_nonzero(nan_m))
    agg["inf_pixels"] += int(np.count_nonzero(inf_m))
    if np.any(nan_m) or np.any(inf_m):
        agg["samples_with_bad"] += 1

    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return
    agg["finite_n"] += int(fin.size)
    agg["finite_sum"] += float(np.sum(fin, dtype=np.float64))
    mn = float(np.min(fin))
    mx = float(np.max(fin))
    agg["global_min"] = min(agg["global_min"], mn)
    agg["global_max"] = max(agg["global_max"], mx)


def main() -> None:
    args = parse_args()
    h5_path = Path(args.hdf5)
    if not h5_path.is_file():
        print(f"HDF5 not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import h5py
    except ImportError:
        print("pip install h5py", file=sys.stderr)
        sys.exit(1)

    agg: Dict[str, Dict[str, Any]] = {}
    examples: DefaultDict[str, List[str]] = defaultdict(list)
    max_ex = 10

    n_groups = 0
    with h5py.File(h5_path, "r") as handle:
        for city in sorted(handle.keys()):
            cg = handle[city]
            if not isinstance(cg, h5py.Group):
                continue
            for sample in sorted(cg.keys()):
                grp = cg[sample]
                if not isinstance(grp, h5py.Group):
                    continue
                if args.max_samples and n_groups >= args.max_samples:
                    break
                n_groups += 1
                if args.progress_every and n_groups % args.progress_every == 0:
                    print(f"[progress] scanned {n_groups} samples...", file=sys.stderr)

                if args.all_datasets:
                    fields = numeric_dataset_keys(grp)
                else:
                    fields = [f for f in FIELDS_DEFAULT if f in grp]

                for field in fields:
                    if field not in agg:
                        agg[field] = new_agg()
                    ds = grp[field]
                    if not isinstance(ds, h5py.Dataset):
                        continue
                    try:
                        arr = np.asarray(ds[...], dtype=np.float64)
                    except (OSError, TypeError, ValueError):
                        continue
                    a = agg[field]
                    a["samples_touched"] += 1
                    merge_stats(a, arr)
                    n_nan = int(np.count_nonzero(np.isnan(arr)))
                    n_inf = int(np.count_nonzero(np.isinf(arr)))
                    if (n_nan or n_inf) and len(examples[field]) < max_ex:
                        examples[field].append(f"{city}/{sample} nan={n_nan} inf={n_inf}")

            if args.max_samples and n_groups >= args.max_samples:
                break

    print(f"File: {h5_path.resolve()}")
    print(f"(city,sample) groups scanned: {n_groups}")
    print(f"Mode: {'all numeric datasets per sample' if args.all_datasets else 'default field list'}\n")

    order = sorted(agg.keys(), key=lambda x: (x not in FIELDS_DEFAULT, x))

    print(
        f"{'field':<22} {'samples':>10} {'w_nan_inf':>10} {'nan_px':>12} {'inf_px':>10} "
        f"{'finite_min':>14} {'finite_max':>14} {'finite_mean':>14}"
    )
    print("-" * 120)
    any_issue = False
    for field in order:
        a = agg[field]
        st = a["samples_touched"]
        sb = a["samples_with_bad"]
        if sb:
            any_issue = True
        fn = a["finite_n"]
        if fn > 0:
            mean_v = a["finite_sum"] / fn
            gmin = a["global_min"] if np.isfinite(a["global_min"]) else float("nan")
            gmax = a["global_max"] if np.isfinite(a["global_max"]) else float("nan")
        else:
            mean_v = float("nan")
            gmin = float("nan")
            gmax = float("nan")
        print(
            f"{field:<22} {st:>10} {sb:>10} {a['nan_pixels']:>12} {a['inf_pixels']:>10} "
            f"{gmin:>14.6g} {gmax:>14.6g} {mean_v:>14.6g}"
        )
    print()

    if any_issue:
        print("Examples (non-finite, up to 10 per field):")
        for field in order:
            xs = examples.get(field) or []
            if not xs:
                continue
            print(f"  [{field}]")
            for line in xs:
                print(f"    {line}")
        print()
    else:
        print("No NaN or Inf in scanned fields.\n")

    print(
        "path_loss fill: CKMHDF5Dataset replaces non-finite pixels with the **max finite** "
        "value in that map (fallback 0 if the map is all non-finite). "
        "path_loss_ignore_nonfinite still zeros their contribution in the loss via mask."
    )


if __name__ == "__main__":
    main()
