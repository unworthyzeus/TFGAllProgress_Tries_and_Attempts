#!/usr/bin/env python3
"""
Inspect ALL metadata/attributes in CKM_Dataset.h5 at every level.
Run this to verify whether antenna height or any other per-sample metadata exists.
Usage: python inspect_hdf5_metadata.py [--hdf5 path/to/CKM_Dataset.h5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def visit_all(name: str, obj: object, depth: int = 0) -> None:
    """Recursively visit every object and print its attributes."""
    indent = "  " * depth
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}[Dataset] {name}")
        print(f"{indent}  shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{indent}[Group] {name}")

    # Print ALL attributes
    if hasattr(obj, "attrs") and len(obj.attrs) > 0:
        for key in sorted(obj.attrs.keys()):
            try:
                val = obj.attrs[key]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                print(f"{indent}  @{key} = {val}")
            except Exception as e:
                print(f"{indent}  @{key} = <error: {e}>")
    elif hasattr(obj, "attrs"):
        print(f"{indent}  (no attributes)")

    if isinstance(obj, h5py.Group):
        for key in sorted(obj.keys()):
            visit_all(key, obj[key], depth + 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect all HDF5 attributes/metadata. Use to verify if antenna height exists."
    )
    parser.add_argument(
        "--hdf5",
        default="CKM_Dataset.h5",
        help="Path to CKM_Dataset.h5",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Also print attributes for first 3 samples in detail",
    )
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        print(f"ERROR: File not found: {hdf5_path}")
        print("Run from TFG_FirstTry1 or provide full path: python inspect_hdf5_metadata.py --hdf5 /path/to/CKM_Dataset.h5")
        return

    print("=" * 60)
    print(f"Inspecting: {hdf5_path}")
    print("=" * 60)

    with h5py.File(hdf5_path, "r") as f:
        # Root attributes
        print("\n--- ROOT attributes ---")
        if len(f.attrs) > 0:
            for k in sorted(f.attrs.keys()):
                print(f"  @{k} = {f.attrs[k]}")
        else:
            print("  (none)")

        # Full tree with attributes
        print("\n--- Full structure (groups/datasets + attributes) ---")
        visit_all("/", f, depth=0)

        # Per-sample detail if requested
        if args.sample:
            print("\n--- Per-sample attributes (first 3 samples) ---")
            count = 0
            for city in sorted(f.keys()):
                for sample in sorted(f[city].keys()):
                    if count >= 3:
                        break
                    grp = f[city][sample]
                    print(f"\n  {city}/{sample}:")
                    if len(grp.attrs) > 0:
                        for k in sorted(grp.attrs.keys()):
                            print(f"    @{k} = {grp.attrs[k]}")
                    else:
                        print("    (no group attributes)")
                    for ds_name in sorted(grp.keys()):
                        ds = grp[ds_name]
                        if len(ds.attrs) > 0:
                            print(f"    {ds_name}.attrs:")
                            for k in sorted(ds.attrs.keys()):
                                print(f"      @{k} = {ds.attrs[k]}")
                    count += 1
                if count >= 3:
                    break

    print("\n" + "=" * 60)
    print("Done. If no 'height', 'antenna_height', 'h_abs', or similar appears above,")
    print("then antenna height is NOT stored in the HDF5 metadata.")
    print("=" * 60)


if __name__ == "__main__":
    main()
