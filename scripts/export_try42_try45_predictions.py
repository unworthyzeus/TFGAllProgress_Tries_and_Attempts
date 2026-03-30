#!/usr/bin/env python3
"""
Export predictions-only assets for the current PMNet prior family:
  - Try 42 path loss -> predictions_fortysecondtry42_path_loss/<split>/...
  - Try 45 path loss -> predictions_fortyfifthtry45_path_loss/<split>/...

This keeps historical folders intact and does not re-export raw_hdf5 unless requested.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from export_dataset_and_predictions import (
    DEFAULT_HDF5,
    DEFAULT_SCALAR_CSV,
    PRACTICE_ROOT,
    ensure_torch_available,
    export_hdf5_dataset,
    predict_ninth_path_loss,
)


TRY42_ROOT = PRACTICE_ROOT / "TFGFortySecondTry42"
TRY42_CONFIG = (
    TRY42_ROOT
    / "experiments"
    / "fortysecondtry42_pmnet_prior_residual"
    / "fortysecondtry42_pmnet_prior_residual.yaml"
)
TRY42_CHECKPOINT = (
    PRACTICE_ROOT
    / "cluster_outputs"
    / "TFGFortySecondTry42"
    / "fortysecondtry42_pmnet_prior_residual_t42_pmnet_prior_residual_1gpu"
    / "best_cgan.pt"
)

TRY45_ROOT = PRACTICE_ROOT / "TFGFortyFifthTry45"
TRY45_CONFIG = (
    TRY45_ROOT
    / "experiments"
    / "fortyfifthtry45_pmnet_moe_enhanced_prior"
    / "fortyfifthtry45_pmnet_moe_enhanced_prior.yaml"
)
TRY45_CHECKPOINT = (
    PRACTICE_ROOT
    / "cluster_outputs"
    / "TFGFortyFifthTry45"
    / "fortyfifthtry45_pmnet_moe_enhanced_prior_t45_pmnet_moe_enhanced_prior_1gpu"
    / "best_cgan.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Try 42 + Try 45 predictions without touching old folders.")
    p.add_argument("--hdf5", type=str, default=str(DEFAULT_HDF5))
    p.add_argument("--scalar-csv", type=str, default=str(DEFAULT_SCALAR_CSV))
    p.add_argument("--dataset-out", type=str, default="D:/Dataset_Imagenes")
    p.add_argument("--split", type=str, default="all", choices=("train", "val", "test", "all"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-skip-dataset-export", action="store_true")
    p.add_argument("--path-loss-viz-scale", type=str, default="gt", choices=("gt", "joint", "independent"))
    p.add_argument("--skip-42", action="store_true", help="Skip Try 42 export.")
    p.add_argument("--skip-45", action="store_true", help="Skip Try 45 export.")
    return p.parse_args()


def _optional_csv(path_s: str) -> Optional[Path]:
    if not path_s:
        return None
    p = Path(path_s)
    return p if p.is_file() else None


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def main() -> None:
    args = parse_args()
    hdf5_path = Path(args.hdf5)
    out_root = Path(args.dataset_out)
    scalar_csv = _optional_csv(args.scalar_csv)

    _require_file(hdf5_path, "HDF5")
    _require_file(TRY42_CONFIG, "Try 42 config")
    _require_file(TRY45_CONFIG, "Try 45 config")
    if not args.skip_42:
        _require_file(TRY42_CHECKPOINT, "Try 42 checkpoint")
    if not args.skip_45:
        _require_file(TRY45_CHECKPOINT, "Try 45 checkpoint")

    out_root.mkdir(parents=True, exist_ok=True)
    if args.no_skip_dataset_export:
        export_hdf5_dataset(hdf5_path, out_root)

    ensure_torch_available()

    if not args.skip_42:
        predict_ninth_path_loss(
            TRY42_ROOT,
            TRY42_CONFIG,
            TRY42_CHECKPOINT,
            hdf5_path,
            scalar_csv,
            out_root,
            args.split,
            args.limit,
            args.device,
            output_label="fortysecondtry42",
            path_loss_viz_scale=str(args.path_loss_viz_scale),
        )

    if not args.skip_45:
        predict_ninth_path_loss(
            TRY45_ROOT,
            TRY45_CONFIG,
            TRY45_CHECKPOINT,
            hdf5_path,
            scalar_csv,
            out_root,
            args.split,
            args.limit,
            args.device,
            output_label="fortyfifthtry45",
            path_loss_viz_scale=str(args.path_loss_viz_scale),
        )


if __name__ == "__main__":
    main()
