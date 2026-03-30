#!/usr/bin/env python3
"""
Export predictions-only assets for the current building-mask family:
  - Try 33 path loss -> predictions_thirtythirdtry33_path_loss/<split>/...
  - Try 36 delay/angular -> predictions_thirtysixthtry36_delay_angular/<split>/...

This keeps the historical folders intact and does not re-export raw_hdf5 unless requested.
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
    predict_spread_multichannel,
)


TRY33_ROOT = PRACTICE_ROOT / "TFGThirtyThirdTry33"
TRY33_CONFIG = (
    TRY33_ROOT
    / "experiments"
    / "thirtythirdtry33_buildingmask_pathloss"
    / "thirtythirdtry33_buildingmask_pathloss.yaml"
)
TRY33_CHECKPOINT = (
    PRACTICE_ROOT
    / "cluster_outputs"
    / "TFGThirtyThirdTry33"
    / "thirtythirdtry33_buildingmask_pathloss_t33_buildingmask_pathloss_2gpu"
    / "best_cgan.pt"
)

TRY36_ROOT = PRACTICE_ROOT / "TFGThirtySixthTry36"
TRY36_CONFIG = (
    TRY36_ROOT
    / "experiments"
    / "thirtysixthtry36_spread_buildingmask"
    / "thirtysixthtry36_spread_buildingmask.yaml"
)
TRY36_CHECKPOINT = (
    PRACTICE_ROOT
    / "cluster_outputs"
    / "TFGThirtySixthTry36"
    / "thirtysixthtry36_spread_buildingmask_t36_spread_buildingmask_2gpu"
    / "best_cgan.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Try 33 + Try 36 predictions without touching old folders.")
    p.add_argument("--hdf5", type=str, default=str(DEFAULT_HDF5))
    p.add_argument("--scalar-csv", type=str, default=str(DEFAULT_SCALAR_CSV))
    p.add_argument("--dataset-out", type=str, default="D:/Dataset_Imagenes")
    p.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("train", "val", "test", "all"),
        help="Inference split. Use 'all' to match the existing Dataset_Imagenes export.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--no-skip-dataset-export",
        action="store_true",
        help="Also re-export raw_hdf5 PNGs from the HDF5. Default is to skip this.",
    )
    p.add_argument(
        "--path-loss-viz-scale",
        type=str,
        default="gt",
        choices=("gt", "joint", "independent"),
    )
    p.add_argument(
        "--spread-include-path-loss",
        action="store_true",
        help="Also export path_loss from the spread checkpoint if that checkpoint contains it.",
    )
    p.add_argument(
        "--skip-path-loss",
        action="store_true",
        help="Skip Try 33 path loss export and only run Try 36 delay/angular.",
    )
    p.add_argument(
        "--skip-spread",
        action="store_true",
        help="Skip Try 36 delay/angular export and only run Try 33 path loss.",
    )
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
    _require_file(TRY33_CONFIG, "Try 33 config")
    _require_file(TRY36_CONFIG, "Try 36 config")

    if not args.skip_path_loss:
        _require_file(TRY33_CHECKPOINT, "Try 33 checkpoint")
    if not args.skip_spread:
        _require_file(TRY36_CHECKPOINT, "Try 36 checkpoint")

    out_root.mkdir(parents=True, exist_ok=True)

    if args.no_skip_dataset_export:
        export_hdf5_dataset(hdf5_path, out_root)

    ensure_torch_available()

    if not args.skip_path_loss:
        predict_ninth_path_loss(
            TRY33_ROOT,
            TRY33_CONFIG,
            TRY33_CHECKPOINT,
            hdf5_path,
            scalar_csv,
            out_root,
            args.split,
            args.limit,
            args.device,
            output_label="thirtythirdtry33",
            path_loss_viz_scale=str(args.path_loss_viz_scale),
        )

    if not args.skip_spread:
        cols = None if args.spread_include_path_loss else ["delay_spread", "angular_spread"]
        predict_spread_multichannel(
            TRY36_ROOT,
            TRY36_CONFIG,
            TRY36_CHECKPOINT,
            hdf5_path,
            out_root,
            args.split,
            args.limit,
            args.device,
            run_label="thirtysixthtry36",
            output_label="thirtysixthtry36",
            columns_only=cols,
        )


if __name__ == "__main__":
    main()
