#!/usr/bin/env python3
"""
Export predictions-only assets for:
  - Try 28 path loss -> predictions_twentyeighthtry28_path_loss/<split>/...
  - Try 26 delay/angular -> predictions_twentysixthtry26_delay_angular/<split>/...

This keeps the historical folders intact:
  - predictions_ninthtry9_path_loss
  - predictions_secondtry2_delay_angular

By default it skips re-exporting raw_hdf5 PNGs. Use --no-skip-dataset-export if needed.
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


TRY28_ROOT = PRACTICE_ROOT / "TFGTwentyEighthTry28"
TRY28_CONFIG = (
    TRY28_ROOT
    / "experiments"
    / "twentyeighthtry28_attention_topology"
    / "twentyeighthtry28_attention_topology.yaml"
)
TRY28_CHECKPOINT = (
    PRACTICE_ROOT
    / "cluster_outputs"
    / "TFGTwentyEighthTry28"
    / "twentyeighthtry28_attention_topology_t28_attention_topology_2gpu"
    / "best_cgan.pt"
)

TRY26_ROOT = PRACTICE_ROOT / "TFGTwentySixthTry26"
TRY26_CONFIG = (
    TRY26_ROOT
    / "experiments"
    / "twentysixthtry26_delay_angular_gradient"
    / "twentysixthtry26_delay_angular_gradient.yaml"
)
TRY26_CHECKPOINT = (
    PRACTICE_ROOT
    / "cluster_outputs"
    / "TFGTwentySixthTry26"
    / "twentysixthtry26_delay_angular_gradient_t26_delay_angular_gradient_2gpu"
    / "best_cgan.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Try 28 + Try 26 predictions without touching old folders.")
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
        help="Also export path_loss from Try 26 if that checkpoint contains it.",
    )
    p.add_argument(
        "--skip-path-loss",
        action="store_true",
        help="Skip Try 28 path loss export and only run Try 26 delay/angular.",
    )
    p.add_argument(
        "--skip-spread",
        action="store_true",
        help="Skip Try 26 delay/angular export and only run Try 28 path loss.",
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
    _require_file(TRY28_CONFIG, "Try 28 config")
    _require_file(TRY28_CHECKPOINT, "Try 28 checkpoint")
    _require_file(TRY26_CONFIG, "Try 26 config")
    _require_file(TRY26_CHECKPOINT, "Try 26 checkpoint")

    out_root.mkdir(parents=True, exist_ok=True)

    if args.no_skip_dataset_export:
        export_hdf5_dataset(hdf5_path, out_root)

    ensure_torch_available()

    if not args.skip_path_loss:
        predict_ninth_path_loss(
            TRY28_ROOT,
            TRY28_CONFIG,
            TRY28_CHECKPOINT,
            hdf5_path,
            scalar_csv,
            out_root,
            args.split,
            args.limit,
            args.device,
            output_label="twentyeighthtry28",
            path_loss_viz_scale=str(args.path_loss_viz_scale),
        )

    if not args.skip_spread:
        cols = None if args.spread_include_path_loss else ["delay_spread", "angular_spread"]
        predict_spread_multichannel(
            TRY26_ROOT,
            TRY26_CONFIG,
            TRY26_CHECKPOINT,
            hdf5_path,
            out_root,
            args.split,
            args.limit,
            args.device,
            run_label="twentysixthtry26",
            output_label="twentysixthtry26",
            columns_only=cols,
        )


if __name__ == "__main__":
    main()
