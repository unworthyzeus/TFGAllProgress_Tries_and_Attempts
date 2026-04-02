#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


TRY_DIR = Path(__file__).resolve().parent
if str(TRY_DIR) not in sys.path:
    sys.path.insert(0, str(TRY_DIR))

import config_utils  # noqa: E402
import data_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute Try 47 obstruction proxy features into an HDF5 cache.")
    p.add_argument("--config", required=True, help="Config path relative to try dir or absolute")
    p.add_argument("--dataset", default="", help="Optional HDF5 override")
    p.add_argument("--out-hdf5", required=True, help="Output HDF5 path for cached obstruction features")
    p.add_argument("--num-workers", type=int, default=max((os.cpu_count() or 4) // 2, 1))
    p.add_argument("--max-inflight", type=int, default=16)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--splits", default="train,val,test", help="Comma-separated split names to include")
    p.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    p.add_argument("--compression-level", type=int, default=1)
    return p.parse_args()


def _resolve_try_relative_path(raw_path: str) -> Path:
    p = Path(str(raw_path))
    return p if p.is_absolute() else (TRY_DIR / p)


def _worker_compute(
    try_dir: str,
    hdf5_path: str,
    image_size: int,
    city: str,
    sample: str,
    input_column: str,
    input_metadata: Dict[str, Any],
    los_input_column: str | None,
    los_metadata: Dict[str, Any],
    non_ground_threshold: float,
    meters_per_pixel: float,
    angle_bins: int,
) -> Tuple[str, str, np.ndarray]:
    sys.path.insert(0, try_dir)
    import data_utils as _du  # type: ignore

    with h5py.File(hdf5_path, "r") as handle:
        grp = handle[city][sample]
        raw_input = np.asarray(grp[input_column][...], dtype=np.float32)
        los_tensor = None
        if los_input_column:
            raw_los = np.asarray(grp[los_input_column][...], dtype=np.float32)
            los_tensor = _du._resize_array(raw_los, image_size, los_metadata)

    non_ground_mask = _du._resize_mask_nearest(raw_input != float(non_ground_threshold), image_size)
    shadow_depth_tensor, distance_since_los_tensor, blocker_height_tensor, blocker_count_tensor = _du._compute_ray_obstruction_proxy_features(
        raw_input,
        los_tensor,
        non_ground_threshold=float(non_ground_threshold),
        meters_per_pixel=float(meters_per_pixel),
        angle_bins=int(angle_bins),
    )
    feats = []
    for tensor in (
        shadow_depth_tensor,
        distance_since_los_tensor,
        blocker_height_tensor,
        blocker_count_tensor,
    ):
        resized = TF.resize(
            tensor,
            [image_size, image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        resized = resized * (1.0 - non_ground_mask).clamp(0.0, 1.0)
        feats.append(resized.squeeze(0).cpu().numpy())

    stacked = np.stack(feats, axis=0)
    stacked = np.clip(np.rint(stacked * 255.0), 0.0, 255.0).astype(np.uint8)
    return city, sample, stacked


def _all_refs_from_splits(cfg: Dict[str, Any], split_names: Iterable[str]) -> List[Tuple[str, str, str]]:
    splits = data_utils.build_dataset_splits_from_config(cfg)
    refs: List[Tuple[str, str, str]] = []
    seen = set()
    for split_name in split_names:
        ds = splits[split_name]
        for city, sample in ds.sample_refs:
            key = (city, sample)
            if key in seen:
                continue
            seen.add(key)
            refs.append((split_name, city, sample))
    return refs


def main() -> None:
    args = parse_args()
    cfg_path = _resolve_try_relative_path(args.config)
    cfg = config_utils.load_config(str(cfg_path))
    if args.dataset:
        cfg["data"]["hdf5_path"] = str(Path(args.dataset).resolve())

    obstruction_cfg = dict(cfg["data"].get("path_loss_obstruction_features", {}))
    if not bool(obstruction_cfg.get("enabled", False)):
        raise RuntimeError("path_loss_obstruction_features.enabled is false in this config.")

    hdf5_path = str(Path(cfg["data"]["hdf5_path"]).resolve())
    cfg["data"]["path_loss_obstruction_features"] = dict(obstruction_cfg)
    cfg["data"]["path_loss_obstruction_features"]["precomputed_hdf5"] = ""
    image_size = int(cfg["data"].get("image_size", 513))
    input_column = str(cfg["data"].get("input_column", "topology_map"))
    input_metadata = dict(cfg["data"].get("input_metadata", {}))
    los_input_column = cfg["data"].get("los_input_column")
    los_metadata = dict(cfg["target_metadata"].get(los_input_column, {})) if los_input_column else {}
    non_ground_threshold = float(cfg["data"].get("non_ground_threshold", 0.0))
    meters_per_pixel = float(obstruction_cfg.get("meters_per_pixel", cfg["data"].get("path_loss_formula_input", {}).get("meters_per_pixel", 1.0)))
    angle_bins = int(obstruction_cfg.get("angle_bins", 720))
    split_names = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    refs = _all_refs_from_splits(cfg, split_names)
    out_path = _resolve_try_relative_path(args.out_hdf5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == "none" else args.compression
    compression_opts = int(args.compression_level) if compression == "gzip" else None

    completed = 0
    skipped = 0
    started = time.time()

    with h5py.File(out_path, "a") as out_h5:
        out_h5.attrs["feature_names"] = np.array(
            [
                "shadow_depth",
                "distance_since_los_break",
                "max_blocker_height",
                "blocker_count",
            ],
            dtype=h5py.string_dtype("utf-8"),
        )
        out_h5.attrs["image_size"] = image_size
        out_h5.attrs["angle_bins"] = angle_bins
        out_h5.attrs["meters_per_pixel"] = meters_per_pixel
        out_h5.attrs["quantization"] = "uint8_0_255"

        todo: List[Tuple[str, str]] = []
        for split_name, city, sample in refs:
            if city in out_h5 and sample in out_h5[city] and "features_u8" in out_h5[city][sample]:
                skipped += 1
                continue
            todo.append((city, sample))

        total = len(refs)
        print(
            f"[precompute] total_refs={total} skipped_existing={skipped} todo={len(todo)} "
            f"workers={args.num_workers} out={out_path}",
            flush=True,
        )

        if not todo:
            return

        inflight = max(int(args.max_inflight), int(args.num_workers), 1)
        pending = {}
        todo_iter = iter(todo)

        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as ex:
            while True:
                while len(pending) < inflight:
                    try:
                        city, sample = next(todo_iter)
                    except StopIteration:
                        break
                    fut = ex.submit(
                        _worker_compute,
                        str(TRY_DIR),
                        hdf5_path,
                        image_size,
                        city,
                        sample,
                        input_column,
                        input_metadata,
                        los_input_column,
                        los_metadata,
                        non_ground_threshold,
                        meters_per_pixel,
                        angle_bins,
                    )
                    pending[fut] = (city, sample)
                if not pending:
                    break
                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    city, sample = pending.pop(fut)
                    city_out, sample_out, arr = fut.result()
                    city_grp = out_h5.require_group(city_out)
                    sample_grp = city_grp.require_group(sample_out)
                    if "features_u8" in sample_grp:
                        del sample_grp["features_u8"]
                    sample_grp.create_dataset(
                        "features_u8",
                        data=arr,
                        compression=compression,
                        compression_opts=compression_opts,
                        shuffle=True,
                        chunks=(1, image_size, image_size),
                    )
                    sample_grp.attrs["channels"] = 4
                    completed += 1
                    if completed == 1 or completed % max(int(args.log_every), 1) == 0 or completed == len(todo):
                        elapsed = time.time() - started
                        print(
                            f"[precompute] completed={completed}/{len(todo)} skipped={skipped} "
                            f"elapsed={elapsed/60.0:.1f} min",
                            flush=True,
                        )

        print(
            f"[precompute] done total_written={completed} skipped_existing={skipped} "
            f"elapsed_min={(time.time() - started)/60.0:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
