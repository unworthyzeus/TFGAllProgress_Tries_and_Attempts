#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


TRY_DIR = Path(__file__).resolve().parent.parent
if str(TRY_DIR) not in sys.path:
    sys.path.insert(0, str(TRY_DIR))

import config_utils
import data_utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute formula-prior inputs into a single HDF5 file.")
    p.add_argument("--config", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--output", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def formula_channel_index(cfg: Dict[str, object]) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):  # type: ignore[index]
        idx += 1
    if bool(cfg["data"].get("distance_map_channel", False)):  # type: ignore[index]
        idx += 1
    return idx


def _resolve_try_relative(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (TRY_DIR / path).resolve()


def main() -> None:
    args = parse_args()
    cfg = config_utils.load_config(args.config)
    config_utils.anchor_data_paths_to_config_file(cfg, args.config)
    cfg["augmentation"] = dict(cfg.get("augmentation", {}))
    cfg["augmentation"]["enable"] = False
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    output_raw = str(args.output).strip() or str(formula_cfg.get("precomputed_hdf5", "")).strip()
    calibration_json_raw = str(formula_cfg.get("regime_calibration_json", "")).strip()
    formula_cfg["cache_enabled"] = False
    formula_cfg["precomputed_hdf5"] = ""
    cfg["data"]["path_loss_formula_input"] = formula_cfg

    if not output_raw:
        raise SystemExit("No output path provided for precomputed prior HDF5.")
    output_path = Path(output_raw)
    if not output_path.is_absolute():
        output_path = (TRY_DIR / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dependency_paths = [Path(args.config).resolve()]
    if calibration_json_raw:
        cal_path = _resolve_try_relative(calibration_json_raw)
        if cal_path.exists():
            dependency_paths.append(cal_path)

    if output_path.exists() and not args.overwrite:
        output_mtime = output_path.stat().st_mtime
        newest_dep_mtime = max(path.stat().st_mtime for path in dependency_paths if path.exists())
        if output_mtime >= newest_dep_mtime:
            print(f"[prior-hdf5] already exists, reusing: {output_path}", flush=True)
            return
        print(
            f"[prior-hdf5] existing file is stale (output mtime={output_mtime:.0f}, newest dependency={newest_dep_mtime:.0f}), regenerating: {output_path}",
            flush=True,
        )

    splits = data_utils.build_dataset_splits_from_config(cfg)
    refs: List[Tuple[str, str, object, int]] = []
    seen: set[Tuple[str, str]] = set()
    for split_name in args.splits:
        if split_name not in splits:
            continue
        ds = splits[split_name]
        for idx, ref in enumerate(ds.sample_refs):
            if ref in seen:
                continue
            seen.add(ref)
            refs.append((ref[0], ref[1], ds, idx))

    if not refs:
        raise SystemExit("No samples selected for prior HDF5 precompute.")

    idx_formula = formula_channel_index(cfg)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    started = time.time()
    with h5py.File(tmp_path, "w") as out_h5:
        out_h5.attrs["source_config"] = str(Path(args.config).resolve())
        out_h5.attrs["source_config_mtime"] = float(Path(args.config).resolve().stat().st_mtime)
        out_h5.attrs["regime_calibration_json"] = calibration_json_raw
        if calibration_json_raw:
            cal_path = _resolve_try_relative(calibration_json_raw)
            if cal_path.exists():
                out_h5.attrs["regime_calibration_json_mtime"] = float(cal_path.stat().st_mtime)
        out_h5.attrs["total_samples"] = len(refs)
        for i, (city, sample, ds, local_idx) in enumerate(refs, start=1):
            item = ds[local_idx]
            x = item[0]
            formula = x[idx_formula : idx_formula + 1].detach().cpu().numpy().astype(np.float16)
            grp = out_h5.require_group(city).require_group(sample)
            if "formula_norm_f16" in grp:
                del grp["formula_norm_f16"]
            grp.create_dataset("formula_norm_f16", data=formula, compression="lzf")
            if i == 1 or i % int(args.log_every) == 0 or i == len(refs):
                elapsed = (time.time() - started) / 60.0
                print(f"[prior-hdf5] {i}/{len(refs)} elapsed={elapsed:.1f} min", flush=True)
    tmp_path.replace(output_path)
    print(f"[prior-hdf5] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
