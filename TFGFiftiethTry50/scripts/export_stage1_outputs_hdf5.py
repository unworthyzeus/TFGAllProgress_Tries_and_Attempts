#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from config_utils import anchor_data_paths_to_config_file, load_config, load_torch_checkpoint, resolve_device
from data_utils import (
    build_dataset_splits_from_config,
    compute_input_channels,
    forward_cgan_generator,
)
from model_pmnet import PMNetResidualRegressor


def formula_channel_index(cfg: Dict[str, Any]) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):
        idx += 1
    if cfg["data"].get("distance_map_channel", False):
        idx += 1
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    if not bool(formula_cfg.get("enabled", False)):
        raise ValueError("Stage1 exporter expects data.path_loss_formula_input.enabled = true")
    return idx


def denormalize(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    return values * scale + offset


def clip_to_target_range(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    clip_min = metadata.get("clip_min")
    clip_max = metadata.get("clip_max")
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    if clip_min is None or clip_max is None:
        return values
    min_norm = (float(clip_min) - offset) / max(scale, 1e-12)
    max_norm = (float(clip_max) - offset) / max(scale, 1e-12)
    return values.clamp(min=min_norm, max=max_norm)


def build_model(cfg: Dict[str, Any]) -> PMNetResidualRegressor:
    return PMNetResidualRegressor(
        in_channels=compute_input_channels(cfg),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"].get("base_channels", 64)),
        encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
    )


def _to_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)
    return array


def export_split(
    *,
    cfg: Dict[str, Any],
    model: PMNetResidualRegressor,
    dataset: Any,
    output_path: Path,
    device: object,
    log_every: int,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"[stage1-hdf5] reusing existing file: {output_path}")
        return

    target_meta = dict(cfg["target_metadata"]["path_loss"])
    formula_idx = formula_channel_index(cfg)
    started = False
    total = len(dataset)
    with h5py.File(output_path, "w") as out_h5:
        out_h5.attrs["source_config"] = str(Path(cfg["_source_config_path"]).resolve())
        out_h5.attrs["source_checkpoint"] = str(cfg["_source_checkpoint_path"])
        out_h5.attrs["total_samples"] = int(total)
        for index, ref in enumerate(tqdm(getattr(dataset, "sample_refs", range(total)), desc=f"export:{output_path.stem}"), start=0):
            item = dataset[index]
            if len(item) == 4:
                x, y, m, scalar_cond = item
            else:
                x, y, m = item
                scalar_cond = None
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            if scalar_cond is not None:
                scalar_cond = scalar_cond.unsqueeze(0).to(device)

            with torch.no_grad():
                residual = forward_cgan_generator(model, x, scalar_cond)
                prior = x[:, formula_idx : formula_idx + 1]
                pred = clip_to_target_range(prior + residual, target_meta)

            pred_db = denormalize(pred, target_meta)
            target_db = denormalize(y[:, :1], target_meta)
            error_db = pred_db - target_db
            abs_error_db = error_db.abs()

            city, sample = ref
            grp = out_h5.require_group(str(city)).require_group(str(sample))
            for name, array in {
                "stage1_pred_norm_f16": _to_numpy_2d(pred).astype(np.float16, copy=False),
                "stage1_pred_db_f16": _to_numpy_2d(pred_db).astype(np.float16, copy=False),
                "stage1_error_db_f16": _to_numpy_2d(error_db).astype(np.float16, copy=False),
                "stage1_abs_error_db_f16": _to_numpy_2d(abs_error_db).astype(np.float16, copy=False),
            }.items():
                if name in grp:
                    del grp[name]
                grp.create_dataset(name, data=array, compression="lzf")

            if not started:
                started = True
            if index == 0 or (index + 1) % max(int(log_every), 1) == 0 or index + 1 == total:
                print(f"[stage1-hdf5] {index + 1}/{total} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Try50 stage1 predictions to HDF5 for the tail refiner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    cfg["_source_config_path"] = str(Path(args.config).resolve())
    cfg["_source_checkpoint_path"] = str(Path(args.checkpoint).resolve())

    device = resolve_device(cfg["runtime"].get("device", "cuda"))
    splits = build_dataset_splits_from_config(cfg)

    model = build_model(cfg).to(device)
    state = load_torch_checkpoint(args.checkpoint, device)
    model_state = state.get("model", state.get("generator"))
    if model_state is None:
        raise RuntimeError("Checkpoint has no model/generator state")
    model.load_state_dict(model_state)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in args.splits:
        if split_name not in splits:
            print(f"[stage1-hdf5] skipping unknown split: {split_name}")
            continue
        output_path = output_dir / f"stage1_outputs_{split_name}.h5"
        export_split(
            cfg=cfg,
            model=model,
            dataset=splits[split_name],
            output_path=output_path,
            device=device,
            log_every=args.log_every,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()