#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.distributed as dist

SCRIPT_DIR = Path(__file__).resolve().parent
TRY_ROOT = SCRIPT_DIR.parent
if str(TRY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY_ROOT))

from config_utils import anchor_data_paths_to_config_file, load_config, load_torch_checkpoint, resolve_device
from data_utils import (
    build_dataset_splits_from_config,
    compute_scalar_cond_dim,
    compute_input_channels,
    forward_cgan_generator,
    return_scalar_cond_from_config,
)
from model_pmhhnet import PMHHNetResidualRegressor, PMHNetResidualRegressor, PMNetResidualRegressor


def distributed_context() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def maybe_init_process_group(device_name: str) -> Tuple[object, int, int, int]:
    rank, world_size, local_rank = distributed_context()
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if str(device_name).startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend)
    if str(device_name).startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = resolve_device(device_name)
    return device, rank, world_size, local_rank


def maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def maybe_destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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
    common = dict(
        in_channels=compute_input_channels(cfg),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"].get("base_channels", 64)),
        encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
    )
    model_arch = str(cfg["model"].get("arch", "pmnet")).lower()
    scalar_dim = int(compute_scalar_cond_dim(cfg)) if return_scalar_cond_from_config(cfg) else 0
    if model_arch == "pmhnet":
        return PMHNetResidualRegressor(
            **common,
            hf_channels=int(cfg["model"].get("hf_channels", max(8, int(cfg["model"].get("base_channels", 64)) // 2))),
        )
    if model_arch == "pmhhnet":
        return PMHHNetResidualRegressor(
            **common,
            hf_channels=int(cfg["model"].get("hf_channels", max(8, int(cfg["model"].get("base_channels", 64)) // 2))),
            scalar_dim=max(1, scalar_dim),
            scalar_hidden_dim=int(cfg["model"].get("scalar_hidden_dim", max(32, int(cfg["model"].get("base_channels", 64)) * 2))),
        )
    return PMNetResidualRegressor(**common)


def _to_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)
    return array


def rank_indices(total: int, rank: int, world_size: int) -> range:
    return range(rank, total, world_size)


def shard_path_for(output_path: Path, rank: int) -> Path:
    return output_path.with_name(f"{output_path.stem}.rank{rank:02d}.h5")


def merge_split_outputs(
    *,
    output_path: Path,
    shard_paths: Sequence[Path],
    source_config: str,
    source_checkpoint: str,
    total_samples: int,
) -> None:
    if output_path.exists():
        output_path.unlink()

    with h5py.File(output_path, "w") as out_h5:
        out_h5.attrs["source_config"] = source_config
        out_h5.attrs["source_checkpoint"] = source_checkpoint
        out_h5.attrs["total_samples"] = int(total_samples)
        for shard_path in shard_paths:
            if not shard_path.exists():
                continue
            with h5py.File(shard_path, "r") as shard_h5:
                for city in shard_h5.keys():
                    dst_city = out_h5.require_group(str(city))
                    src_city = shard_h5[city]
                    for sample in src_city.keys():
                        if sample in dst_city:
                            del dst_city[sample]
                        src_city.copy(src_city[sample], dst_city, name=sample)

    for shard_path in shard_paths:
        if shard_path.exists():
            shard_path.unlink()


def flush_batch(
    *,
    records: List[Tuple[str, str]],
    x_list: List[torch.Tensor],
    y_list: List[torch.Tensor],
    scalar_list: List[Optional[torch.Tensor]],
    shard_h5: h5py.File,
    model: PMNetResidualRegressor,
    formula_idx: int,
    target_meta: Dict[str, Any],
    device: object,
) -> int:
    if not records:
        return 0

    x_batch = torch.stack(x_list, dim=0).to(device)
    y_batch = torch.stack(y_list, dim=0).to(device)
    scalar_batch: Optional[torch.Tensor] = None
    if scalar_list and scalar_list[0] is not None:
        scalar_batch = torch.stack([s for s in scalar_list if s is not None], dim=0).to(device)

    with torch.no_grad():
        model_out = forward_cgan_generator(model, x_batch, scalar_batch)
        residual = model_out[:, :1]
        no_data_logits = model_out[:, 1:2] if model_out.shape[1] > 1 else None
        prior = x_batch[:, formula_idx : formula_idx + 1]
        pred = clip_to_target_range(prior + residual, target_meta)

    pred_db = denormalize(pred, target_meta)
    target_db = denormalize(y_batch[:, :1], target_meta)
    error_db = pred_db - target_db
    abs_error_db = error_db.abs()
    no_data_prob = torch.sigmoid(no_data_logits) if no_data_logits is not None else None

    for row_idx, ref in enumerate(records):
        city, sample = ref
        grp = shard_h5.require_group(str(city)).require_group(str(sample))
        tensors = {
            "stage1_pred_norm_f16": pred[row_idx : row_idx + 1],
            "stage1_pred_db_f16": pred_db[row_idx : row_idx + 1],
            "stage1_error_db_f16": error_db[row_idx : row_idx + 1],
            "stage1_abs_error_db_f16": abs_error_db[row_idx : row_idx + 1],
        }
        if no_data_prob is not None:
            tensors["stage1_no_data_prob_f16"] = no_data_prob[row_idx : row_idx + 1]
        for name, tensor in tensors.items():
            array = _to_numpy_2d(tensor).astype(np.float16, copy=False)
            if name in grp:
                del grp[name]
            grp.create_dataset(name, data=array, compression="lzf")

    return len(records)


def export_split(
    *,
    cfg: Dict[str, Any],
    model: PMNetResidualRegressor,
    dataset: Any,
    output_path: Path,
    device: object,
    log_every: int,
    overwrite: bool,
    batch_size: int,
    rank: int,
    world_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        if rank == 0:
            print(f"[stage1-hdf5] reusing existing file: {output_path}")
        maybe_barrier()
        return

    target_meta = dict(cfg["target_metadata"]["path_loss"])
    formula_idx = formula_channel_index(cfg)
    total = len(dataset)
    sample_refs = list(getattr(dataset, "sample_refs", range(total)))
    local_indices = list(rank_indices(total, rank, world_size))
    shard_path = shard_path_for(output_path, rank)
    if shard_path.exists():
        shard_path.unlink()

    processed = 0
    pending_refs: List[Tuple[str, str]] = []
    pending_x: List[torch.Tensor] = []
    pending_y: List[torch.Tensor] = []
    pending_scalar: List[Optional[torch.Tensor]] = []
    with h5py.File(shard_path, "w") as shard_h5:
        shard_h5.attrs["source_config"] = str(Path(cfg["_source_config_path"]).resolve())
        shard_h5.attrs["source_checkpoint"] = str(cfg["_source_checkpoint_path"])
        shard_h5.attrs["rank"] = int(rank)
        shard_h5.attrs["world_size"] = int(world_size)
        shard_h5.attrs["assigned_samples"] = int(len(local_indices))
        for local_pos, index in enumerate(local_indices, start=1):
            item = dataset[index]
            if len(item) == 4:
                x, y, _m, scalar_cond = item
            else:
                x, y, _m = item
                scalar_cond = None
            pending_refs.append(sample_refs[index])
            pending_x.append(x)
            pending_y.append(y)
            pending_scalar.append(scalar_cond)
            if len(pending_refs) >= max(int(batch_size), 1):
                processed += flush_batch(
                    records=pending_refs,
                    x_list=pending_x,
                    y_list=pending_y,
                    scalar_list=pending_scalar,
                    shard_h5=shard_h5,
                    model=model,
                    formula_idx=formula_idx,
                    target_meta=target_meta,
                    device=device,
                )
                pending_refs, pending_x, pending_y, pending_scalar = [], [], [], []
            if local_pos == 1 or local_pos % max(int(log_every), 1) == 0 or local_pos == len(local_indices):
                print(
                    f"[stage1-hdf5][rank {rank}] {local_pos}/{len(local_indices)} local "
                    f"(global approx {processed}/{total}) -> {shard_path.name}"
                )

        if pending_refs:
            processed += flush_batch(
                records=pending_refs,
                x_list=pending_x,
                y_list=pending_y,
                scalar_list=pending_scalar,
                shard_h5=shard_h5,
                model=model,
                formula_idx=formula_idx,
                target_meta=target_meta,
                device=device,
            )

    maybe_barrier()
    if rank == 0:
        shard_paths = [shard_path_for(output_path, shard_rank) for shard_rank in range(world_size)]
        print(f"[stage1-hdf5] merging {len(shard_paths)} shards -> {output_path}")
        merge_split_outputs(
            output_path=output_path,
            shard_paths=shard_paths,
            source_config=str(Path(cfg["_source_config_path"]).resolve()),
            source_checkpoint=str(cfg["_source_checkpoint_path"]),
            total_samples=total,
        )
        print(f"[stage1-hdf5] finished {output_path}")
    maybe_barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Try49 stage1 predictions to HDF5 for the tail refiner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    cfg["_source_config_path"] = str(Path(args.config).resolve())
    cfg["_source_checkpoint_path"] = str(Path(args.checkpoint).resolve())

    device, rank, world_size, _local_rank = maybe_init_process_group(cfg["runtime"].get("device", "cuda"))
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
            if rank == 0:
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
            batch_size=int(args.batch_size),
            rank=rank,
            world_size=world_size,
        )

    maybe_destroy_process_group()


if __name__ == "__main__":
    main()
