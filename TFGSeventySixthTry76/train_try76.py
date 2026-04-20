"""Try 76 — single-expert training entry point.

Usage:
    python train_try76.py --config experiments/seventysixth_try76_experts/try76_expert_open_sparse_lowrise_los.yaml

On a multi-GPU node, launch via ``torchrun`` (the SLURM scripts under
``cluster/`` do this). Inside, DDP is initialised lazily; single-GPU / CPU
runs work without torchrun.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.config_try76 import Try76Cfg
from src.data_utils import (
    HeightEmbedding,
    Try76Config,
    Try76ExpertDataset,
    build_expert_datasets,
)
from src.losses_try76 import LossWeights, combined_loss
from src.metrics_try76 import masked_rmse_db, masked_mae_db, per_image_histogram
from src.model_try76 import Try76Model, Try76ModelConfig

CODE_MARKER = "try76_nanfix_2026-04-20a"


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def resolve_single_process_device(preferred: str | None = None) -> torch.device:
    preferred = (preferred or "").strip().lower()
    if preferred == "directml":
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("runtime.device=directml requested, but torch_directml is not installed") from exc
        return torch_directml.device()
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    if preferred == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_distributed(preferred_device: str | None = None) -> Tuple[bool, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=os.environ.get("DDP_BACKEND", "nccl"))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return True, rank, world, device
    device = resolve_single_process_device(preferred_device)
    return False, 0, 1, device


def is_main(rank: int) -> bool:
    return rank == 0


def maybe_barrier(ddp: bool) -> None:
    if ddp and dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_lambda(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# Training / validation passes
# ---------------------------------------------------------------------------

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    height_embed: HeightEmbedding,
    weights: LossWeights,
    clamp_lo: float,
    clamp_hi: float,
    grad_accum: int,
    is_train: bool,
    is_main_: bool,
) -> Dict[str, float]:
    model.train(is_train)
    totals = {"total": 0.0, "map_nll": 0.0, "dist_kl": 0.0, "moment_match": 0.0,
              "outlier_budget": 0.0, "rmse_db_loss": 0.0}
    rmse_accum = 0.0
    n_images = 0
    if is_train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    steps_in_epoch = 0
    for step, batch in enumerate(loader):
        steps_in_epoch = step + 1
        batch = to_device(batch, device)
        inputs = batch["inputs"]
        target = batch["target"]
        mask = batch["loss_mask"]
        h = batch["antenna_height_m"]
        h_emb = height_embed(h)

        if is_train:
            out = model(inputs, h_emb)
        else:
            with torch.no_grad():
                out = model(inputs, h_emb)

        def _tensor_stats(t: torch.Tensor, active_mask: torch.Tensor | None = None) -> str:
            if active_mask is not None:
                active = active_mask > 0
                if not torch.any(active):
                    return "empty-mask"
                t = t.masked_select(active)
            return (
                f"shape={tuple(t.shape)} "
                f"finite={bool(torch.isfinite(t).all())} "
                f"min={float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).min()):.4f} "
                f"max={float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).max()):.4f}"
            )

        finite_outputs = all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v))
        if "gmm" in out and isinstance(out["gmm"], dict):
            finite_outputs = finite_outputs and all(torch.isfinite(v).all() for v in out["gmm"].values())
        if not torch.isfinite(inputs).all() or not torch.isfinite(target).all() or not torch.isfinite(mask).all() or not finite_outputs:
            raise RuntimeError(
                "Non-finite tensor before loss. "
                f"sample={list(zip(batch['city'], batch['sample']))} "
                f"inputs[{_tensor_stats(inputs)}] "
                f"target[{_tensor_stats(target, mask)}] "
                f"mask_sum={float(mask.sum()):.1f}"
            )

        loss_terms = combined_loss(target, mask, out, clamp_lo, clamp_hi, weights=weights)
        if not all(torch.isfinite(v).all() for v in loss_terms.values()):
            raise RuntimeError(
                "Non-finite loss terms detected. "
                f"sample={list(zip(batch['city'], batch['sample']))} "
                f"target[{_tensor_stats(target, mask)}] "
                f"pred[{_tensor_stats(out['pred'], mask)}] "
                f"mask_sum={float(mask.sum()):.1f} "
                f"loss_terms={{"
                + ", ".join(f"{k}={float(v.detach())}" for k, v in loss_terms.items())
                + "}}"
            )
        loss = loss_terms["total"] / max(grad_accum, 1)

        if is_train:
            loss.backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        for k in totals:
            if k == "total":
                totals[k] += float(loss_terms["total"].detach().item())
            elif k == "rmse_db_loss":
                totals[k] += float(loss_terms["rmse_db"].item())
            else:
                totals[k] += float(loss_terms[k].item())
        rmse_accum += masked_rmse_db(out["pred"].detach(), target, mask)
        n_images += 1

    if is_train and optimizer is not None and steps_in_epoch > 0 and steps_in_epoch % max(grad_accum, 1) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    denom = max(n_images, 1)
    return {
        **{k: v / denom for k, v in totals.items()},
        "rmse_db": rmse_accum / denom,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = Try76Cfg.load(args.config)
    set_seed(cfg.seed)

    ddp, rank, world, device = init_distributed(cfg.runtime.device)
    main_ = is_main(rank)

    cfg.runtime.output_dir.mkdir(parents=True, exist_ok=True)
    if main_:
        print(
            f"[rank0] code_marker={CODE_MARKER} "
            f"device={device} "
            f"no_data_mask_column={cfg.data.path_loss_no_data_mask_column!r} "
            f"derive_no_data_from_non_ground={cfg.data.derive_no_data_from_non_ground}"
        )
        with open(cfg.runtime.output_dir / "resolved_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "code_marker": CODE_MARKER,
                "seed": cfg.seed,
                "data": {**asdict(cfg.data), "hdf5_path": str(cfg.data.hdf5_path)},
                "model": asdict(cfg.model),
                "training": asdict(cfg.training),
                "runtime": {**asdict(cfg.runtime), "output_dir": str(cfg.runtime.output_dir),
                            "resume_checkpoint": str(cfg.runtime.resume_checkpoint) if cfg.runtime.resume_checkpoint else None},
            }, f, indent=2)

    data_cfg = Try76Config(
        hdf5_path=cfg.data.hdf5_path,
        topology_class=cfg.data.topology_class,
        region_mode=cfg.data.region_mode,
        image_size=cfg.data.image_size,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        split_seed=cfg.data.split_seed,
        path_loss_no_data_mask_column=cfg.data.path_loss_no_data_mask_column,
        derive_no_data_from_non_ground=cfg.data.derive_no_data_from_non_ground,
    )
    train_ds, val_ds, test_ds = build_expert_datasets(data_cfg)
    if main_:
        print(f"[rank0] expert={cfg.data.topology_class}_{cfg.data.region_mode}  "
              f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    def make_loader(ds: Try76ExpertDataset, train: bool) -> DataLoader:
        sampler = DistributedSampler(ds, shuffle=train, drop_last=False) if ddp else None
        return DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=(sampler is None and train),
            sampler=sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=cfg.training.num_workers > 0,
        )

    train_loader = make_loader(train_ds, train=True)
    val_loader = make_loader(val_ds, train=False)

    model_cfg = Try76ModelConfig(
        in_channels=4,
        cond_dim=64,
        height_embed_dim=32,
        base_width=cfg.model.base_width,
        K=cfg.model.K,
        clamp_lo=cfg.model.clamp_lo,
        clamp_hi=cfg.model.clamp_hi,
        outlier_sigma_floor=cfg.model.outlier_sigma_floor,
    )
    model = Try76Model(model_cfg).to(device)
    if ddp:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    if cfg.runtime.resume_checkpoint and cfg.runtime.resume_checkpoint.exists():
        state = torch.load(cfg.runtime.resume_checkpoint, map_location=device)
        (model.module if isinstance(model, DistributedDataParallel) else model).load_state_dict(state["model"], strict=False)
        if main_:
            print(f"[rank0] resumed from {cfg.runtime.resume_checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    height_embed = HeightEmbedding()

    best_score = float("inf")
    patience_ctr = 0
    history: List[Dict] = []

    for epoch in range(cfg.training.epochs):
        if ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        lr = cfg.training.lr * lr_lambda(epoch, cfg.training.warmup_epochs, cfg.training.epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        t0 = time.time()
        train_stats = run_epoch(
            model, train_loader, optimizer, device, height_embed,
            LossWeights(), cfg.model.clamp_lo, cfg.model.clamp_hi,
            cfg.training.grad_accum_steps, is_train=True, is_main_=main_,
        )
        val_stats = run_epoch(
            model, val_loader, None, device, height_embed,
            LossWeights(), cfg.model.clamp_lo, cfg.model.clamp_hi,
            cfg.training.grad_accum_steps, is_train=False, is_main_=main_,
        )
        # Overall objective is RMSE-in-dB; NLL/KL act as regularisers so the
        # model does not collapse on the distribution shape.
        score = val_stats["rmse_db"] + 0.25 * val_stats["dist_kl"] + 0.1 * val_stats["map_nll"]
        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "lr": lr,
            "train": train_stats,
            "val": val_stats,
            "score": score,
            "elapsed_s": dt,
        }
        history.append(row)

        if main_:
            print(f"[epoch {epoch:3d}] lr={lr:.2e}  train RMSE={train_stats['rmse_db']:.2f}  "
                  f"val RMSE={val_stats['rmse_db']:.2f}  val map_nll={val_stats['map_nll']:.3f}  "
                  f"val dist_kl={val_stats['dist_kl']:.3f}  time={dt:.1f}s")
            with open(cfg.runtime.output_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if score < best_score - 1e-4:
                best_score = score
                patience_ctr = 0
                ckpt = {
                    "epoch": epoch,
                    "model": (model.module if isinstance(model, DistributedDataParallel) else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_cfg": asdict(model_cfg),
                    "score": score,
                }
                torch.save(ckpt, cfg.runtime.output_dir / "best_model.pt")
            else:
                patience_ctr += 1

        maybe_barrier(ddp)
        patience_flag = torch.tensor([patience_ctr >= cfg.training.patience], device=device, dtype=torch.int32)
        if ddp:
            dist.broadcast(patience_flag, src=0)
        if int(patience_flag.item()) == 1:
            if main_:
                print(f"[rank0] early stopping at epoch {epoch} (patience {cfg.training.patience})")
            break

    if ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
