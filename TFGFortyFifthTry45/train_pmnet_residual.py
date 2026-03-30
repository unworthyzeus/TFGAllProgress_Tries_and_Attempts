from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config_utils import (
    anchor_data_paths_to_config_file,
    ensure_output_dir,
    is_cuda_device,
    load_config,
    load_torch_checkpoint,
    move_optimizer_state_to_device,
    resolve_device,
)
from data_utils import (
    _antenna_height_bin,
    _city_type_from_thresholds,
    build_dataset_splits_from_config,
    compute_input_channels,
    forward_cgan_generator,
    return_scalar_cond_from_config,
    unpack_cgan_batch,
)
from model_pmnet import PMNetResidualRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_init_distributed(device: object) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    if not dist.is_initialized():
        backend = "nccl" if is_cuda_device(device) else "gloo"
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if is_cuda_device(device):
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    return True, rank, local_rank, dist.get_world_size()


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier_if_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def formula_channel_index(cfg: Dict[str, Any]) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):
        idx += 1
    if cfg["data"].get("distance_map_channel", False):
        idx += 1
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    if not bool(formula_cfg.get("enabled", False)):
        raise ValueError("Try 45 requires data.path_loss_formula_input.enabled = true")
    return idx


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


def denormalize(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    return values * scale + offset


def masked_mse_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    mse_weight: float,
    l1_weight: float,
) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    mse = (((pred - target) ** 2) * mask).sum() / denom
    l1 = (torch.abs(pred - target) * mask).sum() / denom
    return float(mse_weight) * mse + float(l1_weight) * l1


def compute_multiscale_path_loss_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    metadata: Dict[str, Any],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    ms_cfg = dict(cfg.get("multiscale_path_loss", {}))
    if not bool(ms_cfg.get("enabled", False)):
        return torch.tensor(0.0, device=pred.device)

    scales = [int(scale) for scale in ms_cfg.get("scales", [2, 4]) if int(scale) > 1]
    if not scales:
        return torch.tensor(0.0, device=pred.device)

    raw_weights = list(ms_cfg.get("weights", []))
    if raw_weights and len(raw_weights) != len(scales):
        raw_weights = raw_weights[: len(scales)]
    if not raw_weights:
        raw_weights = [1.0] * len(scales)

    pred_db = denormalize(pred, metadata)
    target_db = denormalize(target, metadata)
    scale_db = max(float(metadata.get("scale", 180.0)), 1.0)
    min_valid_ratio = float(ms_cfg.get("min_valid_ratio", 0.5))
    loss_weight = float(ms_cfg.get("loss_weight", 0.5))

    total = torch.tensor(0.0, device=pred.device)
    total_weight = 0.0
    for factor, weight in zip(scales, raw_weights):
        pred_ds = F.avg_pool2d(pred_db, kernel_size=factor, stride=factor)
        target_ds = F.avg_pool2d(target_db, kernel_size=factor, stride=factor)
        mask_ds = F.avg_pool2d(mask, kernel_size=factor, stride=factor)
        valid = (mask_ds >= min_valid_ratio).float()
        denom = valid.sum().clamp_min(1.0)
        mse_ds = (((pred_ds - target_ds) ** 2) * valid).sum() / denom
        total = total + float(weight) * (mse_ds / (scale_db ** 2))
        total_weight += float(weight)

    if total_weight <= 0.0:
        return torch.tensor(0.0, device=pred.device)
    return loss_weight * (total / total_weight)


def build_optimizer(cfg: Dict[str, Any], params: Iterable[torch.nn.Parameter], device: object) -> torch.optim.Optimizer:
    optimizer_name = str(cfg["training"].get("optimizer", "adamw")).lower()
    learning_rate = float(cfg["training"].get("learning_rate", cfg["training"].get("generator_lr", 3e-5)))
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    momentum = float(cfg["training"].get("momentum", 0.0))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            foreach=is_cuda_device(device),
        )
    if optimizer_name == "adam":
        betas = (
            float(cfg["training"].get("beta1", 0.9)),
            float(cfg["training"].get("beta2", 0.999)),
        )
        return torch.optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            foreach=is_cuda_device(device),
        )
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=is_cuda_device(device),
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    name = str(cfg["training"].get("lr_scheduler", "none")).lower()
    if name in {"", "none", "off"}:
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg["training"].get("lr_scheduler_factor", 0.5)),
            patience=int(cfg["training"].get("lr_scheduler_patience", 4)),
            min_lr=float(cfg["training"].get("lr_scheduler_min_lr", 1e-6)),
        )
    raise ValueError(f"Unsupported lr_scheduler '{name}'.")


def resolve_resume_checkpoint(out_dir: Path, configured_resume: str | None) -> Optional[Path]:
    if configured_resume:
        candidate = Path(configured_resume)
        return candidate if candidate.exists() else None
    best = out_dir / "best_cgan.pt"
    if best.exists():
        return best
    epoch_candidates = sorted(out_dir.glob("epoch_*_cgan.pt"))
    if epoch_candidates:
        return epoch_candidates[-1]
    return None


class RegimeAnnotator:
    def __init__(self, dataset: Any, calibration_json: Optional[str]) -> None:
        self.dataset = dataset
        self.hdf5_path = str(dataset.hdf5_path)
        self.input_column = str(dataset.input_column)
        self.non_ground_threshold = float(dataset.non_ground_threshold)
        self.los_input_column = dataset.los_input_column
        self.image_size = int(dataset.image_size)
        self.calibration = None
        if calibration_json:
            cal_path = Path(__file__).resolve().parent / calibration_json
            if cal_path.exists():
                self.calibration = json.loads(cal_path.read_text(encoding="utf-8"))
        self._cache: dict[tuple[str, str], dict[str, str]] = {}

    def _compute_info(self, city: str, sample: str) -> dict[str, str]:
        key = (city, sample)
        if key in self._cache:
            return self._cache[key]
        with h5py.File(self.hdf5_path, "r") as handle:
            grp = handle[city][sample]
            raw_topology = np.asarray(grp[self.input_column][...], dtype=np.float32)
        non_ground = raw_topology != self.non_ground_threshold
        building_density = float(np.mean(non_ground))
        non_zero = raw_topology[non_ground]
        mean_height = float(np.mean(non_zero)) if non_zero.size else 0.0
        antenna_height_m = float(self.dataset._resolve_hdf5_scalar_value(city, sample, "antenna_height_m"))

        city_type = None
        ant_bin = "mid_ant"
        if self.calibration:
            city_type = dict(self.calibration.get("city_type_by_city", {})).get(city)
            if city_type is None:
                city_type = _city_type_from_thresholds(
                    building_density,
                    mean_height,
                    dict(self.calibration.get("city_type_thresholds", {})),
                )
            ant_bin = _antenna_height_bin(antenna_height_m, dict(self.calibration.get("antenna_height_thresholds", {})))
        if city_type is None:
            city_type = "unknown_city_type"
        info = {
            "city_type": city_type,
            "antenna_bin": ant_bin,
            "city": city,
        }
        self._cache[key] = info
        return info

    def info_for_index(self, idx: int) -> dict[str, str]:
        city, sample = self.dataset.sample_refs[idx]
        return self._compute_info(city, sample)


def init_metric_totals() -> dict[str, float]:
    return {
        "count": 0.0,
        "sum_squared_error": 0.0,
        "sum_absolute_error": 0.0,
    }


def update_metric_total(totals: dict[str, float], diff_phys: torch.Tensor) -> None:
    if diff_phys.numel() == 0:
        return
    totals["count"] += float(diff_phys.numel())
    totals["sum_squared_error"] += float(torch.sum(diff_phys ** 2).item())
    totals["sum_absolute_error"] += float(torch.sum(torch.abs(diff_phys)).item())


def finalize_metric_total(totals: dict[str, float], unit: str) -> dict[str, float]:
    count = totals["count"]
    if count <= 0.0:
        return {"mse_physical": float("nan"), "rmse_physical": float("nan"), "mae_physical": float("nan"), "unit": unit}
    mse = totals["sum_squared_error"] / count
    return {
        "mse_physical": float(mse),
        "rmse_physical": float(math.sqrt(mse)),
        "mae_physical": float(totals["sum_absolute_error"] / count),
        "unit": unit,
    }


def evaluate_validation(
    model: nn.Module,
    dataset: Any,
    device: object,
    cfg: Dict[str, Any],
    amp_enabled: bool,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=is_cuda_device(device),
        persistent_workers=False,
    )
    formula_idx = formula_channel_index(cfg)
    meta = dict(cfg["target_metadata"]["path_loss"])
    unit = str(meta.get("unit", "dB"))
    prior_cfg = dict(cfg.get("prior_residual_path_loss", {}))
    clamp_final = bool(prior_cfg.get("clamp_final_output", True))
    annotator = RegimeAnnotator(dataset, dict(cfg["data"].get("path_loss_formula_input", {})).get("regime_calibration_json"))

    totals = init_metric_totals()
    prior_totals = init_metric_totals()
    regime_totals: dict[str, dict[str, float]] = defaultdict(init_metric_totals)

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="val", leave=False)):
            x, y, m, scalar_cond = unpack_cgan_batch(batch, device)
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                residual_pred = forward_cgan_generator(model, x, scalar_cond)
            prior = x[:, formula_idx : formula_idx + 1]
            pred = prior + residual_pred
            if clamp_final:
                pred = clip_to_target_range(pred, meta)

            pred_phys = denormalize(pred, meta)
            target_phys = denormalize(y[:, :1], meta)
            valid_mask = m[:, :1] > 0.0
            diff_phys = (pred_phys - target_phys)[valid_mask]
            update_metric_total(totals, diff_phys)

            prior_phys = denormalize(prior, meta)
            prior_diff_phys = (prior_phys - target_phys)[valid_mask]
            update_metric_total(prior_totals, prior_diff_phys)

            info = annotator.info_for_index(idx)
            los = x[:, 1:2] if cfg["data"].get("los_input_column") else None
            if los is not None:
                los_valid = valid_mask & (los > 0.5)
                nlos_valid = valid_mask & (los <= 0.5)
                update_metric_total(regime_totals["path_loss__los__LoS"], (pred_phys - target_phys)[los_valid])
                update_metric_total(regime_totals["path_loss__los__NLoS"], (pred_phys - target_phys)[nlos_valid])
                update_metric_total(regime_totals["path_loss__prior__los__LoS"], (prior_phys - target_phys)[los_valid])
                update_metric_total(regime_totals["path_loss__prior__los__NLoS"], (prior_phys - target_phys)[nlos_valid])

            city_key = f"path_loss__city_type__{info['city_type']}"
            ant_key = f"path_loss__antenna_bin__{info['antenna_bin']}"
            update_metric_total(regime_totals[city_key], diff_phys)
            update_metric_total(regime_totals[f"path_loss__prior__city_type__{info['city_type']}"], prior_diff_phys)
            update_metric_total(regime_totals[ant_key], diff_phys)
            update_metric_total(regime_totals[f"path_loss__prior__antenna_bin__{info['antenna_bin']}"], prior_diff_phys)

            if los is not None:
                los_valid = valid_mask & (los > 0.5)
                nlos_valid = valid_mask & (los <= 0.5)
                combo_los = f"path_loss__calibration_regime__{info['city_type']}__LoS__{info['antenna_bin']}"
                combo_nlos = f"path_loss__calibration_regime__{info['city_type']}__NLoS__{info['antenna_bin']}"
                combo_prior_los = f"path_loss__prior__calibration_regime__{info['city_type']}__LoS__{info['antenna_bin']}"
                combo_prior_nlos = f"path_loss__prior__calibration_regime__{info['city_type']}__NLoS__{info['antenna_bin']}"
                update_metric_total(regime_totals[combo_los], (pred_phys - target_phys)[los_valid])
                update_metric_total(regime_totals[combo_nlos], (pred_phys - target_phys)[nlos_valid])
                update_metric_total(regime_totals[combo_prior_los], (prior_phys - target_phys)[los_valid])
                update_metric_total(regime_totals[combo_prior_nlos], (prior_phys - target_phys)[nlos_valid])

    summary: dict[str, Any] = {
        "path_loss": finalize_metric_total(totals, unit),
        "path_loss__prior__overall": finalize_metric_total(prior_totals, unit),
        "_regimes": {key: finalize_metric_total(val, unit) for key, val in sorted(regime_totals.items())},
    }
    return summary


def extract_selection_metric(summary: dict[str, Any], metric_path: str) -> float:
    parts = [p for p in metric_path.split(".") if p]
    value: Any = summary
    for part in parts:
        if not isinstance(value, dict):
            raise KeyError(metric_path)
        value = value[part]
    if not isinstance(value, (int, float)):
        raise TypeError(f"Selection metric {metric_path} is not numeric")
    return float(value)


def write_validation_json(out_dir: Path, epoch: int, summary: dict[str, Any], *, best_epoch: int, best_score: float) -> None:
    payload = dict(summary)
    payload["_checkpoint"] = {
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
    }
    latest = out_dir / "validate_metrics_cgan_latest.json"
    epoch_path = out_dir / f"validate_metrics_epoch_{epoch}_cgan.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    epoch_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: object,
    cfg: Dict[str, Any],
    amp_enabled: bool,
) -> float:
    formula_idx = formula_channel_index(cfg)
    meta = dict(cfg["target_metadata"]["path_loss"])
    loss_cfg = dict(cfg["loss"])
    residual_cfg = dict(cfg.get("prior_residual_path_loss", {}))
    moe_cfg = dict(cfg.get("moe_residual", {}))
    clamp_final = bool(residual_cfg.get("clamp_final_output", True))
    clip_grad = float(cfg["training"].get("clip_grad_norm", 0.0))

    model.train()
    running = 0.0
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x, y, m, scalar_cond = unpack_cgan_batch(batch, device)
        target = y[:, :1]
        mask = m[:, :1]
        prior = x[:, formula_idx : formula_idx + 1]
        residual_target = target - prior

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=amp_enabled):
            residual_pred = forward_cgan_generator(model, x, scalar_cond)
            pred = prior + residual_pred
            if clamp_final:
                pred = clip_to_target_range(pred, meta)

            final_loss = masked_mse_l1_loss(
                pred,
                target,
                mask,
                mse_weight=float(loss_cfg.get("mse_weight", 1.0)),
                l1_weight=float(loss_cfg.get("l1_weight", 0.0)),
            )
            residual_loss = masked_mse_l1_loss(
                residual_pred,
                residual_target,
                mask,
                mse_weight=float(residual_cfg.get("mse_weight", 1.0)),
                l1_weight=float(residual_cfg.get("l1_weight", 0.0)),
            )
            multiscale_loss = compute_multiscale_path_loss_loss(pred, target, mask, meta, cfg)
            total_loss = (
                float(loss_cfg.get("lambda_recon", 1.0)) * final_loss
                + float(residual_cfg.get("loss_weight", 0.0)) * residual_loss
                + multiscale_loss
            )
            balance_weight = float(moe_cfg.get("balance_loss_weight", 0.0))
            if balance_weight > 0.0:
                inner_model = model.module if isinstance(model, DistributedDataParallel) else model
                if hasattr(inner_model, "moe_balance_loss"):
                    total_loss = total_loss + balance_weight * inner_model.moe_balance_loss(mask)

        scaler.scale(total_loss).backward()
        if clip_grad > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        running += float(total_loss.item())
        steps += 1
    return running / max(steps, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Try 45 PMNet-MoE residual path-loss model")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg["runtime"]["device"])
    distributed, rank, local_rank, world_size = maybe_init_distributed(device)
    if distributed and is_cuda_device(device):
        device = torch.device("cuda", local_rank)

    out_dir = ensure_output_dir(cfg["runtime"]["output_dir"])
    if is_main_process(rank):
        print(json.dumps({"output_dir": str(out_dir), "world_size": world_size}))

    splits = build_dataset_splits_from_config(cfg)
    train_dataset = splits["train"]
    val_dataset = splits["val"]
    if return_scalar_cond_from_config(cfg):
        raise ValueError("Try 45 expects scalar channels, not scalar FiLM vectors.")

    pin_memory = is_cuda_device(device)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=pin_memory,
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
    )

    model = PMNetResidualRegressor(
        in_channels=compute_input_channels(cfg),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"].get("base_channels", 64)),
        encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        num_experts=int(cfg["model"].get("num_experts", 4)),
        expert_channels=int(cfg["model"].get("expert_channels", cfg["model"].get("base_channels", 64))),
    ).to(device)

    model_for_training: nn.Module = model
    if distributed:
        model_for_training = DistributedDataParallel(
            model,
            device_ids=[local_rank] if is_cuda_device(device) else None,
            output_device=local_rank if is_cuda_device(device) else None,
            find_unused_parameters=False,
        )

    optimizer = build_optimizer(cfg, model_for_training.parameters(), device)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(enabled=bool(cfg["training"].get("amp", True)) and is_cuda_device(device))

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    resume_path = resolve_resume_checkpoint(out_dir, cfg["runtime"].get("resume_checkpoint"))
    if resume_path and resume_path.exists():
        state = load_torch_checkpoint(resume_path, device)
        target_model = model_for_training.module if isinstance(model_for_training, DistributedDataParallel) else model_for_training
        target_model.load_state_dict(state["model"] if "model" in state else state["generator"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        if "scheduler" in state and scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state and scaler is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_score = float(state.get("best_score", best_score))
        best_epoch = int(state.get("best_epoch", best_epoch))
        if is_main_process(rank):
            print(json.dumps({"resume_from": str(resume_path), "start_epoch": start_epoch}))

    amp_enabled = bool(cfg["training"].get("amp", True)) and is_cuda_device(device)
    selection_metric = next(iter(dict(cfg["training"].get("selection_metrics", {"path_loss.rmse_physical": 1.0})).keys()))

    try:
        for epoch in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(model_for_training, train_loader, optimizer, scaler, device, cfg, amp_enabled)
            barrier_if_distributed(distributed)

            if is_main_process(rank):
                val_summary = evaluate_validation(model, val_dataset, device, cfg, amp_enabled)
                val_summary["_train"] = {"loss": float(train_loss)}
                current_score = extract_selection_metric(val_summary, selection_metric)
                if current_score < best_score:
                    best_score = current_score
                    best_epoch = epoch
                    best_payload = {
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "model": model.state_dict(),
                        "generator": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "scaler": scaler.state_dict(),
                        "config_path": args.config,
                    }
                    #torch.save(best_payload, out_dir / "best_cgan.pt")
                    torch.save(best_payload, out_dir / "best_model.pt")

                if epoch % int(cfg["training"].get("save_every", 5)) == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_score": best_score,
                            "model": model.state_dict(),
                            "generator": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler is not None else None,
                            "scaler": scaler.state_dict(),
                            "config_path": args.config,
                        },
                        out_dir / f"epoch_{epoch}_cgan.pt",
                    )

                write_validation_json(out_dir, epoch, val_summary, best_epoch=best_epoch, best_score=best_score)
                print(json.dumps({"epoch": epoch, "train_loss": train_loss, selection_metric: current_score}))
            else:
                current_score = 0.0

            if scheduler is not None:
                if distributed:
                    score_tensor = torch.tensor([float(current_score)], device=device if is_cuda_device(device) else "cpu")
                    dist.broadcast(score_tensor, src=0)
                    current_score = float(score_tensor.item())
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_score)
                else:
                    scheduler.step()

            barrier_if_distributed(distributed)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
