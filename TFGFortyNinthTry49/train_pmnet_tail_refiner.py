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
from torch.utils.data import DataLoader, Dataset, Sampler
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
    return_scalar_cond_from_config,
)
from model_pmnet import PMNetResidualRegressor, UNetResidualRefiner


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
    ms_cfg = dict(cfg.get("loss", {}).get("multiscale_path_loss", cfg.get("multiscale_path_loss", {})))
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


def formula_channel_index(cfg: Dict[str, Any]) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):
        idx += 1
    if cfg["data"].get("distance_map_channel", False):
        idx += 1
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    if not bool(formula_cfg.get("enabled", False)):
        raise ValueError("Tail refiner expects data.path_loss_formula_input.enabled = true")
    return idx


def _scalar_feature_names(cfg: Dict[str, Any]) -> list[str]:
    names = list(cfg["data"].get("scalar_feature_columns", []))
    names.extend(list(dict(cfg["data"].get("constant_scalar_features", {})).keys()))
    return names


def _extract_scalar_channel(base_input: torch.Tensor, cfg: Dict[str, Any], name: str) -> Optional[torch.Tensor]:
    if not bool(cfg.get("model", {}).get("use_scalar_channels", False)):
        return None
    names = _scalar_feature_names(cfg)
    if not names or name not in names:
        return None
    scalar_count = len(names)
    if scalar_count <= 0:
        return None
    input_without_stage1 = base_input.shape[1] - 1
    if input_without_stage1 < scalar_count:
        return None
    scalar_start = input_without_stage1 - scalar_count
    idx = scalar_start + names.index(name)
    if idx < 0 or idx >= input_without_stage1:
        return None
    return base_input[:, idx : idx + 1]


def _tail_focus_weights(
    abs_error_db: torch.Tensor,
    base_input: torch.Tensor,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    tail_cfg = dict(cfg.get("tail_refiner", {})).get("tail_focus", {})
    if not bool(tail_cfg.get("enabled", False)):
        return torch.ones_like(abs_error_db)

    threshold_db = max(float(tail_cfg.get("threshold_db", 6.0)), 1e-3)
    temperature_db = max(float(tail_cfg.get("temperature_db", 2.5)), 1e-3)
    alpha = max(float(tail_cfg.get("alpha", 1.0)), 0.0)
    nlos_boost = max(float(tail_cfg.get("nlos_boost", 0.0)), 0.0)
    antenna_boost = max(float(tail_cfg.get("antenna_boost", 0.0)), 0.0)
    max_weight = max(float(tail_cfg.get("max_weight", 5.0)), 1.0)

    weight = 1.0 + alpha * torch.sigmoid((abs_error_db - threshold_db) / temperature_db)

    if nlos_boost > 0.0 and cfg["data"].get("los_input_column"):
        los_idx = 1
        if base_input.shape[1] > los_idx:
            nlos = (1.0 - base_input[:, los_idx : los_idx + 1].clamp(0.0, 1.0)).clamp(0.0, 1.0)
            weight = weight * (1.0 + nlos_boost * nlos)

    if antenna_boost > 0.0:
        antenna_channel = _extract_scalar_channel(base_input, cfg, "antenna_height_m")
        if antenna_channel is not None:
            low_antenna = (1.0 - antenna_channel.clamp(0.0, 1.0)).clamp(0.0, 1.0)
            weight = weight * (1.0 + antenna_boost * low_antenna)

    return weight.clamp(1.0, max_weight)


def _gate_target_from_error(abs_error_db: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    gate_cfg = dict(cfg.get("tail_refiner", {})).get("gate_target", {})
    threshold_db = max(float(gate_cfg.get("threshold_db", 6.0)), 1e-3)
    temperature_db = max(float(gate_cfg.get("temperature_db", 2.0)), 1e-3)
    return torch.sigmoid((abs_error_db - threshold_db) / temperature_db)


class Stage1OutputDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        stage1_hdf5_path: str,
        *,
        prediction_key: str = "stage1_pred_norm_f16",
        abs_error_key: str = "stage1_abs_error_db_f16",
    ) -> None:
        self.base_dataset = base_dataset
        self.sample_refs = list(getattr(base_dataset, "sample_refs", []))
        self.stage1_hdf5_path = Path(stage1_hdf5_path)
        self.prediction_key = prediction_key
        self.abs_error_key = abs_error_key
        self._handle: Optional[h5py.File] = None
        if not self.stage1_hdf5_path.exists():
            raise FileNotFoundError(f"Missing stage1 HDF5 outputs: {self.stage1_hdf5_path}")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _get_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.stage1_hdf5_path, "r")
        return self._handle

    def _read_stage1_maps(self, city: str, sample: str) -> tuple[torch.Tensor, torch.Tensor]:
        handle = self._get_handle()
        grp = handle[city][sample]
        pred_arr = np.asarray(grp[self.prediction_key][...], dtype=np.float32)
        abs_error_arr = np.asarray(grp[self.abs_error_key][...], dtype=np.float32) if self.abs_error_key in grp else np.zeros_like(pred_arr, dtype=np.float32)
        if pred_arr.ndim == 2:
            pred_arr = pred_arr[None, ...]
        if abs_error_arr.ndim == 2:
            abs_error_arr = abs_error_arr[None, ...]
        pred = torch.from_numpy(pred_arr.astype(np.float32, copy=False))
        abs_error = torch.from_numpy(abs_error_arr.astype(np.float32, copy=False))
        return pred, abs_error

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.base_dataset[idx]
        if len(item) == 4:
            raise ValueError("Tail refiner expects scalar channels, not FiLM vectors.")
        x, y, m = item
        city, sample = self.sample_refs[idx]
        stage1_pred, abs_error = self._read_stage1_maps(city, sample)
        x = torch.cat([x, stage1_pred], dim=0)
        return x, y, m, abs_error


class DistributedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        weights: Sequence[float],
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        seed: int = 42,
    ) -> None:
        self.weights = torch.as_tensor(list(weights), dtype=torch.double)
        if self.weights.numel() == 0:
            raise ValueError("DistributedWeightedSampler requires at least one weight")
        self.num_replicas = int(num_replicas if num_replicas is not None else dist.get_world_size() if dist.is_initialized() else 1)
        self.rank = int(rank if rank is not None else dist.get_rank() if dist.is_initialized() else 0)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / max(self.num_replicas, 1)))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(self.weights, self.total_size, self.replacement, generator=generator).tolist()
        shard = sampled[self.rank : self.total_size : self.num_replicas]
        return iter(shard)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


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


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
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


def init_metric_totals() -> Dict[str, float]:
    return {"sum_sq": 0.0, "sum_abs": 0.0, "count": 0.0}


def update_metric_total(accum: Dict[str, float], diffs: torch.Tensor) -> None:
    if diffs.numel() == 0:
        return
    accum["sum_sq"] += float((diffs * diffs).sum().item())
    accum["sum_abs"] += float(diffs.abs().sum().item())
    accum["count"] += float(diffs.numel())


def finalize_metric_total(totals: Dict[str, float], unit: str = "dB") -> Dict[str, Any]:
    count = max(float(totals.get("count", 0.0)), 1.0)
    mse = float(totals.get("sum_sq", 0.0)) / count
    mae = float(totals.get("sum_abs", 0.0)) / count
    return {
        "mse_physical": mse,
        "rmse_physical": math.sqrt(max(mse, 0.0)),
        "mae_physical": mae,
        "unit": unit,
    }


def extract_selection_metric(summary: Dict[str, Any], metric_path: str) -> float:
    parts = [part for part in metric_path.split(".") if part]
    value: Any = summary
    for part in parts:
        if not isinstance(value, dict):
            raise KeyError(metric_path)
        value = value[part]
    if not isinstance(value, (int, float)):
        raise TypeError(f"Selection metric {metric_path} is not numeric")
    return float(value)


def write_validation_json(out_dir: Path, epoch: int, summary: Dict[str, Any], *, best_epoch: int, best_score: float) -> None:
    payload = dict(summary)
    payload["_checkpoint"] = {
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
    }
    latest = out_dir / "validate_metrics_tail_refiner_latest.json"
    epoch_path = out_dir / f"validate_metrics_epoch_{epoch}_tail_refiner.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    epoch_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_tail_refiner_model(cfg: Dict[str, Any], in_channels: int) -> nn.Module:
    tail_cfg = dict(cfg.get("tail_refiner", {}))
    refiner_arch = str(tail_cfg.get("refiner_arch", "unet")).lower()
    out_channels = int(cfg["model"].get("out_channels", 1))
    if refiner_arch == "pmnet":
        return PMNetResidualRegressor(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(tail_cfg.get("refiner_base_channels", cfg["model"].get("base_channels", 80))),
            encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
            context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
            norm_type=str(cfg["model"].get("norm_type", "group")),
            dropout=float(cfg["model"].get("dropout", 0.0)),
            gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        )
    if refiner_arch == "unet":
        return UNetResidualRefiner(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(tail_cfg.get("refiner_base_channels", 96)),
            norm_type=str(cfg["model"].get("norm_type", "group")),
            dropout=float(cfg["model"].get("dropout", 0.0)),
            gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        )
    raise ValueError(f"Unsupported tail_refiner.refiner_arch '{refiner_arch}'")


def build_sample_weights(dataset: Stage1OutputDataset, cfg: Dict[str, Any]) -> torch.Tensor:
    tail_cfg = dict(cfg.get("tail_refiner", {}))
    oversample_cfg = dict(tail_cfg.get("oversample", {}))
    if not bool(oversample_cfg.get("enabled", False)):
        return torch.ones(len(dataset), dtype=torch.double)

    threshold_db = max(float(oversample_cfg.get("threshold_db", 6.0)), 1e-3)
    temperature_db = max(float(oversample_cfg.get("temperature_db", 2.5)), 1e-3)
    alpha = max(float(oversample_cfg.get("alpha", 1.0)), 0.0)
    nlos_boost = max(float(oversample_cfg.get("nlos_boost", 0.0)), 0.0)
    antenna_boost = max(float(oversample_cfg.get("antenna_boost", 0.0)), 0.0)
    weights: list[float] = []

    for index in range(len(dataset)):
        x, _, m, abs_error = dataset[index]
        valid = (m[:, :1] > 0.0).to(dtype=torch.float32)
        denom = float(valid.sum().item()) if float(valid.sum().item()) > 0 else float(abs_error.numel())
        mean_error = float((abs_error * valid).sum().item()) / max(denom, 1.0)
        weight = 1.0 + alpha * float(torch.sigmoid(torch.tensor((mean_error - threshold_db) / temperature_db)).item())

        if nlos_boost > 0.0 and cfg["data"].get("los_input_column"):
            los_idx = 1
            if x.shape[1] > los_idx:
                nlos_frac = float((1.0 - x[:, los_idx : los_idx + 1].clamp(0.0, 1.0)).mean().item())
                weight *= 1.0 + nlos_boost * nlos_frac

        if antenna_boost > 0.0:
            antenna_channel = _extract_scalar_channel(x, cfg, "antenna_height_m")
            if antenna_channel is not None:
                low_antenna = float((1.0 - antenna_channel.clamp(0.0, 1.0)).mean().item())
                weight *= 1.0 + antenna_boost * low_antenna

        weights.append(max(weight, 1e-6))

    return torch.tensor(weights, dtype=torch.double)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: object,
    cfg: Dict[str, Any],
    amp_enabled: bool,
) -> float:
    tail_cfg = dict(cfg.get("tail_refiner", {}))
    loss_cfg = dict(cfg.get("training", {}))
    meta = dict(cfg["target_metadata"]["path_loss"])
    use_gate = bool(tail_cfg.get("use_gate", False))
    gate_loss_weight = float(tail_cfg.get("gate_loss_weight", 0.0))
    residual_weight = float(tail_cfg.get("residual_weight", 1.0))
    final_weight = float(tail_cfg.get("final_weight", 0.5))
    clip_grad = float(cfg["training"].get("clip_grad_norm", 1.0))
    mse_weight = float(cfg["loss"].get("mse_weight", 1.0))
    l1_weight = float(cfg["loss"].get("l1_weight", 0.1))

    model.train()
    running_loss = 0.0
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x, y, m, abs_error_db = batch
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        abs_error_db = abs_error_db.to(device)

        target = y[:, :1]
        mask = m[:, :1]
        stage1_pred = x[:, -1:]
        residual_target = target - stage1_pred

        weighted_mask = mask * _tail_focus_weights(abs_error_db, x, cfg)
        gate_target = _gate_target_from_error(abs_error_db, cfg) if use_gate else None

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=amp_enabled):
            refiner_out = model(x)
            if use_gate:
                if refiner_out.shape[1] < 2:
                    raise ValueError("tail_refiner.use_gate=true requires model.out_channels >= 2")
                delta = refiner_out[:, :1] * torch.sigmoid(refiner_out[:, 1:2])
                gate_logits = refiner_out[:, 1:2]
            else:
                delta = refiner_out[:, :1]
                gate_logits = None

            final_pred = clip_to_target_range(stage1_pred + delta, meta)
            final_loss = masked_mse_l1_loss(final_pred, target, weighted_mask, mse_weight=mse_weight, l1_weight=l1_weight)
            residual_loss = masked_mse_l1_loss(delta, residual_target, weighted_mask, mse_weight=mse_weight, l1_weight=l1_weight)
            multiscale_loss = compute_multiscale_path_loss_loss(final_pred, target, weighted_mask, meta, cfg)
            total_loss = residual_weight * residual_loss + final_weight * final_loss + multiscale_loss

            if use_gate and gate_logits is not None and gate_target is not None:
                gate_loss_map = F.binary_cross_entropy_with_logits(gate_logits, gate_target, reduction="none")
                gate_loss = (gate_loss_map * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
                total_loss = total_loss + gate_loss_weight * gate_loss

        scaler.scale(total_loss).backward()
        if clip_grad > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(total_loss.item())
        steps += 1

    return running_loss / max(steps, 1)


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    device: object,
    cfg: Dict[str, Any],
    amp_enabled: bool,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    meta = dict(cfg["target_metadata"]["path_loss"])
    tail_cfg = dict(cfg.get("tail_refiner", {}))
    use_gate = bool(tail_cfg.get("use_gate", False))

    totals = init_metric_totals()
    stage1_totals = init_metric_totals()
    los_totals = init_metric_totals()
    nlos_totals = init_metric_totals()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False, disable=distributed and not is_main_process(rank)):
            x, y, m, abs_error_db = batch
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            abs_error_db = abs_error_db.to(device)
            target = y[:, :1]
            mask = m[:, :1]
            stage1_pred = x[:, -1:]
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                refiner_out = model(x)
                if use_gate:
                    delta = refiner_out[:, :1] * torch.sigmoid(refiner_out[:, 1:2])
                else:
                    delta = refiner_out[:, :1]
                final_pred = clip_to_target_range(stage1_pred + delta, meta)

            pred_phys = denormalize(final_pred, meta)
            stage1_phys = denormalize(stage1_pred, meta)
            target_phys = denormalize(target[:, :1], meta)
            valid_mask = mask[:, :1] > 0.0

            update_metric_total(totals, (pred_phys - target_phys)[valid_mask])
            update_metric_total(stage1_totals, (stage1_phys - target_phys)[valid_mask])

            if cfg["data"].get("los_input_column") and x.shape[1] > 1:
                los = x[:, 1:2]
                los_valid = valid_mask & (los > 0.5)
                nlos_valid = valid_mask & (los <= 0.5)
                update_metric_total(los_totals, (pred_phys - target_phys)[los_valid])
                update_metric_total(nlos_totals, (pred_phys - target_phys)[nlos_valid])

    if distributed and dist.is_initialized():
        payload = {
            "totals": totals,
            "stage1_totals": stage1_totals,
            "los_totals": los_totals,
            "nlos_totals": nlos_totals,
        }
        gathered: list[dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, payload)
        if not is_main_process(rank):
            return {}
        totals = init_metric_totals()
        stage1_totals = init_metric_totals()
        los_totals = init_metric_totals()
        nlos_totals = init_metric_totals()
        for part in gathered:
            for key, value in part["totals"].items():
                totals[key] += float(value)
            for key, value in part["stage1_totals"].items():
                stage1_totals[key] += float(value)
            for key, value in part["los_totals"].items():
                los_totals[key] += float(value)
            for key, value in part["nlos_totals"].items():
                nlos_totals[key] += float(value)

    summary: Dict[str, Any] = {
        "path_loss": finalize_metric_total(totals, meta.get("unit", "dB")),
        "path_loss__stage1__overall": finalize_metric_total(stage1_totals, meta.get("unit", "dB")),
    }
    if los_totals["count"] > 0.0:
        summary["path_loss__los__LoS"] = finalize_metric_total(los_totals, meta.get("unit", "dB"))
    if nlos_totals["count"] > 0.0:
        summary["path_loss__los__NLoS"] = finalize_metric_total(nlos_totals, meta.get("unit", "dB"))
    return summary


def resolve_resume_checkpoint(out_dir: Path, configured_resume: str | None) -> Optional[Path]:
    if configured_resume:
        candidate = Path(configured_resume)
        return candidate if candidate.exists() else None
    best = out_dir / "best_tail_refiner.pt"
    if best.exists():
        return best
    epoch_candidates = sorted(out_dir.glob("epoch_*_tail_refiner.pt"))
    if epoch_candidates:
        return epoch_candidates[-1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Try49 stage2 tail-refiner without GANs")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    set_seed(int(cfg.get("seed", 42)))

    if return_scalar_cond_from_config(cfg):
        raise ValueError("Tail refiner expects scalar channels, not scalar FiLM vectors.")

    device = resolve_device(cfg["runtime"].get("device", "cuda"))
    distributed, rank, local_rank, world_size = maybe_init_distributed(device)
    if distributed and is_cuda_device(device):
        device = torch.device("cuda", local_rank)

    out_dir = ensure_output_dir(cfg["runtime"]["output_dir"])
    if is_main_process(rank):
        print(json.dumps({"output_dir": str(out_dir), "world_size": world_size}))

    splits = build_dataset_splits_from_config(cfg)
    tail_cfg = dict(cfg.get("tail_refiner", {}))
    train_hdf5 = str(tail_cfg.get("stage1_outputs_train_hdf5", "")).strip()
    val_hdf5 = str(tail_cfg.get("stage1_outputs_val_hdf5", "")).strip()
    if not train_hdf5 or not val_hdf5:
        raise ValueError("tail_refiner.stage1_outputs_train_hdf5 and tail_refiner.stage1_outputs_val_hdf5 are required")

    train_dataset = Stage1OutputDataset(
        splits["train"],
        train_hdf5,
        prediction_key=str(tail_cfg.get("prediction_key", "stage1_pred_norm_f16")),
        abs_error_key=str(tail_cfg.get("abs_error_key", "stage1_abs_error_db_f16")),
    )
    val_dataset = Stage1OutputDataset(
        splits["val"],
        val_hdf5,
        prediction_key=str(tail_cfg.get("prediction_key", "stage1_pred_norm_f16")),
        abs_error_key=str(tail_cfg.get("abs_error_key", "stage1_abs_error_db_f16")),
    )

    input_channels = compute_input_channels(cfg) + 1
    model = build_tail_refiner_model(cfg, input_channels).to(device)

    if bool(tail_cfg.get("oversample", {}).get("enabled", False)):
        sample_weights = build_sample_weights(train_dataset, cfg)
        if distributed:
            train_sampler = DistributedWeightedSampler(
                sample_weights,
                num_replicas=world_size,
                rank=rank,
                replacement=True,
                seed=int(cfg.get("seed", 42)),
            )
        else:
            train_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"].get("batch_size", 1)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg["data"].get("num_workers", 6)),
        pin_memory=is_cuda_device(device),
        persistent_workers=int(cfg["data"].get("num_workers", 6)) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["data"].get("val_num_workers", cfg["data"].get("num_workers", 6))),
        pin_memory=is_cuda_device(device),
        persistent_workers=bool(cfg["data"].get("val_persistent_workers", cfg["data"].get("persistent_workers", False)))
        and int(cfg["data"].get("val_num_workers", cfg["data"].get("num_workers", 6))) > 0,
    )

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if is_cuda_device(device) else None,
            output_device=local_rank if is_cuda_device(device) else None,
            find_unused_parameters=False,
        )

    optimizer = build_optimizer(cfg, model.parameters(), device)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(enabled=bool(cfg["training"].get("amp", True)) and is_cuda_device(device))

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    resume_path = resolve_resume_checkpoint(out_dir, cfg["runtime"].get("resume_checkpoint"))
    if resume_path and resume_path.exists():
        state = load_torch_checkpoint(resume_path, device)
        target_model = model.module if isinstance(model, DistributedDataParallel) else model
        target_model.load_state_dict(state["model"] if "model" in state else state["generator"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        if "scheduler" in state and scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
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
            if isinstance(train_sampler, (DistributedSampler, DistributedWeightedSampler)):
                train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, amp_enabled)
            barrier_if_distributed(distributed)

            val_summary = evaluate_validation(
                model.module if isinstance(model, DistributedDataParallel) else model,
                val_loader,
                device,
                cfg,
                amp_enabled,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
            )

            if is_main_process(rank):
                val_summary["_train"] = {"tail_refiner_loss": float(train_loss)}
                try:
                    current_score = extract_selection_metric(val_summary, selection_metric)
                except Exception:
                    current_score = float(val_summary["path_loss"]["rmse_physical"])
                if current_score < best_score:
                    best_score = current_score
                    best_epoch = epoch
                    target_model = model.module if isinstance(model, DistributedDataParallel) else model
                    best_payload = {
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "model": target_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "scaler": scaler.state_dict(),
                        "config_path": args.config,
                    }
                    torch.save(best_payload, out_dir / "best_tail_refiner.pt")

                if epoch % int(cfg["training"].get("save_every", 5)) == 0:
                    target_model = model.module if isinstance(model, DistributedDataParallel) else model
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_score": best_score,
                            "model": target_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler is not None else None,
                            "scaler": scaler.state_dict(),
                            "config_path": args.config,
                        },
                        out_dir / f"epoch_{epoch}_tail_refiner.pt",
                    )

                write_validation_json(out_dir, epoch, val_summary, best_epoch=best_epoch, best_score=best_score)
                print(json.dumps({"epoch": epoch, "tail_refiner_loss": train_loss, selection_metric: current_score}))
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

            if is_cuda_device(device):
                torch.cuda.empty_cache()

            barrier_if_distributed(distributed)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()