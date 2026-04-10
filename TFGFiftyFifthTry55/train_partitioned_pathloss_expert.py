from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Subset
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
    _infer_city_type_simple,
    build_dataset_splits_from_config,
    compute_scalar_cond_dim,
    compute_input_channels,
    forward_cgan_generator,
    return_scalar_cond_from_config,
    unpack_cgan_batch,
)
from model_pmhhnet import PMHHNetResidualRegressor, PMHNetResidualRegressor, PMNetResidualRegressor, PatchDiscriminator, UNetResidualRefiner


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


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def resolve_ema_decay(cfg: Dict[str, Any]) -> float:
    decay = float(cfg.get("training", {}).get("ema_decay", 0.99))
    if decay == 1.0:
        return 0.0
    if decay < 0.0 or decay > 1.0:
        raise ValueError("training.ema_decay must be in [0, 1], with 1 meaning disabled")
    return decay


def create_ema_model(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def update_ema_model(ema_model: nn.Module, source_model: nn.Module, decay: float) -> None:
    if decay <= 0.0:
        return
    ema_state = ema_model.state_dict()
    source_state = source_model.state_dict()
    with torch.no_grad():
        for name, source_value in source_state.items():
            ema_value = ema_state[name]
            if torch.is_floating_point(source_value):
                ema_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(source_value)


def formula_channel_index(cfg: Dict[str, Any]) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):
        idx += 1
    if cfg["data"].get("distance_map_channel", False):
        idx += 1
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    if not bool(formula_cfg.get("enabled", False)):
        raise ValueError("Try 42 requires data.path_loss_formula_input.enabled = true")
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


def masked_rmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    mse = (((pred - target) ** 2) * mask).sum() / denom
    return torch.sqrt(mse + float(eps))


def _no_data_aux_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("no_data_auxiliary", {}))


def _split_model_output(
    output: torch.Tensor,
    *,
    aux_enabled: bool,
    gate_enabled: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if output.ndim != 4 or output.shape[1] < 1:
        raise ValueError("Expected model output with shape [B, C, H, W]")
    residual = output[:, :1]
    cursor = 1
    gate_logits: Optional[torch.Tensor] = None
    if gate_enabled:
        if output.shape[1] < 2:
            raise ValueError("Gate-enabled output requires at least 2 channels")
        gate_logits = output[:, 1:2]
        cursor = 2
    aux_logits: Optional[torch.Tensor] = None
    if aux_enabled and output.shape[1] > cursor:
        aux_logits = output[:, cursor : cursor + 1]
    return residual, gate_logits, aux_logits


def _derive_no_data_target(mask: torch.Tensor) -> torch.Tensor:
    return (mask[:, :1] <= 0.0).float()


def _extract_no_data_target(target: torch.Tensor, mask: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    aux_cfg = _no_data_aux_cfg(cfg)
    if bool(aux_cfg.get("enabled", False)) and target.shape[1] > 1:
        return target[:, 1:2]
    return _derive_no_data_target(mask)


def _compute_no_data_loss(
    aux_logits: Optional[torch.Tensor],
    no_data_target: torch.Tensor,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    aux_cfg = _no_data_aux_cfg(cfg)
    if not bool(aux_cfg.get("enabled", False)) or aux_logits is None:
        return torch.zeros((), device=no_data_target.device)
    positive_weight = max(float(aux_cfg.get("positive_weight", 1.0)), 1e-6)
    if abs(positive_weight - 1.0) < 1e-6:
        return F.binary_cross_entropy_with_logits(aux_logits, no_data_target)
    pos_weight = torch.tensor([positive_weight], dtype=aux_logits.dtype, device=aux_logits.device)
    return F.binary_cross_entropy_with_logits(aux_logits, no_data_target, pos_weight=pos_weight)


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


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    lr = float(learning_rate)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    if isinstance(getattr(optimizer, "defaults", None), dict):
        optimizer.defaults["lr"] = lr


def set_optimizer_weight_decay(optimizer: torch.optim.Optimizer, weight_decay: float) -> None:
    decay = float(weight_decay)
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = decay
    if isinstance(getattr(optimizer, "defaults", None), dict):
        optimizer.defaults["weight_decay"] = decay


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


def build_adversarial_loss(loss_name: str) -> nn.Module:
    mode = str(loss_name).lower()
    if mode == "bce":
        return nn.BCEWithLogitsLoss()
    if mode == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported adversarial loss '{loss_name}'")


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def resolve_resume_checkpoint(out_dir: Path, configured_resume: str | None) -> Optional[Path]:
    if configured_resume:
        candidate = Path(configured_resume)
        return candidate if candidate.exists() else None
    epoch_candidates = sorted(out_dir.glob("epoch_*_model.pt"))
    if epoch_candidates:
        return epoch_candidates[-1]
    legacy_epoch_candidates = sorted(out_dir.glob("epoch_*_cgan.pt"))
    if legacy_epoch_candidates:
        return legacy_epoch_candidates[-1]
    best = out_dir / "best_model.pt"
    if best.exists():
        return best
    legacy_best = out_dir / "best_cgan.pt"
    if legacy_best.exists():
        return legacy_best
    return None


def _build_pmnet_from_cfg(cfg: Dict[str, Any], in_channels: int) -> nn.Module:
    arch = str(cfg.get("model", {}).get("arch", "pmnet")).lower()
    scalar_dim = int(compute_scalar_cond_dim(cfg)) if return_scalar_cond_from_config(cfg) else 0
    common = dict(
        in_channels=in_channels,
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"].get("base_channels", 64)),
        encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
    )
    if arch == "pmhnet":
        return PMHNetResidualRegressor(
            **common,
            hf_channels=int(cfg["model"].get("hf_channels", max(8, int(cfg["model"].get("base_channels", 64)) // 2))),
        )
    if arch == "pmhhnet":
        return PMHHNetResidualRegressor(
            **common,
            hf_channels=int(cfg["model"].get("hf_channels", max(8, int(cfg["model"].get("base_channels", 64)) // 2))),
            scalar_dim=max(1, scalar_dim),
            scalar_hidden_dim=int(cfg["model"].get("scalar_hidden_dim", max(32, int(cfg["model"].get("base_channels", 64)) * 2))),
        )
    return PMNetResidualRegressor(
        **common,
    )


def _build_refiner_from_cfg(cfg: Dict[str, Any], in_channels: int) -> nn.Module:
    sep_cfg = dict(cfg.get("separated_refiner", {}))
    arch = str(sep_cfg.get("refiner_arch", "unet")).lower()
    if arch == "pmnet":
        return _build_pmnet_from_cfg(cfg, in_channels)
    if arch == "unet":
        return UNetResidualRefiner(
            in_channels=in_channels,
            out_channels=int(sep_cfg.get("refiner_out_channels", cfg["model"]["out_channels"])),
            base_channels=int(sep_cfg.get("refiner_base_channels", 48)),
            norm_type=str(cfg["model"].get("norm_type", "group")),
            dropout=float(cfg["model"].get("dropout", 0.0)),
            gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        )
    raise ValueError(f"Unsupported separated_refiner.refiner_arch '{arch}'")


def _checkpoint_model_state(state: Dict[str, Any]) -> Dict[str, Any]:
    if "model" in state:
        return state["model"]
    if "generator" in state:
        return state["generator"]
    raise KeyError("Checkpoint missing model/generator state")


def _rewind_training_state_from_checkpoint(
    checkpoint_path: Path,
    device: object,
    generator_model: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    scaler_g: amp.GradScaler,
    scheduler_g: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    ema_generator: Optional[nn.Module],
    discriminator: Optional[nn.Module],
    optimizer_d: Optional[torch.optim.Optimizer],
    scaler_d: Optional[amp.GradScaler],
) -> None:
    state = load_torch_checkpoint(checkpoint_path, device)
    target_generator = unwrap_model(generator_model)
    target_generator.load_state_dict(_checkpoint_model_state(state))
    if ema_generator is not None:
        if "generator_ema" in state and state["generator_ema"] is not None:
            ema_generator.load_state_dict(state["generator_ema"])
        else:
            ema_generator.load_state_dict(target_generator.state_dict())
    if discriminator is not None and "discriminator" in state and state["discriminator"] is not None:
        target_discriminator = unwrap_model(discriminator)
        target_discriminator.load_state_dict(state["discriminator"])
    if "optimizer_g" in state and state["optimizer_g"] is not None:
        optimizer_g.load_state_dict(state["optimizer_g"])
        move_optimizer_state_to_device(optimizer_g, device)
    elif "optimizer" in state and state["optimizer"] is not None:
        optimizer_g.load_state_dict(state["optimizer"])
        move_optimizer_state_to_device(optimizer_g, device)
    if "optimizer_d" in state and optimizer_d is not None and state["optimizer_d"] is not None:
        optimizer_d.load_state_dict(state["optimizer_d"])
        move_optimizer_state_to_device(optimizer_d, device)
    if "scaler_g" in state and state["scaler_g"] is not None:
        scaler_g.load_state_dict(state["scaler_g"])
    elif "scaler" in state and state["scaler"] is not None:
        scaler_g.load_state_dict(state["scaler"])
    if "scaler_d" in state and scaler_d is not None and state["scaler_d"] is not None:
        scaler_d.load_state_dict(state["scaler_d"])
    if scheduler_g is not None:
        if "scheduler_g" in state and state["scheduler_g"] is not None:
            scheduler_g.load_state_dict(state["scheduler_g"])
        elif "scheduler" in state and state["scheduler"] is not None:
            scheduler_g.load_state_dict(state["scheduler"])


def _load_generator_weights(module: nn.Module, checkpoint_path: Path, device: object) -> None:
    state = load_torch_checkpoint(checkpoint_path, device)
    module.load_state_dict(_checkpoint_model_state(state), strict=True)


def _scalar_feature_names(cfg: Dict[str, Any]) -> list[str]:
    names = list(cfg["data"].get("scalar_feature_columns", []))
    names.extend(list(dict(cfg["data"].get("constant_scalar_features", {})).keys()))
    return names


def _extract_scalar_input_channel(input_batch: torch.Tensor, cfg: Dict[str, Any], name: str) -> Optional[torch.Tensor]:
    if not bool(cfg.get("model", {}).get("use_scalar_channels", False)):
        return None
    scalar_names = _scalar_feature_names(cfg)
    if not scalar_names or name not in scalar_names:
        return None
    scalar_count = len(scalar_names)
    total_channels = int(compute_input_channels(cfg))
    prefix_channels = max(total_channels - scalar_count, 0)
    idx = prefix_channels + scalar_names.index(name)
    if idx < 0 or idx >= input_batch.shape[1]:
        return None
    return input_batch[:, idx : idx + 1]


def _compose_residual_prediction_with_aux(
    trainable_generator: nn.Module,
    x: torch.Tensor,
    scalar_cond: Optional[torch.Tensor],
    prior: torch.Tensor,
    *,
    separated_mode: bool,
    base_generator: Optional[nn.Module],
    use_gate: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not separated_mode:
        raw_out = forward_cgan_generator(trainable_generator, x, scalar_cond)
        residual, _, aux_logits = _split_model_output(raw_out, aux_enabled=True, gate_enabled=False)
        return residual, None, None, aux_logits
    if base_generator is None:
        raise RuntimeError("Separated mode enabled but base_generator is missing")
    with torch.no_grad():
        base_out = forward_cgan_generator(base_generator, x, scalar_cond)
        base_residual, _, _ = _split_model_output(base_out, aux_enabled=True, gate_enabled=False)
    base_pred = prior + base_residual
    refiner_in = torch.cat([x, base_pred, base_residual], dim=1)
    refiner_out = forward_cgan_generator(trainable_generator, refiner_in, None)
    delta_raw, gate_logits, aux_logits = _split_model_output(refiner_out, aux_enabled=True, gate_enabled=use_gate)
    if use_gate and gate_logits is not None:
        delta = delta_raw * torch.sigmoid(gate_logits)
    else:
        delta = delta_raw
    return base_residual + delta, gate_logits, base_residual, aux_logits


def _compose_residual_prediction(
    trainable_generator: nn.Module,
    x: torch.Tensor,
    scalar_cond: Optional[torch.Tensor],
    prior: torch.Tensor,
    *,
    separated_mode: bool,
    base_generator: Optional[nn.Module],
    use_gate: bool = False,
) -> torch.Tensor:
    residual_pred, _, _, _ = _compose_residual_prediction_with_aux(
        trainable_generator,
        x,
        scalar_cond,
        prior,
        separated_mode=separated_mode,
        base_generator=base_generator,
        use_gate=use_gate,
    )
    return residual_pred


def _tail_focus_weights(
    stage1_error: torch.Tensor,
    x: torch.Tensor,
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
) -> torch.Tensor:
    sep_cfg = dict(cfg.get("separated_refiner", {}))
    tail_cfg = dict(sep_cfg.get("tail_focus", {}))
    if not bool(tail_cfg.get("enabled", False)):
        return torch.ones_like(stage1_error)

    scale = max(float(meta.get("scale", 180.0)), 1e-6)
    threshold_db = float(tail_cfg.get("threshold_db", 6.0))
    temperature_db = max(float(tail_cfg.get("temperature_db", 2.5)), 1e-3)
    alpha = max(float(tail_cfg.get("alpha", 1.0)), 0.0)
    nlos_boost = max(float(tail_cfg.get("nlos_boost", 0.0)), 0.0)
    antenna_boost = max(float(tail_cfg.get("antenna_boost", 0.0)), 0.0)
    max_weight = max(float(tail_cfg.get("max_weight", 5.0)), 1.0)

    hard_score = torch.sigmoid((stage1_error.abs() - (threshold_db / scale)) / max(temperature_db / scale, 1e-6))
    weight = 1.0 + alpha * hard_score

    if nlos_boost > 0.0 and cfg["data"].get("los_input_column"):
        los_idx = 1
        if x.shape[1] > los_idx:
            nlos = (1.0 - x[:, los_idx : los_idx + 1].clamp(0.0, 1.0)).clamp(0.0, 1.0)
            weight = weight * (1.0 + nlos_boost * nlos)

    if antenna_boost > 0.0:
        antenna_channel = _extract_scalar_input_channel(x, cfg, "antenna_height_m")
        if antenna_channel is not None:
            low_antenna = (1.0 - antenna_channel.clamp(0.0, 1.0)).clamp(0.0, 1.0)
            weight = weight * (1.0 + antenna_boost * low_antenna)

    return weight.clamp(1.0, max_weight)


class AutomaticCityTypeResolver:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        input_meta = dict(cfg["data"].get("input_metadata", {}))
        self.input_scale = float(input_meta.get("scale", 1.0))
        self.input_offset = float(input_meta.get("offset", 0.0))
        self.non_ground_threshold = float(cfg["data"].get("non_ground_threshold", 0.0))
        self.city_type_thresholds: dict[str, float] = {}
        formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
        calibration_json = formula_cfg.get("regime_calibration_json")
        if calibration_json:
            cal_path = Path(__file__).resolve().parent / str(calibration_json)
            if cal_path.exists():
                calibration = json.loads(cal_path.read_text(encoding="utf-8"))
                self.city_type_thresholds = dict(calibration.get("city_type_thresholds", {}))

    def infer(self, input_batch: torch.Tensor) -> str:
        topo_batch = input_batch[:, :1] if input_batch.ndim == 4 else input_batch[:1]
        raw_topology = topo_batch.detach().float().cpu().numpy() * self.input_scale + self.input_offset
        raw_topology = np.squeeze(raw_topology, axis=0)
        non_ground = raw_topology != self.non_ground_threshold
        building_density = float(np.mean(non_ground))
        non_zero = raw_topology[non_ground]
        mean_height = float(np.mean(non_zero)) if non_zero.size else 0.0
        if self.city_type_thresholds:
            return _city_type_from_thresholds(building_density, mean_height, self.city_type_thresholds)
        return _infer_city_type_simple(building_density, mean_height)


def _apply_regime_reweighting(
    mask: torch.Tensor,
    input_batch: torch.Tensor,
    cfg: Dict[str, Any],
    city_type_resolver: Optional[AutomaticCityTypeResolver],
) -> torch.Tensor:
    rw_cfg = dict(cfg.get("training", {}).get("regime_reweighting", {}))
    if not bool(rw_cfg.get("enabled", False)):
        return mask

    los_weight = max(float(rw_cfg.get("los_weight", 1.0)), 1e-6)
    nlos_weight = max(float(rw_cfg.get("nlos_weight", 1.0)), 1e-6)
    low_antenna_boost = max(float(rw_cfg.get("low_antenna_boost", 0.0)), 0.0)
    city_weights = {str(k): float(v) for k, v in dict(rw_cfg.get("city_type_weights", {})).items()}
    default_city_weight = max(float(rw_cfg.get("default_city_weight", 1.0)), 1e-6)
    min_weight = max(float(rw_cfg.get("min_weight", 0.25)), 1e-6)
    max_weight = max(float(rw_cfg.get("max_weight", 4.0)), min_weight)

    weight = torch.ones_like(mask)
    if cfg["data"].get("los_input_column"):
        los_idx = 1
        if input_batch.shape[1] > los_idx:
            los = input_batch[:, los_idx : los_idx + 1].clamp(0.0, 1.0)
            nlos = (1.0 - los).clamp(0.0, 1.0)
            weight = weight * (los * los_weight + nlos * nlos_weight)

    if low_antenna_boost > 0.0:
        antenna_channel = _extract_scalar_input_channel(input_batch, cfg, "antenna_height_m")
        if antenna_channel is not None:
            low_antenna = (1.0 - antenna_channel.clamp(0.0, 1.0)).clamp(0.0, 1.0)
            weight = weight * (1.0 + low_antenna_boost * low_antenna)

    if city_type_resolver is not None:
        city_type = city_type_resolver.infer(input_batch)
        weight = weight * max(city_weights.get(city_type, default_city_weight), 1e-6)

    return mask * weight.clamp(min=min_weight, max=max_weight)


def _gate_target_from_error(stage1_error: torch.Tensor, cfg: Dict[str, Any], meta: Dict[str, Any]) -> torch.Tensor:
    sep_cfg = dict(cfg.get("separated_refiner", {}))
    gate_cfg = dict(sep_cfg.get("gate_target", {}))
    scale = max(float(meta.get("scale", 180.0)), 1e-6)
    threshold_db = float(gate_cfg.get("threshold_db", 6.0))
    temperature_db = max(float(gate_cfg.get("temperature_db", 2.0)), 1e-3)
    return torch.sigmoid((stage1_error.abs() - (threshold_db / scale)) / max(temperature_db / scale, 1e-6))


class RegimeAnnotator:
    def __init__(self, dataset: Any, calibration_json: Optional[str], cfg: Dict[str, Any]) -> None:
        self.dataset = dataset
        input_meta = dict(cfg["data"].get("input_metadata", {}))
        self.input_scale = float(input_meta.get("scale", 1.0))
        self.input_offset = float(input_meta.get("offset", 0.0))
        self.non_ground_threshold = float(cfg["data"].get("non_ground_threshold", 0.0))
        self.scalar_columns = list(cfg["data"].get("scalar_feature_columns", []))
        self.scalar_norms = dict(cfg["data"].get("scalar_feature_norms", {}))
        self.uses_scalar_channels = bool(cfg.get("model", {}).get("use_scalar_channels", False))
        self.calibration = None
        if calibration_json:
            cal_path = Path(__file__).resolve().parent / calibration_json
            if cal_path.exists():
                self.calibration = json.loads(cal_path.read_text(encoding="utf-8"))
        self.city_type_by_city = dict(self.calibration.get("city_type_by_city", {})) if self.calibration else {}
        self.city_type_thresholds = dict(self.calibration.get("city_type_thresholds", {})) if self.calibration else {}
        self.antenna_height_thresholds = dict(self.calibration.get("antenna_height_thresholds", {})) if self.calibration else {}
        self._city_cache: dict[str, str] = {}

    def _city_type_from_batch_topology(self, input_batch: torch.Tensor) -> str:
        topo_batch = input_batch[:, :1] if input_batch.ndim == 4 else input_batch[:1]
        raw_topology = topo_batch.detach().float().cpu().numpy() * self.input_scale + self.input_offset
        raw_topology = np.squeeze(raw_topology, axis=0)
        non_ground = raw_topology != self.non_ground_threshold
        building_density = float(np.mean(non_ground))
        non_zero = raw_topology[non_ground]
        mean_height = float(np.mean(non_zero)) if non_zero.size else 0.0
        if self.city_type_thresholds:
            return _city_type_from_thresholds(building_density, mean_height, self.city_type_thresholds)
        return "unknown_city_type"

    def _antenna_height_from_scalar(self, scalar_cond: Optional[torch.Tensor]) -> float:
        if scalar_cond is None or scalar_cond.numel() == 0:
            return 0.0
        if "antenna_height_m" not in self.scalar_columns:
            return 0.0
        idx = self.scalar_columns.index("antenna_height_m")
        norm = float(self.scalar_norms.get("antenna_height_m", 1.0))
        return float(scalar_cond[0, idx].detach().float().cpu().item() * norm)

    def _antenna_height_from_input(self, input_batch: torch.Tensor) -> float:
        if not self.uses_scalar_channels or "antenna_height_m" not in self.scalar_columns:
            return 0.0
        scalar_count = len(self.scalar_columns)
        if scalar_count <= 0 or input_batch.shape[1] < scalar_count:
            return 0.0
        idx = self.scalar_columns.index("antenna_height_m")
        channel_idx = input_batch.shape[1] - scalar_count + idx
        norm = float(self.scalar_norms.get("antenna_height_m", 1.0))
        return float(input_batch[0, channel_idx, 0, 0].detach().float().cpu().item() * norm)

    def info_for_index(self, idx: int, input_batch: torch.Tensor, scalar_cond: Optional[torch.Tensor]) -> dict[str, str]:
        city, sample = self.dataset.sample_refs[idx]
        city_type = self.city_type_by_city.get(city)
        if city_type is None:
            city_type = self._city_cache.get(city)
            if city_type is None:
                city_type = self._city_type_from_batch_topology(input_batch)
                self._city_cache[city] = city_type
        antenna_height_m = self._antenna_height_from_scalar(scalar_cond)
        if antenna_height_m <= 0.0:
            antenna_height_m = self._antenna_height_from_input(input_batch)
        ant_bin = _antenna_height_bin(antenna_height_m, self.antenna_height_thresholds) if self.antenna_height_thresholds else "mid_ant"
        return {"city_type": city_type, "antenna_bin": ant_bin, "city": city}


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


def update_metric_total_quantized_u8(
    totals: dict[str, float],
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    pred_q = torch.clamp(torch.round(pred_phys), 0.0, 255.0)
    target_q = torch.clamp(torch.round(target_phys), 0.0, 255.0)
    update_metric_total(totals, (pred_q - target_q)[valid_mask])


def finalize_metric_total(
    totals: dict[str, float],
    unit: str,
    *,
    total_count: Optional[float] = None,
) -> dict[str, float]:
    count = float(totals.get("count", 0.0))
    if count <= 0.0:
        payload: dict[str, float] = {
            "mse_physical": float("nan"),
            "rmse_physical": float("nan"),
            "mae_physical": float("nan"),
            "count": 0,
            "unit": unit,
        }
        if total_count is not None and total_count > 0.0:
            payload["fraction_of_valid_pixels"] = 0.0
        return payload
    mse = totals["sum_squared_error"] / count
    payload = {
        "mse_physical": float(mse),
        "rmse_physical": float(math.sqrt(mse)),
        "mae_physical": float(totals["sum_absolute_error"] / count),
        "count": int(round(count)),
        "unit": unit,
    }
    if total_count is not None and total_count > 0.0:
        payload["fraction_of_valid_pixels"] = float(count / total_count)
    return payload


def attach_quantized_metric_fields(
    payload: dict[str, float],
    quantized_totals: dict[str, float],
    unit: str,
    *,
    total_count: Optional[float] = None,
) -> dict[str, float]:
    quantized = finalize_metric_total(quantized_totals, unit, total_count=total_count)
    payload["mse_physical_quantized_u8"] = float(quantized.get("mse_physical", float("nan")))
    payload["rmse_physical_quantized_u8"] = float(quantized.get("rmse_physical", float("nan")))
    payload["mae_physical_quantized_u8"] = float(quantized.get("mae_physical", float("nan")))
    return payload


def init_binary_totals() -> dict[str, float]:
    return {
        "count": 0.0,
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "sum_bce": 0.0,
        "sum_squared_error": 0.0,
        "sum_absolute_error": 0.0,
    }


def update_binary_totals(binary_totals: dict[str, float], logits: torch.Tensor, target: torch.Tensor) -> None:
    if logits.numel() == 0 or target.numel() == 0:
        return
    probs = torch.sigmoid(logits)
    pred = probs >= 0.5
    tgt = target >= 0.5
    binary_totals["count"] += float(target.numel())
    binary_totals["tp"] += float((pred & tgt).sum().item())
    binary_totals["tn"] += float((~pred & ~tgt).sum().item())
    binary_totals["fp"] += float((pred & ~tgt).sum().item())
    binary_totals["fn"] += float((~pred & tgt).sum().item())
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    binary_totals["sum_bce"] += float(bce.item()) * float(target.numel())
    diff = probs - target
    binary_totals["sum_squared_error"] += float(torch.sum(diff ** 2).item())
    binary_totals["sum_absolute_error"] += float(torch.sum(torch.abs(diff)).item())


def finalize_binary_totals(binary_totals: dict[str, float]) -> dict[str, float]:
    count = max(float(binary_totals.get("count", 0.0)), 0.0)
    if count <= 0.0:
        return {
            "count": 0,
            "bce": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "iou": float("nan"),
            "positive_fraction_target": float("nan"),
            "positive_fraction_pred": float("nan"),
        }
    tp = float(binary_totals.get("tp", 0.0))
    tn = float(binary_totals.get("tn", 0.0))
    fp = float(binary_totals.get("fp", 0.0))
    fn = float(binary_totals.get("fn", 0.0))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    mse = float(binary_totals.get("sum_squared_error", 0.0) / count)
    return {
        "count": int(round(count)),
        "bce": float(binary_totals.get("sum_bce", 0.0) / count),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(binary_totals.get("sum_absolute_error", 0.0) / count),
        "accuracy": float((tp + tn) / count),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "positive_fraction_target": float((tp + fn) / count),
        "positive_fraction_pred": float((tp + fp) / count),
    }


def build_experiment_summary(cfg: Dict[str, Any], dataset: Optional[Any] = None) -> dict[str, Any]:
    data_cfg = dict(cfg.get("data", {}))
    model_cfg = dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    aux_cfg = _no_data_aux_cfg(cfg)
    partition_filter = dict(data_cfg.get("partition_filter", {}))
    explicit_no_data_column = str(data_cfg.get("path_loss_no_data_mask_column", "")).strip()
    if dataset is not None and hasattr(dataset, "describe_no_data_target_source"):
        no_data_target_source = str(dataset.describe_no_data_target_source())
    elif explicit_no_data_column:
        no_data_target_source = f"hdf5:{explicit_no_data_column}"
    elif bool(data_cfg.get("derive_no_data_from_non_ground", False)):
        no_data_target_source = "fallback:non_ground_mask"
    else:
        no_data_target_source = "disabled"
    return {
        "topology_class": partition_filter.get("topology_class"),
        "focused_city_type": topology_class_to_focus_city_type(partition_filter.get("topology_class")),
        "split_mode": data_cfg.get("split_mode"),
        "sample_count": int(len(dataset)) if dataset is not None else None,
        "image_size": int(data_cfg.get("image_size", 0)),
        "batch_size": int(train_cfg.get("batch_size", 0)),
        "model_arch": str(model_cfg.get("arch", "pmnet")),
        "base_channels": int(model_cfg.get("base_channels", 0)),
        "hf_channels": int(model_cfg.get("hf_channels", 0)) if "hf_channels" in model_cfg else None,
        "out_channels": int(model_cfg.get("out_channels", 1)),
        "use_scalar_film": bool(model_cfg.get("use_scalar_film", False)),
        "learning_rate": float(train_cfg.get("learning_rate", 0.0)),
        "generator_objective": str(train_cfg.get("generator_objective", "legacy")),
        "save_every": int(train_cfg.get("save_every", 0)),
        "no_data_aux_enabled": bool(aux_cfg.get("enabled", False)),
        "no_data_target_source": no_data_target_source,
    }


def topology_class_to_focus_city_type(topology_class: Optional[str]) -> Optional[str]:
    value = str(topology_class or "").strip()
    if not value:
        return None
    if value.startswith("open_sparse_"):
        return "open_lowrise"
    if value.startswith("mixed_compact_"):
        return "mixed_midrise"
    if value.startswith("dense_block_"):
        return "dense_highrise"
    return None


def filter_regime_summary_for_topology_class(
    regimes: Dict[str, Dict[str, Any]],
    topology_class: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    focus_city_type = topology_class_to_focus_city_type(topology_class)
    if not focus_city_type:
        return dict(regimes)

    keep_exact = {
        "path_loss__los__LoS",
        "path_loss__los__NLoS",
        "path_loss__prior__los__LoS",
        "path_loss__prior__los__NLoS",
    }
    keep_prefixes = (
        f"path_loss__city_type__{focus_city_type}",
        f"path_loss__prior__city_type__{focus_city_type}",
        f"path_loss__calibration_regime__{focus_city_type}__",
        f"path_loss__prior__calibration_regime__{focus_city_type}__",
    )

    filtered: Dict[str, Dict[str, Any]] = {}
    for key, value in regimes.items():
        if key in keep_exact or any(key.startswith(prefix) for prefix in keep_prefixes):
            filtered[key] = value
    return filtered


def build_validation_payload(
    summary: dict[str, Any],
    *,
    epoch: int,
    best_epoch: int,
    best_score: float,
    selection_metric: str,
    current_score: float,
) -> dict[str, Any]:
    experiment = dict(summary.get("experiment", {}))
    payload: dict[str, Any] = {
        "metrics": {
            "path_loss": dict(summary.get("path_loss", {})),
            "train_path_loss": dict(summary.get("_train_metrics", {})),
            "prior_path_loss": dict(summary.get("path_loss__prior__overall", {})),
            "improvement_vs_prior": dict(summary.get("improvement_vs_prior", {})),
        },
        "focus": {
            "topology_class": experiment.get("topology_class"),
            "routed_city_type": experiment.get("focused_city_type"),
            "regimes": dict(summary.get("_regimes", {})),
        },
        "experiment": experiment,
        "support": dict(summary.get("_support", {})),
        "runtime": dict(summary.get("_train", {})),
        "checkpoint": {
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
        },
        "selection": {
            "metric": str(selection_metric),
            "current_score": float(current_score),
            "is_best_epoch": bool(epoch == best_epoch),
        },
    }
    if "no_data" in summary:
        payload["metrics"]["no_data"] = dict(summary["no_data"])
    return payload


def write_live_train_progress(
    out_dir: Path,
    payload: dict[str, Any],
) -> None:
    (out_dir / "train_progress_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_validation(
    model: nn.Module,
    dataset: Any,
    device: object,
    cfg: Dict[str, Any],
    amp_enabled: bool,
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    separated_mode: bool = False,
    base_generator: Optional[nn.Module] = None,
) -> dict[str, Any]:
    val_num_workers = int(cfg["data"].get("val_num_workers", cfg["data"].get("num_workers", 0)))
    sample_indices = list(range(len(dataset)))
    eval_dataset = dataset
    if distributed:
        base_sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
        sample_indices = list(iter(base_sampler))
        eval_dataset = Subset(dataset, sample_indices)
    loader_kwargs: dict[str, Any] = {}
    if val_num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))
    loader = DataLoader(
        eval_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=is_cuda_device(device),
        persistent_workers=bool(cfg["data"].get("val_persistent_workers", cfg["data"].get("persistent_workers", False))) and val_num_workers > 0,
        **loader_kwargs,
    )
    formula_idx = formula_channel_index(cfg)
    meta = dict(cfg["target_metadata"]["path_loss"])
    unit = str(meta.get("unit", "dB"))
    prior_cfg = dict(cfg.get("prior_residual_path_loss", {}))
    sep_cfg = dict(cfg.get("separated_refiner", {}))
    clamp_final = bool(prior_cfg.get("clamp_final_output", True))
    use_gate = bool(sep_cfg.get("use_gate", False))
    annotator = RegimeAnnotator(
        dataset,
        dict(cfg["data"].get("path_loss_formula_input", {})).get("regime_calibration_json"),
        cfg,
    )

    totals = init_metric_totals()
    totals_quantized = init_metric_totals()
    prior_totals = init_metric_totals()
    prior_totals_quantized = init_metric_totals()
    regime_totals: dict[str, dict[str, float]] = defaultdict(init_metric_totals)
    aux_cfg = _no_data_aux_cfg(cfg)
    no_data_enabled = bool(aux_cfg.get("enabled", False))
    no_data_totals = init_binary_totals()

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="val", leave=False, disable=distributed and not is_main_process(rank))):
            x, y, m, scalar_cond = unpack_cgan_batch(batch, device)
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                residual_pred, _, _, no_data_logits = _compose_residual_prediction_with_aux(
                    model,
                    x,
                    scalar_cond,
                    x[:, formula_idx : formula_idx + 1],
                    separated_mode=separated_mode,
                    base_generator=base_generator,
                    use_gate=use_gate,
                )
            prior = x[:, formula_idx : formula_idx + 1]
            pred = prior + residual_pred
            if clamp_final:
                pred = clip_to_target_range(pred, meta)

            pred_phys = denormalize(pred, meta)
            target_phys = denormalize(y[:, :1], meta)
            valid_mask = m[:, :1] > 0.0
            diff_phys = (pred_phys - target_phys)[valid_mask]
            update_metric_total(totals, diff_phys)
            update_metric_total_quantized_u8(totals_quantized, pred_phys, target_phys, valid_mask)

            prior_phys = denormalize(prior, meta)
            prior_diff_phys = (prior_phys - target_phys)[valid_mask]
            update_metric_total(prior_totals, prior_diff_phys)
            update_metric_total_quantized_u8(prior_totals_quantized, prior_phys, target_phys, valid_mask)
            if no_data_enabled and no_data_logits is not None:
                no_data_target = _extract_no_data_target(y, m, cfg)
                update_binary_totals(no_data_totals, no_data_logits, no_data_target)

            info = annotator.info_for_index(sample_indices[idx], x, scalar_cond)
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

    if distributed and dist.is_initialized():
        payload = {
            "totals": totals,
            "totals_quantized": totals_quantized,
            "prior_totals": prior_totals,
            "prior_totals_quantized": prior_totals_quantized,
            "regime_totals": dict(regime_totals),
            "no_data_totals": no_data_totals,
        }
        gathered: list[dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, payload)
        if not is_main_process(rank):
            return {}
        totals = init_metric_totals()
        totals_quantized = init_metric_totals()
        prior_totals = init_metric_totals()
        prior_totals_quantized = init_metric_totals()
        merged_regimes: dict[str, dict[str, float]] = defaultdict(init_metric_totals)
        for part in gathered:
            for k, v in part["totals"].items():
                totals[k] += float(v)
            for k, v in part["totals_quantized"].items():
                totals_quantized[k] += float(v)
            for k, v in part["prior_totals"].items():
                prior_totals[k] += float(v)
            for k, v in part["prior_totals_quantized"].items():
                prior_totals_quantized[k] += float(v)
            for key, reg in dict(part["regime_totals"]).items():
                bucket = merged_regimes[key]
                for k, v in reg.items():
                    bucket[k] += float(v)
            for k, v in dict(part.get("no_data_totals", {})).items():
                no_data_totals[k] += float(v)
        regime_totals = merged_regimes

    total_count = float(totals.get("count", 0.0))
    los_count = float(regime_totals.get("path_loss__los__LoS", {}).get("count", 0.0))
    nlos_count = float(regime_totals.get("path_loss__los__NLoS", {}).get("count", 0.0))
    regime_summary = {
        key: finalize_metric_total(val, unit, total_count=total_count)
        for key, val in sorted(regime_totals.items())
    }
    topology_class = dict(cfg.get("data", {})).get("partition_filter", {}).get("topology_class")
    filtered_regimes = filter_regime_summary_for_topology_class(regime_summary, topology_class)

    path_loss_summary = finalize_metric_total(totals, unit, total_count=total_count)
    attach_quantized_metric_fields(path_loss_summary, totals_quantized, unit, total_count=total_count)
    prior_summary = finalize_metric_total(prior_totals, unit, total_count=total_count)
    attach_quantized_metric_fields(prior_summary, prior_totals_quantized, unit, total_count=total_count)

    summary: dict[str, Any] = {
        "path_loss": path_loss_summary,
        "path_loss__prior__overall": prior_summary,
        "experiment": build_experiment_summary(cfg, dataset),
        "_support": {
            "sample_count": int(len(dataset)),
            "valid_pixel_count": int(round(total_count)),
            "los_valid_pixel_count": int(round(los_count)),
            "nlos_valid_pixel_count": int(round(nlos_count)),
            "los_fraction": float(los_count / total_count) if total_count > 0.0 else float("nan"),
            "nlos_fraction": float(nlos_count / total_count) if total_count > 0.0 else float("nan"),
        },
        "_regimes": filtered_regimes,
    }
    if no_data_enabled:
        summary["no_data"] = finalize_binary_totals(no_data_totals)
        summary["_support"]["no_data_positive_fraction"] = float(summary["no_data"]["positive_fraction_target"])
    current_rmse = float(summary["path_loss"]["rmse_physical"])
    prior_rmse = float(summary["path_loss__prior__overall"]["rmse_physical"])
    current_mae = float(summary["path_loss"]["mae_physical"])
    prior_mae = float(summary["path_loss__prior__overall"]["mae_physical"])
    summary["improvement_vs_prior"] = {
        "rmse_gain_db": float(prior_rmse - current_rmse),
        "rmse_relative_gain_pct": float(((prior_rmse - current_rmse) / prior_rmse) * 100.0) if math.isfinite(prior_rmse) and abs(prior_rmse) > 1e-12 else float("nan"),
        "mae_gain_db": float(prior_mae - current_mae),
        "mae_relative_gain_pct": float(((prior_mae - current_mae) / prior_mae) * 100.0) if math.isfinite(prior_mae) and abs(prior_mae) > 1e-12 else float("nan"),
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


def write_validation_json(
    out_dir: Path,
    epoch: int,
    summary: dict[str, Any],
    *,
    best_epoch: int,
    best_score: float,
    selection_metric: str,
    current_score: float,
) -> None:
    payload = build_validation_payload(
        summary,
        epoch=epoch,
        best_epoch=best_epoch,
        best_score=best_score,
        selection_metric=selection_metric,
        current_score=current_score,
    )
    latest = out_dir / "validate_metrics_latest.json"
    epoch_path = out_dir / f"validate_metrics_epoch_{epoch}.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    epoch_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_one_epoch(
    generator: nn.Module,
    discriminator: Optional[nn.Module],
    loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: Optional[torch.optim.Optimizer],
    scaler_g: amp.GradScaler,
    scaler_d: Optional[amp.GradScaler],
    device: object,
    cfg: Dict[str, Any],
    amp_enabled: bool,
    *,
    ema_generator: Optional[nn.Module] = None,
    ema_decay: float = 0.0,
    separated_mode: bool = False,
    base_generator: Optional[nn.Module] = None,
    city_type_resolver: Optional[AutomaticCityTypeResolver] = None,
    distributed: bool = False,
    rank: int = 0,
    epoch: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> tuple[float, float, float, dict[str, Any]]:
    formula_idx = formula_channel_index(cfg)
    meta = dict(cfg["target_metadata"]["path_loss"])
    loss_cfg = dict(cfg["loss"])
    residual_cfg = dict(cfg.get("prior_residual_path_loss", {}))
    sep_cfg = dict(cfg.get("separated_refiner", {}))
    clamp_final = bool(residual_cfg.get("clamp_final_output", True))
    optimize_residual_only = bool(residual_cfg.get("optimize_residual_only", False))
    use_gate = bool(sep_cfg.get("use_gate", False))
    gate_loss_weight = float(sep_cfg.get("gate_loss_weight", 0.0))
    clip_grad = float(cfg["training"].get("clip_grad_norm", 0.0))
    lambda_gan = float(loss_cfg.get("lambda_gan", 0.0))
    lambda_recon = float(loss_cfg.get("lambda_recon", 1.0))
    adv_criterion = build_adversarial_loss(str(loss_cfg.get("adversarial_loss", "bce"))).to(device)
    use_gan = lambda_gan > 0.0 and discriminator is not None and optimizer_d is not None and scaler_d is not None

    generator.train()
    if discriminator is not None:
        discriminator.train()
    source_generator = unwrap_model(generator)
    running_g = 0.0
    running_d = 0.0
    running_no_data = 0.0
    train_metric_totals = init_metric_totals()
    train_metric_totals_quantized = init_metric_totals()
    steps = 0
    epoch_started = time.perf_counter()
    progress_every = 25
    for batch in tqdm(loader, desc="train", leave=False, disable=distributed and not is_main_process(rank)):
        x, y, m, scalar_cond = unpack_cgan_batch(batch, device)
        target = y[:, :1]
        mask = m[:, :1]
        prior = x[:, formula_idx : formula_idx + 1]
        residual_target = target - prior

        with torch.no_grad():
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                residual_fake_det = _compose_residual_prediction(
                    generator,
                    x,
                    scalar_cond,
                    prior,
                    separated_mode=separated_mode,
                    base_generator=base_generator,
                    use_gate=use_gate,
                )
                fake_det = prior + residual_fake_det
                if clamp_final:
                    fake_det = clip_to_target_range(fake_det, meta)

        if use_gan:
            if discriminator is None or optimizer_d is None or scaler_d is None:
                raise RuntimeError("use_gan=true requires discriminator, optimizer_d, and scaler_d")
            set_requires_grad(discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                real_logits = discriminator(x, target)
                fake_logits = discriminator(x, fake_det.detach())
                real_labels = torch.ones_like(real_logits)
                fake_labels = torch.zeros_like(fake_logits)
                d_loss_real = adv_criterion(real_logits, real_labels)
                d_loss_fake = adv_criterion(fake_logits, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
            scaler_d.scale(d_loss).backward()
            if clip_grad > 0.0:
                scaler_d.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            d_loss = torch.zeros((), device=target.device)

        if discriminator is not None:
            set_requires_grad(discriminator, False)
        optimizer_g.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=amp_enabled):
            residual_pred, gate_logits, base_residual, no_data_logits = _compose_residual_prediction_with_aux(
                generator,
                x,
                scalar_cond,
                prior,
                separated_mode=separated_mode,
                base_generator=base_generator,
                use_gate=use_gate,
            )
            pred = prior + residual_pred
            if clamp_final:
                pred = clip_to_target_range(pred, meta)

            weighted_mask = mask
            gate_target = None
            gate_loss = torch.zeros((), device=target.device)
            if separated_mode and base_residual is not None:
                stage1_error = residual_target - base_residual
                weighted_mask = mask * _tail_focus_weights(stage1_error, x, cfg, meta)
                if use_gate:
                    gate_target = _gate_target_from_error(stage1_error, cfg, meta)
            objective_mode = str(cfg.get("training", {}).get("generator_objective", "legacy")).lower()
            if objective_mode != "full_map_rmse_only":
                weighted_mask = _apply_regime_reweighting(weighted_mask, x, cfg, city_type_resolver)

            if objective_mode == "full_map_rmse_only":
                final_loss = torch.zeros((), device=target.device)
                residual_loss = torch.zeros((), device=target.device)
                multiscale_loss = torch.zeros((), device=target.device)
            else:
                final_loss = masked_mse_l1_loss(
                    pred,
                    target,
                    weighted_mask,
                    mse_weight=float(loss_cfg.get("mse_weight", 1.0)),
                    l1_weight=float(loss_cfg.get("l1_weight", 0.0)),
                )
                residual_loss = masked_mse_l1_loss(
                    residual_pred,
                    residual_target,
                    weighted_mask,
                    mse_weight=float(residual_cfg.get("mse_weight", 1.0)),
                    l1_weight=float(residual_cfg.get("l1_weight", 0.0)),
                )
                multiscale_loss = compute_multiscale_path_loss_loss(pred, target, mask, meta, cfg)
            no_data_target = _extract_no_data_target(y, m, cfg)
            no_data_loss = _compute_no_data_loss(no_data_logits, no_data_target, cfg)
            if use_gate and gate_logits is not None and gate_target is not None:
                gate_loss_map = F.binary_cross_entropy_with_logits(gate_logits, gate_target, reduction="none")
                gate_loss = (gate_loss_map * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)

            if use_gan and discriminator is not None:
                fake_logits_for_g = discriminator(x, pred)
                gan_loss = adv_criterion(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
            else:
                gan_loss = torch.zeros((), device=target.device)

            if objective_mode == "full_map_rmse_only":
                g_loss = masked_rmse_loss(pred, target, mask)
            elif optimize_residual_only:
                residual_only_weight = float(residual_cfg.get("loss_weight", 1.0))
                final_only_weight = float(residual_cfg.get("final_loss_weight_when_residual_only", 0.0))
                multiscale_only_weight = float(residual_cfg.get("multiscale_loss_weight_when_residual_only", 0.0))
                g_loss = (
                    residual_only_weight * residual_loss
                    + final_only_weight * final_loss
                    + multiscale_only_weight * multiscale_loss
                )
            else:
                g_loss = (
                    lambda_recon * final_loss
                    + float(residual_cfg.get("loss_weight", 0.0)) * residual_loss
                    + multiscale_loss
                    + lambda_gan * gan_loss
                )

            if gate_loss_weight > 0.0:
                g_loss = g_loss + gate_loss_weight * gate_loss
            if bool(_no_data_aux_cfg(cfg).get("enabled", False)):
                g_loss = g_loss + float(_no_data_aux_cfg(cfg).get("loss_weight", 0.0)) * no_data_loss

        scaler_g.scale(g_loss).backward()
        if clip_grad > 0.0:
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        if ema_generator is not None and ema_decay > 0.0:
            update_ema_model(ema_generator, source_generator, ema_decay)

        with torch.no_grad():
            pred_phys = denormalize(pred.detach(), meta)
            target_phys = denormalize(target.detach(), meta)
            valid_mask = mask > 0.0
            diff_phys = (pred_phys - target_phys)[valid_mask]
            update_metric_total(train_metric_totals, diff_phys)
            update_metric_total_quantized_u8(train_metric_totals_quantized, pred_phys, target_phys, valid_mask)

        running_g += float(g_loss.item())
        running_d += float(d_loss.item())
        running_no_data += float(no_data_loss.item())
        steps += 1
        if is_main_process(rank) and out_dir is not None and (steps == 1 or steps % progress_every == 0 or steps == len(loader)):
            elapsed = max(time.perf_counter() - epoch_started, 1e-9)
            live_payload = {
                "epoch": int(epoch) if epoch is not None else None,
                "step": int(steps),
                "total_steps": int(len(loader)),
                "progress_fraction": float(steps / max(len(loader), 1)),
                "elapsed_seconds": float(elapsed),
                "train_batches_per_second": float(steps / elapsed),
                "seconds_per_train_batch": float(elapsed / steps),
                "generator_loss_running": float(running_g / steps),
                "no_data_loss_running": float(running_no_data / steps),
                "train_rmse_physical_running": float(
                    finalize_metric_total(train_metric_totals, str(meta.get("unit", "dB"))).get("rmse_physical", float("nan"))
                ),
                "train_rmse_physical_quantized_u8_running": float(
                    finalize_metric_total(train_metric_totals_quantized, str(meta.get("unit", "dB"))).get("rmse_physical", float("nan"))
                ),
                "learning_rate": float(optimizer_g.param_groups[0]["lr"]),
                "generator_objective": str(cfg.get("training", {}).get("generator_objective", "legacy")),
            }
            if use_gan:
                live_payload["discriminator_loss_running"] = float(running_d / steps)
            write_live_train_progress(out_dir, live_payload)
    denom = max(steps, 1)
    train_metrics = finalize_metric_total(train_metric_totals, str(meta.get("unit", "dB")))
    attach_quantized_metric_fields(train_metrics, train_metric_totals_quantized, str(meta.get("unit", "dB")))
    return running_g / denom, running_d / denom, running_no_data / denom, train_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Try 55 topology expert on top of the calibrated prior")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg["runtime"]["device"])
    distributed, rank, local_rank, world_size = maybe_init_distributed(device)
    if distributed and is_cuda_device(device):
        device = torch.device("cuda", local_rank)
    if is_cuda_device(device):
        torch.backends.cudnn.benchmark = True

    out_dir = ensure_output_dir(cfg["runtime"]["output_dir"])
    if is_main_process(rank):
        print(json.dumps({"output_dir": str(out_dir), "world_size": world_size}))

    splits = build_dataset_splits_from_config(cfg)
    train_dataset = splits["train"]
    val_dataset = splits["val"]

    pin_memory = is_cuda_device(device)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader_kwargs: dict[str, Any] = {}
    if int(cfg["data"]["num_workers"]) > 0:
        train_loader_kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=pin_memory,
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
        **train_loader_kwargs,
    )
    city_type_resolver = AutomaticCityTypeResolver(cfg)

    sep_cfg = dict(cfg.get("separated_refiner", {}))
    separated_mode = bool(sep_cfg.get("enabled", False))
    input_channels = compute_input_channels(cfg)

    base_generator: Optional[nn.Module] = None
    if separated_mode:
        base_generator = _build_pmnet_from_cfg(cfg, input_channels).to(device)
        base_ckpt_raw = sep_cfg.get("base_checkpoint") or cfg["runtime"].get("resume_checkpoint")
        if not base_ckpt_raw:
            raise ValueError("separated_refiner.enabled=true requires separated_refiner.base_checkpoint")
        base_ckpt = Path(str(base_ckpt_raw))
        if not base_ckpt.exists():
            base_ckpt = (Path(__file__).resolve().parent / str(base_ckpt_raw)).resolve()
        if not base_ckpt.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_raw}")
        _load_generator_weights(base_generator, base_ckpt, device)
        base_generator.eval()
        set_requires_grad(base_generator, False)
        generator = _build_refiner_from_cfg(cfg, input_channels + 2).to(device)
    else:
        generator = _build_pmnet_from_cfg(cfg, input_channels).to(device)

    ema_decay = resolve_ema_decay(cfg)
    ema_generator: Optional[nn.Module] = create_ema_model(generator) if ema_decay > 0.0 else None

    loss_cfg = dict(cfg.get("loss", {}))
    use_gan_training = float(loss_cfg.get("lambda_gan", 0.0)) > 0.0

    discriminator: Optional[nn.Module]
    if use_gan_training:
        discriminator = PatchDiscriminator(
            in_channels=input_channels,
            target_channels=int(cfg["model"]["out_channels"]),
            base_channels=int(cfg["model"].get("disc_base_channels", 24)),
            norm_type=str(cfg["model"].get("disc_norm_type", cfg["model"].get("norm_type", "group"))),
            input_downsample_factor=int(cfg["model"].get("disc_input_downsample_factor", 2)),
        ).to(device)
    else:
        discriminator = None

    generator_for_training: nn.Module = generator
    discriminator_for_training: Optional[nn.Module] = discriminator
    if distributed:
        generator_for_training = DistributedDataParallel(
            generator,
            device_ids=[local_rank] if is_cuda_device(device) else None,
            output_device=local_rank if is_cuda_device(device) else None,
            find_unused_parameters=False,
        )
        if discriminator is not None:
            discriminator_for_training = DistributedDataParallel(
                discriminator,
                device_ids=[local_rank] if is_cuda_device(device) else None,
                output_device=local_rank if is_cuda_device(device) else None,
                find_unused_parameters=False,
            )

    optimizer_g = build_optimizer(cfg, generator_for_training.parameters(), device)
    optimizer_d: Optional[torch.optim.Optimizer] = None
    if discriminator_for_training is not None:
        opt_name_d = str(cfg["training"].get("discriminator_optimizer", cfg["training"].get("optimizer", "adam"))).lower()
        if opt_name_d == "adamw":
            optimizer_d = torch.optim.AdamW(
                discriminator_for_training.parameters(),
                lr=float(cfg["training"].get("discriminator_lr", 3e-5)),
                weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
                betas=(float(cfg["training"].get("beta1", 0.5)), float(cfg["training"].get("beta2", 0.999))),
                foreach=is_cuda_device(device),
            )
        else:
            optimizer_d = torch.optim.Adam(
                discriminator_for_training.parameters(),
                lr=float(cfg["training"].get("discriminator_lr", 3e-5)),
                weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
                betas=(float(cfg["training"].get("beta1", 0.5)), float(cfg["training"].get("beta2", 0.999))),
                foreach=is_cuda_device(device),
            )
    scheduler_g = build_scheduler(cfg, optimizer_g)
    scaler_g = amp.GradScaler(enabled=bool(cfg["training"].get("amp", True)) and is_cuda_device(device))
    scaler_d: Optional[amp.GradScaler] = None
    if discriminator_for_training is not None:
        scaler_d = amp.GradScaler(enabled=bool(cfg["training"].get("amp", True)) and is_cuda_device(device))

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    resume_cfg_key = sep_cfg.get("refiner_resume_checkpoint") if separated_mode else cfg["runtime"].get("resume_checkpoint")
    resume_path = resolve_resume_checkpoint(out_dir, resume_cfg_key)
    if resume_path and resume_path.exists():
        state = load_torch_checkpoint(resume_path, device)
        target_generator = generator_for_training.module if isinstance(generator_for_training, DistributedDataParallel) else generator_for_training
        target_generator.load_state_dict(_checkpoint_model_state(state))
        if "discriminator" in state and discriminator_for_training is not None:
            target_discriminator = discriminator_for_training.module if isinstance(discriminator_for_training, DistributedDataParallel) else discriminator_for_training
            target_discriminator.load_state_dict(state["discriminator"])
        if ema_generator is not None:
            if "generator_ema" in state:
                ema_generator.load_state_dict(state["generator_ema"])
            else:
                ema_generator.load_state_dict(target_generator.state_dict())
        if "optimizer_g" in state:
            optimizer_g.load_state_dict(state["optimizer_g"])
            move_optimizer_state_to_device(optimizer_g, device)
        elif "optimizer" in state:
            optimizer_g.load_state_dict(state["optimizer"])
            move_optimizer_state_to_device(optimizer_g, device)
        if "optimizer_d" in state and optimizer_d is not None:
            optimizer_d.load_state_dict(state["optimizer_d"])
            move_optimizer_state_to_device(optimizer_d, device)
        if "scheduler_g" in state and scheduler_g is not None:
            scheduler_g.load_state_dict(state["scheduler_g"])
        elif "scheduler" in state and scheduler_g is not None:
            scheduler_g.load_state_dict(state["scheduler"])
        if "scaler_g" in state:
            scaler_g.load_state_dict(state["scaler_g"])
        elif "scaler" in state:
            scaler_g.load_state_dict(state["scaler"])
        if "scaler_d" in state and scaler_d is not None:
            scaler_d.load_state_dict(state["scaler_d"])
        resume_generator_lr = float(cfg["training"].get("learning_rate", cfg["training"].get("generator_lr", 3e-5)))
        resume_weight_decay = float(cfg["training"].get("weight_decay", 0.0))
        set_optimizer_learning_rate(optimizer_g, resume_generator_lr)
        set_optimizer_weight_decay(optimizer_g, resume_weight_decay)
        if scheduler_g is not None and hasattr(scheduler_g, "base_lrs"):
            scheduler_g.base_lrs = [resume_generator_lr for _ in getattr(scheduler_g, "base_lrs", [])]
        if optimizer_d is not None:
            resume_discriminator_lr = float(cfg["training"].get("discriminator_lr", resume_generator_lr))
            set_optimizer_learning_rate(optimizer_d, resume_discriminator_lr)
            set_optimizer_weight_decay(optimizer_d, resume_weight_decay)
        start_epoch = int(state.get("epoch", 0)) + 1
        best_score = float(state.get("best_score", best_score))
        best_epoch = int(state.get("best_epoch", best_epoch))
        if is_main_process(rank):
            print(
                json.dumps(
                    {
                        "resume_from": str(resume_path),
                        "start_epoch": start_epoch,
                        "learning_rate": resume_generator_lr,
                        "weight_decay": resume_weight_decay,
                        "discriminator_learning_rate": float(cfg["training"].get("discriminator_lr", resume_generator_lr)) if optimizer_d is not None else None,
                    }
                )
            )

    es_cfg = cfg['training'].get('early_stopping') or {}
    es_enabled = bool(es_cfg.get('enabled', False))
    es_patience = int(es_cfg.get('patience', 10))
    es_min_delta = float(es_cfg.get('min_delta', 0.0))
    es_rewind_to_best_model = bool(es_cfg.get('rewind_to_best_model', False))
    epochs_without_improvement = 0

    amp_enabled = bool(cfg["training"].get("amp", True)) and is_cuda_device(device)
    selection_metric = next(iter(dict(cfg["training"].get("selection_metrics", {"path_loss.rmse_physical": 1.0})).keys()))

    try:
        for epoch in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_started = time.perf_counter()
            train_g_loss, train_d_loss, train_no_data_loss, train_metrics = train_one_epoch(
                generator_for_training,
                discriminator_for_training,
                train_loader,
                optimizer_g,
                optimizer_d,
                scaler_g,
                scaler_d,
                device,
                cfg,
                amp_enabled,
                ema_generator=ema_generator,
                ema_decay=ema_decay,
                separated_mode=separated_mode,
                base_generator=base_generator,
                city_type_resolver=city_type_resolver,
                distributed=distributed,
                rank=rank,
                epoch=epoch,
                out_dir=out_dir,
            )
            train_seconds = time.perf_counter() - train_started
            barrier_if_distributed(distributed)

            validation_generator = ema_generator if ema_generator is not None else generator
            val_started = time.perf_counter()
            val_summary = evaluate_validation(
                validation_generator,
                val_dataset,
                device,
                cfg,
                amp_enabled,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                separated_mode=separated_mode,
                base_generator=base_generator,
            )
            val_seconds = time.perf_counter() - val_started

            if is_main_process(rank):
                train_batches = max(len(train_loader), 1)
                val_samples = max(len(val_dataset), 1)
                train_payload = {
                    "generator_loss": float(train_g_loss),
                    "no_data_loss": float(train_no_data_loss),
                    "learning_rate": float(optimizer_g.param_groups[0]["lr"]),
                    "generator_objective": str(cfg["training"].get("generator_objective", "legacy")),
                    "train_seconds": float(train_seconds),
                    "val_seconds": float(val_seconds),
                    "epoch_seconds": float(train_seconds + val_seconds),
                    "train_batches_per_second": float(train_batches / max(train_seconds, 1e-12)),
                    "seconds_per_train_batch": float(train_seconds / train_batches),
                    "val_samples_per_second": float(val_samples / max(val_seconds, 1e-12)),
                    "seconds_per_val_sample": float(val_seconds / val_samples),
                    "gan_enabled": bool(use_gan_training),
                }
                if use_gan_training:
                    train_payload["discriminator_loss"] = float(train_d_loss)
                val_summary["_train"] = train_payload
                val_summary["_train_metrics"] = train_metrics
                current_score = extract_selection_metric(val_summary, selection_metric)
                prev_best_score = float(best_score)
                if es_enabled and es_patience > 0:
                    if current_score < (prev_best_score - es_min_delta):
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                if current_score < best_score:
                    best_score = current_score
                    best_epoch = epoch
                    best_payload = {
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "model": generator.state_dict(),
                        "generator": generator.state_dict(),
                        "ema_decay": ema_decay,
                        "optimizer_g": optimizer_g.state_dict(),
                        "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
                        "scaler_g": scaler_g.state_dict(),
                        "config_path": args.config,
                    }
                    if ema_generator is not None:
                        best_payload["generator_ema"] = ema_generator.state_dict()
                    if discriminator is not None:
                        best_payload["discriminator"] = discriminator.state_dict()
                    if optimizer_d is not None:
                        best_payload["optimizer_d"] = optimizer_d.state_dict()
                    if scaler_d is not None:
                        best_payload["scaler_d"] = scaler_d.state_dict()
                    torch.save(best_payload, out_dir / "best_model.pt")

                epoch_ckpt_path = out_dir / f"epoch_{epoch}_model.pt"
                epoch_payload = {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "model": generator.state_dict(),
                    "generator": generator.state_dict(),
                    "ema_decay": ema_decay,
                    "optimizer_g": optimizer_g.state_dict(),
                    "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
                    "scaler_g": scaler_g.state_dict(),
                    "config_path": args.config,
                }
                if ema_generator is not None:
                    epoch_payload["generator_ema"] = ema_generator.state_dict()
                if discriminator is not None:
                    epoch_payload["discriminator"] = discriminator.state_dict()
                if optimizer_d is not None:
                    epoch_payload["optimizer_d"] = optimizer_d.state_dict()
                if scaler_d is not None:
                    epoch_payload["scaler_d"] = scaler_d.state_dict()
                torch.save(epoch_payload, epoch_ckpt_path)
                prev_epoch_ckpt_path = out_dir / f"epoch_{epoch - 1}_model.pt"
                if prev_epoch_ckpt_path.exists():
                    prev_epoch_ckpt_path.unlink()

                write_validation_json(
                    out_dir,
                    epoch,
                    val_summary,
                    best_epoch=best_epoch,
                    best_score=best_score,
                    selection_metric=selection_metric,
                    current_score=current_score,
                )
                progress_payload = {
                    "epoch": epoch,
                    "generator_loss": train_g_loss,
                    selection_metric: current_score,
                }
                if use_gan_training:
                    progress_payload["discriminator_loss"] = train_d_loss
                print(json.dumps(progress_payload))
            else:
                current_score = 0.0

            control_tensor = torch.tensor([0], device=device, dtype=torch.int32)
            if distributed:
                if is_main_process(rank) and es_enabled and es_patience > 0 and epochs_without_improvement >= es_patience:
                    if es_rewind_to_best_model:
                        control_tensor[0] = 2
                        print(
                            json.dumps(
                                {
                                    "rewind_to_best_model": True,
                                    "epochs_without_improvement": int(epochs_without_improvement),
                                    "patience": int(es_patience),
                                }
                            )
                        )
                    else:
                        control_tensor[0] = 1
                        print(
                            json.dumps(
                                {
                                    "early_stopping": True,
                                    "epochs_without_improvement": int(epochs_without_improvement),
                                    "patience": int(es_patience),
                                }
                            )
                        )
                dist.broadcast(control_tensor, src=0)
            else:
                if is_main_process(rank) and es_enabled and es_patience > 0 and epochs_without_improvement >= es_patience:
                    if es_rewind_to_best_model:
                        control_tensor[0] = 2
                        print(
                            json.dumps(
                                {
                                    "rewind_to_best_model": True,
                                    "epochs_without_improvement": int(epochs_without_improvement),
                                    "patience": int(es_patience),
                                }
                            )
                        )
                    else:
                        control_tensor[0] = 1
                        print(
                            json.dumps(
                                {
                                    "early_stopping": True,
                                    "epochs_without_improvement": int(epochs_without_improvement),
                                    "patience": int(es_patience),
                                }
                            )
                        )

            control_value = int(control_tensor.item())
            if control_value == 2:
                best_path = out_dir / "best_model.pt"
                if not best_path.exists():
                    raise FileNotFoundError(f"Best checkpoint not found for rewind: {best_path}")
                _rewind_training_state_from_checkpoint(
                    best_path,
                    device,
                    generator_for_training,
                    optimizer_g,
                    scaler_g,
                    scheduler_g,
                    ema_generator,
                    discriminator_for_training,
                    optimizer_d,
                    scaler_d,
                )
                epochs_without_improvement = 0
                if is_cuda_device(device):
                    torch.cuda.empty_cache()
                barrier_if_distributed(distributed)
                continue
            if control_value == 1:
                if is_cuda_device(device):
                    torch.cuda.empty_cache()
                barrier_if_distributed(distributed)
                break

            if scheduler_g is not None:
                if distributed:
                    score_tensor = torch.tensor([float(current_score)], device=device if is_cuda_device(device) else "cpu")
                    dist.broadcast(score_tensor, src=0)
                    current_score = float(score_tensor.item())
                if isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_g.step(current_score)
                else:
                    scheduler_g.step()

            if is_cuda_device(device):
                torch.cuda.empty_cache()

            barrier_if_distributed(distributed)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
