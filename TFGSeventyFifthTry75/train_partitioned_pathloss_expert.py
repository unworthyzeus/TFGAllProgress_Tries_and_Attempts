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
from model_pmhhnet import PMHHNetResidualRegressor, PMHNetResidualRegressor, PMNetResidualRegressor, PatchDiscriminator, UNetResidualRefiner, UNetResidualRefinerH


class _IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to append the sample index as the last element of each item."""

    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self._ds = dataset

    def __len__(self) -> int:
        return len(self._ds)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        item = self._ds[idx]
        if isinstance(item, tuple):
            return item + (torch.tensor(idx, dtype=torch.long),)
        return item, torch.tensor(idx, dtype=torch.long)


class SampleRMSEBuffer:
    """Per-sample RMSE tracker for RL-style prioritized loss reweighting.

    Each training step, the sample's RMSE is recorded via an EMA. At the next
    step touching that sample the loss is scaled by:

        w_i = clip( (rmse_i / mean_rmse)^gamma,  min_weight, max_weight )

    Harder samples (high RMSE relative to the dataset mean) get more gradient;
    easy samples get less. Equivalent to prioritized experience replay but for
    supervised image regression.
    """

    def __init__(
        self,
        n_samples: int,
        ema_alpha: float = 0.7,
        gamma: float = 1.5,
        min_weight: float = 0.5,
        max_weight: float = 3.0,
    ) -> None:
        self.buffer = torch.ones(n_samples, dtype=torch.float32)
        self.ema_alpha = float(ema_alpha)
        self.gamma = float(gamma)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self._initialized = torch.zeros(n_samples, dtype=torch.bool)

    def update(self, indices: Sequence[int], rmse_values: Sequence[float]) -> None:
        for i, rmse in zip(indices, rmse_values):
            idx = int(i)
            if not math.isfinite(rmse) or rmse <= 0.0:
                continue
            if not self._initialized[idx]:
                self.buffer[idx] = float(rmse)
                self._initialized[idx] = True
            else:
                self.buffer[idx] = (
                    self.ema_alpha * self.buffer[idx]
                    + (1.0 - self.ema_alpha) * float(rmse)
                )

    def weight_for(self, indices: Sequence[int]) -> float:
        """Return mean weight for the given sample indices."""
        vals = self.buffer[list(indices)]
        mean_val = float(self.buffer[self._initialized].mean().clamp_min(1e-6)) if self._initialized.any() else 1.0
        weights = (vals / mean_val).pow(self.gamma).clamp(self.min_weight, self.max_weight)
        return float(weights.mean())

    def sync_distributed(self, distributed: bool) -> None:
        """All-reduce buffer across DDP ranks (max so each rank contributes its updates)."""
        if distributed and dist.is_available() and dist.is_initialized():
            t = self.buffer.cuda()
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            self.buffer = t.cpu()


def _unpack_train_batch(
    batch: Tuple[Any, ...],
    device: object,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list]]:
    """Unpack a training batch that may include sample indices (from _IndexedDataset)."""
    sample_indices: Optional[list] = None
    if batch and isinstance(batch[-1], torch.Tensor) and batch[-1].dtype == torch.long and batch[-1].ndim == 1:
        sample_indices = batch[-1].tolist()
        batch = batch[:-1]
    x, y, m, sc = unpack_cgan_batch(batch, device)
    return x, y, m, sc, sample_indices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cuda_training_backends(device: object) -> None:
    """Best-effort CUDA throughput tuning for fixed-resolution conv training (cluster GPUs)."""
    if not is_cuda_device(device):
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


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
                source_finite = torch.isfinite(source_value)
                ema_finite = torch.isfinite(ema_value)
                if not bool(source_finite.all()):
                    # Keep the last healthy EMA weights instead of poisoning validation
                    # with a transient non-finite optimizer update.
                    continue
                if not bool(ema_finite.all()):
                    ema_value.copy_(source_value)
                    continue
                ema_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(source_value)


def _update_swa_model(swa_model: nn.Module, source_model: nn.Module, swa_count: int) -> None:
    """Uniform-weight Stochastic Weight Averaging (Izmailov et al. NeurIPS 2018).

    Maintains a running arithmetic mean of model weights across epochs, which finds
    wider loss-landscape optima and generalises better to unseen cities than EMA.
    ``swa_count`` is the number of models averaged so far (before this update).
    """
    if swa_count <= 0:
        # First SWA update: copy weights directly
        with torch.no_grad():
            for swa_p, src_p in zip(swa_model.parameters(), source_model.parameters()):
                swa_p.copy_(src_p)
        for swa_b, src_b in zip(swa_model.buffers(), source_model.buffers()):
            swa_b.copy_(src_b)
        return
    alpha = 1.0 / (swa_count + 1)
    swa_state = swa_model.state_dict()
    source_state = source_model.state_dict()
    with torch.no_grad():
        for name, src_val in source_state.items():
            swa_val = swa_state[name]
            if torch.is_floating_point(src_val):
                # running mean: new_avg = old_avg + (new - old_avg) / (n+1)
                swa_val.add_(src_val - swa_val, alpha=alpha)
            else:
                swa_val.copy_(src_val)



_WEIGHT_DECAY_EXEMPT_MODULES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def _split_weight_decay_parameters(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for submodule in module.modules():
        exempt_module = isinstance(submodule, _WEIGHT_DECAY_EXEMPT_MODULES)
        for param_name, param in submodule.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if exempt_module or param_name == "bias" or param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    return decay_params, no_decay_params


def configure_optimizer_weight_decay(optimizer: torch.optim.Optimizer, module: nn.Module, weight_decay: float) -> None:
    decay_params, no_decay_params = _split_weight_decay_parameters(module)
    setattr(optimizer, "_manual_weight_decay_params", tuple(decay_params))
    setattr(optimizer, "_manual_no_weight_decay_params", tuple(no_decay_params))
    setattr(optimizer, "_manual_weight_decay_param_count", len(decay_params))
    setattr(optimizer, "_manual_no_weight_decay_param_count", len(no_decay_params))
    setattr(optimizer, "_manual_weight_decay_total_count", len(decay_params) + len(no_decay_params))
    setattr(optimizer, "_manual_weight_decay_mode", "selective_excluding_bias_and_norm")
    set_optimizer_weight_decay(optimizer, weight_decay)


def apply_optimizer_weight_decay(optimizer: torch.optim.Optimizer) -> None:
    decay = float(getattr(optimizer, "_manual_weight_decay", 0.0))
    if decay <= 0.0:
        return
    decay_params = getattr(optimizer, "_manual_weight_decay_params", ())
    if not decay_params:
        return
    if not optimizer.param_groups:
        return
    lr = float(optimizer.param_groups[0].get("lr", 0.0))
    if lr <= 0.0:
        return
    scale = 1.0 - lr * decay
    if scale == 1.0:
        return
    with torch.no_grad():
        for param in decay_params:
            param.mul_(scale)


def formula_channel_index(cfg: Dict[str, Any]) -> int:
    idx = 1
    if bool(cfg["data"].get("los_input_column")) and bool(cfg["data"].get("use_los_as_input", True)):
        idx += 1
    if cfg["data"].get("distance_map_channel", False):
        idx += 1
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    if not bool(formula_cfg.get("enabled", False)):
        raise ValueError("Try 42 requires data.path_loss_formula_input.enabled = true")
    return idx


def uses_formula_prior(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("data", {}).get("path_loss_formula_input", {}).get("enabled", False))


def uses_los_input_channel(cfg: Dict[str, Any]) -> bool:
    data_cfg = dict(cfg.get("data", {}))
    return bool(data_cfg.get("los_input_column")) and bool(data_cfg.get("use_los_as_input", True))


def uses_absolute_path_loss_prediction(cfg: Dict[str, Any]) -> bool:
    prior_cfg = dict(cfg.get("prior_residual_path_loss", {}))
    if "absolute_prediction" in prior_cfg:
        return bool(prior_cfg.get("absolute_prediction", False))
    return not bool(prior_cfg.get("enabled", False)) and not uses_formula_prior(cfg)


def extract_formula_prior_or_zero(
    input_batch: torch.Tensor,
    cfg: Dict[str, Any],
    reference: torch.Tensor,
) -> torch.Tensor:
    if uses_formula_prior(cfg):
        idx = formula_channel_index(cfg)
        return input_batch[:, idx : idx + 1]
    return torch.zeros_like(reference[:, :1])


def clip_to_target_range(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    """Clamp model outputs in normalized target space.

    If ``clip_min_db`` and ``clip_max_db`` are set (physical units, same as
    ``scale``/``offset`` convention), they override ``clip_min``/``clip_max``.
    """
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    clip_min_db = metadata.get("clip_min_db")
    clip_max_db = metadata.get("clip_max_db")
    if clip_min_db is not None and clip_max_db is not None:
        min_norm = (float(clip_min_db) - offset) / max(scale, 1e-12)
        max_norm = (float(clip_max_db) - offset) / max(scale, 1e-12)
        return values.clamp(min=min_norm, max=max_norm)
    clip_min = metadata.get("clip_min")
    clip_max = metadata.get("clip_max")
    if clip_min is None or clip_max is None:
        return values
    min_norm = (float(clip_min) - offset) / max(scale, 1e-12)
    max_norm = (float(clip_max) - offset) / max(scale, 1e-12)
    return values.clamp(min=min_norm, max=max_norm)


def denormalize(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    return values * scale + offset


def sanitize_masked_target(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked-out / non-finite target pixels with safe finite values.

    Dataset loaders may mark invalid supervision as NaN so excluded regions do
    not look like a real 0 dB target. Losses in this trainer multiply by the
    mask after arithmetic, so we need a finite tensor here to avoid 0 * NaN.
    """
    finite_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(mask > 0.0, finite_target, torch.zeros_like(finite_target))


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


def _gaussian_kernel_1d(
    sigma_px: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
    truncate: float = 3.0,
) -> torch.Tensor:
    sigma = max(float(sigma_px), 1.0e-6)
    radius = max(int(math.ceil(truncate * sigma)), 1)
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1.0e-12)
    return kernel


def _gaussian_blur_2d(image: torch.Tensor, sigma_px: float) -> torch.Tensor:
    if sigma_px <= 0.0:
        return image
    if image.ndim != 4:
        raise ValueError("Expected image tensor with shape [B, C, H, W]")

    kernel_1d = _gaussian_kernel_1d(
        sigma_px,
        dtype=image.dtype,
        device=image.device,
    )
    radius = int((kernel_1d.numel() - 1) // 2)
    channels = int(image.shape[1])

    kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)

    pad_mode = "reflect"
    if image.shape[-1] <= radius or image.shape[-2] <= radius:
        pad_mode = "replicate"

    blurred = F.conv2d(
        F.pad(image, (radius, radius, 0, 0), mode=pad_mode),
        kernel_x,
        groups=channels,
    )
    blurred = F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode=pad_mode),
        kernel_y,
        groups=channels,
    )
    return blurred


def _highpass_residual_db(values_db: torch.Tensor, sigma_px: float) -> torch.Tensor:
    return values_db - _gaussian_blur_2d(values_db, sigma_px)


def _local_std_2d(image: torch.Tensor, sigma_px: float) -> torch.Tensor:
    mean = _gaussian_blur_2d(image, sigma_px)
    var = _gaussian_blur_2d((image - mean) ** 2, sigma_px)
    return torch.sqrt(var.clamp_min(0.0) + 1.0e-8)


def _resolve_region_mask(
    mask: torch.Tensor,
    input_batch: torch.Tensor,
    cfg: Dict[str, Any],
    threshold: float,
    region: str,
) -> torch.Tensor:
    """Return a [B,1,H,W] mask for the target region (los|nlos).

    If the LoS input channel is present (use_los_as_input=true and cfg/input
    agree), combine ``mask`` with the LoS channel thresholded accordingly.
    Otherwise fall back to ``mask`` alone -- appropriate when the caller has
    already filtered pixels via the expert's region mask (e.g. los_region_mask_mode).
    """
    has_los_channel = (
        uses_los_input_channel(cfg)
        and input_batch.ndim == 4
        and input_batch.shape[1] >= 2
    )
    if not has_los_channel:
        return mask
    los = input_batch[:, 1:2].clamp(0.0, 1.0)
    if region == "los":
        sel = (los > threshold).to(mask.dtype)
    else:
        sel = (los <= threshold).to(mask.dtype)
    return mask * sel


def compute_los_highpass_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_batch: torch.Tensor,
    metadata: Dict[str, Any],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    hp_cfg = dict(cfg.get("los_highpass_loss", {}))
    if not bool(hp_cfg.get("enabled", False)):
        return torch.zeros((), device=pred.device)

    # Multi-scale: sigma_px_list takes precedence; else fall back to sigma_px.
    sigma_list = hp_cfg.get("sigma_px_list", None)
    if sigma_list is None:
        sigma_list = [float(hp_cfg.get("sigma_px", 3.0))]
    sigma_list = [max(float(s), 0.0) for s in sigma_list]
    threshold = float(hp_cfg.get("los_threshold", 0.5))
    los_valid = _resolve_region_mask(mask, input_batch, cfg, threshold, "los")
    denom = los_valid.sum().clamp_min(1.0)

    pred_db = denormalize(pred, metadata)
    target_db = denormalize(target, metadata)
    scale_db = max(float(metadata.get("scale", 180.0)), 1.0)

    mode = str(hp_cfg.get("mode", "l2")).lower()
    std_window_px = float(hp_cfg.get("local_std_window_px", 7.0))

    total = torch.zeros((), device=pred.device)
    for sigma_px in sigma_list:
        pred_hp = _highpass_residual_db(pred_db, sigma_px)
        target_hp = _highpass_residual_db(target_db, sigma_px)

        if mode == "local_std":
            # Phase-insensitive: match local std of the highpass residual.
            pred_std = _local_std_2d(pred_hp, std_window_px)
            tgt_std = _local_std_2d(target_hp, std_window_px)
            diff = pred_std - tgt_std
            base = ((diff ** 2) * los_valid).sum() / denom
            normalized = base / scale_db  # dB² -> dB-like via /scale
        elif mode == "l1":
            diff_hp = pred_hp - target_hp
            base = (torch.abs(diff_hp) * los_valid).sum() / denom
            normalized = base / scale_db
        elif mode == "rmse":
            diff_hp = pred_hp - target_hp
            mse = ((diff_hp ** 2) * los_valid).sum() / denom
            normalized = torch.sqrt(mse + 1.0e-12) / scale_db
        else:  # l2
            diff_hp = pred_hp - target_hp
            base = ((diff_hp ** 2) * los_valid).sum() / denom
            normalized = base / scale_db

        total = total + normalized

    total = total / max(len(sigma_list), 1)
    loss_weight = float(hp_cfg.get("loss_weight", 0.1))
    return loss_weight * total


def _sobel_gradients(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 4:
        raise ValueError("Expected tensor with shape [B, C, H, W]")
    channels = int(values.shape[1])
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=values.dtype,
        device=values.device,
    ) / 8.0
    ky = kx.t()
    wx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    wy = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    pad_mode = "reflect"
    if values.shape[-2] < 3 or values.shape[-1] < 3:
        pad_mode = "replicate"
    padded = F.pad(values, (1, 1, 1, 1), mode=pad_mode)
    gx = F.conv2d(padded, wx, groups=channels)
    gy = F.conv2d(padded, wy, groups=channels)
    return gx, gy


def compute_los_gradient_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_batch: torch.Tensor,
    metadata: Dict[str, Any],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    grad_cfg = dict(cfg.get("los_gradient_magnitude_loss", {}))
    if not bool(grad_cfg.get("enabled", False)):
        return torch.zeros((), device=pred.device)

    threshold = float(grad_cfg.get("los_threshold", 0.5))
    eps = max(float(grad_cfg.get("eps", 1.0e-6)), 1.0e-12)
    los_valid = _resolve_region_mask(mask, input_batch, cfg, threshold, "los")
    denom = los_valid.sum().clamp_min(1.0)

    pred_db = denormalize(pred, metadata)
    target_db = denormalize(target, metadata)
    pred_gx, pred_gy = _sobel_gradients(pred_db)
    tgt_gx, tgt_gy = _sobel_gradients(target_db)
    pred_mag = torch.sqrt(pred_gx.pow(2) + pred_gy.pow(2) + eps)
    tgt_mag = torch.sqrt(tgt_gx.pow(2) + tgt_gy.pow(2) + eps)

    mse = (((pred_mag - tgt_mag) ** 2) * los_valid).sum() / denom
    scale_db = max(float(metadata.get("scale", 180.0)), 1.0)
    normalized = mse / scale_db
    return float(grad_cfg.get("loss_weight", 0.05)) * normalized


def compute_los_laplacian_pyramid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_batch: torch.Tensor,
    metadata: Dict[str, Any],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    lap_cfg = dict(cfg.get("los_laplacian_pyramid_loss", {}))
    if not bool(lap_cfg.get("enabled", False)):
        return torch.zeros((), device=pred.device)

    levels = max(int(lap_cfg.get("levels", 3)), 1)
    threshold = float(lap_cfg.get("los_threshold", 0.5))
    min_valid_ratio = float(lap_cfg.get("min_valid_ratio", 0.25))
    coarse_emphasis = bool(lap_cfg.get("coarse_emphasis", True))

    pred_level = denormalize(pred, metadata)
    target_level = denormalize(target, metadata)
    los_mask_level = _resolve_region_mask(mask, input_batch, cfg, threshold, "los")

    total = torch.zeros((), device=pred.device)
    total_weight = 0.0
    for level in range(levels):
        if pred_level.shape[-2] < 2 or pred_level.shape[-1] < 2:
            break

        pred_down = F.avg_pool2d(pred_level, kernel_size=2, stride=2)
        target_down = F.avg_pool2d(target_level, kernel_size=2, stride=2)
        pred_up = F.interpolate(pred_down, size=pred_level.shape[-2:], mode="bilinear", align_corners=False)
        target_up = F.interpolate(target_down, size=target_level.shape[-2:], mode="bilinear", align_corners=False)
        lap_pred = pred_level - pred_up
        lap_tgt = target_level - target_up

        valid = (los_mask_level >= min_valid_ratio).to(los_mask_level.dtype)
        denom = valid.sum().clamp_min(1.0)
        mse = (((lap_pred - lap_tgt) ** 2) * valid).sum() / denom

        level_weight = (2.0 ** level) if coarse_emphasis else (1.0 / (2.0 ** level))
        total = total + level_weight * mse
        total_weight += float(level_weight)

        pred_level = pred_down
        target_level = target_down
        los_mask_level = F.avg_pool2d(los_mask_level, kernel_size=2, stride=2)

    if total_weight <= 0.0:
        return torch.zeros((), device=pred.device)
    scale_db = max(float(metadata.get("scale", 180.0)), 1.0)
    normalized = (total / total_weight) / scale_db
    return float(lap_cfg.get("loss_weight", 0.05)) * normalized


def compute_nlos_dog_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_batch: torch.Tensor,
    metadata: Dict[str, Any],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    dog_cfg = dict(cfg.get("nlos_dog_loss", {}))
    if not bool(dog_cfg.get("enabled", False)):
        return torch.zeros((), device=pred.device)

    threshold = float(dog_cfg.get("los_threshold", 0.5))
    sigma_small = max(float(dog_cfg.get("sigma_small_px", 2.0)), 0.0)
    sigma_large = max(float(dog_cfg.get("sigma_large_px", 6.0)), sigma_small + 1.0e-3)
    nlos_valid = _resolve_region_mask(mask, input_batch, cfg, threshold, "nlos")
    denom = nlos_valid.sum().clamp_min(1.0)

    pred_db = denormalize(pred, metadata)
    target_db = denormalize(target, metadata)
    pred_dog = _gaussian_blur_2d(pred_db, sigma_small) - _gaussian_blur_2d(pred_db, sigma_large)
    target_dog = _gaussian_blur_2d(target_db, sigma_small) - _gaussian_blur_2d(target_db, sigma_large)
    mse = (((pred_dog - target_dog) ** 2) * nlos_valid).sum() / denom

    scale_db = max(float(metadata.get("scale", 180.0)), 1.0)
    normalized = mse / (scale_db ** 2)
    return float(dog_cfg.get("loss_weight", 0.08)) * normalized


def compute_nlos_gradient_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_batch: torch.Tensor,
    metadata: Dict[str, Any],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    grad_cfg = dict(cfg.get("nlos_gradmag_loss", {}))
    if not bool(grad_cfg.get("enabled", False)):
        return torch.zeros((), device=pred.device)

    threshold = float(grad_cfg.get("los_threshold", 0.5))
    eps = max(float(grad_cfg.get("eps", 1.0e-6)), 1.0e-12)
    boundary_boost = max(float(grad_cfg.get("boundary_boost", 1.0)), 1.0)

    nlos_valid = _resolve_region_mask(mask, input_batch, cfg, threshold, "nlos")

    pred_db = denormalize(pred, metadata)
    target_db = denormalize(target, metadata)
    pred_gx, pred_gy = _sobel_gradients(pred_db)
    tgt_gx, tgt_gy = _sobel_gradients(target_db)
    pred_mag = torch.sqrt(pred_gx.pow(2) + pred_gy.pow(2) + eps)
    tgt_mag = torch.sqrt(tgt_gx.pow(2) + tgt_gy.pow(2) + eps)

    weight_map = nlos_valid
    if boundary_boost > 1.0 and uses_los_input_channel(cfg) and input_batch.shape[1] >= 2:
        los = input_batch[:, 1:2].clamp(0.0, 1.0)
        los_bin = (los > threshold).to(pred.dtype)
        edge_x, edge_y = _sobel_gradients(los_bin)
        edge_mag = torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + eps)
        edge_mag = torch.clamp(edge_mag * 4.0, min=0.0, max=1.0)
        weight_map = weight_map * (1.0 + (boundary_boost - 1.0) * edge_mag)

    denom = weight_map.sum().clamp_min(1.0)
    mse = (((pred_mag - tgt_mag) ** 2) * weight_map).sum() / denom

    scale_db = max(float(metadata.get("scale", 180.0)), 1.0)
    normalized = mse / (scale_db ** 2)
    return float(grad_cfg.get("loss_weight", 0.06)) * normalized


def compute_nlos_laplacian_pyramid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_batch: torch.Tensor,
    metadata: Dict[str, Any],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    lap_cfg = dict(cfg.get("nlos_laplacian_pyramid_loss", {}))
    if not bool(lap_cfg.get("enabled", False)):
        return torch.zeros((), device=pred.device)

    levels = max(int(lap_cfg.get("levels", 3)), 1)
    threshold = float(lap_cfg.get("los_threshold", 0.5))
    min_valid_ratio = float(lap_cfg.get("min_valid_ratio", 0.25))
    coarse_emphasis = bool(lap_cfg.get("coarse_emphasis", True))

    pred_level = denormalize(pred, metadata)
    target_level = denormalize(target, metadata)
    nlos_mask_level = _resolve_region_mask(mask, input_batch, cfg, threshold, "nlos")

    total = torch.zeros((), device=pred.device)
    total_weight = 0.0
    for level in range(levels):
        if pred_level.shape[-2] < 2 or pred_level.shape[-1] < 2:
            break

        pred_down = F.avg_pool2d(pred_level, kernel_size=2, stride=2)
        target_down = F.avg_pool2d(target_level, kernel_size=2, stride=2)
        pred_up = F.interpolate(pred_down, size=pred_level.shape[-2:], mode="bilinear", align_corners=False)
        target_up = F.interpolate(target_down, size=target_level.shape[-2:], mode="bilinear", align_corners=False)
        lap_pred = pred_level - pred_up
        lap_tgt = target_level - target_up

        valid = (nlos_mask_level >= min_valid_ratio).to(nlos_mask_level.dtype)
        denom = valid.sum().clamp_min(1.0)
        mse = (((lap_pred - lap_tgt) ** 2) * valid).sum() / denom

        level_weight = (2.0 ** level) if coarse_emphasis else (1.0 / (2.0 ** level))
        total = total + level_weight * mse
        total_weight += float(level_weight)

        pred_level = pred_down
        target_level = target_down
        nlos_mask_level = F.avg_pool2d(nlos_mask_level, kernel_size=2, stride=2)

    if total_weight <= 0.0:
        return torch.zeros((), device=pred.device)
    scale_db = max(float(metadata.get("scale", 180.0)), 1.0)
    normalized = (total / total_weight) / (scale_db ** 2)
    return float(lap_cfg.get("loss_weight", 0.08)) * normalized


def effective_huber_delta(delta: float, path_loss_meta: Dict[str, Any], loss_cfg: Dict[str, Any]) -> float:
    """Map config ``huber_delta`` to the scale of ``|pred - target|`` (normalized path loss).

    Targets are trained in normalized units with ``path_loss.scale`` (e.g. 180 dB full span).
    Project docs treat ``huber_delta`` as **physical dB** at which Huber switches from quadratic
    to linear. In that case ``delta_eff = huber_delta / scale``.

    Set ``loss.huber_delta_normalized: true`` to treat ``huber_delta`` as already in normalized
    residual units (legacy behaviour where large values made the loss purely quadratic).
    """
    if bool(loss_cfg.get("huber_delta_normalized", False)):
        return float(delta)
    scale = max(float(path_loss_meta.get("scale", 1.0)), 1e-12)
    return float(delta) / scale


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 6.0,
) -> torch.Tensor:
    """Element-wise Huber / smooth transition; ``delta`` must be in the same units as ``pred-target``."""
    abs_diff = (pred - target).abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    huber = 0.5 * quadratic.pow(2) + delta * (abs_diff - quadratic)
    denom = mask.sum().clamp_min(1.0)
    return (huber * mask).sum() / denom


def masked_huber_loss_physical_scaled(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    metadata: Dict[str, Any],
    *,
    delta_db: float = 6.0,
) -> torch.Tensor:
    pred_db = denormalize(pred, metadata)
    target_db = denormalize(target, metadata)
    scale_db = max(float(metadata.get("scale", 180.0)), 1.0)
    return masked_huber_loss(pred_db, target_db, mask, delta=float(delta_db)) / scale_db


def _masked_hard_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """Huber loss on normalized residuals, up-weighted on large errors (NLoS focus).

    ``w = 1 + alpha * |pred - target|^gamma`` applied elementwise before masking.
    """
    abs_diff = (pred - target).abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    huber = 0.5 * quadratic.pow(2) + delta * (abs_diff - quadratic)
    w = 1.0 + float(alpha) * abs_diff.pow(float(gamma))
    denom = mask.sum().clamp_min(1.0)
    return (w * huber * mask).sum() / denom


def _cutmix_box(h: int, w: int, lam: float) -> tuple[int, int, int, int]:
    """Random bounding box for CutMix; lam is the fraction to *keep* from the original."""
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(round(h * cut_ratio))
    cut_w = int(round(w * cut_ratio))
    cy = random.randint(0, h)
    cx = random.randint(0, w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    return y1, y2, x1, x2


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


def compute_nlos_focus_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_batch: torch.Tensor,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    focus_cfg = dict(cfg.get("nlos_focus_loss", {}))
    if not bool(focus_cfg.get("enabled", False)):
        return torch.zeros((), device=pred.device)
    if not uses_los_input_channel(cfg):
        return torch.zeros((), device=pred.device)
    if input_batch.shape[1] < 2:
        return torch.zeros((), device=pred.device)

    los = input_batch[:, 1:2].clamp(0.0, 1.0)
    nlos_mask = mask * (los <= 0.5).to(mask.dtype)
    mode = str(focus_cfg.get("mode", "rmse")).lower()
    if mode == "hard_huber":
        delta_raw = float(
            focus_cfg.get(
                "huber_delta",
                cfg.get("loss", {}).get("huber_delta", 6.0),
            )
        )
        meta_pl = dict(cfg.get("target_metadata", {}).get("path_loss", {}))
        loss_cfg = dict(cfg.get("loss", {}))
        delta = effective_huber_delta(delta_raw, meta_pl, loss_cfg)
        alpha = float(focus_cfg.get("hard_huber_alpha", 1.0))
        gamma = float(focus_cfg.get("hard_huber_gamma", 1.0))
        return _masked_hard_huber_loss(
            pred, target, nlos_mask, delta=delta, alpha=alpha, gamma=gamma
        )
    if mode == "rmse":
        return masked_rmse_loss(pred, target, nlos_mask)
    return masked_mse_l1_loss(
        pred,
        target,
        nlos_mask,
        mse_weight=float(focus_cfg.get("mse_weight", 1.0)),
        l1_weight=float(focus_cfg.get("l1_weight", 0.0)),
    )


def build_optimizer(
    cfg: Dict[str, Any],
    module: nn.Module,
    device: object,
    *,
    optimizer_key: str = "optimizer",
    learning_rate_key: str = "learning_rate",
) -> torch.optim.Optimizer:
    optimizer_name = str(cfg["training"].get(optimizer_key, cfg["training"].get("optimizer", "adamw"))).lower()
    learning_rate = float(cfg["training"].get(learning_rate_key, cfg["training"].get("learning_rate", cfg["training"].get("generator_lr", 3e-5))))
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    momentum = float(cfg["training"].get("momentum", 0.0))

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            foreach=is_cuda_device(device),
        )
    elif optimizer_name == "adam":
        betas = (
            float(cfg["training"].get("beta1", 0.9)),
            float(cfg["training"].get("beta2", 0.999)),
        )
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            betas=betas,
            foreach=is_cuda_device(device),
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            module.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            momentum=momentum,
            foreach=is_cuda_device(device),
        )
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")

    configure_optimizer_weight_decay(optimizer, module, weight_decay)
    return optimizer


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    lr = float(learning_rate)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    if isinstance(getattr(optimizer, "defaults", None), dict):
        optimizer.defaults["lr"] = lr


def set_optimizer_weight_decay(optimizer: torch.optim.Optimizer, weight_decay: float) -> None:
    decay = float(weight_decay)
    setattr(optimizer, "_manual_weight_decay", decay)
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = 0.0
    if isinstance(getattr(optimizer, "defaults", None), dict):
        optimizer.defaults["weight_decay"] = 0.0


def apply_optimizer_hparams_from_cfg(
    optimizer: torch.optim.Optimizer,
    *,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    momentum: float,
) -> None:
    set_optimizer_learning_rate(optimizer, learning_rate)
    set_optimizer_weight_decay(optimizer, weight_decay)
    for param_group in optimizer.param_groups:
        if "betas" in param_group:
            param_group["betas"] = (float(beta1), float(beta2))
        if "momentum" in param_group:
            param_group["momentum"] = float(momentum)
    if isinstance(getattr(optimizer, "defaults", None), dict):
        if "betas" in optimizer.defaults:
            optimizer.defaults["betas"] = (float(beta1), float(beta2))
        if "momentum" in optimizer.defaults:
            optimizer.defaults["momentum"] = float(momentum)


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
    if name in {"cosine_warm_restarts", "cosine_annealing_warm_restarts", "sgdr"}:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(cfg["training"].get("lr_scheduler_T0", 40)),
            T_mult=int(cfg["training"].get("lr_scheduler_Tmult", 2)),
            eta_min=float(cfg["training"].get("lr_scheduler_eta_min", 1e-6)),
        )
    if name in {"cosine", "cosine_annealing", "cosine_annealing_lr"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["training"].get("lr_scheduler_T_max", cfg["training"].get("epochs", 300))),
            eta_min=float(cfg["training"].get("lr_scheduler_eta_min", 1e-6)),
        )
    raise ValueError(f"Unsupported lr_scheduler '{name}'.")


def _apply_warmup_lr(
    optimizer: torch.optim.Optimizer,
    optimizer_step: int,
    warmup_steps: int,
    base_lr: float,
    *,
    start_factor: float = 0.5,
) -> None:
    """Linearly ramp LR from start_factor*base_lr to base_lr over warmup optimizer steps."""
    if warmup_steps <= 0 or optimizer_step > warmup_steps:
        return
    start_factor = float(min(max(start_factor, 0.0), 1.0))
    alpha = max(optimizer_step, 1) / warmup_steps
    lr = base_lr * (start_factor + (1.0 - start_factor) * alpha)
    for pg in optimizer.param_groups:
        pg["lr"] = lr


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
    if arch == "unet_h":
        from data_utils import compute_scalar_cond_dim
        scalar_dim = int(compute_scalar_cond_dim(cfg))
        return UNetResidualRefinerH(
            in_channels=in_channels,
            out_channels=int(sep_cfg.get("refiner_out_channels", cfg["model"]["out_channels"])),
            base_channels=int(sep_cfg.get("refiner_base_channels", 48)),
            norm_type=str(cfg["model"].get("norm_type", "group")),
            dropout=float(cfg["model"].get("dropout", 0.0)),
            gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
            scalar_dim=max(scalar_dim, 1),
        )
    raise ValueError(f"Unsupported separated_refiner.refiner_arch '{arch}'")


def _resolve_absolute_output_bias_db(cfg: Dict[str, Any]) -> float:
    model_cfg = dict(cfg.get("model", {}))
    requested = model_cfg.get("absolute_output_bias_init_db", "auto")
    meta = dict(cfg.get("target_metadata", {}).get("path_loss", {}))
    if isinstance(requested, str) and requested.strip().lower() == "auto":
        clip_min = float(meta.get("clip_min", meta.get("offset", 0.0)))
        clip_max = float(meta.get("clip_max", clip_min + float(meta.get("scale", 180.0))))
        return 0.5 * (clip_min + clip_max)
    return float(requested)


def initialize_absolute_prediction_head(module: nn.Module, cfg: Dict[str, Any]) -> bool:
    if not uses_absolute_path_loss_prediction(cfg):
        return False
    final_conv = next(
        (
            m
            for m in reversed(list(module.modules()))
            if isinstance(m, nn.Conv2d) and m.bias is not None
        ),
        None,
    )
    if final_conv is None:
        return False
    meta = dict(cfg.get("target_metadata", {}).get("path_loss", {}))
    scale = max(float(meta.get("scale", 180.0)), 1e-12)
    offset = float(meta.get("offset", 0.0))
    init_db = _resolve_absolute_output_bias_db(cfg)
    init_norm = (init_db - offset) / scale
    with torch.no_grad():
        final_conv.weight.zero_()
        final_conv.bias.fill_(float(init_norm))
    return True


def _checkpoint_model_state(state: Dict[str, Any]) -> Dict[str, Any]:
    if "model" in state:
        return state["model"]
    if "generator" in state:
        return state["generator"]
    raise KeyError("Checkpoint missing model/generator state")


def _net2net_widen_tensor(old_t: torch.Tensor, new_shape: tuple[int, ...]) -> torch.Tensor:
    if old_t.shape == new_shape:
        return old_t.clone()
    if old_t.ndim != len(new_shape):
        raise ValueError(f"ndim mismatch: {tuple(old_t.shape)} vs {new_shape}")
    result = old_t.clone()
    for dim, (old_size, new_size) in enumerate(zip(old_t.shape, new_shape)):
        if old_size == new_size:
            continue
        if new_size != 2 * old_size:
            raise ValueError(f"dim {dim}: not 2x ({old_size}->{new_size})")
        result = torch.cat([result, result], dim=dim)
        if dim >= 1 and result.ndim >= 2 and result.is_floating_point():
            result = result * 0.5
    return result


def _partial_copy_tensor(old_t: torch.Tensor, new_shape: tuple[int, ...], like_t: torch.Tensor) -> torch.Tensor:
    result = like_t.new_zeros(new_shape)
    source = old_t.to(device=like_t.device, dtype=like_t.dtype)
    if source.ndim != len(new_shape):
        return result
    slices = tuple(slice(0, min(old_size, new_size)) for old_size, new_size in zip(source.shape, new_shape))
    result[slices] = source[slices]
    return result


def _remap_checkpoint_tensor(old_t: torch.Tensor, target_t: torch.Tensor) -> tuple[torch.Tensor, str]:
    source_is_float = torch.is_floating_point(old_t)
    source = old_t.to(device=target_t.device, dtype=target_t.dtype)
    if source.shape == target_t.shape:
        return source.clone(), "copy"
    if source_is_float and source.ndim == target_t.ndim:
        try:
            return _net2net_widen_tensor(source, tuple(target_t.shape)), "net2net"
        except ValueError:
            pass
    return _partial_copy_tensor(source, tuple(target_t.shape), target_t), "partial"


def _remap_checkpoint_state(
    module: nn.Module,
    checkpoint_state: Dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, int], bool]:
    target_module = unwrap_model(module)
    target_state = target_module.state_dict()
    remapped_state: dict[str, torch.Tensor] = {}
    stats = {"copied": 0, "net2net": 0, "partial": 0, "new": 0, "extra": 0}
    exact_match = True

    for key, target_tensor in target_state.items():
        source_tensor = checkpoint_state.get(key)
        if source_tensor is None or not torch.is_tensor(source_tensor):
            remapped_state[key] = target_tensor.clone()
            stats["new"] += 1
            exact_match = False
            continue

        remapped_tensor, mode = _remap_checkpoint_tensor(source_tensor, target_tensor)
        remapped_state[key] = remapped_tensor
        if mode == "copy":
            stats["copied"] += 1
        elif mode == "net2net":
            stats["net2net"] += 1
            exact_match = False
        else:
            stats["partial"] += 1
            exact_match = False

    extra_keys = set(checkpoint_state.keys()) - set(target_state.keys())
    if extra_keys:
        stats["extra"] = len(extra_keys)
        exact_match = False

    return remapped_state, stats, exact_match


def _format_remap_stats(stats: dict[str, int]) -> str:
    return (
        f"copied={stats['copied']} net2net={stats['net2net']} "
        f"partial={stats['partial']} new={stats['new']} extra={stats['extra']}"
    )


def _load_adaptive_state_dict(module: nn.Module, checkpoint_state: Dict[str, Any]) -> tuple[dict[str, int], bool]:
    remapped_state, stats, exact_match = _remap_checkpoint_state(module, checkpoint_state)
    unwrap_model(module).load_state_dict(remapped_state, strict=True)
    return stats, exact_match


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
) -> bool:
    state = load_torch_checkpoint(checkpoint_path, device)
    layout_is_exact = True
    target_generator = unwrap_model(generator_model)
    _, gen_exact = _load_adaptive_state_dict(target_generator, _checkpoint_model_state(state))
    layout_is_exact &= gen_exact
    if ema_generator is not None:
        if "generator_ema" in state and state["generator_ema"] is not None:
            _, ema_exact = _load_adaptive_state_dict(ema_generator, state["generator_ema"])
            layout_is_exact &= ema_exact
        else:
            ema_generator.load_state_dict(target_generator.state_dict())
    if discriminator is not None and "discriminator" in state and state["discriminator"] is not None:
        target_discriminator = unwrap_model(discriminator)
        _, disc_exact = _load_adaptive_state_dict(target_discriminator, state["discriminator"])
        layout_is_exact &= disc_exact
    if layout_is_exact:
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
    return layout_is_exact


def _load_generator_weights(module: nn.Module, checkpoint_path: Path, device: object) -> tuple[dict[str, int], bool]:
    state = load_torch_checkpoint(checkpoint_path, device)
    return _load_adaptive_state_dict(module, _checkpoint_model_state(state))


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
    absolute_prediction: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not separated_mode:
        raw_out = forward_cgan_generator(trainable_generator, x, scalar_cond)
        model_out, _, aux_logits = _split_model_output(raw_out, aux_enabled=True, gate_enabled=False)
        return model_out, None, None, aux_logits
    if base_generator is None:
        raise RuntimeError("Separated mode enabled but base_generator is missing")
    if absolute_prediction:
        raise RuntimeError("absolute_prediction=true is not supported with separated_mode")
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
    absolute_prediction: bool = False,
) -> torch.Tensor:
    residual_pred, _, _, _ = _compose_residual_prediction_with_aux(
        trainable_generator,
        x,
        scalar_cond,
        prior,
        separated_mode=separated_mode,
        base_generator=base_generator,
        use_gate=use_gate,
        absolute_prediction=absolute_prediction,
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

    if nlos_boost > 0.0 and uses_los_input_channel(cfg):
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

    def _infer_from_topology(self, topology_tensor: torch.Tensor) -> str:
        raw_topology = topology_tensor.detach().float().cpu().numpy() * self.input_scale + self.input_offset
        raw_topology = np.squeeze(raw_topology)
        non_ground = raw_topology != self.non_ground_threshold
        building_density = float(np.mean(non_ground))
        non_zero = raw_topology[non_ground]
        mean_height = float(np.mean(non_zero)) if non_zero.size else 0.0
        if self.city_type_thresholds:
            return _city_type_from_thresholds(building_density, mean_height, self.city_type_thresholds)
        return _infer_city_type_simple(building_density, mean_height)

    def infer(self, input_batch: torch.Tensor) -> str:
        if input_batch.ndim == 4:
            return self._infer_from_topology(input_batch[0, :1])
        if input_batch.ndim == 3:
            return self._infer_from_topology(input_batch[:1])
        raise ValueError(f"Expected input batch with 3 or 4 dims, got shape={tuple(input_batch.shape)}")

    def infer_batch(self, input_batch: torch.Tensor) -> list[str]:
        if input_batch.ndim == 4:
            return [self._infer_from_topology(input_batch[i, :1]) for i in range(input_batch.shape[0])]
        return [self.infer(input_batch)]


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
    if uses_los_input_channel(cfg):
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
        city_types = city_type_resolver.infer_batch(input_batch)
        city_weight_values = [
            max(city_weights.get(city_type, default_city_weight), 1e-6)
            for city_type in city_types
        ]
        city_weight_tensor = torch.tensor(city_weight_values, dtype=weight.dtype, device=weight.device).view(-1, 1, 1, 1)
        weight = weight * city_weight_tensor

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


def update_fullres_metric_totals_for_sample(
    dataset: Any,
    sample_index: int,
    pred_norm: torch.Tensor,
    meta: dict[str, Any],
    totals: dict[str, float],
    quantized_totals: dict[str, float],
) -> None:
    if not hasattr(dataset, "sample_refs") or not hasattr(dataset, "_get_handle"):
        return
    if sample_index < 0 or sample_index >= len(dataset.sample_refs):
        return
    city, sample = dataset.sample_refs[sample_index]
    handle = dataset._get_handle()
    if city not in handle or sample not in handle[city]:
        return
    grp = handle[city][sample]
    field_name = str(getattr(dataset, "target_field_map", {}).get("path_loss", "path_loss"))
    if field_name not in grp:
        return

    raw_target = np.asarray(grp[field_name][...], dtype=np.float32)
    finite_mask = np.isfinite(raw_target)
    finite_vals = raw_target[finite_mask]
    fill_val = float(np.max(finite_vals)) if finite_vals.size > 0 else 0.0
    raw_fixed = np.where(finite_mask, raw_target, fill_val).astype(np.float32)

    valid_mask_np = np.ones_like(raw_fixed, dtype=bool)
    if bool(getattr(dataset, "exclude_non_ground_targets", False)):
        input_column = str(getattr(dataset, "input_column", "topology_map"))
        if input_column in grp:
            raw_input = np.asarray(grp[input_column][...], dtype=np.float32)
            valid_mask_np &= raw_input == float(getattr(dataset, "non_ground_threshold", 0.0))
    saturation_db = getattr(dataset, "path_loss_saturation_db", None)
    if saturation_db is not None:
        valid_mask_np &= raw_fixed < float(saturation_db)
    if bool(getattr(dataset, "path_loss_ignore_nonfinite", True)):
        valid_mask_np &= finite_mask
    if not np.any(valid_mask_np):
        return

    target_hw = tuple(int(v) for v in raw_fixed.shape)
    pred_up = F.interpolate(pred_norm, size=target_hw, mode="bilinear", align_corners=False)
    pred_phys = denormalize(pred_up, meta)
    target_phys = torch.from_numpy(raw_fixed).to(device=pred_phys.device, dtype=pred_phys.dtype).unsqueeze(0).unsqueeze(0)
    valid_mask = torch.from_numpy(valid_mask_np).to(device=pred_phys.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    diff_phys = (pred_phys - target_phys)[valid_mask]
    update_metric_total(totals, diff_phys)
    update_metric_total_quantized_u8(quantized_totals, pred_phys, target_phys, valid_mask)


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
    partition_filter = dict(data_cfg.get("partition_filter", {}))
    return {
        "topology_class": partition_filter.get("topology_class"),
        "los_sample_filter": data_cfg.get("los_sample_filter"),
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
        "path_loss__los_highpass__LoS",
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
    uses_ema: bool = False,
    ema_decay: float = 0.0,
) -> dict[str, Any]:
    experiment = dict(summary.get("experiment", {}))
    train_metrics = dict(summary.get("_train_metrics", {}))
    payload: dict[str, Any] = {
        "metrics": {
            "path_loss": dict(summary.get("path_loss", {})),
            "train_path_loss": train_metrics,
            "prior_path_loss": dict(summary.get("path_loss__prior__overall", {})),
            "improvement_vs_prior": dict(summary.get("improvement_vs_prior", {})),
        },
        "focus": {
            "topology_class": experiment.get("topology_class"),
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
        "model_info": {
            "val_uses_ema": uses_ema,
            "ema_decay": ema_decay,
            "train_metrics_use_online_weights": True,
            "note": "train RMSE uses online (non-EMA) weights in train() mode; val RMSE uses EMA weights in eval() mode. Val < Train is expected when EMA is active.",
        },
    }
    if "no_data" in summary:
        payload["metrics"]["no_data"] = dict(summary["no_data"])
    if "path_loss_513" in summary:
        payload["metrics"]["path_loss_513"] = dict(summary["path_loss_513"])
    if "selection_proxy" in summary:
        payload["selection_proxy"] = dict(summary["selection_proxy"])
    return payload


def write_live_train_progress(
    out_dir: Path,
    payload: dict[str, Any],
) -> None:
    (out_dir / "train_progress_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_validation_loader(
    dataset: Any,
    device: object,
    cfg: Dict[str, Any],
    *,
    distributed: bool = False,
) -> tuple[DataLoader, list[int]]:
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
        batch_size=int(cfg["data"].get("val_batch_size", cfg["training"]["batch_size"])),
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=is_cuda_device(device),
        persistent_workers=bool(cfg["data"].get("val_persistent_workers", cfg["data"].get("persistent_workers", False))) and val_num_workers > 0,
        **loader_kwargs,
    )
    return loader, sample_indices


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
    loader: Optional[DataLoader] = None,
    sample_indices: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    if loader is None or sample_indices is None:
        loader, sample_indices = build_validation_loader(
            dataset,
            device,
            cfg,
            distributed=distributed,
        )
    else:
        sample_indices = list(sample_indices)
    use_formula_prior = uses_formula_prior(cfg)
    absolute_prediction = uses_absolute_path_loss_prediction(cfg)
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
    totals_fullres = init_metric_totals()
    totals_fullres_quantized = init_metric_totals()
    prior_totals = init_metric_totals()
    prior_totals_quantized = init_metric_totals()
    regime_totals: dict[str, dict[str, float]] = defaultdict(init_metric_totals)
    aux_cfg = _no_data_aux_cfg(cfg)
    no_data_enabled = bool(aux_cfg.get("enabled", False))
    no_data_totals = init_binary_totals()
    los_highpass_cfg = dict(cfg.get("los_highpass_loss", {}))
    los_highpass_sigma_px = max(float(los_highpass_cfg.get("sigma_px", 3.0)), 0.0)
    los_highpass_threshold = float(los_highpass_cfg.get("los_threshold", 0.5))

    model.eval()
    sample_cursor = 0
    # Use no_grad (not inference_mode): DirectML and some backends error on
    # "Cannot set version_counter for inference tensor" when inference tensors
    # from the batch mix with module buffers (e.g. sinusoidal freq table * scalar).
    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader, desc="val", leave=False, disable=distributed and not is_main_process(rank))):
            x, y, m, scalar_cond = unpack_cgan_batch(batch, device)
            prior = extract_formula_prior_or_zero(x, cfg, y[:, :1])
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                residual_pred, _, _, no_data_logits = _compose_residual_prediction_with_aux(
                    model,
                    x,
                    scalar_cond,
                    prior,
                    separated_mode=separated_mode,
                    base_generator=base_generator,
                    use_gate=use_gate,
                    absolute_prediction=absolute_prediction,
                )
            pred = residual_pred if absolute_prediction else (prior + residual_pred)
            if clamp_final:
                pred = clip_to_target_range(pred, meta)

            valid_mask = m[:, :1] > 0.0
            safe_target = sanitize_masked_target(y[:, :1], m[:, :1])
            pred_phys = denormalize(pred, meta)
            target_phys = denormalize(safe_target, meta)
            diff_phys = (pred_phys - target_phys)[valid_mask]
            update_metric_total(totals, diff_phys)
            update_metric_total_quantized_u8(totals_quantized, pred_phys, target_phys, valid_mask)

            los = x[:, 1:2].clamp(0.0, 1.0) if uses_los_input_channel(cfg) and x.shape[1] >= 2 else None
            los_highpass_diff_phys: Optional[torch.Tensor] = None
            if los is not None:
                pred_hp_phys = _highpass_residual_db(pred_phys, los_highpass_sigma_px)
                target_hp_phys = _highpass_residual_db(target_phys, los_highpass_sigma_px)
                los_highpass_diff_phys = pred_hp_phys - target_hp_phys

            if use_formula_prior:
                prior_phys = denormalize(prior, meta)
                prior_diff_phys = (prior_phys - target_phys)[valid_mask]
                update_metric_total(prior_totals, prior_diff_phys)
                update_metric_total_quantized_u8(prior_totals_quantized, prior_phys, target_phys, valid_mask)
            if no_data_enabled and no_data_logits is not None:
                nd_target = _extract_no_data_target(y, m, cfg)
                update_binary_totals(no_data_totals, no_data_logits, nd_target)

            batch_size = int(x.shape[0])
            for sample_offset in range(batch_size):
                base_index = int(sample_indices[sample_cursor + sample_offset])
                scalar_cond_sample = scalar_cond[sample_offset : sample_offset + 1] if scalar_cond is not None else None
                info = annotator.info_for_index(
                    base_index,
                    x[sample_offset : sample_offset + 1],
                    scalar_cond_sample,
                )
                sample_valid_mask = valid_mask[sample_offset : sample_offset + 1]
                sample_diff_phys = (pred_phys - target_phys)[sample_offset : sample_offset + 1][sample_valid_mask]
                if sample_diff_phys.numel() > 0:
                    city_key = f"path_loss__city_type__{info['city_type']}"
                    ant_key = f"path_loss__antenna_bin__{info['antenna_bin']}"
                    update_metric_total(regime_totals[city_key], sample_diff_phys)
                    update_metric_total(regime_totals[ant_key], sample_diff_phys)
                if use_formula_prior:
                    sample_prior_diff_phys = (prior_phys - target_phys)[sample_offset : sample_offset + 1][sample_valid_mask]
                    if sample_prior_diff_phys.numel() > 0:
                        update_metric_total(regime_totals[f"path_loss__prior__city_type__{info['city_type']}"], sample_prior_diff_phys)
                        update_metric_total(regime_totals[f"path_loss__prior__antenna_bin__{info['antenna_bin']}"], sample_prior_diff_phys)

                if los is not None:
                    sample_los = los[sample_offset : sample_offset + 1]
                    los_valid = sample_valid_mask & (sample_los > los_highpass_threshold)
                    nlos_valid = sample_valid_mask & (sample_los <= los_highpass_threshold)
                    update_metric_total(regime_totals["path_loss__los__LoS"], (pred_phys - target_phys)[sample_offset : sample_offset + 1][los_valid])
                    update_metric_total(regime_totals["path_loss__los__NLoS"], (pred_phys - target_phys)[sample_offset : sample_offset + 1][nlos_valid])
                    if los_highpass_diff_phys is not None:
                        update_metric_total(
                            regime_totals["path_loss__los_highpass__LoS"],
                            los_highpass_diff_phys[sample_offset : sample_offset + 1][los_valid],
                        )
                    combo_los = f"path_loss__calibration_regime__{info['city_type']}__LoS__{info['antenna_bin']}"
                    combo_nlos = f"path_loss__calibration_regime__{info['city_type']}__NLoS__{info['antenna_bin']}"
                    update_metric_total(regime_totals[combo_los], (pred_phys - target_phys)[sample_offset : sample_offset + 1][los_valid])
                    update_metric_total(regime_totals[combo_nlos], (pred_phys - target_phys)[sample_offset : sample_offset + 1][nlos_valid])
                    if use_formula_prior:
                        update_metric_total(regime_totals["path_loss__prior__los__LoS"], (prior_phys - target_phys)[sample_offset : sample_offset + 1][los_valid])
                        update_metric_total(regime_totals["path_loss__prior__los__NLoS"], (prior_phys - target_phys)[sample_offset : sample_offset + 1][nlos_valid])
                        combo_prior_los = f"path_loss__prior__calibration_regime__{info['city_type']}__LoS__{info['antenna_bin']}"
                        combo_prior_nlos = f"path_loss__prior__calibration_regime__{info['city_type']}__NLoS__{info['antenna_bin']}"
                        update_metric_total(regime_totals[combo_prior_los], (prior_phys - target_phys)[sample_offset : sample_offset + 1][los_valid])
                        update_metric_total(regime_totals[combo_prior_nlos], (prior_phys - target_phys)[sample_offset : sample_offset + 1][nlos_valid])

                update_fullres_metric_totals_for_sample(
                    dataset,
                    base_index,
                    pred[sample_offset : sample_offset + 1],
                    meta,
                    totals_fullres,
                    totals_fullres_quantized,
                )
            sample_cursor += batch_size

    if distributed and dist.is_initialized():
        payload = {
            "totals": totals,
            "totals_quantized": totals_quantized,
            "totals_fullres": totals_fullres,
            "totals_fullres_quantized": totals_fullres_quantized,
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
        totals_fullres = init_metric_totals()
        totals_fullres_quantized = init_metric_totals()
        prior_totals = init_metric_totals()
        prior_totals_quantized = init_metric_totals()
        merged_regimes: dict[str, dict[str, float]] = defaultdict(init_metric_totals)
        for part in gathered:
            for k, v in part["totals"].items():
                totals[k] += float(v)
            for k, v in part["totals_quantized"].items():
                totals_quantized[k] += float(v)
            for k, v in part["totals_fullres"].items():
                totals_fullres[k] += float(v)
            for k, v in part["totals_fullres_quantized"].items():
                totals_fullres_quantized[k] += float(v)
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
    fullres_total_count = float(totals_fullres.get("count", 0.0))
    los_count = float(regime_totals.get("path_loss__los__LoS", {}).get("count", 0.0))
    nlos_count = float(regime_totals.get("path_loss__los__NLoS", {}).get("count", 0.0))
    los_highpass_count = float(regime_totals.get("path_loss__los_highpass__LoS", {}).get("count", 0.0))
    regime_summary = {
        key: finalize_metric_total(val, unit, total_count=total_count)
        for key, val in sorted(regime_totals.items())
    }
    topology_class = dict(cfg.get("data", {})).get("partition_filter", {}).get("topology_class")
    filtered_regimes = filter_regime_summary_for_topology_class(regime_summary, topology_class)

    path_loss_summary = finalize_metric_total(totals, unit, total_count=total_count)
    attach_quantized_metric_fields(path_loss_summary, totals_quantized, unit, total_count=total_count)
    path_loss_summary_513 = finalize_metric_total(totals_fullres, unit, total_count=fullres_total_count)
    attach_quantized_metric_fields(path_loss_summary_513, totals_fullres_quantized, unit, total_count=fullres_total_count)
    summary: dict[str, Any] = {
        "path_loss": path_loss_summary,
        "path_loss_513": path_loss_summary_513,
        "experiment": build_experiment_summary(cfg, dataset),
        "_support": {
            "sample_count": int(len(dataset)),
            "valid_pixel_count": int(round(total_count)),
            "valid_pixel_count_513": int(round(fullres_total_count)),
            "los_valid_pixel_count": int(round(los_count)),
            "nlos_valid_pixel_count": int(round(nlos_count)),
            "los_highpass_valid_pixel_count": int(round(los_highpass_count)),
            "los_fraction": float(los_count / total_count) if total_count > 0.0 else float("nan"),
            "nlos_fraction": float(nlos_count / total_count) if total_count > 0.0 else float("nan"),
        },
        "_regimes": filtered_regimes,
    }
    if no_data_enabled:
        summary["no_data"] = finalize_binary_totals(no_data_totals)
        summary["_support"]["no_data_positive_fraction"] = float(summary["no_data"]["positive_fraction_target"])
    if use_formula_prior:
        prior_summary = finalize_metric_total(prior_totals, unit, total_count=total_count)
        attach_quantized_metric_fields(prior_summary, prior_totals_quantized, unit, total_count=total_count)
        summary["path_loss__prior__overall"] = prior_summary
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
    selection_alpha = float(cfg.get("nlos_focus_loss", {}).get("selection_alpha", cfg.get("training", {}).get("selection_nlos_alpha", 0.25)))
    nlos_rmse = float(regime_summary.get("path_loss__los__NLoS", {}).get("rmse_physical", path_loss_summary.get("rmse_physical", float("nan"))))
    overall_rmse = float(path_loss_summary.get("rmse_physical", float("nan")))
    los_highpass_rmse = float(regime_summary.get("path_loss__los_highpass__LoS", {}).get("rmse_physical", float("nan")))
    path_loss_summary["los_highpass_rmse_physical"] = los_highpass_rmse
    summary["selection_proxy"] = {
        "overall_rmse_physical": overall_rmse,
        "nlos_rmse_physical": nlos_rmse,
        "los_highpass_rmse_physical": los_highpass_rmse,
        "alpha": selection_alpha,
        "composite_nlos_weighted_rmse": float(overall_rmse + selection_alpha * nlos_rmse),
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
    uses_ema: bool = False,
    ema_decay: float = 0.0,
) -> None:
    payload = build_validation_payload(
        summary,
        epoch=epoch,
        best_epoch=best_epoch,
        best_score=best_score,
        selection_metric=selection_metric,
        current_score=current_score,
        uses_ema=uses_ema,
        ema_decay=ema_decay,
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
    warmup_optimizer_steps: int = 0,
    warmup_start_factor: float = 0.5,
    base_lr: float = 3e-4,
    optimizer_step_offset: int = 0,
    rmse_buffer: Optional["SampleRMSEBuffer"] = None,
) -> tuple[float, float, dict[str, Any], dict[str, float], int]:
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
    nlos_focus_cfg = dict(cfg.get("nlos_focus_loss", {}))
    nlos_focus_weight = float(nlos_focus_cfg.get("loss_weight", 0.0)) if bool(nlos_focus_cfg.get("enabled", False)) else 0.0
    adv_criterion = build_adversarial_loss(str(loss_cfg.get("adversarial_loss", "bce"))).to(device)
    use_gan = lambda_gan > 0.0 and discriminator is not None and optimizer_d is not None and scaler_d is not None

    generator.train()
    if discriminator is not None:
        discriminator.train()
    source_generator = unwrap_model(generator)
    running_g = 0.0
    running_d = 0.0
    running_final = 0.0
    running_residual = 0.0
    running_multiscale = 0.0
    running_gate = 0.0
    running_gan = 0.0
    running_nlos_focus = 0.0
    running_los_highpass = 0.0
    running_los_gradmag = 0.0
    running_los_laplacian = 0.0
    running_nlos_dog = 0.0
    running_nlos_gradmag = 0.0
    running_nlos_laplacian = 0.0
    running_term_final = 0.0
    running_term_residual = 0.0
    running_term_multiscale = 0.0
    running_term_gate = 0.0
    running_term_gan = 0.0
    running_term_nlos_focus = 0.0
    running_term_los_highpass = 0.0
    running_term_los_gradmag = 0.0
    running_term_los_laplacian = 0.0
    running_term_nlos_dog = 0.0
    running_term_nlos_gradmag = 0.0
    running_term_nlos_laplacian = 0.0
    running_main_map_regression = 0.0
    train_metric_totals = init_metric_totals()
    train_metric_totals_quantized = init_metric_totals()
    steps = 0
    epoch_started = time.perf_counter()
    progress_every = 25

    grad_accum_steps = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    cutmix_prob = float(cfg["training"].get("cutmix_prob", 0.0))
    cutmix_alpha = float(cfg["training"].get("cutmix_alpha", 1.0))
    nlos_reweight_factor = float(cfg["training"].get("nlos_reweight_factor", 1.0))
    loss_type = str(cfg.get("loss", {}).get("loss_type", "mse")).lower()
    huber_delta_cfg = float(cfg.get("loss", {}).get("huber_delta", 6.0))
    huber_delta_eff = effective_huber_delta(huber_delta_cfg, meta, loss_cfg)
    has_los_channel = uses_los_input_channel(cfg)
    full_map_scale = float(loss_cfg.get("full_map_generator_scale", 1.0))
    objective_lower = str(cfg.get("training", {}).get("generator_objective", "legacy")).lower()
    absolute_prediction = uses_absolute_path_loss_prediction(cfg)

    prev_cutmix_buf: dict[str, torch.Tensor | None] = {"x": None, "y": None, "m": None, "sc": None}
    accum_count = 0
    optimizer_steps_completed = 0

    _srr_cfg = dict(cfg.get("sample_rmse_reweight", {}))
    _srr_enabled = bool(_srr_cfg.get("enabled", False)) and rmse_buffer is not None

    for batch in tqdm(loader, desc="train", leave=False, disable=distributed and not is_main_process(rank)):
        x, y, m, scalar_cond, _batch_indices = _unpack_train_batch(batch, device)

        # --- CutMix: mix with buffered previous sample ---
        # When FiLM uses a global height vector, spatial CutMix pastes another sample's
        # maps (formula / tx-depth / elevation were built with that height) while
        # scalar_cond still matches the current sample — inconsistent supervision.
        do_cutmix = (
            cutmix_prob > 0.0
            and prev_cutmix_buf["x"] is not None
            and random.random() < cutmix_prob
        )
        if do_cutmix and return_scalar_cond_from_config(cfg) and scalar_cond is not None:
            do_cutmix = False
        if do_cutmix:
            lam = random.betavariate(cutmix_alpha, cutmix_alpha) if cutmix_alpha > 0.0 else 0.5
            _, _, h, w = x.shape
            by1, by2, bx1, bx2 = _cutmix_box(h, w, lam)
            px, py, pm = prev_cutmix_buf["x"], prev_cutmix_buf["y"], prev_cutmix_buf["m"]
            if px is not None and py is not None and pm is not None:
                if px.shape[2:] == x.shape[2:]:
                    x[:, :, by1:by2, bx1:bx2] = px[:, :, by1:by2, bx1:bx2]
                    y[:, :, by1:by2, bx1:bx2] = py[:, :, by1:by2, bx1:bx2]
                    m[:, :, by1:by2, bx1:bx2] = pm[:, :, by1:by2, bx1:bx2]
        prev_cutmix_buf["x"] = x.detach().clone()
        prev_cutmix_buf["y"] = y.detach().clone()
        prev_cutmix_buf["m"] = m.detach().clone()

        mask = m[:, :1]
        target = sanitize_masked_target(y[:, :1], mask)
        # --- Target label noise (addresses uint8 1-dB quantisation, improves NLoS tail) ---
        _tgt_noise_cfg = dict(cfg.get("training", {}).get("target_noise", {}))
        if bool(_tgt_noise_cfg.get("enabled", False)):
            _tn_sigma_db = float(_tgt_noise_cfg.get("sigma_db", 0.5))
            _tn_scale = max(float(cfg["target_metadata"]["path_loss"].get("scale", 180.0)), 1e-6)
            target = target + torch.randn_like(target) * (_tn_sigma_db / _tn_scale)

        # --- NLoS pixel reweighting in loss mask ---
        if nlos_reweight_factor > 1.0 and has_los_channel and x.shape[1] >= 2:
            los_channel = x[:, 1:2].clamp(0.0, 1.0)
            nlos_pixels = (los_channel <= 0.5).to(mask.dtype)
            mask = mask * (1.0 + (nlos_reweight_factor - 1.0) * nlos_pixels)

        prior = extract_formula_prior_or_zero(x, cfg, target)
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
                    absolute_prediction=absolute_prediction,
                )
                fake_det = residual_fake_det if absolute_prediction else (prior + residual_fake_det)
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
            apply_optimizer_weight_decay(optimizer_d)
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            d_loss = torch.zeros((), device=target.device)

        if discriminator is not None:
            set_requires_grad(discriminator, False)
        if accum_count == 0:
            optimizer_g.zero_grad(set_to_none=True)
        main_map_reg = torch.zeros((), device=target.device)
        with amp.autocast(device_type="cuda", enabled=amp_enabled):
            residual_pred, gate_logits, base_residual, no_data_logits = _compose_residual_prediction_with_aux(
                generator,
                x,
                scalar_cond,
                prior,
                separated_mode=separated_mode,
                base_generator=base_generator,
                use_gate=use_gate,
                absolute_prediction=absolute_prediction,
            )
            pred = residual_pred if absolute_prediction else (prior + residual_pred)
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
            nlos_focus_loss = compute_nlos_focus_loss(pred, target, mask, x, cfg)
            los_highpass_loss = compute_los_highpass_loss(pred, target, mask, x, meta, cfg)
            los_gradmag_loss = compute_los_gradient_magnitude_loss(pred, target, mask, x, meta, cfg)
            los_laplacian_loss = compute_los_laplacian_pyramid_loss(pred, target, mask, x, meta, cfg)
            nlos_dog_loss = compute_nlos_dog_loss(pred, target, mask, x, meta, cfg)
            nlos_gradmag_loss = compute_nlos_gradient_magnitude_loss(pred, target, mask, x, meta, cfg)
            nlos_laplacian_loss = compute_nlos_laplacian_pyramid_loss(pred, target, mask, x, meta, cfg)
            no_data_loss = torch.zeros((), device=target.device)
            if use_gate and gate_logits is not None and gate_target is not None:
                gate_loss_map = F.binary_cross_entropy_with_logits(gate_logits, gate_target, reduction="none")
                gate_loss = (gate_loss_map * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)

            if use_gan and discriminator is not None:
                fake_logits_for_g = discriminator(x, pred)
                gan_loss = adv_criterion(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
            else:
                gan_loss = torch.zeros((), device=target.device)

            if objective_mode == "full_map_rmse_only":
                if loss_type == "huber":
                    if absolute_prediction and bool(loss_cfg.get("full_map_huber_physical_db", True)):
                        main_map_reg = masked_huber_loss_physical_scaled(
                            pred,
                            target,
                            mask,
                            meta,
                            delta_db=huber_delta_cfg,
                        )
                    else:
                        main_map_reg = masked_huber_loss(pred, target, mask, delta=huber_delta_eff)
                else:
                    main_map_reg = masked_rmse_loss(pred, target, mask)
                ms_loss = compute_multiscale_path_loss_loss(pred, target, mask, meta, cfg)
                main_scaled = main_map_reg * full_map_scale
                ms_scaled = ms_loss * full_map_scale
                g_loss = main_scaled + ms_scaled
                term_final = torch.zeros((), device=target.device)
                term_residual = torch.zeros((), device=target.device)
                term_multiscale = ms_scaled.detach()
            elif optimize_residual_only:
                residual_only_weight = float(residual_cfg.get("loss_weight", 1.0))
                final_only_weight = float(residual_cfg.get("final_loss_weight_when_residual_only", 0.0))
                multiscale_only_weight = float(residual_cfg.get("multiscale_loss_weight_when_residual_only", 0.0))
                term_final = final_only_weight * final_loss
                term_residual = residual_only_weight * residual_loss
                term_multiscale = multiscale_only_weight * multiscale_loss
                g_loss = (
                    term_residual
                    + term_final
                    + term_multiscale
                )
            else:
                term_final = lambda_recon * final_loss
                term_residual = float(residual_cfg.get("loss_weight", 0.0)) * residual_loss
                term_multiscale = multiscale_loss
                g_loss = (
                    term_final
                    + term_residual
                    + term_multiscale
                    + lambda_gan * gan_loss
                )
            term_gan = lambda_gan * gan_loss

            term_gate = torch.zeros((), device=target.device)
            if gate_loss_weight > 0.0:
                term_gate = gate_loss_weight * gate_loss
                g_loss = g_loss + term_gate
            term_nlos_focus = torch.zeros((), device=target.device)
            if nlos_focus_weight > 0.0:
                term_nlos_focus = nlos_focus_weight * nlos_focus_loss
                g_loss = g_loss + term_nlos_focus
            term_los_highpass = los_highpass_loss
            g_loss = g_loss + term_los_highpass
            term_los_gradmag = los_gradmag_loss
            g_loss = g_loss + term_los_gradmag
            term_los_laplacian = los_laplacian_loss
            g_loss = g_loss + term_los_laplacian
            term_nlos_dog = nlos_dog_loss
            g_loss = g_loss + term_nlos_dog
            term_nlos_gradmag = nlos_gradmag_loss
            g_loss = g_loss + term_nlos_gradmag
            term_nlos_laplacian = nlos_laplacian_loss
            g_loss = g_loss + term_nlos_laplacian
            if bool(_no_data_aux_cfg(cfg).get("enabled", False)) and no_data_logits is not None:
                no_data_target = _extract_no_data_target(y, m, cfg)
                no_data_loss = _compute_no_data_loss(no_data_logits, no_data_target, cfg)
                g_loss = g_loss + float(_no_data_aux_cfg(cfg).get("loss_weight", 0.0)) * no_data_loss

        # --- RL-style per-sample loss reweighting ---
        if _srr_enabled and _batch_indices is not None:
            _sample_weight = rmse_buffer.weight_for(_batch_indices)  # type: ignore[union-attr]
            g_loss = g_loss * _sample_weight
            # Update buffer with current batch RMSE (physical dB, outside autocast)
            with torch.no_grad():
                _pred_phys_srr = denormalize(pred.detach(), meta)
                _tgt_phys_srr = denormalize(target.detach(), meta)
                _mask_srr = (mask > 0.0).squeeze(1)  # (B, H, W)
                _batch_size_srr = _pred_phys_srr.shape[0]
                _per_sample_rmse: list[float] = []
                for _bi in range(_batch_size_srr):
                    _valid = _mask_srr[_bi]
                    _diff = (_pred_phys_srr[_bi, 0] - _tgt_phys_srr[_bi, 0])[_valid]
                    _per_sample_rmse.append(float(_diff.pow(2).mean().sqrt()) if _valid.any() else float("nan"))
                rmse_buffer.update(_batch_indices, _per_sample_rmse)

        scaled_loss = g_loss / grad_accum_steps
        scaler_g.scale(scaled_loss).backward()
        accum_count += 1

        if accum_count >= grad_accum_steps or (steps + 1) == len(loader):
            current_optimizer_step = optimizer_step_offset + optimizer_steps_completed + 1
            if warmup_optimizer_steps > 0 and current_optimizer_step <= warmup_optimizer_steps:
                _apply_warmup_lr(
                    optimizer_g,
                    current_optimizer_step,
                    warmup_optimizer_steps,
                    base_lr,
                    start_factor=warmup_start_factor,
                )
            if clip_grad > 0.0:
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad)
            apply_optimizer_weight_decay(optimizer_g)
            scaler_g.step(optimizer_g)
            scaler_g.update()
            optimizer_g.zero_grad(set_to_none=True)
            accum_count = 0
            optimizer_steps_completed += 1

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
        running_final += float(final_loss.item())
        running_residual += float(residual_loss.item())
        running_multiscale += float(multiscale_loss.item())
        running_gate += float(gate_loss.item())
        running_gan += float(gan_loss.item())
        running_nlos_focus += float(nlos_focus_loss.item())
        running_los_highpass += float(los_highpass_loss.item())
        running_los_gradmag += float(los_gradmag_loss.item())
        running_los_laplacian += float(los_laplacian_loss.item())
        running_nlos_dog += float(nlos_dog_loss.item())
        running_nlos_gradmag += float(nlos_gradmag_loss.item())
        running_nlos_laplacian += float(nlos_laplacian_loss.item())
        running_term_final += float(term_final.item())
        running_term_residual += float(term_residual.item())
        running_term_multiscale += float(term_multiscale.item())
        running_term_gate += float(term_gate.item())
        running_term_gan += float(term_gan.item())
        running_term_nlos_focus += float(term_nlos_focus.item())
        running_term_los_highpass += float(term_los_highpass.item())
        running_term_los_gradmag += float(term_los_gradmag.item())
        running_term_los_laplacian += float(term_los_laplacian.item())
        running_term_nlos_dog += float(term_nlos_dog.item())
        running_term_nlos_gradmag += float(term_nlos_gradmag.item())
        running_term_nlos_laplacian += float(term_nlos_laplacian.item())
        running_main_map_regression += float(
            (main_map_reg * full_map_scale).detach().item()
            if objective_lower == "full_map_rmse_only"
            else main_map_reg.detach().item()
        )
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
                "train_rmse_physical_running": float(
                    finalize_metric_total(train_metric_totals, str(meta.get("unit", "dB"))).get("rmse_physical", float("nan"))
                ),
                "train_rmse_physical_quantized_u8_running": float(
                    finalize_metric_total(train_metric_totals_quantized, str(meta.get("unit", "dB"))).get("rmse_physical", float("nan"))
                ),
                "learning_rate": float(optimizer_g.param_groups[0]["lr"]),
                "generator_objective": str(cfg.get("training", {}).get("generator_objective", "legacy")),
                "loss_components_running": {
                    "final_loss": float(running_final / steps),
                    "residual_loss": float(running_residual / steps),
                    "multiscale_loss": float(running_multiscale / steps),
                    "gate_loss": float(running_gate / steps),
                    "gan_loss": float(running_gan / steps),
                    "nlos_focus_loss": float(running_nlos_focus / steps),
                    "los_highpass_loss": float(running_los_highpass / steps),
                    "los_gradmag_loss": float(running_los_gradmag / steps),
                    "los_laplacian_loss": float(running_los_laplacian / steps),
                    "nlos_dog_loss": float(running_nlos_dog / steps),
                    "nlos_gradmag_loss": float(running_nlos_gradmag / steps),
                    "nlos_laplacian_loss": float(running_nlos_laplacian / steps),
                    "term_final": float(running_term_final / steps),
                    "term_residual": float(running_term_residual / steps),
                    "term_multiscale": float(running_term_multiscale / steps),
                    "term_gate": float(running_term_gate / steps),
                    "term_gan": float(running_term_gan / steps),
                    "term_nlos_focus": float(running_term_nlos_focus / steps),
                    "term_los_highpass": float(running_term_los_highpass / steps),
                    "term_los_gradmag": float(running_term_los_gradmag / steps),
                    "term_los_laplacian": float(running_term_los_laplacian / steps),
                    "term_nlos_dog": float(running_term_nlos_dog / steps),
                    "term_nlos_gradmag": float(running_term_nlos_gradmag / steps),
                    "term_nlos_laplacian": float(running_term_nlos_laplacian / steps),
                    "generator_loss_total": float(running_g / steps),
                    "main_map_regression_loss": float(running_main_map_regression / steps),
                },
            }
            if use_gan:
                live_payload["discriminator_loss_running"] = float(running_d / steps)
            write_live_train_progress(out_dir, live_payload)
    denom = max(steps, 1)
    train_metrics = finalize_metric_total(train_metric_totals, str(meta.get("unit", "dB")))
    attach_quantized_metric_fields(train_metrics, train_metric_totals_quantized, str(meta.get("unit", "dB")))
    train_loss_components = {
        "final_loss": float(running_final / denom),
        "residual_loss": float(running_residual / denom),
        "multiscale_loss": float(running_multiscale / denom),
        "gate_loss": float(running_gate / denom),
        "gan_loss": float(running_gan / denom),
        "nlos_focus_loss": float(running_nlos_focus / denom),
        "los_highpass_loss": float(running_los_highpass / denom),
        "los_gradmag_loss": float(running_los_gradmag / denom),
        "los_laplacian_loss": float(running_los_laplacian / denom),
        "nlos_dog_loss": float(running_nlos_dog / denom),
        "nlos_gradmag_loss": float(running_nlos_gradmag / denom),
        "nlos_laplacian_loss": float(running_nlos_laplacian / denom),
        "term_final": float(running_term_final / denom),
        "term_residual": float(running_term_residual / denom),
        "term_multiscale": float(running_term_multiscale / denom),
        "term_gate": float(running_term_gate / denom),
        "term_gan": float(running_term_gan / denom),
        "term_nlos_focus": float(running_term_nlos_focus / denom),
        "term_los_highpass": float(running_term_los_highpass / denom),
        "term_los_gradmag": float(running_term_los_gradmag / denom),
        "term_los_laplacian": float(running_term_los_laplacian / denom),
        "term_nlos_dog": float(running_term_nlos_dog / denom),
        "term_nlos_gradmag": float(running_term_nlos_gradmag / denom),
        "term_nlos_laplacian": float(running_term_nlos_laplacian / denom),
        "generator_loss_total": float(running_g / denom),
        "main_map_regression_loss": float(running_main_map_regression / denom),
    }
    if _srr_enabled and rmse_buffer is not None:
        rmse_buffer.sync_distributed(distributed)
    return running_g / denom, running_d / denom, train_metrics, train_loss_components, optimizer_steps_completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Try 61 topology expert for direct path-loss prediction with explicit NLoS emphasis")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="If set, overrides training.epochs (e.g. short ablation smoke runs).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs_override is not None:
        cfg.setdefault("training", {})["epochs"] = max(1, int(args.epochs_override))
    anchor_data_paths_to_config_file(cfg, args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg["runtime"]["device"])
    distributed, rank, local_rank, world_size = maybe_init_distributed(device)
    if distributed and is_cuda_device(device):
        device = torch.device("cuda", local_rank)
    configure_cuda_training_backends(device)

    out_dir = ensure_output_dir(cfg["runtime"]["output_dir"])
    if is_main_process(rank):
        print(json.dumps({"output_dir": str(out_dir), "world_size": world_size}))

    splits = build_dataset_splits_from_config(cfg)
    train_dataset = splits["train"]
    val_dataset = splits["val"]

    # --- RL-style per-sample reweighting setup ---
    _srr_main_cfg = dict(cfg.get("sample_rmse_reweight", {}))
    _srr_main_enabled = bool(_srr_main_cfg.get("enabled", False))
    rmse_buffer: Optional[SampleRMSEBuffer] = None
    if _srr_main_enabled:
        train_dataset = _IndexedDataset(train_dataset)
        rmse_buffer = SampleRMSEBuffer(
            n_samples=len(train_dataset),
            ema_alpha=float(_srr_main_cfg.get("ema_alpha", 0.7)),
            gamma=float(_srr_main_cfg.get("gamma", 1.5)),
            min_weight=float(_srr_main_cfg.get("min_weight", 0.5)),
            max_weight=float(_srr_main_cfg.get("max_weight", 3.0)),
        )
        if is_main_process(rank):
            print(json.dumps({"sample_rmse_reweight": {"enabled": True, "n_samples": len(train_dataset),
                "ema_alpha": rmse_buffer.ema_alpha, "gamma": rmse_buffer.gamma}}))

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
        persistent_workers=bool(cfg["data"].get("persistent_workers", True))
        and int(cfg["data"]["num_workers"]) > 0,
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
        base_stats, base_exact = _load_generator_weights(base_generator, base_ckpt, device)
        if is_main_process(rank) and not base_exact:
            print(f"[base] Adapted checkpoint: {_format_remap_stats(base_stats)}")
        base_generator.eval()
        set_requires_grad(base_generator, False)
        generator = _build_refiner_from_cfg(cfg, input_channels + 2).to(device)
    else:
        generator = _build_pmnet_from_cfg(cfg, input_channels).to(device)
        absolute_head_initialized = initialize_absolute_prediction_head(generator, cfg)
        if absolute_head_initialized and is_main_process(rank):
            print(
                json.dumps(
                    {
                        "absolute_prediction_init": {
                            "enabled": True,
                            "bias_init_db": _resolve_absolute_output_bias_db(cfg),
                            "strategy": "zero_final_conv_weights_plus_constant_bias",
                        }
                    }
                )
            )

    ema_decay = resolve_ema_decay(cfg)
    ema_generator: Optional[nn.Module] = create_ema_model(generator) if ema_decay > 0.0 else None
    # --- SWA init (Stochastic Weight Averaging for city-holdout generalisation) ---
    _swa_cfg = dict(cfg.get("training", {}).get("swa", {}))
    _swa_enabled = bool(_swa_cfg.get("enabled", False))
    _swa_start_fraction = float(_swa_cfg.get("start_fraction", 0.6))
    _total_epochs = int(cfg["training"]["epochs"])
    _swa_start_epoch = max(1, int(_swa_start_fraction * _total_epochs))
    swa_generator: Optional[nn.Module] = create_ema_model(generator) if _swa_enabled else None
    _swa_count = 0

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

    optimizer_g = build_optimizer(cfg, generator_for_training, device)
    generator_beta1 = float(cfg["training"].get("beta1", 0.5))
    generator_beta2 = float(cfg["training"].get("beta2", 0.999))
    generator_momentum = float(cfg["training"].get("momentum", 0.0))
    optimizer_d: Optional[torch.optim.Optimizer] = None
    if discriminator_for_training is not None:
        optimizer_d = build_optimizer(
            cfg,
            discriminator_for_training,
            device,
            optimizer_key="discriminator_optimizer",
            learning_rate_key="discriminator_lr",
        )
    scheduler_g = build_scheduler(cfg, optimizer_g)
    scaler_g = amp.GradScaler(enabled=bool(cfg["training"].get("amp", True)) and is_cuda_device(device))
    scaler_d: Optional[amp.GradScaler] = None
    if discriminator_for_training is not None:
        scaler_d = amp.GradScaler(enabled=bool(cfg["training"].get("amp", True)) and is_cuda_device(device))

    val_loader, val_sample_indices = build_validation_loader(
        val_dataset,
        device,
        cfg,
        distributed=distributed,
    )

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    resume_cfg_key = sep_cfg.get("refiner_resume_checkpoint") if separated_mode else cfg["runtime"].get("resume_checkpoint")
    resume_path = resolve_resume_checkpoint(out_dir, resume_cfg_key)
    weights_only_resume = bool(cfg.get("training", {}).get("resume_weights_only", False))
    resume_layout_is_exact = True
    if resume_path and resume_path.exists():
        state = load_torch_checkpoint(resume_path, device)
        if not weights_only_resume:
            start_epoch = int(state.get("epoch", 0)) + 1
            best_score = float(state.get("best_score", best_score))
            best_epoch = int(state.get("best_epoch", best_epoch))
        target_generator = generator_for_training.module if isinstance(generator_for_training, DistributedDataParallel) else generator_for_training
        gen_stats, gen_exact = _load_adaptive_state_dict(target_generator, _checkpoint_model_state(state))
        resume_layout_is_exact &= gen_exact
        if is_main_process(rank) and not gen_exact:
            print(f"[resume] Adapted generator checkpoint: {_format_remap_stats(gen_stats)}")
        if "discriminator" in state and discriminator_for_training is not None:
            target_discriminator = discriminator_for_training.module if isinstance(discriminator_for_training, DistributedDataParallel) else discriminator_for_training
            disc_stats, disc_exact = _load_adaptive_state_dict(target_discriminator, state["discriminator"])
            resume_layout_is_exact &= disc_exact
            if is_main_process(rank) and not disc_exact:
                print(f"[resume] Adapted discriminator checkpoint: {_format_remap_stats(disc_stats)}")
        if ema_generator is not None:
            if "generator_ema" in state and state["generator_ema"] is not None:
                ema_stats, ema_exact = _load_adaptive_state_dict(ema_generator, state["generator_ema"])
                resume_layout_is_exact &= ema_exact
                if is_main_process(rank) and not ema_exact:
                    print(f"[resume] Adapted EMA checkpoint: {_format_remap_stats(ema_stats)}")
            else:
                ema_generator.load_state_dict(target_generator.state_dict())
        if not weights_only_resume and resume_layout_is_exact:
            if "optimizer_g" in state:
                optimizer_g.load_state_dict(state["optimizer_g"])
                move_optimizer_state_to_device(optimizer_g, device)
            elif "optimizer" in state:
                optimizer_g.load_state_dict(state["optimizer"])
                move_optimizer_state_to_device(optimizer_g, device)
        if not weights_only_resume:
            if "optimizer_d" in state and optimizer_d is not None:
                optimizer_d.load_state_dict(state["optimizer_d"])
                move_optimizer_state_to_device(optimizer_d, device)
        restored_scheduler_g = False
        if not weights_only_resume:
            if "scheduler_g" in state and scheduler_g is not None:
                scheduler_state = state["scheduler_g"]
                if scheduler_state is not None:
                    try:
                        scheduler_g.load_state_dict(scheduler_state)
                        restored_scheduler_g = True
                    except Exception as e:
                        if is_main_process(rank):
                            print(f"[resume] Scheduler state incompatible (type changed?), reinitializing: {e}")
            elif "scheduler" in state and scheduler_g is not None:
                scheduler_state = state["scheduler"]
                if scheduler_state is not None:
                    try:
                        scheduler_g.load_state_dict(scheduler_state)
                        restored_scheduler_g = True
                    except Exception as e:
                        if is_main_process(rank):
                            print(f"[resume] Scheduler state incompatible (type changed?), reinitializing: {e}")
            if "scaler_g" in state:
                scaler_g.load_state_dict(state["scaler_g"])
            elif "scaler" in state:
                scaler_g.load_state_dict(state["scaler"])
            if "scaler_d" in state and scaler_d is not None:
                scaler_d.load_state_dict(state["scaler_d"])
        elif not weights_only_resume and is_main_process(rank) and not resume_layout_is_exact:
            print("[resume] Checkpoint layout differs from the current model; optimizer/scheduler/scaler state will be reinitialized.")
        config_generator_lr = float(cfg["training"].get("learning_rate", cfg["training"].get("generator_lr", 3e-5)))
        resume_weight_decay = float(cfg["training"].get("weight_decay", 0.0))
        apply_optimizer_hparams_from_cfg(
            optimizer_g,
            learning_rate=config_generator_lr,
            weight_decay=resume_weight_decay,
            beta1=generator_beta1,
            beta2=generator_beta2,
            momentum=generator_momentum,
        )
        if scheduler_g is not None and hasattr(scheduler_g, "base_lrs"):
            old_base = scheduler_g.base_lrs[0] if scheduler_g.base_lrs else None
            scheduler_g.base_lrs = [config_generator_lr for _ in scheduler_g.base_lrs]
            is_plateau = isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau)
            if not restored_scheduler_g:
                if not is_plateau:
                    for _ff in range(start_epoch - 1):
                        scheduler_g.step()
                if is_main_process(rank):
                    print(f"[resume] Scheduler fast-forwarded to epoch {start_epoch - 1}, lr={optimizer_g.param_groups[0]['lr']:.2e}")
            else:
                if not is_plateau:
                    scheduler_g.step()
                if is_main_process(rank):
                    print(f"[resume] Scheduler state restored, base_lrs overridden {old_base:.2e} -> {config_generator_lr:.2e}, lr={optimizer_g.param_groups[0]['lr']:.2e}")
        if optimizer_d is not None:
            config_discriminator_lr = float(cfg["training"].get("discriminator_lr", config_generator_lr))
            resume_discriminator_lr = float(optimizer_d.param_groups[0].get("lr", config_discriminator_lr))
            apply_optimizer_hparams_from_cfg(
                optimizer_d,
                learning_rate=resume_discriminator_lr,
                weight_decay=resume_weight_decay,
                beta1=generator_beta1,
                beta2=generator_beta2,
                momentum=generator_momentum,
            )
        if scheduler_g is not None and isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
            cfg_min_lr = float(cfg["training"].get("lr_scheduler_min_lr", 1e-6))
            if hasattr(scheduler_g, "min_lrs"):
                scheduler_g.min_lrs = [cfg_min_lr for _ in optimizer_g.param_groups]
            for pg in optimizer_g.param_groups:
                if float(pg["lr"]) < cfg_min_lr:
                    pg["lr"] = cfg_min_lr
        if scheduler_g is not None and isinstance(scheduler_g, torch.optim.lr_scheduler.CosineAnnealingLR):
            eta_c = float(cfg["training"].get("lr_scheduler_eta_min", 1e-6))
            curr_lr = float(optimizer_g.param_groups[0]["lr"])
            if curr_lr + 1e-15 < eta_c:
                for pg in optimizer_g.param_groups:
                    pg["lr"] = max(float(pg["lr"]), eta_c)
                if is_main_process(rank):
                    print(f"[resume] Clamped generator LR to cosine eta_min={eta_c:.2e} (scheduler lr was below floor).")
        if is_main_process(rank):
            print(
                json.dumps(
                    {
                        "resume_from": str(resume_path),
                        "weights_only": weights_only_resume,
                        "start_epoch": start_epoch,
                        "learning_rate_config": config_generator_lr,
                        "weight_decay": resume_weight_decay,
                        "discriminator_learning_rate": float(cfg["training"].get("discriminator_lr", config_generator_lr)) if optimizer_d is not None else None,
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

    if is_main_process(rank) and is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(device)
        alloc_mb = torch.cuda.memory_allocated(device) / 1e6
        reserved_mb = torch.cuda.memory_reserved(device) / 1e6
        total_mb = torch.cuda.get_device_properties(device).total_memory / 1e6
        print(json.dumps({"vram_pre_train_mb": {"allocated": round(alloc_mb, 1), "reserved": round(reserved_mb, 1), "total": round(total_mb, 1)}}))

    warmup_optimizer_steps = int(cfg["training"].get("lr_warmup_optimizer_steps", 0))
    warmup_start_factor = float(cfg["training"].get("lr_warmup_start_factor", 0.5))
    base_lr = float(cfg["training"].get("learning_rate", 3e-4))
    grad_accum_steps = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    optimizer_step_offset = max(start_epoch - 1, 0) * math.ceil(len(train_loader) / grad_accum_steps)

    try:
        for epoch in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_started = time.perf_counter()
            train_g_loss, train_d_loss, train_metrics, train_loss_components, optimizer_steps_this_epoch = train_one_epoch(
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
                warmup_optimizer_steps=warmup_optimizer_steps,
                warmup_start_factor=warmup_start_factor,
                base_lr=base_lr,
                optimizer_step_offset=optimizer_step_offset,
                rmse_buffer=rmse_buffer,
            )
            optimizer_step_offset += optimizer_steps_this_epoch
            train_seconds = time.perf_counter() - train_started
            if epoch == start_epoch and is_main_process(rank) and is_cuda_device(device):
                peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
                alloc_mb = torch.cuda.memory_allocated(device) / 1e6
                reserved_mb = torch.cuda.memory_reserved(device) / 1e6
                print(json.dumps({"vram_after_epoch1_mb": {"peak": round(peak_mb, 1), "allocated": round(alloc_mb, 1), "reserved": round(reserved_mb, 1)}}))
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
                loader=val_loader,
                sample_indices=val_sample_indices,
            )
            val_seconds = time.perf_counter() - val_started

            if is_main_process(rank):
                train_batches = max(len(train_loader), 1)
                val_samples = max(len(val_dataset), 1)
                train_payload = {
                    "generator_loss": float(train_g_loss),
                    "loss_components": train_loss_components,
                    "loss_flags": {
                        "optimize_residual_only": bool(cfg.get("prior_residual_path_loss", {}).get("optimize_residual_only", False)),
                        "multiscale_enabled": bool(cfg.get("multiscale_path_loss", {}).get("enabled", False)),
                        "gan_enabled": bool(cfg.get("loss", {}).get("lambda_gan", 0.0) > 0.0),
                        "gate_enabled": bool(cfg.get("separated_refiner", {}).get("use_gate", False)),
                        "lambda_recon": float(cfg.get("loss", {}).get("lambda_recon", 0.0)),
                        "residual_loss_weight": float(cfg.get("prior_residual_path_loss", {}).get("loss_weight", 0.0)),
                        "gate_loss_weight": float(cfg.get("separated_refiner", {}).get("gate_loss_weight", 0.0)),
                        "nlos_focus_loss_weight": (
                            float(cfg.get("nlos_focus_loss", {}).get("loss_weight", 0.0))
                            if bool(cfg.get("nlos_focus_loss", {}).get("enabled", False))
                            else 0.0
                        ),
                    },
                    "learning_rate": float(optimizer_g.param_groups[0]["lr"]),
                    "weight_decay": float(getattr(optimizer_g, "_manual_weight_decay", cfg["training"].get("weight_decay", 0.0))),
                    "weight_decay_mode": str(getattr(optimizer_g, "_manual_weight_decay_mode", "selective_excluding_bias_and_norm")),
                    "weight_decay_param_counts": {
                        "decay": int(getattr(optimizer_g, "_manual_weight_decay_param_count", 0)),
                        "no_decay": int(getattr(optimizer_g, "_manual_no_weight_decay_param_count", 0)),
                        "total": int(getattr(optimizer_g, "_manual_weight_decay_total_count", 0)),
                    },
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

                if bool(cfg["training"].get("save_validation_json_each_epoch", True)):
                    write_validation_json(
                        out_dir,
                        epoch,
                        val_summary,
                        best_epoch=best_epoch,
                        best_score=best_score,
                        selection_metric=selection_metric,
                        current_score=current_score,
                        uses_ema=ema_generator is not None,
                        ema_decay=ema_decay,
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
                rewind_layout_is_exact = _rewind_training_state_from_checkpoint(
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
                if not rewind_layout_is_exact and is_main_process(rank):
                    print("[rewind] Best checkpoint layout differs from the current model; optimizer/scheduler/scaler state was reinitialized.")
                apply_optimizer_hparams_from_cfg(
                    optimizer_g,
                    learning_rate=float(cfg["training"].get("learning_rate", cfg["training"].get("generator_lr", 3e-5))),
                    weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
                    beta1=generator_beta1,
                    beta2=generator_beta2,
                    momentum=generator_momentum,
                )
                if scheduler_g is not None and hasattr(scheduler_g, "base_lrs"):
                    scheduler_g.base_lrs = [
                        float(cfg["training"].get("learning_rate", cfg["training"].get("generator_lr", 3e-5)))
                        for _ in getattr(scheduler_g, "base_lrs", [])
                    ]
                if optimizer_d is not None:
                    apply_optimizer_hparams_from_cfg(
                        optimizer_d,
                        learning_rate=float(cfg["training"].get("discriminator_lr", cfg["training"].get("learning_rate", cfg["training"].get("generator_lr", 3e-5)))),
                        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
                        beta1=generator_beta1,
                        beta2=generator_beta2,
                        momentum=generator_momentum,
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
                    _lr_smooth_factor = float(cfg.get('training', {}).get('lr_scheduler_score_ema', 0.0))
                    if not hasattr(scheduler_g, '_smoothed_score'):
                        scheduler_g._smoothed_score = current_score
                    if _lr_smooth_factor > 0.0:
                        scheduler_g._smoothed_score = (_lr_smooth_factor * scheduler_g._smoothed_score
                                                       + (1.0 - _lr_smooth_factor) * current_score)
                    else:
                        scheduler_g._smoothed_score = current_score
                    scheduler_g.step(scheduler_g._smoothed_score)
                else:
                    scheduler_g.step()

            if is_cuda_device(device):
                torch.cuda.empty_cache()

            barrier_if_distributed(distributed)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
