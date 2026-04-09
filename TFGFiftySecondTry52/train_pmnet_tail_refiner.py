from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import re
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
    _infer_city_type_simple,
    build_dataset_splits_from_config,
    compute_input_channels,
    return_scalar_cond_from_config,
)
from model_pmnet import CityTypeRoutedNLoSMoERegressor, GlobalContextUNetRefiner, PMNetResidualRegressor, UNetResidualRefiner


CITY_TYPE_EXPERT_ORDER = ["open_lowrise", "mixed_midrise", "dense_highrise"]


def _stage1_model_arch(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("model", {}).get("arch", "pmnet")).lower()


def _city_type_to_expert_index(city_type: str) -> int:
    try:
        return CITY_TYPE_EXPERT_ORDER.index(str(city_type))
    except ValueError:
        return CITY_TYPE_EXPERT_ORDER.index("mixed_midrise")


def _select_city_expert_map(expert_maps: torch.Tensor, city_type: str) -> torch.Tensor:
    if expert_maps.ndim != 4:
        raise ValueError(f"Expected expert_maps [B,E,H,W], got {tuple(expert_maps.shape)}")
    expert_idx = _city_type_to_expert_index(city_type)
    expert_idx = min(max(expert_idx, 0), expert_maps.shape[1] - 1)
    return expert_maps[:, expert_idx : expert_idx + 1]


def _los_nlos_masks(input_batch: torch.Tensor, cfg: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg["data"].get("los_input_column") and input_batch.shape[1] > 1:
        los = input_batch[:, 1:2].clamp(0.0, 1.0)
        nlos = (1.0 - los).clamp(0.0, 1.0)
        return los, nlos
    ones = torch.ones_like(input_batch[:, :1])
    return ones, ones


def _build_stage1_model(cfg: Dict[str, Any], in_channels: int) -> nn.Module:
    if _stage1_model_arch(cfg) == "city_routed_nlos_moe":
        return CityTypeRoutedNLoSMoERegressor(
            in_channels=in_channels,
            out_channels=int(cfg["model"].get("out_channels", len(CITY_TYPE_EXPERT_ORDER))),
            base_channels=int(cfg["model"].get("base_channels", 48)),
            encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [1, 2, 2, 2])),
            context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4])),
            norm_type=str(cfg["model"].get("norm_type", "group")),
            dropout=float(cfg["model"].get("dropout", 0.0)),
            gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
            attention_heads=int(cfg["model"].get("attention_heads", 4)),
            attention_pool_size=int(cfg["model"].get("attention_pool_size", 8)),
        )
    return PMNetResidualRegressor(
        in_channels=in_channels,
        out_channels=int(cfg["model"].get("out_channels", 1)),
        base_channels=int(cfg["model"].get("base_channels", 64)),
        encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
    )


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


def _masked_gradient_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    denom = (mask_dx.sum() + mask_dy.sum()).clamp_min(1.0)
    loss_dx = (torch.abs(pred_dx - target_dx) * mask_dx).sum()
    loss_dy = (torch.abs(pred_dy - target_dy) * mask_dy).sum()
    return (loss_dx + loss_dy) / denom


def _masked_laplacian_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3)
    pred_lap = F.conv2d(pred, kernel, padding=1)
    target_lap = F.conv2d(target, kernel, padding=1)
    valid = (F.conv2d(mask, torch.ones_like(kernel), padding=1) >= 5.0).to(dtype=pred.dtype)
    denom = valid.sum().clamp_min(1.0)
    return (torch.abs(pred_lap - target_lap) * valid).sum() / denom


def compute_high_frequency_path_loss_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    hf_cfg = dict(cfg.get("tail_refiner", {}).get("high_frequency_loss", {}))
    if not bool(hf_cfg.get("enabled", False)):
        return torch.tensor(0.0, device=pred.device)
    lap_w = float(hf_cfg.get("laplacian_weight", 0.0))
    grad_w = float(hf_cfg.get("gradient_weight", 0.0))
    total = torch.tensor(0.0, device=pred.device)
    if lap_w > 0.0:
        total = total + lap_w * _masked_laplacian_l1_loss(pred, target, mask)
    if grad_w > 0.0:
        total = total + grad_w * _masked_gradient_l1_loss(pred, target, mask)
    return total


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
    base_input: torch.Tensor,
    cfg: Dict[str, Any],
    city_type_resolver: Optional[AutomaticCityTypeResolver],
) -> torch.Tensor:
    rw_cfg = dict(cfg.get("tail_refiner", {}).get("regime_reweighting", {}))
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
        if base_input.shape[1] > los_idx:
            los = base_input[:, los_idx : los_idx + 1].clamp(0.0, 1.0)
            nlos = (1.0 - los).clamp(0.0, 1.0)
            weight = weight * (los * los_weight + nlos * nlos_weight)

    if low_antenna_boost > 0.0:
        antenna_channel = _extract_scalar_channel(base_input, cfg, "antenna_height_m")
        if antenna_channel is not None:
            low_antenna = (1.0 - antenna_channel.clamp(0.0, 1.0)).clamp(0.0, 1.0)
            weight = weight * (1.0 + low_antenna_boost * low_antenna)

    if city_type_resolver is not None:
        city_type = city_type_resolver.infer(base_input)
        weight = weight * max(city_weights.get(city_type, default_city_weight), 1e-6)

    return mask * weight.clamp(min=min_weight, max=max_weight)


def _gate_target_from_error(abs_error_db: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    gate_cfg = dict(cfg.get("tail_refiner", {})).get("gate_target", {})
    threshold_db = max(float(gate_cfg.get("threshold_db", 6.0)), 1e-3)
    temperature_db = max(float(gate_cfg.get("temperature_db", 2.0)), 1e-3)
    return torch.sigmoid((abs_error_db - threshold_db) / temperature_db)


class BaseTailRefinerDataset(Dataset):
    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset
        self.sample_refs = list(getattr(base_dataset, "sample_refs", []))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.base_dataset[idx]
        if len(item) == 4:
            raise ValueError("Tail refiner expects scalar channels, not FiLM vectors.")
        x, y, m = item
        return x, y, m


class Stage1Teacher:
    def __init__(self, cfg: Dict[str, Any], stage1_cfg: Dict[str, Any], checkpoint_path: str, device: object) -> None:
        self.cfg = cfg
        self.stage1_cfg = stage1_cfg
        self.device = device
        self.meta = dict(cfg["target_metadata"]["path_loss"])
        self.formula_idx = formula_channel_index(stage1_cfg)
        self.city_type_resolver = AutomaticCityTypeResolver(stage1_cfg)
        in_channels = compute_input_channels(stage1_cfg)
        self.model = _build_stage1_model(stage1_cfg, in_channels).to(device)
        state = load_torch_checkpoint(Path(checkpoint_path), device)
        model_state = state.get("model", state.get("generator", state))
        self.model.load_state_dict(model_state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _predict_from_raw(self, x: torch.Tensor, raw: torch.Tensor, prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if _stage1_model_arch(self.stage1_cfg) == "city_routed_nlos_moe":
            city_type = self.city_type_resolver.infer(x)
            _, nlos = _los_nlos_masks(x, self.stage1_cfg)
            delta = _select_city_expert_map(raw, city_type)
            pred = clip_to_target_range(prior + nlos * delta, self.meta)
            return pred, delta
        residual = raw[:, :1]
        pred = clip_to_target_range(prior + residual, self.meta)
        return pred, residual

    def infer(self, x: torch.Tensor, target: torch.Tensor, amp_enabled: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prior = x[:, self.formula_idx : self.formula_idx + 1]
        with torch.no_grad():
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                raw = self.model(x)
                pred, _ = self._predict_from_raw(x, raw, prior)
        abs_error_db = (denormalize(pred, self.meta) - denormalize(target, self.meta)).abs()
        return pred, abs_error_db, prior

    def predict_only(self, x: torch.Tensor, amp_enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
        prior = x[:, self.formula_idx : self.formula_idx + 1]
        with torch.no_grad():
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                raw = self.model(x)
                pred, _ = self._predict_from_raw(x, raw, prior)
        return pred, prior


class Stage2Teacher:
    def __init__(self, cfg: Dict[str, Any], stage2_cfg: Dict[str, Any], checkpoint_path: str, device: object) -> None:
        self.cfg = cfg
        self.stage2_cfg = stage2_cfg
        self.device = device
        self.meta = dict(cfg["target_metadata"]["path_loss"])

        tail_cfg = dict(stage2_cfg.get("tail_refiner", {}))
        stage1_config_raw = str(tail_cfg.get("stage1_config", "")).strip()
        stage1_checkpoint_raw = str(tail_cfg.get("stage1_checkpoint", "")).strip()
        if not stage1_config_raw or not stage1_checkpoint_raw:
            raise ValueError("Stage2 teacher requires stage1_config and stage1_checkpoint in teacher config")

        try_root = Path(__file__).resolve().parent
        stage1_config_path = Path(stage1_config_raw)
        if not stage1_config_path.is_absolute():
            stage1_config_path = (try_root / stage1_config_raw).resolve()
        stage1_cfg = load_config(str(stage1_config_path))
        anchor_data_paths_to_config_file(stage1_cfg, str(stage1_config_path))

        stage1_checkpoint_path = Path(stage1_checkpoint_raw)
        if not stage1_checkpoint_path.is_absolute():
            stage1_checkpoint_path = (try_root / stage1_checkpoint_raw).resolve()

        self.stage1_teacher = Stage1Teacher(cfg, stage1_cfg, str(stage1_checkpoint_path), device)
        input_channels = compute_input_channels(stage2_cfg) + 1
        self.model = build_tail_refiner_model(stage2_cfg, input_channels).to(device)
        state = load_torch_checkpoint(Path(checkpoint_path), device)
        model_state = state.get("model", state.get("generator", state))
        self.model.load_state_dict(model_state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def infer(self, x: torch.Tensor, target: torch.Tensor, amp_enabled: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stage1_pred, _, prior = self.stage1_teacher.infer(x, target, amp_enabled)
        x_aug = torch.cat([x, stage1_pred], dim=1)
        with torch.no_grad():
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                refiner_out = self.model(x_aug)
                delta = refiner_out[:, :1]
                pred = clip_to_target_range(stage1_pred + delta, self.meta)
        abs_error_db = (denormalize(pred, self.meta) - denormalize(target, self.meta)).abs()
        return pred, abs_error_db, prior

    def predict_only(self, x: torch.Tensor, amp_enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
        stage1_pred, prior = self.stage1_teacher.predict_only(x, amp_enabled)
        x_aug = torch.cat([x, stage1_pred], dim=1)
        with torch.no_grad():
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                refiner_out = self.model(x_aug)
                delta = refiner_out[:, :1]
                pred = clip_to_target_range(stage1_pred + delta, self.meta)
        return pred, prior


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


def finalize_metric_total(
    totals: Dict[str, float],
    unit: str = "dB",
    *,
    total_count: Optional[float] = None,
) -> Dict[str, Any]:
    raw_count = float(totals.get("count", 0.0))
    count = max(raw_count, 1.0)
    mse = float(totals.get("sum_sq", 0.0)) / count
    mae = float(totals.get("sum_abs", 0.0)) / count
    payload: Dict[str, Any] = {
        "mse_physical": mse,
        "rmse_physical": math.sqrt(max(mse, 0.0)),
        "mae_physical": mae,
        "count": int(round(raw_count)),
        "unit": unit,
    }
    if total_count is not None and total_count > 0.0:
        payload["fraction_of_valid_pixels"] = float(raw_count / total_count)
    return payload


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
    if refiner_arch == "global_context_unet":
        return GlobalContextUNetRefiner(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(tail_cfg.get("refiner_base_channels", 40)),
            norm_type=str(cfg["model"].get("norm_type", "group")),
            dropout=float(cfg["model"].get("dropout", 0.0)),
            gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
            attention_heads=int(tail_cfg.get("attention_heads", 4)),
            attention_pool_size=int(tail_cfg.get("attention_pool_size", 8)),
        )
    raise ValueError(f"Unsupported tail_refiner.refiner_arch '{refiner_arch}'")


def build_sample_weights(dataset: BaseTailRefinerDataset, cfg: Dict[str, Any]) -> torch.Tensor:
    tail_cfg = dict(cfg.get("tail_refiner", {}))
    oversample_cfg = dict(tail_cfg.get("oversample", {}))
    if not bool(oversample_cfg.get("enabled", False)):
        return torch.ones(len(dataset), dtype=torch.double)

    nlos_boost = max(float(oversample_cfg.get("nlos_boost", 0.0)), 0.0)
    antenna_boost = max(float(oversample_cfg.get("antenna_boost", 0.0)), 0.0)
    weights: list[float] = []

    base_dataset = getattr(dataset, "base_dataset", None)
    sample_refs = list(getattr(dataset, "sample_refs", []))
    hdf5_path = getattr(base_dataset, "hdf5_path", None)
    antenna_norm = float(dict(cfg.get("data", {}).get("scalar_feature_norms", {})).get("antenna_height_m", 120.0))
    los_field = str(cfg.get("data", {}).get("los_input_column", "los_mask"))

    if hdf5_path is not None and sample_refs:
        with h5py.File(str(hdf5_path), "r") as handle:
            for city, sample in sample_refs:
                grp = handle[city][sample]
                weight = 1.0

                if nlos_boost > 0.0 and los_field in grp:
                    los_map = np.asarray(grp[los_field][...], dtype=np.float32)
                    if los_map.size > 0:
                        max_val = float(np.max(los_map))
                        if max_val > 1.0:
                            los_map = los_map / max(max_val, 1.0)
                        nlos_frac = float(np.mean(los_map <= 0.5))
                        weight *= 1.0 + nlos_boost * nlos_frac

                if antenna_boost > 0.0 and "uav_height" in grp:
                    ant = float(np.asarray(grp["uav_height"][...]).reshape(-1)[0])
                    low_antenna = float(np.clip(1.0 - (ant / max(antenna_norm, 1e-6)), 0.0, 1.0))
                    weight *= 1.0 + antenna_boost * low_antenna

                weights.append(max(weight, 1e-6))
        return torch.tensor(weights, dtype=torch.double)

    for index in range(len(dataset)):
        x, _, _ = dataset[index]
        weight = 1.0

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
    stage1_teacher: Stage1Teacher,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: object,
    cfg: Dict[str, Any],
    amp_enabled: bool,
    city_type_resolver: Optional[AutomaticCityTypeResolver] = None,
) -> tuple[float, Dict[str, float]]:
    tail_cfg = dict(cfg.get("tail_refiner", {}))
    loss_cfg = dict(cfg.get("training", {}))
    meta = dict(cfg["target_metadata"]["path_loss"])
    use_gate = bool(tail_cfg.get("use_gate", False))
    nlos_only = bool(tail_cfg.get("nlos_only", False))
    gate_loss_weight = float(tail_cfg.get("gate_loss_weight", 0.0))
    residual_weight = float(tail_cfg.get("residual_weight", 1.0))
    final_weight = float(tail_cfg.get("final_weight", 0.5))
    clip_grad = float(cfg["training"].get("clip_grad_norm", 1.0))
    mse_weight = float(cfg["loss"].get("mse_weight", 1.0))
    l1_weight = float(cfg["loss"].get("l1_weight", 0.1))

    model.train()
    running_loss = 0.0
    running_components: Dict[str, float] = {
        "final_loss": 0.0,
        "residual_loss": 0.0,
        "multiscale_loss": 0.0,
        "high_frequency_loss": 0.0,
        "gate_loss": 0.0,
        "term_final": 0.0,
        "term_residual": 0.0,
        "term_multiscale": 0.0,
        "term_high_frequency": 0.0,
        "term_gate": 0.0,
        "total_loss": 0.0,
    }
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x, y, m = batch
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        target = y[:, :1]
        mask = m[:, :1]
        stage1_pred, abs_error_db, _ = stage1_teacher.infer(x, target, amp_enabled)
        x_aug = torch.cat([x, stage1_pred], dim=1)
        residual_target = target - stage1_pred
        _, nlos_mask = _los_nlos_masks(x_aug, cfg)

        weighted_mask = mask * _tail_focus_weights(abs_error_db, x_aug, cfg)
        weighted_mask = _apply_regime_reweighting(weighted_mask, x_aug, cfg, city_type_resolver)
        if nlos_only:
            weighted_mask = weighted_mask * nlos_mask
        gate_target = _gate_target_from_error(abs_error_db, cfg) if use_gate else None

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=amp_enabled):
            refiner_out = model(x_aug)
            if use_gate:
                if refiner_out.shape[1] < 2:
                    raise ValueError("tail_refiner.use_gate=true requires model.out_channels >= 2")
                delta = refiner_out[:, :1] * torch.sigmoid(refiner_out[:, 1:2])
                gate_logits = refiner_out[:, 1:2]
            else:
                delta = refiner_out[:, :1]
                gate_logits = None

            if nlos_only:
                final_pred = clip_to_target_range(stage1_pred + nlos_mask * delta, meta)
            else:
                final_pred = clip_to_target_range(stage1_pred + delta, meta)
            final_loss = masked_mse_l1_loss(final_pred, target, weighted_mask, mse_weight=mse_weight, l1_weight=l1_weight)
            residual_loss = masked_mse_l1_loss(delta, residual_target, weighted_mask, mse_weight=mse_weight, l1_weight=l1_weight)
            multiscale_loss = compute_multiscale_path_loss_loss(final_pred, target, weighted_mask, meta, cfg)
            high_frequency_loss = compute_high_frequency_path_loss_loss(final_pred, target, weighted_mask, cfg)
            total_loss = residual_weight * residual_loss + final_weight * final_loss + multiscale_loss + high_frequency_loss

            if use_gate and gate_logits is not None and gate_target is not None:
                gate_loss_map = F.binary_cross_entropy_with_logits(gate_logits, gate_target, reduction="none")
                gate_loss = (gate_loss_map * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
                total_loss = total_loss + gate_loss_weight * gate_loss
            else:
                gate_loss = torch.zeros((), device=target.device)

            final_loss_value = float(final_loss.detach().item())
            residual_loss_value = float(residual_loss.detach().item())
            multiscale_loss_value = float(multiscale_loss.detach().item())
            high_frequency_loss_value = float(high_frequency_loss.detach().item())
            gate_loss_value = float(gate_loss.detach().item())
            term_final = final_weight * final_loss_value
            term_residual = residual_weight * residual_loss_value
            term_multiscale = multiscale_loss_value
            term_high_frequency = high_frequency_loss_value
            term_gate = gate_loss_weight * gate_loss_value if gate_loss_weight > 0.0 else 0.0

        scaler.scale(total_loss).backward()
        if clip_grad > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(total_loss.item())
        running_components["final_loss"] += final_loss_value
        running_components["residual_loss"] += residual_loss_value
        running_components["multiscale_loss"] += multiscale_loss_value
        running_components["high_frequency_loss"] += high_frequency_loss_value
        running_components["gate_loss"] += gate_loss_value
        running_components["term_final"] += term_final
        running_components["term_residual"] += term_residual
        running_components["term_multiscale"] += term_multiscale
        running_components["term_high_frequency"] += term_high_frequency
        running_components["term_gate"] += term_gate
        running_components["total_loss"] += float(total_loss.item())
        steps += 1

    denom = max(steps, 1)
    avg_components = {key: float(value / denom) for key, value in running_components.items()}
    return running_loss / denom, avg_components


def evaluate_validation(
    model: nn.Module,
    stage1_teacher: Stage1Teacher,
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
    nlos_only = bool(tail_cfg.get("nlos_only", False))

    totals = init_metric_totals()
    stage1_totals = init_metric_totals()
    los_totals = init_metric_totals()
    nlos_totals = init_metric_totals()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False, disable=distributed and not is_main_process(rank)):
            x, y, m = batch
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            target = y[:, :1]
            mask = m[:, :1]
            stage1_pred, _ = stage1_teacher.predict_only(x, amp_enabled)
            x_aug = torch.cat([x, stage1_pred], dim=1)
            _, nlos_mask = _los_nlos_masks(x_aug, cfg)
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                refiner_out = model(x_aug)
                if use_gate:
                    delta = refiner_out[:, :1] * torch.sigmoid(refiner_out[:, 1:2])
                else:
                    delta = refiner_out[:, :1]
                if nlos_only:
                    final_pred = clip_to_target_range(stage1_pred + nlos_mask * delta, meta)
                else:
                    final_pred = clip_to_target_range(stage1_pred + delta, meta)

            pred_phys = denormalize(final_pred, meta)
            stage1_phys = denormalize(stage1_pred, meta)
            target_phys = denormalize(target[:, :1], meta)
            valid_mask = mask[:, :1] > 0.0

            update_metric_total(totals, (pred_phys - target_phys)[valid_mask])
            update_metric_total(stage1_totals, (stage1_phys - target_phys)[valid_mask])

            if cfg["data"].get("los_input_column") and x_aug.shape[1] > 1:
                los = x_aug[:, 1:2]
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

    total_count = float(totals.get("count", 0.0))
    los_count = float(los_totals.get("count", 0.0))
    nlos_count = float(nlos_totals.get("count", 0.0))
    summary: Dict[str, Any] = {
        "path_loss": finalize_metric_total(totals, meta.get("unit", "dB"), total_count=total_count),
        "path_loss__stage1__overall": finalize_metric_total(stage1_totals, meta.get("unit", "dB"), total_count=total_count),
        "_support": {
            "valid_pixel_count": int(round(total_count)),
            "los_valid_pixel_count": int(round(los_count)),
            "nlos_valid_pixel_count": int(round(nlos_count)),
            "los_fraction": float(los_count / total_count) if total_count > 0.0 else float("nan"),
            "nlos_fraction": float(nlos_count / total_count) if total_count > 0.0 else float("nan"),
        },
    }
    if los_totals["count"] > 0.0:
        summary["path_loss__los__LoS"] = finalize_metric_total(los_totals, meta.get("unit", "dB"), total_count=total_count)
    if nlos_totals["count"] > 0.0:
        summary["path_loss__los__NLoS"] = finalize_metric_total(nlos_totals, meta.get("unit", "dB"), total_count=total_count)
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


def _extract_epoch_number(path: Path) -> int:
    match = re.search(r"epoch_(\d+)_", path.name)
    if match:
        return int(match.group(1))
    return -1


def _latest_epoch_checkpoint(search_dir: Path, patterns: Sequence[str]) -> Optional[Path]:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(search_dir.glob(pattern))
    checkpoints = [path for path in candidates if path.is_file()]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda path: (_extract_epoch_number(path), path.name))
    return checkpoints[-1]


def resolve_teacher_checkpoint(configured_checkpoint: str, try_root: Path) -> Path:
    configured_path = Path(configured_checkpoint)
    if not configured_path.is_absolute():
        configured_path = (try_root / configured_path).resolve()
    if configured_path.exists() and configured_path.is_file():
        return configured_path

    search_dir = configured_path if configured_path.is_dir() else configured_path.parent
    if not search_dir.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {configured_checkpoint}")

    configured_name = configured_path.name.lower()
    if "tail_refiner" in configured_name:
        preferred_patterns = ["epoch_*_tail_refiner.pt"]
    elif "cgan" in configured_name:
        preferred_patterns = ["epoch_*_cgan.pt"]
    else:
        preferred_patterns = []
    fallback_patterns = preferred_patterns + ["epoch_*_tail_refiner.pt", "epoch_*_cgan.pt"]
    resolved = _latest_epoch_checkpoint(search_dir, fallback_patterns)
    if resolved is None:
        raise FileNotFoundError(f"Teacher checkpoint not found: {configured_checkpoint}")
    return resolved


def build_tail_loss_flags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tail_cfg = dict(cfg.get("tail_refiner", {}))
    ms_cfg = dict(cfg.get("loss", {}).get("multiscale_path_loss", cfg.get("multiscale_path_loss", {})))
    hf_cfg = dict(tail_cfg.get("high_frequency_loss", {}))
    gate_weight = float(tail_cfg.get("gate_loss_weight", 0.0))
    return {
        "nlos_only": bool(tail_cfg.get("nlos_only", False)),
        "use_gate": bool(tail_cfg.get("use_gate", False)),
        "gate_enabled": bool(tail_cfg.get("use_gate", False)) and gate_weight > 0.0,
        "multiscale_enabled": bool(ms_cfg.get("enabled", False)),
        "high_frequency_enabled": bool(hf_cfg.get("enabled", False)),
        "residual_weight": float(tail_cfg.get("residual_weight", 1.0)),
        "final_weight": float(tail_cfg.get("final_weight", 0.5)),
        "gate_loss_weight": gate_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Try52 stage2 tail-refiner without GANs")
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
        print("[refiner] building dataset splits", flush=True)

    splits = build_dataset_splits_from_config(cfg)
    if is_main_process(rank):
        print(
            json.dumps(
                {
                    "refiner_train_samples": len(splits["train"]),
                    "refiner_val_samples": len(splits["val"]),
                }
            ),
            flush=True,
        )
    tail_cfg = dict(cfg.get("tail_refiner", {}))
    teacher_kind = str(tail_cfg.get("teacher_kind", "stage1")).lower()
    teacher_config_raw = str(tail_cfg.get("teacher_config", tail_cfg.get("stage1_config", ""))).strip()
    teacher_checkpoint_raw = str(tail_cfg.get("teacher_checkpoint", tail_cfg.get("stage1_checkpoint", ""))).strip()
    if not teacher_config_raw or not teacher_checkpoint_raw:
        raise ValueError("tail_refiner.teacher_config and tail_refiner.teacher_checkpoint are required")
    try_root = Path(__file__).resolve().parent
    teacher_config_path = Path(teacher_config_raw)
    if not teacher_config_path.is_absolute():
        teacher_config_path = (try_root / teacher_config_raw).resolve()
    if not teacher_config_path.exists():
        raise FileNotFoundError(f"Teacher config not found: {teacher_config_raw}")
    teacher_cfg = load_config(str(teacher_config_path))
    anchor_data_paths_to_config_file(teacher_cfg, str(teacher_config_path))
    teacher_checkpoint_path = resolve_teacher_checkpoint(teacher_checkpoint_raw, try_root)
    configured_teacher_checkpoint_path = Path(teacher_checkpoint_raw)
    if not configured_teacher_checkpoint_path.is_absolute():
        configured_teacher_checkpoint_path = (try_root / configured_teacher_checkpoint_path).resolve()
    if is_main_process(rank) and teacher_checkpoint_path != configured_teacher_checkpoint_path:
        print(
            json.dumps(
                {
                    "teacher_checkpoint_requested": teacher_checkpoint_raw,
                    "teacher_checkpoint_resolved": str(teacher_checkpoint_path),
                }
            ),
            flush=True,
        )

    train_dataset = BaseTailRefinerDataset(splits["train"])
    val_dataset = BaseTailRefinerDataset(splits["val"])
    if is_main_process(rank):
        print(f"[refiner] building {teacher_kind} teacher", flush=True)

    input_channels = compute_input_channels(cfg) + 1
    model = build_tail_refiner_model(cfg, input_channels).to(device)
    if teacher_kind == "stage2":
        stage1_teacher = Stage2Teacher(cfg, teacher_cfg, str(teacher_checkpoint_path), device)
    else:
        stage1_teacher = Stage1Teacher(cfg, teacher_cfg, str(teacher_checkpoint_path), device)
    if is_main_process(rank):
        print(f"[refiner] {teacher_kind} teacher ready", flush=True)

    if bool(tail_cfg.get("oversample", {}).get("enabled", False)):
        if is_main_process(rank):
            print("[refiner] building oversample weights", flush=True)
        sample_weights = build_sample_weights(train_dataset, cfg)
        if is_main_process(rank):
            print("[refiner] oversample weights ready", flush=True)
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
    city_type_resolver = AutomaticCityTypeResolver(cfg)
    if is_main_process(rank):
        print("[refiner] dataloaders ready", flush=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["data"].get("val_batch_size", 1)),
        shuffle=False,
        sampler=val_sampler,
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
    validate_only = bool(cfg.get("runtime", {}).get("validate_only", False))

    if validate_only:
        val_epoch = max(start_epoch - 1, 0)
        val_summary = evaluate_validation(
            model.module if isinstance(model, DistributedDataParallel) else model,
            stage1_teacher,
            val_loader,
            device,
            cfg,
            amp_enabled,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        if is_main_process(rank):
            write_validation_json(out_dir, val_epoch, val_summary, best_epoch=best_epoch, best_score=best_score)
            print(json.dumps({"validate_only": True, "epoch": val_epoch, **val_summary}))
        cleanup_distributed(distributed)
        return

    try:
        for epoch in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
            if isinstance(train_sampler, (DistributedSampler, DistributedWeightedSampler)):
                train_sampler.set_epoch(epoch)

            train_loss, train_loss_components = train_one_epoch(
                model,
                stage1_teacher,
                train_loader,
                optimizer,
                scaler,
                device,
                cfg,
                amp_enabled,
                city_type_resolver=city_type_resolver,
            )
            barrier_if_distributed(distributed)

            val_summary = evaluate_validation(
                model.module if isinstance(model, DistributedDataParallel) else model,
                stage1_teacher,
                val_loader,
                device,
                cfg,
                amp_enabled,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
            )

            if is_main_process(rank):
                val_summary["_train"] = {
                    "tail_refiner_loss": float(train_loss),
                    "loss_components": train_loss_components,
                    "loss_flags": build_tail_loss_flags(cfg),
                }
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

                target_model = model.module if isinstance(model, DistributedDataParallel) else model
                epoch_ckpt_path = out_dir / f"epoch_{epoch}_tail_refiner.pt"
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
                    epoch_ckpt_path,
                )
                prev_epoch_ckpt_path = out_dir / f"epoch_{epoch - 1}_tail_refiner.pt"
                if prev_epoch_ckpt_path.exists():
                    prev_epoch_ckpt_path.unlink()

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

