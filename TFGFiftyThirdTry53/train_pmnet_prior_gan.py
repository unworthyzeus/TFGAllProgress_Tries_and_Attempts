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

import numpy as np
import h5py
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
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
    compute_input_channels,
    forward_cgan_generator,
    return_scalar_cond_from_config,
    unpack_cgan_batch,
)
from model_pmnet import (
    CityTypeRoutedNLoSMoERegressor,
    PMNetResidualRegressor,
    PatchDiscriminator,
    UNetResidualRefiner,
)


CITY_TYPE_EXPERT_ORDER = ["open_lowrise", "mixed_midrise", "dense_highrise"]


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
        raise ValueError("This branch requires data.path_loss_formula_input.enabled = true")
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
    search_dir = out_dir
    if configured_resume:
        candidate = Path(configured_resume)
        if candidate.exists():
            return candidate
        if candidate.parent != Path("."):
            search_dir = candidate.parent
    best = search_dir / "best_cgan.pt"
    if best.exists():
        return best
    epoch_candidates = sorted(search_dir.glob("epoch_*_cgan.pt"))
    if epoch_candidates:
        return epoch_candidates[-1]
    return None


def _model_arch(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("model", {}).get("arch", "pmnet")).lower()


def _build_pmnet_from_cfg(cfg: Dict[str, Any], in_channels: int) -> nn.Module:
    arch = _model_arch(cfg)
    if arch == "city_routed_nlos_moe":
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
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"].get("base_channels", 64)),
        encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
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
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not separated_mode:
        residual = forward_cgan_generator(trainable_generator, x, scalar_cond)
        return residual, None, None
    if base_generator is None:
        raise RuntimeError("Separated mode enabled but base_generator is missing")
    with torch.no_grad():
        base_residual = forward_cgan_generator(base_generator, x, scalar_cond)
    base_pred = prior + base_residual
    refiner_in = torch.cat([x, base_pred, base_residual], dim=1)
    refiner_out = forward_cgan_generator(trainable_generator, refiner_in, None)
    if use_gate:
        if refiner_out.shape[1] < 2:
            raise ValueError("separated_refiner.use_gate=true requires refiner_out_channels >= 2")
        delta_raw = refiner_out[:, :1]
        gate_logits = refiner_out[:, 1:2]
        delta = delta_raw * torch.sigmoid(gate_logits)
    else:
        delta = refiner_out[:, :1]
        gate_logits = None
    return base_residual + delta, gate_logits, base_residual


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
    residual_pred, _, _ = _compose_residual_prediction_with_aux(
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

    def _infer_from_raw_topology(self, raw_topology: np.ndarray) -> str:
        non_ground = raw_topology != self.non_ground_threshold
        building_density = float(np.mean(non_ground))
        non_zero = raw_topology[non_ground]
        mean_height = float(np.mean(non_zero)) if non_zero.size else 0.0
        if self.city_type_thresholds:
            return _city_type_from_thresholds(building_density, mean_height, self.city_type_thresholds)
        return _infer_city_type_simple(building_density, mean_height)

    def infer_batch(self, input_batch: torch.Tensor) -> list[str]:
        topo_batch = input_batch[:, :1] if input_batch.ndim == 4 else input_batch[:1].unsqueeze(0)
        raw_topology_batch = topo_batch.detach().float().cpu().numpy() * self.input_scale + self.input_offset
        city_types: list[str] = []
        for sample in raw_topology_batch:
            city_types.append(self._infer_from_raw_topology(np.squeeze(sample, axis=0)))
        return city_types

    def infer(self, input_batch: torch.Tensor) -> str:
        return self.infer_batch(input_batch)[0]


class HomogeneousCityTypeBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        buckets: Dict[str, list[int]],
        batch_size: int,
        *,
        shuffle: bool,
        num_replicas: int,
        rank: int,
        drop_last: bool = False,
    ) -> None:
        self.buckets = {key: list(values) for key, values in buckets.items() if values}
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._global_num_batches = self._compute_global_num_batches()

    def _compute_global_num_batches(self) -> int:
        total = 0
        for indices in self.buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        if total == 0:
            return 0
        return math.ceil(total / self.num_replicas) * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self._global_num_batches == 0:
            return 0
        return self._global_num_batches // self.num_replicas

    def __iter__(self):
        rng = random.Random(self.epoch)
        all_batches: list[list[int]] = []
        for key, indices in self.buckets.items():
            bucket_indices = list(indices)
            if self.shuffle:
                rng.shuffle(bucket_indices)
            batches = [
                bucket_indices[start : start + self.batch_size]
                for start in range(0, len(bucket_indices), self.batch_size)
            ]
            if self.drop_last:
                batches = [batch for batch in batches if len(batch) == self.batch_size]
            all_batches.extend(batches)
        if self.shuffle:
            rng.shuffle(all_batches)
        if not all_batches:
            return iter(())
        while len(all_batches) < self._global_num_batches:
            all_batches.append(list(all_batches[len(all_batches) % len(all_batches)]))
        all_batches = all_batches[: self._global_num_batches]
        return iter(all_batches[self.rank :: self.num_replicas])


def build_city_type_buckets(dataset: Any, cfg: Dict[str, Any]) -> Dict[str, list[int]]:
    if not hasattr(dataset, "sample_refs") or not hasattr(dataset, "hdf5_path"):
        raise ValueError("homogeneous_city_type_batches requires an HDF5 dataset with sample_refs")
    resolver = AutomaticCityTypeResolver(cfg)
    input_column = str(cfg["data"].get("input_column", "topology_map"))
    buckets: Dict[str, list[int]] = defaultdict(list)
    city_cache: Dict[str, str] = {}
    with h5py.File(getattr(dataset, "hdf5_path"), "r") as handle:
        for idx, (city, sample) in enumerate(getattr(dataset, "sample_refs")):
            city_type = city_cache.get(city)
            if city_type is None:
                raw_topology = np.asarray(handle[city][sample][input_column][...], dtype=np.float32)
                city_type = resolver._infer_from_raw_topology(raw_topology)
                city_cache[city] = city_type
            buckets[city_type].append(idx)
    return buckets


def _city_type_to_expert_index(city_type: str) -> int:
    try:
        return CITY_TYPE_EXPERT_ORDER.index(str(city_type))
    except ValueError:
        return CITY_TYPE_EXPERT_ORDER.index("mixed_midrise")


def _select_city_expert_map(expert_maps: torch.Tensor, city_type: str | Sequence[str]) -> torch.Tensor:
    if expert_maps.ndim != 4:
        raise ValueError(f"Expected expert_maps [B,E,H,W], got {tuple(expert_maps.shape)}")
    if isinstance(city_type, str):
        expert_idx = _city_type_to_expert_index(city_type)
        expert_idx = min(max(expert_idx, 0), expert_maps.shape[1] - 1)
        return expert_maps[:, expert_idx : expert_idx + 1]

    city_types = list(city_type)
    if len(city_types) != expert_maps.shape[0]:
        raise ValueError(
            f"Expected {expert_maps.shape[0]} city-type labels, got {len(city_types)}"
        )
    indices = torch.as_tensor(
        [_city_type_to_expert_index(label) for label in city_types],
        device=expert_maps.device,
        dtype=torch.long,
    )
    indices = indices.clamp(min=0, max=expert_maps.shape[1] - 1)
    gather_index = indices.view(-1, 1, 1, 1).expand(-1, 1, expert_maps.shape[2], expert_maps.shape[3])
    return torch.gather(expert_maps, 1, gather_index)


def _los_nlos_masks(input_batch: torch.Tensor, cfg: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg["data"].get("los_input_column") and input_batch.shape[1] > 1:
        los = input_batch[:, 1:2].clamp(0.0, 1.0)
        nlos = (1.0 - los).clamp(0.0, 1.0)
        return los, nlos
    ones = torch.ones_like(input_batch[:, :1])
    return ones, ones


def _compose_final_prediction(
    raw_output: torch.Tensor,
    prior_or_stage1: torch.Tensor,
    input_batch: torch.Tensor,
    cfg: Dict[str, Any],
    city_type_resolver: Optional[AutomaticCityTypeResolver],
) -> tuple[torch.Tensor, torch.Tensor, str]:
    city_types = city_type_resolver.infer_batch(input_batch) if city_type_resolver is not None else ["mixed_midrise"] * input_batch.shape[0]
    city_type = city_types[0] if len(set(city_types)) == 1 else "mixed_midrise"
    _, nlos = _los_nlos_masks(input_batch, cfg)
    if _model_arch(cfg) == "city_routed_nlos_moe":
        selected_delta = _select_city_expert_map(raw_output, city_types)
        final_pred = prior_or_stage1 + nlos * selected_delta
        return final_pred, selected_delta, city_type
    final_pred = prior_or_stage1 + raw_output[:, :1]
    return final_pred, raw_output[:, :1], city_type


class RegimeBanditReweighter:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        bandit_cfg = dict(cfg.get("training", {}).get("bandit_reweighting", {}))
        self.enabled = bool(bandit_cfg.get("enabled", False))
        self.ema = float(bandit_cfg.get("ema", 0.9))
        self.exploration = float(bandit_cfg.get("exploration", 0.05))
        self.min_weight = float(bandit_cfg.get("min_weight", 0.8))
        self.max_weight = float(bandit_cfg.get("max_weight", 1.8))
        self.temperature = max(float(bandit_cfg.get("temperature", 1.0)), 1e-6)
        self.state = {
            name: {"ema_loss": 1.0, "count": 0}
            for name in CITY_TYPE_EXPERT_ORDER
        }

    def weight_for(self, city_type: str) -> float:
        if not self.enabled:
            return 1.0
        losses = np.asarray([float(v["ema_loss"]) for v in self.state.values()], dtype=np.float64)
        centered = losses - float(np.mean(losses))
        logits = centered / max(float(np.std(losses)) * self.temperature, 1e-6)
        probs = np.exp(logits - np.max(logits))
        probs = probs / max(float(np.sum(probs)), 1e-12)
        idx = _city_type_to_expert_index(city_type)
        chosen = float(probs[min(idx, len(probs) - 1)])
        uniform = 1.0 / max(len(CITY_TYPE_EXPERT_ORDER), 1)
        mixed = (1.0 - self.exploration) * chosen + self.exploration * uniform
        normalized = mixed / max(uniform, 1e-12)
        return float(np.clip(normalized, self.min_weight, self.max_weight))

    def update(self, city_type: str, loss_value: float) -> None:
        if not self.enabled:
            return
        key = CITY_TYPE_EXPERT_ORDER[_city_type_to_expert_index(city_type)]
        slot = self.state[key]
        slot["ema_loss"] = self.ema * float(slot["ema_loss"]) + (1.0 - self.ema) * float(loss_value)
        slot["count"] = int(slot["count"]) + 1

    def summary(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "_enabled": bool(self.enabled),
            "_signal": "final_loss_physical_proxy",
        }
        payload.update({
            key: {
                "ema_loss": float(val["ema_loss"]),
                "count": int(val["count"]),
                "weight": float(self.weight_for(key)),
            }
            for key, val in self.state.items()
        })
        return payload


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
        city_types = city_type_resolver.infer_batch(input_batch)
        sample_weights = torch.as_tensor(
            [max(city_weights.get(city_type, default_city_weight), 1e-6) for city_type in city_types],
            device=weight.device,
            dtype=weight.dtype,
        ).view(-1, 1, 1, 1)
        weight = weight * sample_weights

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
        batch_size=1,
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
    clamp_final = bool(prior_cfg.get("clamp_final_output", True))
    annotator = RegimeAnnotator(
        dataset,
        dict(cfg["data"].get("path_loss_formula_input", {})).get("regime_calibration_json"),
        cfg,
    )

    totals = init_metric_totals()
    prior_totals = init_metric_totals()
    regime_totals: dict[str, dict[str, float]] = defaultdict(init_metric_totals)

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="val", leave=False, disable=distributed and not is_main_process(rank))):
            x, y, m, scalar_cond = unpack_cgan_batch(batch, device)
            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                residual_pred = _compose_residual_prediction(
                    model,
                    x,
                    scalar_cond,
                    x[:, formula_idx : formula_idx + 1],
                    separated_mode=separated_mode,
                    base_generator=base_generator,
                )
            prior = x[:, formula_idx : formula_idx + 1]
            info = annotator.info_for_index(sample_indices[idx], x, scalar_cond)
            if _model_arch(cfg) == "city_routed_nlos_moe":
                _, nlos_mask = _los_nlos_masks(x, cfg)
                selected_residual = _select_city_expert_map(residual_pred, info["city_type"])
                pred = prior + nlos_mask * selected_residual
            else:
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
            "prior_totals": prior_totals,
            "regime_totals": dict(regime_totals),
        }
        gathered: list[dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, payload)
        if not is_main_process(rank):
            return {}
        totals = init_metric_totals()
        prior_totals = init_metric_totals()
        merged_regimes: dict[str, dict[str, float]] = defaultdict(init_metric_totals)
        for part in gathered:
            for k, v in part["totals"].items():
                totals[k] += float(v)
            for k, v in part["prior_totals"].items():
                prior_totals[k] += float(v)
            for key, reg in dict(part["regime_totals"]).items():
                bucket = merged_regimes[key]
                for k, v in reg.items():
                    bucket[k] += float(v)
        regime_totals = merged_regimes

    total_count = float(totals.get("count", 0.0))
    los_count = float(regime_totals.get("path_loss__los__LoS", {}).get("count", 0.0))
    nlos_count = float(regime_totals.get("path_loss__los__NLoS", {}).get("count", 0.0))
    summary: dict[str, Any] = {
        "path_loss": finalize_metric_total(totals, unit, total_count=total_count),
        "path_loss__prior__overall": finalize_metric_total(prior_totals, unit, total_count=total_count),
        "_support": {
            "valid_pixel_count": int(round(total_count)),
            "los_valid_pixel_count": int(round(los_count)),
            "nlos_valid_pixel_count": int(round(nlos_count)),
            "los_fraction": float(los_count / total_count) if total_count > 0.0 else float("nan"),
            "nlos_fraction": float(nlos_count / total_count) if total_count > 0.0 else float("nan"),
        },
        "_regimes": {
            key: finalize_metric_total(val, unit, total_count=total_count)
            for key, val in sorted(regime_totals.items())
        },
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


def build_stage1_loss_flags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    loss_cfg = dict(cfg.get("loss", {}))
    residual_cfg = dict(cfg.get("prior_residual_path_loss", {}))
    sep_cfg = dict(cfg.get("separated_refiner", {}))
    multiscale_cfg = dict(cfg.get("multiscale_path_loss", {}))
    gate_weight = float(sep_cfg.get("gate_loss_weight", 0.0))
    return {
        "optimize_residual_only": bool(residual_cfg.get("optimize_residual_only", False)),
        "multiscale_enabled": bool(multiscale_cfg.get("enabled", False)),
        "gan_enabled": float(loss_cfg.get("lambda_gan", 0.0)) > 0.0,
        "gate_enabled": bool(sep_cfg.get("use_gate", False)) and gate_weight > 0.0,
        "lambda_recon": float(loss_cfg.get("lambda_recon", 1.0)),
        "lambda_gan": float(loss_cfg.get("lambda_gan", 0.0)),
        "residual_loss_weight": float(residual_cfg.get("loss_weight", 1.0)),
        "final_loss_weight_when_residual_only": float(
            residual_cfg.get("final_loss_weight_when_residual_only", 0.0)
        ),
        "multiscale_loss_weight_when_residual_only": float(
            residual_cfg.get("multiscale_loss_weight_when_residual_only", 0.0)
        ),
        "gate_loss_weight": gate_weight,
    }


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
    separated_mode: bool = False,
    base_generator: Optional[nn.Module] = None,
    city_type_resolver: Optional[AutomaticCityTypeResolver] = None,
    bandit_reweighter: Optional[RegimeBanditReweighter] = None,
) -> tuple[float, float, Dict[str, float]]:
    formula_idx = formula_channel_index(cfg)
    meta = dict(cfg["target_metadata"]["path_loss"])
    loss_cfg = dict(cfg["loss"])
    residual_cfg = dict(cfg.get("prior_residual_path_loss", {}))
    sep_cfg = dict(cfg.get("separated_refiner", {}))
    clamp_final = bool(residual_cfg.get("clamp_final_output", True))
    optimize_residual_only = bool(residual_cfg.get("optimize_residual_only", False))
    residual_only_final_weight = float(residual_cfg.get("final_loss_weight_when_residual_only", 0.0))
    residual_only_multiscale_weight = float(
        residual_cfg.get("multiscale_loss_weight_when_residual_only", 0.0)
    )
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
    running_g = 0.0
    running_d = 0.0
    running_components: Dict[str, float] = {
        "final_loss": 0.0,
        "residual_loss": 0.0,
        "multiscale_loss": 0.0,
        "gan_loss": 0.0,
        "gate_loss": 0.0,
        "term_final": 0.0,
        "term_residual": 0.0,
        "term_multiscale": 0.0,
        "term_gan": 0.0,
        "term_gate": 0.0,
        "generator_loss_total": 0.0,
    }
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x, y, m, scalar_cond = unpack_cgan_batch(batch, device)
        target = y[:, :1]
        mask = m[:, :1]
        prior = x[:, formula_idx : formula_idx + 1]
        residual_target = target - prior
        los_mask, nlos_mask = _los_nlos_masks(x, cfg)

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
                fake_det, _, _ = _compose_final_prediction(
                    residual_fake_det,
                    prior,
                    x,
                    cfg,
                    city_type_resolver,
                )
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
            residual_pred, gate_logits, base_residual = _compose_residual_prediction_with_aux(
                generator,
                x,
                scalar_cond,
                prior,
                separated_mode=separated_mode,
                base_generator=base_generator,
                use_gate=use_gate,
            )
            pred, selected_residual, city_type = _compose_final_prediction(
                residual_pred,
                prior,
                x,
                cfg,
                city_type_resolver,
            )
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
            weighted_mask = _apply_regime_reweighting(weighted_mask, x, cfg, city_type_resolver)
            if bool(cfg.get("model", {}).get("hard_los_passthrough", False)) or _model_arch(cfg) == "city_routed_nlos_moe":
                weighted_mask = weighted_mask * nlos_mask
            if bandit_reweighter is not None:
                weighted_mask = weighted_mask * bandit_reweighter.weight_for(city_type)

            final_loss = masked_mse_l1_loss(
                pred,
                target,
                weighted_mask,
                mse_weight=float(loss_cfg.get("mse_weight", 1.0)),
                l1_weight=float(loss_cfg.get("l1_weight", 0.0)),
            )
            residual_loss = masked_mse_l1_loss(
                selected_residual,
                residual_target,
                weighted_mask,
                mse_weight=float(residual_cfg.get("mse_weight", 1.0)),
                l1_weight=float(residual_cfg.get("l1_weight", 0.0)),
            )
            multiscale_loss = compute_multiscale_path_loss_loss(pred, target, weighted_mask, meta, cfg)
            if use_gate and gate_logits is not None and gate_target is not None:
                gate_loss_map = F.binary_cross_entropy_with_logits(gate_logits, gate_target, reduction="none")
                gate_loss = (gate_loss_map * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)

            if use_gan and discriminator is not None:
                fake_logits_for_g = discriminator(x, pred)
                gan_loss = adv_criterion(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
            else:
                gan_loss = torch.zeros((), device=target.device)

            if optimize_residual_only:
                g_loss = (
                    float(residual_cfg.get("loss_weight", 1.0)) * residual_loss
                    + residual_only_final_weight * final_loss
                    + residual_only_multiscale_weight * multiscale_loss
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

            final_loss_value = float(final_loss.detach().item())
            residual_loss_value = float(residual_loss.detach().item())
            multiscale_loss_value = float(multiscale_loss.detach().item())
            gan_loss_value = float(gan_loss.detach().item())
            gate_loss_value = float(gate_loss.detach().item())
            residual_term_weight = float(
                residual_cfg.get("loss_weight", 1.0 if optimize_residual_only else 0.0)
            )
            if optimize_residual_only:
                term_final = residual_only_final_weight * final_loss_value
                term_residual = residual_term_weight * residual_loss_value
                term_multiscale = residual_only_multiscale_weight * multiscale_loss_value
                term_gan = 0.0
            else:
                term_final = lambda_recon * final_loss_value
                term_residual = residual_term_weight * residual_loss_value
                term_multiscale = multiscale_loss_value
                term_gan = lambda_gan * gan_loss_value
            term_gate = gate_loss_weight * gate_loss_value if gate_loss_weight > 0.0 else 0.0

        scaler_g.scale(g_loss).backward()
        if clip_grad > 0.0:
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        running_g += float(g_loss.item())
        running_d += float(d_loss.item())
        running_components["final_loss"] += final_loss_value
        running_components["residual_loss"] += residual_loss_value
        running_components["multiscale_loss"] += multiscale_loss_value
        running_components["gan_loss"] += gan_loss_value
        running_components["gate_loss"] += gate_loss_value
        running_components["term_final"] += term_final
        running_components["term_residual"] += term_residual
        running_components["term_multiscale"] += term_multiscale
        running_components["term_gan"] += term_gan
        running_components["term_gate"] += term_gate
        running_components["generator_loss_total"] += float(g_loss.item())
        if bandit_reweighter is not None:
            bandit_reweighter.update(city_type, final_loss_value)
        steps += 1
    denom = max(steps, 1)
    avg_components = {key: float(value / denom) for key, value in running_components.items()}
    return running_g / denom, running_d / denom, avg_components


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Try 52 PMNet prior+residual path-loss model")
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
        raise ValueError("Try 42 expects scalar channels, not scalar FiLM vectors.")

    pin_memory = is_cuda_device(device)
    train_batch_size = int(cfg["training"]["batch_size"])
    homogeneous_city_type_batches = bool(cfg["training"].get("homogeneous_city_type_batches", False))
    train_sampler: Optional[Any]
    train_loader_kwargs = {
        "num_workers": int(cfg["data"]["num_workers"]),
        "pin_memory": pin_memory,
        "persistent_workers": int(cfg["data"]["num_workers"]) > 0,
    }
    if homogeneous_city_type_batches and train_batch_size > 1:
        city_type_buckets = build_city_type_buckets(train_dataset, cfg)
        train_sampler = HomogeneousCityTypeBatchSampler(
            city_type_buckets,
            batch_size=train_batch_size,
            shuffle=True,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            **train_loader_kwargs,
        )
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **train_loader_kwargs,
        )
    city_type_resolver = AutomaticCityTypeResolver(cfg)
    bandit_reweighter = RegimeBanditReweighter(cfg)

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

            train_g_loss, train_d_loss, train_loss_components = train_one_epoch(
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
                separated_mode=separated_mode,
                base_generator=base_generator,
                city_type_resolver=city_type_resolver,
                bandit_reweighter=bandit_reweighter,
            )
            barrier_if_distributed(distributed)

            val_summary = evaluate_validation(
                generator,
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

            if is_main_process(rank):
                val_summary["_train"] = {
                    "generator_loss": float(train_g_loss),
                    "discriminator_loss": float(train_d_loss),
                    "loss_components": train_loss_components,
                    "loss_flags": build_stage1_loss_flags(cfg),
                    "bandit_reweighting": bandit_reweighter.summary(),
                }
                current_score = extract_selection_metric(val_summary, selection_metric)
                if current_score < best_score:
                    best_score = current_score
                    best_epoch = epoch
                    best_payload = {
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "model": generator.state_dict(),
                        "generator": generator.state_dict(),
                        "optimizer_g": optimizer_g.state_dict(),
                        "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
                        "scaler_g": scaler_g.state_dict(),
                        "config_path": args.config,
                    }
                    if discriminator is not None:
                        best_payload["discriminator"] = discriminator.state_dict()
                    if optimizer_d is not None:
                        best_payload["optimizer_d"] = optimizer_d.state_dict()
                    if scaler_d is not None:
                        best_payload["scaler_d"] = scaler_d.state_dict()
                    torch.save(best_payload, out_dir / "best_cgan.pt")

                epoch_ckpt_path = out_dir / f"epoch_{epoch}_cgan.pt"
                epoch_payload = {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "model": generator.state_dict(),
                    "generator": generator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
                    "scaler_g": scaler_g.state_dict(),
                    "config_path": args.config,
                }
                if discriminator is not None:
                    epoch_payload["discriminator"] = discriminator.state_dict()
                if optimizer_d is not None:
                    epoch_payload["optimizer_d"] = optimizer_d.state_dict()
                if scaler_d is not None:
                    epoch_payload["scaler_d"] = scaler_d.state_dict()
                torch.save(epoch_payload, epoch_ckpt_path)
                prev_epoch_ckpt_path = out_dir / f"epoch_{epoch - 1}_cgan.pt"
                if prev_epoch_ckpt_path.exists():
                    prev_epoch_ckpt_path.unlink()

                write_validation_json(out_dir, epoch, val_summary, best_epoch=best_epoch, best_score=best_score)
                print(json.dumps({"epoch": epoch, "generator_loss": train_g_loss, "discriminator_loss": train_d_loss, selection_metric: current_score}))
            else:
                current_score = 0.0

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
