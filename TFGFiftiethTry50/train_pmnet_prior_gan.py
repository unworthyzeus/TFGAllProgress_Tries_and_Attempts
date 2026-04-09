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
    build_dataset_splits_from_config,
    compute_input_channels,
    forward_cgan_generator,
    return_scalar_cond_from_config,
    unpack_cgan_batch,
)
from model_pmnet import PMNetResidualRegressor, PatchDiscriminator, UNetResidualRefiner


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
        raise ValueError("Try 50 stage 1 requires data.path_loss_formula_input.enabled = true")
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


def _build_pmnet_from_cfg(cfg: Dict[str, Any], in_channels: int) -> nn.Module:
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
) -> tuple[float, float]:
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
    running_g = 0.0
    running_d = 0.0
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
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
            residual_pred, gate_logits, base_residual = _compose_residual_prediction_with_aux(
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
            if use_gate and gate_logits is not None and gate_target is not None:
                gate_loss_map = F.binary_cross_entropy_with_logits(gate_logits, gate_target, reduction="none")
                gate_loss = (gate_loss_map * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)

            if use_gan and discriminator is not None:
                fake_logits_for_g = discriminator(x, pred)
                gan_loss = adv_criterion(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
            else:
                gan_loss = torch.zeros((), device=target.device)

            if optimize_residual_only:
                g_loss = float(residual_cfg.get("loss_weight", 1.0)) * residual_loss
            else:
                g_loss = (
                    lambda_recon * final_loss
                    + float(residual_cfg.get("loss_weight", 0.0)) * residual_loss
                    + multiscale_loss
                    + lambda_gan * gan_loss
                )

            if gate_loss_weight > 0.0:
                g_loss = g_loss + gate_loss_weight * gate_loss

        scaler_g.scale(g_loss).backward()
        if clip_grad > 0.0:
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        running_g += float(g_loss.item())
        running_d += float(d_loss.item())
        steps += 1
    denom = max(steps, 1)
    return running_g / denom, running_d / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Try 49 PMNet prior+residual path-loss model with light PatchGAN")
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
        raise ValueError("Try 50 expects scalar channels, not scalar FiLM vectors.")

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

    discriminator = PatchDiscriminator(
        in_channels=input_channels,
        target_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"].get("disc_base_channels", 24)),
        norm_type=str(cfg["model"].get("disc_norm_type", cfg["model"].get("norm_type", "group"))),
        input_downsample_factor=int(cfg["model"].get("disc_input_downsample_factor", 2)),
    ).to(device)

    generator_for_training: nn.Module = generator
    discriminator_for_training: nn.Module = discriminator
    if distributed:
        generator_for_training = DistributedDataParallel(
            generator,
            device_ids=[local_rank] if is_cuda_device(device) else None,
            output_device=local_rank if is_cuda_device(device) else None,
            find_unused_parameters=False,
        )
        discriminator_for_training = DistributedDataParallel(
            discriminator,
            device_ids=[local_rank] if is_cuda_device(device) else None,
            output_device=local_rank if is_cuda_device(device) else None,
            find_unused_parameters=False,
        )

    optimizer_g = build_optimizer(cfg, generator_for_training.parameters(), device)
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
        if "discriminator" in state:
            target_discriminator = discriminator_for_training.module if isinstance(discriminator_for_training, DistributedDataParallel) else discriminator_for_training
            target_discriminator.load_state_dict(state["discriminator"])
        if "optimizer_g" in state:
            optimizer_g.load_state_dict(state["optimizer_g"])
            move_optimizer_state_to_device(optimizer_g, device)
        elif "optimizer" in state:
            optimizer_g.load_state_dict(state["optimizer"])
            move_optimizer_state_to_device(optimizer_g, device)
        if "optimizer_d" in state:
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
        if "scaler_d" in state:
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

            train_g_loss, train_d_loss = train_one_epoch(
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
                val_summary["_train"] = {"generator_loss": float(train_g_loss), "discriminator_loss": float(train_d_loss)}
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
                        "discriminator": discriminator.state_dict(),
                        "optimizer_g": optimizer_g.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
                        "scaler_g": scaler_g.state_dict(),
                        "scaler_d": scaler_d.state_dict(),
                        "config_path": args.config,
                    }
                    torch.save(best_payload, out_dir / "best_cgan.pt")

                epoch_ckpt_path = out_dir / f"epoch_{epoch}_cgan.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "model": generator.state_dict(),
                        "generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer_g": optimizer_g.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
                        "scaler_g": scaler_g.state_dict(),
                        "scaler_d": scaler_d.state_dict(),
                        "config_path": args.config,
                    },
                    epoch_ckpt_path,
                )
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
