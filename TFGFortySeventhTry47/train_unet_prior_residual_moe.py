from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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
    compute_scalar_cond_dim,
    forward_cgan_generator,
    return_scalar_cond_from_config,
)
from model_unet_residual_moe import UNetPriorResidualMoE


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
    if not bool(cfg["data"].get("path_loss_formula_input", {}).get("enabled", False)):
        raise ValueError("Try 47 requires data.path_loss_formula_input.enabled = true")
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
    learning_rate = float(cfg["training"].get("learning_rate", 3e-5))
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, foreach=is_cuda_device(device))
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, foreach=is_cuda_device(device))
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


def unpack_try47_batch(
    batch: Tuple[Any, ...],
    device: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict[str, Any]]]:
    meta: Optional[dict[str, Any]] = None
    if len(batch) == 5:
        x, y, m, sc, meta = batch
        return x.to(device), y.to(device), m.to(device), sc.to(device), meta
    if len(batch) == 4:
        x, y, m, last = batch
        if isinstance(last, dict):
            meta = last
            return x.to(device), y.to(device), m.to(device), None, meta
        return x.to(device), y.to(device), m.to(device), last.to(device), None
    x, y, m = batch
    return x.to(device), y.to(device), m.to(device), None, None


class RegimeAnnotator:
    def __init__(self, dataset: Any, calibration_json: Optional[str]) -> None:
        self.dataset = dataset
        self.hdf5_path = str(dataset.hdf5_path)
        self.input_column = str(dataset.input_column)
        self.non_ground_threshold = float(dataset.non_ground_threshold)
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
            antenna_height_m = float(np.asarray(grp["uav_height"][...], dtype=np.float32).reshape(-1)[0])
        non_ground = raw_topology != self.non_ground_threshold
        building_density = float(np.mean(non_ground))
        non_zero = raw_topology[non_ground]
        mean_height = float(np.mean(non_zero)) if non_zero.size else 0.0

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
        info = {"city_type": city_type, "antenna_bin": ant_bin, "city": city}
        self._cache[key] = info
        return info

    def info_for_index(self, idx: int) -> dict[str, str]:
        city, sample = self.dataset.sample_refs[idx]
        return self._compute_info(city, sample)


def init_metric_totals() -> dict[str, float]:
    return {"count": 0.0, "sum_squared_error": 0.0, "sum_absolute_error": 0.0}


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


def compute_nlos_shadow_bands(los_map: torch.Tensor, kernel_size: int = 41) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel = max(int(kernel_size), 1)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    nlos = (los_map <= 0.5).to(dtype=torch.float32)
    support = F.avg_pool2d(nlos, kernel_size=kernel, stride=1, padding=pad)
    shallow = (support > 0.0) & (support <= 0.33)
    medium = (support > 0.33) & (support <= 0.66)
    deep = support > 0.66
    return shallow, medium, deep


def evaluate_validation(
    model: nn.Module,
    dataset: Any,
    device: object,
    cfg: Dict[str, Any],
    amp_enabled: bool,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=is_cuda_device(device))
    formula_idx = formula_channel_index(cfg)
    meta = dict(cfg["target_metadata"]["path_loss"])
    residual_cfg = dict(cfg.get("prior_residual_path_loss", {}))
    clamp_final = bool(residual_cfg.get("clamp_final_output", True))
    calibration_json = cfg["data"].get("path_loss_formula_input", {}).get("regime_calibration_json")
    annotator = RegimeAnnotator(dataset, calibration_json)

    overall = init_metric_totals()
    prior_totals = init_metric_totals()
    regime_totals: dict[str, dict[str, float]] = defaultdict(init_metric_totals)

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="val", leave=False)):
            x, y, m, scalar_cond, _ = unpack_try47_batch(batch, device)
            target = y[:, :1]
            mask = m[:, :1]
            prior = x[:, formula_idx : formula_idx + 1]

            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                residual_pred = forward_cgan_generator(model, x, scalar_cond)
                pred = prior + residual_pred
                if clamp_final:
                    pred = clip_to_target_range(pred, meta)

            pred_phys = denormalize(pred, meta)
            target_phys = denormalize(target, meta)
            prior_phys = denormalize(prior, meta)
            valid_mask = mask > 0.0
            diff_phys = (pred_phys - target_phys)[valid_mask]
            prior_diff_phys = (prior_phys - target_phys)[valid_mask]
            update_metric_total(overall, diff_phys)
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
                nlos_shallow, nlos_medium, nlos_deep = compute_nlos_shadow_bands(los)
                for label, subtype_mask in (
                    ("shallow_shadow", nlos_valid & nlos_shallow),
                    ("medium_shadow", nlos_valid & nlos_medium),
                    ("deep_shadow", nlos_valid & nlos_deep),
                ):
                    update_metric_total(regime_totals[f"path_loss__nlos_shadow__{label}"], (pred_phys - target_phys)[subtype_mask])
                    update_metric_total(regime_totals[f"path_loss__prior__nlos_shadow__{label}"], (prior_phys - target_phys)[subtype_mask])

            update_metric_total(regime_totals[f"path_loss__city_type__{info['city_type']}"], diff_phys)
            update_metric_total(regime_totals[f"path_loss__antenna_bin__{info['antenna_bin']}"], diff_phys)
            update_metric_total(regime_totals[f"path_loss__prior__city_type__{info['city_type']}"], prior_diff_phys)
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

    summary = {"path_loss": finalize_metric_total(overall, meta.get("unit", "dB"))}
    summary["_prior"] = {"path_loss": finalize_metric_total(prior_totals, meta.get("unit", "dB"))}
    summary["_regimes"] = {name: finalize_metric_total(totals, meta.get("unit", "dB")) for name, totals in sorted(regime_totals.items())}
    return summary


def extract_selection_metric(summary: dict[str, Any], metric_path: str) -> float:
    current: Any = summary
    for part in metric_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Selection metric {metric_path} not found in validation summary")
        current = current[part]
    if not isinstance(current, (int, float)):
        raise TypeError(f"Selection metric {metric_path} is not numeric")
    return float(current)


def write_validation_json(out_dir: Path, epoch: int, summary: dict[str, Any], *, best_epoch: int, best_score: float) -> None:
    payload = dict(summary)
    payload["_checkpoint"] = {"epoch": int(epoch), "best_epoch": int(best_epoch), "best_score": float(best_score)}
    latest = out_dir / "validate_metrics_cgan_latest.json"
    epoch_path = out_dir / f"validate_metrics_epoch_{epoch}_cgan.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    epoch_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _meta_item(meta: Optional[dict[str, Any]], key: str, batch_index: int, default: str) -> str:
    if not meta or key not in meta:
        return default
    value = meta[key]
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return str(value[min(batch_index, len(value) - 1)])
    return str(value)


def add_nlos_combo_losses(
    total_loss: torch.Tensor,
    nlos_residual_pred: torch.Tensor,
    residual_target: torch.Tensor,
    mask: torch.Tensor,
    los_mask: torch.Tensor,
    meta: Optional[dict[str, Any]],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    combo_cfg = dict(cfg.get("nlos_combo_loss", {}))
    if not bool(combo_cfg.get("enabled", False)):
        return total_loss

    shadow_kernel = int(combo_cfg.get("shadow_kernel_size", 41))
    shallow_mask, medium_mask, deep_mask = compute_nlos_shadow_bands(los_mask, kernel_size=shadow_kernel)
    mse_weight = float(combo_cfg.get("mse_weight", 1.0))
    l1_weight = float(combo_cfg.get("l1_weight", 0.05))
    city_bonus = dict(combo_cfg.get("city_bonus", {}))
    ant_shadow_weights = dict(combo_cfg.get("antenna_shadow_weights", {}))

    for batch_index in range(nlos_residual_pred.shape[0]):
        antenna_bin = _meta_item(meta, "antenna_bin", batch_index, "mid_ant")
        city_type = _meta_item(meta, "city_type", batch_index, "mixed_midrise")
        base_valid = mask[batch_index : batch_index + 1] * (los_mask[batch_index : batch_index + 1] <= 0.5).float()
        if base_valid.sum().item() <= 0:
            continue
        shadow_masks = {
            "shallow_shadow": base_valid * shallow_mask[batch_index : batch_index + 1].float(),
            "medium_shadow": base_valid * medium_mask[batch_index : batch_index + 1].float(),
            "deep_shadow": base_valid * deep_mask[batch_index : batch_index + 1].float(),
        }
        city_scale = 1.0 + float(city_bonus.get(city_type, 0.0))
        for label, submask in shadow_masks.items():
            if submask.sum().item() <= 0:
                continue
            combo_key = f"{antenna_bin}__{label}"
            weight = float(ant_shadow_weights.get(combo_key, 0.0))
            if weight <= 0.0:
                continue
            combo_loss = masked_mse_l1_loss(
                nlos_residual_pred[batch_index : batch_index + 1],
                residual_target[batch_index : batch_index + 1],
                submask,
                mse_weight=mse_weight,
                l1_weight=l1_weight,
            )
            total_loss = total_loss + city_scale * weight * combo_loss
    return total_loss


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
    branch_cfg = dict(cfg.get("los_nlos_branching", {}))
    clamp_final = bool(residual_cfg.get("clamp_final_output", True))
    clip_grad = float(cfg["training"].get("clip_grad_norm", 0.0))

    model.train()
    running = 0.0
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x, y, m, scalar_cond, regime_meta = unpack_try47_batch(batch, device)
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

            inner_model = model.module if isinstance(model, DistributedDataParallel) else model
            los_mask = x[:, 1:2] if cfg["data"].get("los_input_column") else None
            if bool(branch_cfg.get("enabled", True)) and los_mask is not None and hasattr(inner_model, "last_branch_outputs"):
                los_base_pred, _nlos_delta_pred, nlos_full_pred = inner_model.last_branch_outputs()
                if los_base_pred is not None and nlos_full_pred is not None:
                    los_valid = mask * (los_mask > 0.5).float()
                    nlos_valid = mask * (los_mask <= 0.5).float()
                    los_branch_loss = masked_mse_l1_loss(
                        los_base_pred,
                        residual_target,
                        los_valid,
                        mse_weight=float(branch_cfg.get("los_mse_weight", 1.0)),
                        l1_weight=float(branch_cfg.get("los_l1_weight", 0.02)),
                    )
                    nlos_branch_loss = masked_mse_l1_loss(
                        nlos_full_pred,
                        residual_target,
                        nlos_valid,
                        mse_weight=float(branch_cfg.get("nlos_mse_weight", 1.0)),
                        l1_weight=float(branch_cfg.get("nlos_l1_weight", 0.05)),
                    )
                    total_loss = total_loss + float(branch_cfg.get("los_loss_weight", 0.12)) * los_branch_loss
                    total_loss = total_loss + float(branch_cfg.get("nlos_loss_weight", 0.95)) * nlos_branch_loss
                    total_loss = add_nlos_combo_losses(total_loss, nlos_full_pred, residual_target, mask, los_mask, regime_meta, cfg)

            balance_weight = float(moe_cfg.get("balance_loss_weight", 0.0))
            if balance_weight > 0.0 and hasattr(inner_model, "moe_balance_loss"):
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
    parser = argparse.ArgumentParser(description="Train Try 47 U-Net prior+residual with NLoS expert branch")
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

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    pin_memory = is_cuda_device(device)
    num_workers = int(cfg["data"].get("num_workers", 0))
    persistent_workers = bool(cfg["data"].get("persistent_workers", num_workers > 0))
    prefetch_factor_raw = cfg["data"].get("prefetch_factor", None)
    train_loader_kwargs = dict(
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    if num_workers > 0 and prefetch_factor_raw is not None:
        train_loader_kwargs["prefetch_factor"] = int(prefetch_factor_raw)
    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )

    model = UNetPriorResidualMoE(
        in_channels=compute_input_channels(cfg),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"].get("base_channels", 96)),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        scalar_cond_dim=compute_scalar_cond_dim(cfg) if return_scalar_cond_from_config(cfg) else 0,
        scalar_film_hidden=int(cfg["model"].get("scalar_film_hidden", 192)),
        upsample_mode=str(cfg["model"].get("upsample_mode", "bilinear")),
        num_experts=int(cfg["model"].get("num_experts", 4)),
        expert_channels=int(cfg["model"].get("expert_channels", 96)),
        los_channel_index=int(cfg["model"].get("los_channel_index", 1)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
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
        if "scheduler" in state and scheduler is not None and state["scheduler"] is not None:
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
