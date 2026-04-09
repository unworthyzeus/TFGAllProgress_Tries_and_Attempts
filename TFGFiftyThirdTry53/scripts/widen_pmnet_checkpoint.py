#!/usr/bin/env python3
"""Adapt PMNet checkpoint weights from one base_channels to another.

This script converts a trained PMNet checkpoint to a new base_channels setting
so training can continue while preserving learned structure. It supports both
widening and shrinking.

Example:
python TFGFortyEighthTry48/scripts/widen_pmnet_checkpoint.py \
  --src-config TFGFortyEighthTry48/experiments/fortyeighthtry48_pmnet_prior_gan_fastbatch/fortyeighthtry48_pmnet_prior_stage1_warmstart_try42.yaml \
  --dst-config TFGFortyEighthTry48/experiments/fortyeighthtry48_pmnet_prior_gan_fastbatch/fortyeighthtry48_pmnet_prior_stage1_warmstart_try42_widen96_cont.yaml \
  --src-ckpt /scratch/nas/3/gmoreno/TFGpractice/TFGFortyEighthTry48/outputs/fortyeighthtry48_pmnet_prior_stage1_warmstart_try42_t48_stage1_warm_t42_4gpu/best_cgan.pt \
  --out-ckpt /scratch/nas/3/gmoreno/TFGpractice/TFGFortyEighthTry48/outputs/fortyeighthtry48_pmnet_prior_stage1_warmstart_try42_t48_stage1_warm_t42_4gpu/widened_72_to_96_epoch22_seed.ckpt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from model_pmnet import CityTypeRoutedNLoSMoERegressor, GlobalContextUNetRefiner, PMNetResidualRegressor, UNetResidualRefiner  # noqa: E402


def compute_input_channels(cfg: Dict[str, Any]) -> int:
    in_channels = 1
    data_cfg = dict(cfg.get("data", {}))
    if data_cfg.get("los_input_column"):
        in_channels += 1
    if bool(data_cfg.get("distance_map_channel", False)):
        in_channels += 1
    if bool(data_cfg.get("path_loss_formula_input", {}).get("enabled", False)):
        in_channels += 1
        if bool(data_cfg.get("path_loss_formula_input", {}).get("include_confidence_channel", False)):
            in_channels += 1
    if bool(data_cfg.get("path_loss_obstruction_features", {}).get("enabled", False)):
        obstruction_cfg = dict(data_cfg.get("path_loss_obstruction_features", {}))
        if bool(obstruction_cfg.get("include_shadow_depth", True)):
            in_channels += 1
        if bool(obstruction_cfg.get("include_distance_since_los_break", True)):
            in_channels += 1
        if bool(obstruction_cfg.get("include_max_blocker_height", True)):
            in_channels += 1
        if bool(obstruction_cfg.get("include_blocker_count", True)):
            in_channels += 1
    if bool(cfg.get("model", {}).get("use_scalar_channels", False)):
        in_channels += len(list(data_cfg.get("scalar_feature_columns", [])))
        in_channels += len(dict(data_cfg.get("constant_scalar_features", {})))
    return in_channels


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} is not a mapping")
    return data


def _extract_epoch_number(path: Path) -> int:
    match = re.search(r"epoch_(\d+)_", path.name)
    if match:
        return int(match.group(1))
    return -1


def resolve_checkpoint_path(configured_checkpoint: str) -> Path:
    candidate = Path(configured_checkpoint)
    if candidate.exists():
        return candidate

    search_dir = candidate if candidate.is_dir() else candidate.parent
    if not search_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {configured_checkpoint}")

    best = search_dir / "best_cgan.pt"
    if best.exists():
        return best

    epoch_candidates = [path for path in search_dir.glob("epoch_*_cgan.pt") if path.is_file()]
    if epoch_candidates:
        epoch_candidates.sort(key=lambda path: (_extract_epoch_number(path), path.name))
        return epoch_candidates[-1]

    raise FileNotFoundError(f"Checkpoint not found: {configured_checkpoint}")


def expand_dim_repeat(t: torch.Tensor, target_size: int, dim: int) -> torch.Tensor:
    src_size = t.shape[dim]
    if src_size == target_size:
        return t.clone()
    if src_size > target_size:
        idx = [slice(None)] * t.ndim
        idx[dim] = slice(0, target_size)
        return t[tuple(idx)].clone()

    chunks = [t]
    remain = target_size - src_size
    start = 0
    while remain > 0:
        take = min(remain, src_size)
        idx = [slice(None)] * t.ndim
        idx[dim] = slice(start, start + take)
        chunks.append(t[tuple(idx)].clone())
        remain -= take
        start = (start + take) % src_size
    return torch.cat(chunks, dim=dim)


def adapt_tensor(src_tensor: torch.Tensor, dst_shape: torch.Size) -> torch.Tensor:
    out = src_tensor.clone()
    if out.ndim == 1 and out.shape[0] != dst_shape[0]:
        out = expand_dim_repeat(out, dst_shape[0], 0)
    elif out.ndim == 4:
        if out.shape[0] != dst_shape[0]:
            out = expand_dim_repeat(out, dst_shape[0], 0)
        if out.shape[1] != dst_shape[1]:
            out = expand_dim_repeat(out, dst_shape[1], 1)
    elif out.ndim == 2:
        if out.shape[0] != dst_shape[0]:
            out = expand_dim_repeat(out, dst_shape[0], 0)
        if out.shape[1] != dst_shape[1]:
            out = expand_dim_repeat(out, dst_shape[1], 1)

    if tuple(out.shape) != tuple(dst_shape):
        raise RuntimeError(f"Could not adapt tensor from {tuple(src_tensor.shape)} to {tuple(dst_shape)}")
    return out


def build_model(cfg: Dict[str, Any], model_kind: str) -> torch.nn.Module:
    kind = str(model_kind).lower()
    if kind == "auto":
        kind = "tail_refiner" if "tail_refiner" in cfg else "stage1"
    if kind == "stage1":
        arch = str(cfg.get("model", {}).get("arch", "pmnet")).lower()
        if arch == "city_routed_nlos_moe":
            return CityTypeRoutedNLoSMoERegressor(
                in_channels=compute_input_channels(cfg),
                out_channels=int(cfg["model"].get("out_channels", 3)),
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
            in_channels=compute_input_channels(cfg),
            out_channels=int(cfg["model"]["out_channels"]),
            base_channels=int(cfg["model"].get("base_channels", 64)),
            encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
            context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
            norm_type=str(cfg["model"].get("norm_type", "group")),
            dropout=float(cfg["model"].get("dropout", 0.0)),
            gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        )
    if kind == "tail_refiner":
        tail_cfg = dict(cfg.get("tail_refiner", {}))
        refiner_arch = str(tail_cfg.get("refiner_arch", "unet")).lower()
        in_channels = compute_input_channels(cfg) + 1
        out_channels = int(cfg["model"].get("out_channels", 1))
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
                base_channels=int(tail_cfg.get("refiner_base_channels", 64)),
                norm_type=str(cfg["model"].get("norm_type", "group")),
                dropout=float(cfg["model"].get("dropout", 0.0)),
                gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
                attention_heads=int(tail_cfg.get("attention_heads", 4)),
                attention_pool_size=int(tail_cfg.get("attention_pool_size", 8)),
            )
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
        raise ValueError(f"Unsupported tail refiner arch '{refiner_arch}'")
    raise ValueError(f"Unsupported model kind '{model_kind}'")


def main() -> None:
    p = argparse.ArgumentParser(description="Adapt a PMNet checkpoint to a different base_channels model")
    p.add_argument("--src-config", default="", help="Optional source config used to validate the checkpoint architecture.")
    p.add_argument("--dst-config", required=True)
    p.add_argument("--src-ckpt", required=True)
    p.add_argument("--out-ckpt", required=True)
    p.add_argument("--model-kind", choices=["auto", "stage1", "tail_refiner"], default="auto")
    args = p.parse_args()

    src_cfg = load_yaml(Path(args.src_config).resolve()) if args.src_config else None
    dst_cfg = load_yaml(Path(args.dst_config).resolve())

    dst_model = build_model(dst_cfg, args.model_kind)

    resolved_src_ckpt = resolve_checkpoint_path(args.src_ckpt)
    ckpt = torch.load(resolved_src_ckpt, map_location="cpu")
    src_state = ckpt.get("model", ckpt.get("generator"))
    if src_state is None:
        raise RuntimeError("Checkpoint has no model/generator state")

    if src_cfg is not None:
        src_model = build_model(src_cfg, args.model_kind)
        src_model.load_state_dict(src_state, strict=True)
        src_sd = src_model.state_dict()
    else:
        src_sd = src_state
    dst_sd = dst_model.state_dict()

    new_sd: Dict[str, torch.Tensor] = {}
    adapted = 0
    copied = 0

    for key, dst_tensor in dst_sd.items():
        if key not in src_sd:
            new_sd[key] = dst_tensor
            continue
        src_tensor = src_sd[key]
        if tuple(src_tensor.shape) == tuple(dst_tensor.shape):
            new_sd[key] = src_tensor.clone()
            copied += 1
        else:
            new_sd[key] = adapt_tensor(src_tensor, dst_tensor.shape)
            adapted += 1

    dst_model.load_state_dict(new_sd, strict=True)

    new_ckpt: Dict[str, Any] = {
        "epoch": int(ckpt.get("epoch", 0)),
        "best_epoch": int(ckpt.get("best_epoch", ckpt.get("epoch", 0))),
        "best_score": float(ckpt.get("best_score", float("inf"))),
        "model": dst_model.state_dict(),
        "generator": dst_model.state_dict(),
        "config_path": str(args.dst_config),
        "widened_from": str(resolved_src_ckpt),
        "widening_note": f"copied={copied}, adapted={adapted}",
    }

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, out_path)

    print(f"[ok] saved widened checkpoint to {out_path}")
    print(f"[ok] copied={copied} adapted={adapted}")


if __name__ == "__main__":
    main()
