#!/usr/bin/env python3
"""Widen PMNet checkpoint weights from one base_channels to another.

This script converts a trained 72-channel PMNet checkpoint to a 96-channel checkpoint
so training can continue with a wider model while preserving learned structure.

Example:
python TFGFortyEighthTry48/scripts/widen_pmnet_checkpoint.py \
  --src-config TFGFortyEighthTry48/experiments/fortyeighthtry48_pmnet_prior_gan_fastbatch/fortyeighthtry48_pmnet_prior_stage1_warmstart_try42.yaml \
  --dst-config TFGFortyEighthTry48/experiments/fortyeighthtry48_pmnet_prior_gan_fastbatch/fortyeighthtry48_pmnet_prior_stage1_warmstart_try42_widen96_cont.yaml \
  --src-ckpt /scratch/nas/3/gmoreno/TFGpractice/TFGFortyEighthTry48/outputs/fortyeighthtry48_pmnet_prior_stage1_warmstart_try42_t48_stage1_warm_t42_4gpu/best_cgan.pt \
  --out-ckpt /scratch/nas/3/gmoreno/TFGpractice/TFGFortyEighthTry48/outputs/fortyeighthtry48_pmnet_prior_stage1_warmstart_try42_t48_stage1_warm_t42_4gpu/widened_72_to_96_epoch22_seed.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from model_pmnet import PMNetResidualRegressor  # noqa: E402


def compute_input_channels(cfg: Dict[str, Any]) -> int:
    in_channels = 1
    data_cfg = dict(cfg.get("data", {}))
    if data_cfg.get("los_input_column"):
        in_channels += 1
    if bool(data_cfg.get("distance_map_channel", False)):
        in_channels += 1
    if bool(data_cfg.get("path_loss_formula_input", {}).get("enabled", False)):
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


def build_model(cfg: Dict[str, Any]) -> PMNetResidualRegressor:
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


def main() -> None:
    p = argparse.ArgumentParser(description="Widen PMNet checkpoint to a larger base_channels model")
    p.add_argument("--src-config", required=True)
    p.add_argument("--dst-config", required=True)
    p.add_argument("--src-ckpt", required=True)
    p.add_argument("--out-ckpt", required=True)
    args = p.parse_args()

    src_cfg = load_yaml(Path(args.src_config).resolve())
    dst_cfg = load_yaml(Path(args.dst_config).resolve())

    src_model = build_model(src_cfg)
    dst_model = build_model(dst_cfg)

    ckpt = torch.load(args.src_ckpt, map_location="cpu")
    src_state = ckpt.get("model", ckpt.get("generator"))
    if src_state is None:
        raise RuntimeError("Checkpoint has no model/generator state")

    src_model.load_state_dict(src_state, strict=True)

    src_sd = src_model.state_dict()
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
        "widened_from": str(args.src_ckpt),
        "widening_note": f"copied={copied}, adapted={adapted}",
    }

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, out_path)

    print(f"[ok] saved widened checkpoint to {out_path}")
    print(f"[ok] copied={copied} adapted={adapted}")


if __name__ == "__main__":
    main()
