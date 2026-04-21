#!/usr/bin/env python3
"""De-widen a PMHHNet checkpoint by truncating channels to a smaller base_channels.

Slices the first N channels of each parameter (warm-start pruning). Input
checkpoint is never modified. Optimizer / scheduler / scaler state is dropped
because shapes no longer match — they re-initialize on resume. Fine-tuning is
expected after de-widening.

Example
-------
Make a 36-channel copy of the Try 75 LoS checkpoint, targeted at the local
DirectML config:

    python scripts/dewiden_base_channels.py \\
        --in-ckpt  C:/TFG/Final_Models_TFG/los/all_experts_los_try75.pt \\
        --out-ckpt C:/TFG/Final_Models_TFG/los/all_experts_los_try75_base36.pt \\
        --config   experiments/seventyfifth_try75_experts/try75_expert_allcity_los_local_directml.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

TRY_DIR = Path(__file__).resolve().parent.parent
if str(TRY_DIR) not in sys.path:
    sys.path.insert(0, str(TRY_DIR))

import config_utils  # noqa: E402
from data_utils import compute_scalar_cond_dim, return_scalar_cond_from_config  # noqa: E402
from model_pmhhnet import (  # noqa: E402
    PMHHNetResidualRegressor,
    PMHNetResidualRegressor,
    PMNetResidualRegressor,
)


STATE_DICT_KEYS_DEFAULT = ("model", "generator", "generator_ema")
OPT_KEYS_TO_DROP = (
    "optimizer_g",
    "optimizer_d",
    "scheduler_g",
    "scheduler_d",
    "scaler_g",
    "scaler_d",
)


def build_model_from_cfg(cfg: Dict, in_channels: int) -> torch.nn.Module:
    arch = str(cfg.get("model", {}).get("arch", "pmnet")).lower()
    scalar_dim = int(compute_scalar_cond_dim(cfg)) if return_scalar_cond_from_config(cfg) else 0
    base_ch = int(cfg["model"].get("base_channels", 64))
    common = dict(
        in_channels=in_channels,
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=base_ch,
        encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
    )
    if arch == "pmhnet":
        return PMHNetResidualRegressor(
            **common,
            hf_channels=int(cfg["model"].get("hf_channels", max(8, base_ch // 2))),
        )
    if arch == "pmhhnet":
        return PMHHNetResidualRegressor(
            **common,
            hf_channels=int(cfg["model"].get("hf_channels", max(8, base_ch // 2))),
            scalar_dim=max(1, scalar_dim),
            scalar_hidden_dim=int(cfg["model"].get("scalar_hidden_dim", max(32, base_ch * 2))),
        )
    return PMNetResidualRegressor(**common)


def truncate_state_dict(
    old_sd: Dict[str, torch.Tensor],
    target_sd: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[Tuple[str, tuple, tuple]], List[Tuple[str, tuple, tuple]]]:
    """Return a state_dict matching target_sd shapes, filled from old_sd via first-N slicing per dim."""
    out: Dict[str, torch.Tensor] = {}
    missing_in_old: List[str] = []
    shape_mismatch: List[Tuple[str, tuple, tuple]] = []
    truncated: List[Tuple[str, tuple, tuple]] = []

    for key, new_t in target_sd.items():
        if key not in old_sd:
            missing_in_old.append(key)
            out[key] = new_t.clone()
            continue
        old_t = old_sd[key]
        if old_t.shape == new_t.shape:
            out[key] = old_t.clone()
            continue
        if old_t.ndim != new_t.ndim:
            shape_mismatch.append((key, tuple(old_t.shape), tuple(new_t.shape)))
            out[key] = new_t.clone()
            continue
        # Any dim where new > old would mean we're widening instead of de-widening for that tensor.
        if any(n > o for o, n in zip(old_t.shape, new_t.shape)):
            shape_mismatch.append((key, tuple(old_t.shape), tuple(new_t.shape)))
            out[key] = new_t.clone()
            continue
        slices = tuple(slice(0, n) for n in new_t.shape)
        sliced = old_t[slices].contiguous().clone()
        out[key] = sliced
        truncated.append((key, tuple(old_t.shape), tuple(new_t.shape)))

    return out, missing_in_old, shape_mismatch, truncated


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-ckpt", required=True, help="Source checkpoint (left untouched)")
    p.add_argument("--out-ckpt", required=True, help="Destination checkpoint path")
    p.add_argument("--config", required=True, help="Config YAML with TARGET model settings (smaller base_channels)")
    p.add_argument(
        "--state-keys",
        default=",".join(STATE_DICT_KEYS_DEFAULT),
        help=f"Comma-separated state_dict keys to de-widen (default: {','.join(STATE_DICT_KEYS_DEFAULT)})",
    )
    p.add_argument("--force", action="store_true", help="Overwrite --out-ckpt if it exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_ckpt).resolve()
    out_path = Path(args.out_ckpt).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"--in-ckpt not found: {in_path}")
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists (use --force to overwrite)")
    if out_path.resolve() == in_path.resolve():
        raise ValueError("Refusing to overwrite the input checkpoint")

    cfg = config_utils.load_config(str(args.config))
    in_channels = int(cfg["model"].get("in_channels", 10))
    target_base = int(cfg["model"].get("base_channels", 0))

    print(f"[dewiden] config: {args.config}")
    print(f"[dewiden] target arch={cfg['model'].get('arch')} base_channels={target_base} in_channels={in_channels}")
    print(f"[dewiden] loading source: {in_path}")
    old_ckpt = torch.load(str(in_path), map_location="cpu", weights_only=False)

    print("[dewiden] building fresh target model to read reference shapes")
    target_model = build_model_from_cfg(cfg, in_channels=in_channels)
    target_sd = target_model.state_dict()
    target_param_count = sum(t.numel() for t in target_sd.values())
    print(f"[dewiden] target model parameter count: {target_param_count:,}")

    keys_to_truncate = [k.strip() for k in str(args.state_keys).split(",") if k.strip()]
    new_ckpt: Dict = {}

    for key, val in old_ckpt.items():
        if key in OPT_KEYS_TO_DROP:
            print(f"[dewiden] DROP  {key:18s} (shape mismatch after de-widening)")
            continue
        if key in keys_to_truncate and isinstance(val, dict):
            print(f"[dewiden] TRUNC {key:18s} ({len(val)} tensors)")
            new_sd, missing, mismatch, truncated = truncate_state_dict(val, target_sd)
            new_ckpt[key] = new_sd
            print(f"           kept-as-is:  {len(val) - len(missing) - len(mismatch) - len(truncated)}")
            print(f"           truncated :  {len(truncated)}")
            print(f"           missing   :  {len(missing)} (filled with fresh init)")
            print(f"           mismatch  :  {len(mismatch)} (filled with fresh init)")
            for k, old_sh, new_sh in mismatch[:5]:
                print(f"             ! {k}: old={old_sh} new={new_sh}")
        else:
            new_ckpt[key] = val

    new_ckpt["dewiden_source"] = str(in_path)
    new_ckpt["dewiden_base_channels"] = target_base
    new_ckpt["dewiden_config"] = str(Path(args.config).resolve())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dewiden] saving: {out_path}")
    torch.save(new_ckpt, str(out_path))
    print("[dewiden] done")


if __name__ == "__main__":
    main()
