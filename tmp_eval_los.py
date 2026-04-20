"""
Eval runner: full val split (partition_filter removed), LoS-only output.
Auto-detects base_channels / hf_channels from checkpoint weights so YAML mismatches don't crash.
"""
from __future__ import annotations
import argparse, json, sys, os

sys.path.insert(0, os.getcwd())

import torch

from config_utils import anchor_data_paths_to_config_file, load_config, load_torch_checkpoint, resolve_device
from data_utils import build_dataset_splits_from_config, compute_input_channels
from train_partitioned_pathloss_expert import _build_pmnet_from_cfg, evaluate_validation


def patch_cfg_from_checkpoint(cfg: dict, state: dict) -> None:
    """Override base_channels / hf_channels in cfg using actual checkpoint weight shapes."""
    # stem.0.block.0.weight has shape [base_channels, in_ch, k, k]
    if "stem.0.block.0.weight" in state:
        actual_base = state["stem.0.block.0.weight"].shape[0]
        yaml_base = cfg.get("model", {}).get("base_channels", actual_base)
        if actual_base != yaml_base:
            print(f"  [patch] base_channels: {yaml_base} -> {actual_base} (from checkpoint)", flush=True)
            cfg["model"]["base_channels"] = actual_base

    # hf_project.0.block.0.weight has shape [hf_channels, in_ch, k, k]
    if "hf_project.0.block.0.weight" in state:
        actual_hf = state["hf_project.0.block.0.weight"].shape[0]
        yaml_hf = cfg.get("model", {}).get("hf_channels", actual_hf)
        if actual_hf != yaml_hf:
            print(f"  [patch] hf_channels: {yaml_hf} -> {actual_hf} (from checkpoint)", flush=True)
            cfg["model"]["hf_channels"] = actual_hf

    # out_channels: head.2.weight has shape [out_channels, C, 1, 1]
    if "head.2.weight" in state:
        actual_out = state["head.2.weight"].shape[0]
        yaml_out = cfg.get("model", {}).get("out_channels", actual_out)
        if actual_out != yaml_out:
            print(f"  [patch] out_channels: {yaml_out} -> {actual_out} (from checkpoint)", flush=True)
            cfg["model"]["out_channels"] = actual_out


def keep_los_only(d: dict) -> dict:
    """Keep only keys whose path contains 'los' but not 'nlos', plus non-tagged top-level scalars."""
    out = {}
    for k, v in d.items():
        kl = k.lower()
        if isinstance(v, dict):
            sub = keep_los_only(v)
            if sub:
                out[k] = sub
        elif 'los' in kl and 'nlos' not in kl:
            out[k] = v
        elif not isinstance(v, dict) and 'nlos' not in kl and 'los' not in kl:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    cfg.setdefault("runtime", {})["device"] = args.device

    # Evaluate on FULL val split — drop topology partition filter
    cfg.get("data", {}).pop("partition_filter", None)

    device = resolve_device(cfg["runtime"]["device"])

    # Load checkpoint early so we can patch cfg before building model
    state = load_torch_checkpoint(args.checkpoint, device)
    model_state = state.get("model") or state.get("generator")  # no EMA for eval
    if model_state is None:
        raise ValueError(f"Could not find model weights in checkpoint keys: {list(state.keys())}")

    patch_cfg_from_checkpoint(cfg, model_state)

    splits = build_dataset_splits_from_config(cfg)
    if args.split not in splits:
        raise ValueError(f"Split '{args.split}' not available. Got: {list(splits.keys())}")

    in_ch = int(compute_input_channels(cfg))
    model = _build_pmnet_from_cfg(cfg, in_ch).to(device)
    model.load_state_dict(model_state)
    model.eval()

    amp = bool(cfg["training"].get("amp", True)) and (getattr(device, "type", str(device)) == "cuda")
    summary = evaluate_validation(model, splits[args.split], device, cfg, amp)

    los_summary = keep_los_only(summary)
    print(json.dumps(los_summary, indent=2))


if __name__ == "__main__":
    main()
