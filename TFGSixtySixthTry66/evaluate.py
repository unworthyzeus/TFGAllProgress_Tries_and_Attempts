from __future__ import annotations

import argparse
import json

import torch

from config_utils import anchor_data_paths_to_config_file, load_config, load_torch_checkpoint, resolve_device
from data_utils import build_dataset_splits_from_config, compute_input_channels
from train_partitioned_pathloss_expert import _build_pmnet_from_cfg, evaluate_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate path-loss model (PMNet / PMHNet / PMHHNet) using the same validation path as training."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Load generator_ema from checkpoint if present (training validates with EMA when enabled).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    device = resolve_device(cfg["runtime"]["device"])
    splits = build_dataset_splits_from_config(cfg)
    if args.split not in splits:
        raise ValueError(f"Split '{args.split}' is not available.")

    in_ch = int(compute_input_channels(cfg))
    model = _build_pmnet_from_cfg(cfg, in_ch).to(device)

    state = load_torch_checkpoint(args.checkpoint, device)
    if args.use_ema and "generator_ema" in state:
        model.load_state_dict(state["generator_ema"])
    else:
        model.load_state_dict(state["model"] if "model" in state else state["generator"])
    model.eval()

    summary = evaluate_validation(
        model,
        splits[args.split],
        device,
        cfg,
        bool(cfg["training"].get("amp", True)) and (getattr(device, "type", str(device)) == "cuda"),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
