from __future__ import annotations

import argparse
import json

import torch

from config_utils import anchor_data_paths_to_config_file, load_config, load_torch_checkpoint, resolve_device
from data_utils import build_dataset_splits_from_config, compute_input_channels
from model_pmhhnet import PMNetResidualRegressor
from train_pmnet_residual import evaluate_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Try 42 PMNet residual model")
    parser.add_argument("--config", type=str, default="experiments/fortysecondtry42_pmnet_prior_residual/fortysecondtry42_pmnet_prior_residual.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    device = resolve_device(cfg["runtime"]["device"])
    splits = build_dataset_splits_from_config(cfg)
    if args.split not in splits:
        raise ValueError(f"Split '{args.split}' is not available.")

    model = PMNetResidualRegressor(
        in_channels=compute_input_channels(cfg),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        gradient_checkpointing=False,
    ).to(device)

    state = load_torch_checkpoint(args.checkpoint, device)
    model.load_state_dict(state["model"] if "model" in state else state["generator"])
    model.eval()

    summary = evaluate_validation(
        model,
        splits[args.split],
        device,
        cfg,
        bool(cfg["training"].get("amp", True)) and (getattr(device, "type", device) == "cuda"),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
