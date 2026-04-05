from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config_utils import anchor_data_paths_to_config_file, load_config
from data_utils import (
    _build_prior_confidence_channel,
    build_dataset_splits_from_config,
    compute_input_channels,
)
from model_pmnet import PMNetResidualRegressor


def _formula_channel_index(cfg: dict) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):
        idx += 1
    if cfg["data"].get("distance_map_channel", False):
        idx += 1
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Try50 local smoke test for prior and confidence channel")
    parser.add_argument("--config", required=True, help="Path to Try50 YAML config")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to sample")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index in selected split")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    anchor_data_paths_to_config_file(cfg, str(cfg_path))

    expected_in = compute_input_channels(cfg)
    print(f"[INFO] expected input channels from config: {expected_in}")

    x = y = mask = None
    cfg_h5 = str(cfg.get("data", {}).get("hdf5_path", "")).strip()
    h5_exists = bool(cfg_h5) and Path(cfg_h5).exists()
    if h5_exists:
        splits = build_dataset_splits_from_config(cfg)
        if args.split not in splits:
            raise ValueError(f"Split '{args.split}' not available in this config")
        ds = splits[args.split]

        sample = ds[args.sample_index]
        if len(sample) == 4:
            x, y, mask, scalar_cond = sample
            print(f"[INFO] scalar_cond shape: {tuple(scalar_cond.shape)}")
        else:
            x, y, mask = sample
        print("[INFO] dataset-backed smoke path used")
    else:
        print("[WARN] HDF5 dataset not found locally, running synthetic smoke path")
        h = int(cfg["data"].get("image_size", 513))
        w = h
        x = torch.rand(expected_in, h, w)
        y = torch.rand(1, h, w)
        mask = torch.ones(1, h, w)

    print(f"[INFO] x shape: {tuple(x.shape)}")
    print(f"[INFO] y shape: {tuple(y.shape)}")
    print(f"[INFO] mask shape: {tuple(mask.shape)}")

    assert x.shape[0] == expected_in, f"Channel mismatch: tensor has {x.shape[0]} but expected {expected_in}"

    formula_enabled = bool(cfg["data"].get("path_loss_formula_input", {}).get("enabled", False))
    conf_enabled = bool(cfg["data"].get("path_loss_formula_input", {}).get("include_confidence_channel", False))
    if formula_enabled:
        formula_idx = _formula_channel_index(cfg)
        formula_map = x[formula_idx]
        print(
            "[INFO] formula map stats min={:.5f} max={:.5f} mean={:.5f}".format(
                float(formula_map.min()), float(formula_map.max()), float(formula_map.mean())
            )
        )
        if conf_enabled:
            conf_map = x[formula_idx + 1]
            print(
                "[INFO] confidence map stats min={:.5f} max={:.5f} mean={:.5f}".format(
                    float(conf_map.min()), float(conf_map.max()), float(conf_map.mean())
                )
            )

            # Independent confidence map sanity from synthetic primitives.
            los_prob = torch.rand(1, x.shape[-2], x.shape[-1])
            distance_norm = torch.rand(1, x.shape[-2], x.shape[-1])
            topology = torch.rand(1, x.shape[-2], x.shape[-1])
            confidence_check = _build_prior_confidence_channel(
                los_tensor=los_prob,
                distance_tensor=distance_norm,
                topology_tensor=topology,
                non_ground_threshold=float(cfg["data"].get("non_ground_threshold", 0.0)),
                kernel_size=int(cfg["data"]["path_loss_formula_input"].get("confidence_kernel_size", 31)),
            )
            assert confidence_check.shape == los_prob.shape, "Confidence channel shape mismatch"
            assert float(confidence_check.min()) >= 0.0 and float(confidence_check.max()) <= 1.0, (
                "Confidence channel out of expected [0, 1] range"
            )
            print("[INFO] synthetic confidence-map builder sanity passed")

    model = PMNetResidualRegressor(
        in_channels=expected_in,
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"].get("base_channels", 64)),
        encoder_blocks=tuple(cfg["model"].get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=tuple(cfg["model"].get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(cfg["model"].get("norm_type", "group")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
    ).eval()

    with torch.no_grad():
        pred = model(x.unsqueeze(0))

    print(f"[INFO] forward output shape: {tuple(pred.shape)}")
    if not torch.isfinite(pred).all():
        raise RuntimeError("Non-finite values detected in PMNet forward output")

    print("[PASS] Try50 prior smoke test completed successfully.")


if __name__ == "__main__":
    main()
