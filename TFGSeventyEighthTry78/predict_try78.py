"""Try 78 — unified inference script.

Loads all three model families (LoS path-loss, NLoS path-loss, spreads) into
VRAM simultaneously and produces all output maps from a single forward pass
per topology class.

Routing:
    path_loss[LoS pixels]  <- LoS expert (try74/PMHNet)
    path_loss[NLoS pixels] <- NLoS expert (try76/distribution-first)
    delay_spread           <- spreads expert (try77/spike+GMM)
    angular_spread         <- spreads expert (try77/spike+GMM)

Usage example:
    python predict_try78.py \\
        --topology-map /path/to/topology_map.npy \\
        --los-mask /path/to/los_mask.npy \\
        --height 50.0 \\
        --topology-class open_sparse_lowrise \\
        --los-checkpoint los_pathloss/outputs/try78_expert_open_sparse_lowrise_los/best_model.pt \\
        --nlos-checkpoint nlos_pathloss/outputs/try78_expert_open_sparse_lowrise_nlos/best_model.pt \\
        --delay-checkpoint spreads/outputs/try78_expert_open_sparse_lowrise_delay_spread/best_model.pt \\
        --angular-checkpoint spreads/outputs/try78_expert_open_sparse_lowrise_angular_spread/best_model.pt \\
        --output-dir /path/to/outputs

All maps are written to ``--output-dir`` as .npy files:
    path_loss_combined.npy   (dB, NaN where no-data)
    delay_spread.npy         (ns, NaN where no-data)
    angular_spread.npy       (deg, NaN where no-data)
    los_mask_used.npy        (float [0,1])
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# sys.path manipulation so each family's modules are importable without install
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_LOS_DIR = str(_HERE / "los_pathloss")
_NLOS_DIR = str(_HERE / "nlos_pathloss")
_SPREADS_DIR = str(_HERE / "spreads")

# Insert at the front so these shadow any system-installed versions.
for _d in [_SPREADS_DIR, _NLOS_DIR, _LOS_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ---------------------------------------------------------------------------
# Import model classes from each family
# ---------------------------------------------------------------------------
# LoS family (los_pathloss/)
from model_pmhhnet import PMHNetResidualRegressor, PMHHNetResidualRegressor  # noqa: E402

# NLoS family (nlos_pathloss/)
# We temporarily remove los_pathloss from path so "model" resolves to nlos/model
# rather than los/model_pmhhnet. We do this via a scoped import.
if _LOS_DIR in sys.path:
    sys.path.remove(_LOS_DIR)

from model import Try76Model, Try76ModelConfig as NLosModelConfig  # noqa: E402
from data_utils import HeightEmbedding as NLosHeightEmbedding  # noqa: E402

# Now add los back and remove nlos so spreads "model" resolves correctly
sys.path.insert(0, _LOS_DIR)
if _NLOS_DIR in sys.path:
    sys.path.remove(_NLOS_DIR)

from model import Try77Model, Try77ModelConfig as SpreadModelConfig  # noqa: E402
from data_utils import HeightEmbedding as SpreadHeightEmbedding  # noqa: E402

# Restore full path
if _NLOS_DIR not in sys.path:
    sys.path.insert(0, _NLOS_DIR)


# ---------------------------------------------------------------------------
# Topology classification (matches try76/try77 TOPOLOGY_THRESHOLDS)
# ---------------------------------------------------------------------------

TOPOLOGY_THRESHOLDS = {
    "density_q1": 0.12,
    "density_q2": 0.28,
    "height_q1": 12.0,
    "height_q2": 28.0,
}

TOPOLOGY_CLASSES = (
    "open_sparse_lowrise",
    "open_sparse_vertical",
    "mixed_compact_lowrise",
    "mixed_compact_midrise",
    "dense_block_midrise",
    "dense_block_highrise",
)


def classify_topology(topo_m: np.ndarray, non_ground_threshold: float = 0.0) -> str:
    non_ground = topo_m != float(non_ground_threshold)
    density = float(np.mean(non_ground)) if non_ground.size else 0.0
    heights = topo_m[non_ground]
    mean_h = float(np.mean(heights)) if heights.size else 0.0
    d1 = TOPOLOGY_THRESHOLDS["density_q1"]
    d2 = TOPOLOGY_THRESHOLDS["density_q2"]
    h1 = TOPOLOGY_THRESHOLDS["height_q1"]
    h2 = TOPOLOGY_THRESHOLDS["height_q2"]
    if density <= d1:
        return "open_sparse_lowrise" if mean_h <= h1 else "open_sparse_vertical"
    if density >= d2:
        return "dense_block_midrise" if mean_h <= h2 else "dense_block_highrise"
    return "mixed_compact_lowrise" if mean_h <= h1 else "mixed_compact_midrise"


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_los_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a try74-style PMHNet checkpoint."""
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    # Try to infer model config from checkpoint
    model_cfg = state.get("model_cfg", {})
    arch = model_cfg.get("arch", "pmhnet")
    base_channels = int(model_cfg.get("base_channels", 60))
    hf_channels = int(model_cfg.get("hf_channels", 20))
    in_channels = int(model_cfg.get("in_channels", 9))
    encoder_blocks = model_cfg.get("encoder_blocks", [2, 2, 2, 2])
    context_dilations = model_cfg.get("context_dilations", [1, 2, 4, 8])
    norm_type = model_cfg.get("norm_type", "group")
    dropout = float(model_cfg.get("dropout", 0.0))

    if arch in ("pmhhnet", "pmhhnet_residual"):
        model = PMHHNetResidualRegressor(
            in_channels=in_channels,
            out_channels=1,
            base_channels=base_channels,
            encoder_blocks=encoder_blocks,
            context_dilations=context_dilations,
            norm_type=norm_type,
            dropout=dropout,
            hf_channels=hf_channels,
        )
    else:  # pmhnet (default for try74 LoS experts)
        model = PMHNetResidualRegressor(
            in_channels=in_channels,
            out_channels=1,
            base_channels=base_channels,
            encoder_blocks=encoder_blocks,
            context_dilations=context_dilations,
            norm_type=norm_type,
            dropout=dropout,
            hf_channels=hf_channels,
        )

    model_state = state.get("model", state.get("generator", state))
    if isinstance(model_state, dict) and any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=False)
    model.to(device).eval()
    return model


def _load_nlos_model(checkpoint_path: Path, device: torch.device) -> Try76Model:
    """Load a try76-style NLoS distribution model checkpoint."""
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model_cfg_dict = state.get("model_cfg", {})
    model_cfg = NLosModelConfig(
        in_channels=int(model_cfg_dict.get("in_channels", 4)),
        cond_dim=int(model_cfg_dict.get("cond_dim", 64)),
        height_embed_dim=int(model_cfg_dict.get("height_embed_dim", 32)),
        base_width=int(model_cfg_dict.get("base_width", 48)),
        K=int(model_cfg_dict.get("K", 5)),
        clamp_lo=float(model_cfg_dict.get("clamp_lo", 30.0)),
        clamp_hi=float(model_cfg_dict.get("clamp_hi", 178.0)),
        outlier_sigma_floor=float(model_cfg_dict.get("outlier_sigma_floor", 15.0)),
    )
    model = Try76Model(model_cfg)
    model_state = state.get("model", state)
    if isinstance(model_state, dict) and any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=False)
    model.to(device).eval()
    return model


def _load_spread_model(checkpoint_path: Path, device: torch.device) -> Try77Model:
    """Load a try77-style spread distribution model checkpoint."""
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model_cfg_dict = state.get("model_cfg", {})
    model_cfg = SpreadModelConfig(
        in_channels=int(model_cfg_dict.get("in_channels", 4)),
        cond_dim=int(model_cfg_dict.get("cond_dim", 64)),
        height_embed_dim=int(model_cfg_dict.get("height_embed_dim", 32)),
        base_width=int(model_cfg_dict.get("base_width", 48)),
        K=int(model_cfg_dict.get("K", 5)),
        clamp_lo=float(model_cfg_dict.get("clamp_lo", 0.0)),
        clamp_hi=float(model_cfg_dict.get("clamp_hi", 400.0)),
        sigma_min=float(model_cfg_dict.get("sigma_min", 1.0)),
        sigma_max=float(model_cfg_dict.get("sigma_max", 120.0)),
        spike_mu_max=float(model_cfg_dict.get("spike_mu_max", 10.0)),
        spike_sigma_min=float(model_cfg_dict.get("spike_sigma_min", 0.3)),
        spike_sigma_max=float(model_cfg_dict.get("spike_sigma_max", 5.0)),
    )
    model = Try77Model(model_cfg)
    model_state = state.get("model", state)
    if isinstance(model_state, dict) and any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=False)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Input preparation helpers
# ---------------------------------------------------------------------------

def _load_map(path_or_scalar, dtype=np.float32) -> np.ndarray:
    """Load a .npy file or parse a scalar string."""
    p = str(path_or_scalar)
    if p.endswith(".npy"):
        return np.load(p).astype(dtype)
    # Try treating as a scalar
    return np.array(float(p), dtype=dtype)


def _prepare_los_inputs(
    topology_map: np.ndarray,
    los_mask: np.ndarray,
    height: float,
    image_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build the 9-channel input tensor expected by the try74 PMHNet LoS expert.

    Channel layout (matches train_partitioned_pathloss_expert.py / data_utils.py):
      0: topology_map / 255.0     (normalised height map)
      1: los_mask                 (binary LoS indicator)
      2: distance_map             (Euclidean distance from TX, normalised)
      3: tx_depth_map             (set to 0 — not available at inference)
      4: elevation_angle_map      (set to 0 — not available at inference)
      5: building_mask            (topology_map != 0, binary)
      6-8: obstruction features   (set to 0 — not available at inference)

    NOTE: If your checkpoint was trained with fewer or different channels,
    adjust the channel count to match. The LoS mask routing in the inference
    loop will still work regardless of model internals.
    """
    H, W = topology_map.shape[-2], topology_map.shape[-1]

    topo_norm = (topology_map / 255.0).astype(np.float32)
    los = los_mask.astype(np.float32)
    ground = (topology_map == 0.0).astype(np.float32)

    # Distance map: Euclidean distance from image centre, normalised to [0, 1]
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = float(np.sqrt(cy ** 2 + cx ** 2)) or 1.0
    dist_map = dist_map / max_dist

    building_mask = (topology_map != 0.0).astype(np.float32)

    # Pack to (1, 9, H, W) — channels 3-8 are zeros (not available offline)
    channels = np.stack([
        topo_norm,
        los,
        dist_map,
        np.zeros_like(topo_norm),  # tx_depth
        np.zeros_like(topo_norm),  # elevation_angle
        building_mask,
        np.zeros_like(topo_norm),  # obstruction ch1
        np.zeros_like(topo_norm),  # obstruction ch2
        np.zeros_like(topo_norm),  # obstruction ch3
    ], axis=0)  # (9, H, W)

    inp = torch.from_numpy(channels).unsqueeze(0).to(device)  # (1, 9, H, W)

    # try74 PMHNet is height-unaware (no scalar_cond) for the pmhnet arch
    return inp, None


def _prepare_dist_inputs(
    topology_map: np.ndarray,
    los_mask: np.ndarray,
    height: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the 4-channel input tensor + height embedding for try76/try77 models.

    Channel layout (matches try76/try77 data_utils.py):
      0: topology_input = topology_map * ground / 90.0
      1: los = los_mask * ground
      2: nlos = (1 - los_mask) * ground
      3: ground_mask

    Returns (inputs, height_embed) tensors both on ``device``.
    """
    TOPOLOGY_NORM_M = 90.0

    ground = (topology_map == 0.0).astype(np.float32)
    topology_input = topology_map * ground / max(TOPOLOGY_NORM_M, 1e-3)
    los = los_mask * ground
    nlos = (1.0 - los_mask) * ground

    channels = np.stack([topology_input, los, nlos, ground], axis=0)  # (4, H, W)
    inp = torch.from_numpy(channels).unsqueeze(0).to(device)  # (1, 4, H, W)

    h_tensor = torch.tensor([height], dtype=torch.float32).to(device)
    height_embed = NLosHeightEmbedding()(h_tensor)  # (1, 32)

    return inp, height_embed


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------

def run_inference(
    topology_map: np.ndarray,
    los_mask: np.ndarray,
    height: float,
    los_model: torch.nn.Module,
    nlos_model: Try76Model,
    delay_model: Try77Model,
    angular_model: Try77Model,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run all four models on one sample and combine outputs.

    Returns a dict with keys:
        path_loss_combined  — (H, W) float32, NaN where invalid
        path_loss_los       — (H, W) float32
        path_loss_nlos      — (H, W) float32
        delay_spread        — (H, W) float32, NaN where invalid
        angular_spread      — (H, W) float32, NaN where invalid
        los_mask_used       — (H, W) float32
    """
    H, W = topology_map.shape

    # --- LoS path-loss ---
    los_inp, los_scalar = _prepare_los_inputs(topology_map, los_mask, height, H, device)
    with torch.no_grad():
        if los_scalar is not None:
            los_pred = los_model(los_inp, los_scalar)
        else:
            los_pred = los_model(los_inp)
    los_pred_np = los_pred.squeeze().cpu().numpy()  # (H, W)

    # --- NLoS path-loss ---
    nlos_inp, nlos_h_emb = _prepare_dist_inputs(topology_map, los_mask, height, device)
    with torch.no_grad():
        nlos_out = nlos_model(nlos_inp, nlos_h_emb)
    nlos_pred_np = nlos_out["pred"].squeeze().cpu().numpy()  # (H, W)

    # --- Delay spread ---
    # Reuse the same input preparation; height embedding is identical
    delay_inp, delay_h_emb = _prepare_dist_inputs(topology_map, los_mask, height, device)
    with torch.no_grad():
        delay_out = delay_model(delay_inp, delay_h_emb)
    delay_pred_np = delay_out["pred"].squeeze().cpu().numpy()  # (H, W)

    # --- Angular spread ---
    angular_inp, angular_h_emb = _prepare_dist_inputs(topology_map, los_mask, height, device)
    with torch.no_grad():
        angular_out = angular_model(angular_inp, angular_h_emb)
    angular_pred_np = angular_out["pred"].squeeze().cpu().numpy()  # (H, W)

    # --- Combine path-loss using LoS mask ---
    ground_mask = (topology_map == 0.0)
    los_binary = (los_mask > 0.5) & ground_mask
    nlos_binary = (los_mask <= 0.5) & ground_mask

    path_loss_combined = np.full((H, W), np.nan, dtype=np.float32)
    path_loss_combined[los_binary] = los_pred_np[los_binary]
    path_loss_combined[nlos_binary] = nlos_pred_np[nlos_binary]

    # Mark non-ground as NaN for spread maps
    delay_out_map = np.where(ground_mask, delay_pred_np, np.nan).astype(np.float32)
    angular_out_map = np.where(ground_mask, angular_pred_np, np.nan).astype(np.float32)

    return {
        "path_loss_combined": path_loss_combined,
        "path_loss_los": los_pred_np.astype(np.float32),
        "path_loss_nlos": nlos_pred_np.astype(np.float32),
        "delay_spread": delay_out_map,
        "angular_spread": angular_out_map,
        "los_mask_used": los_mask.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Try 78 unified inference: loads LoS (try74), NLoS (try76), and spread (try77) "
            "experts and produces combined output maps."
        )
    )
    parser.add_argument("--topology-map", type=str, required=True,
                        help="Path to topology_map .npy (H x W float32, heights in metres, ground=0).")
    parser.add_argument("--los-mask", type=str, required=True,
                        help="Path to los_mask .npy (H x W float32, 1=LoS, 0=NLoS).")
    parser.add_argument("--height", type=float, required=True,
                        help="Antenna height in metres (scalar).")
    parser.add_argument("--topology-class", type=str, default=None,
                        choices=list(TOPOLOGY_CLASSES),
                        help="Topology class of the sample (auto-detected if omitted).")
    # Checkpoint paths
    parser.add_argument("--los-checkpoint", type=str, required=True,
                        help="Path to LoS path-loss expert checkpoint (.pt).")
    parser.add_argument("--nlos-checkpoint", type=str, required=True,
                        help="Path to NLoS path-loss expert checkpoint (.pt).")
    parser.add_argument("--delay-checkpoint", type=str, required=True,
                        help="Path to delay-spread expert checkpoint (.pt).")
    parser.add_argument("--angular-checkpoint", type=str, required=True,
                        help="Path to angular-spread expert checkpoint (.pt).")
    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where output .npy files will be written.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device (auto, cuda, cpu). Default: auto.")
    args = parser.parse_args()

    # Device
    device_str = args.device.lower()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[try78] device={device}")

    # Load inputs
    topology_map = _load_map(args.topology_map)
    los_mask = _load_map(args.los_mask)
    height = float(args.height)

    if topology_map.ndim == 3:
        topology_map = topology_map.squeeze(0)
    if los_mask.ndim == 3:
        los_mask = los_mask.squeeze(0)

    # Auto-detect topology class
    topo_class = args.topology_class
    if topo_class is None:
        topo_class = classify_topology(topology_map)
        print(f"[try78] auto-detected topology_class={topo_class}")
    else:
        print(f"[try78] topology_class={topo_class} (user-specified)")

    # Load all four models
    print("[try78] loading LoS model ...")
    los_model = _load_los_model(Path(args.los_checkpoint), device)
    print("[try78] loading NLoS model ...")
    nlos_model = _load_nlos_model(Path(args.nlos_checkpoint), device)
    print("[try78] loading delay-spread model ...")
    delay_model = _load_spread_model(Path(args.delay_checkpoint), device)
    print("[try78] loading angular-spread model ...")
    angular_model = _load_spread_model(Path(args.angular_checkpoint), device)

    print("[try78] running inference ...")
    outputs = run_inference(
        topology_map=topology_map,
        los_mask=los_mask,
        height=height,
        los_model=los_model,
        nlos_model=nlos_model,
        delay_model=delay_model,
        angular_model=angular_model,
        device=device,
    )

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in outputs.items():
        out_path = out_dir / f"{name}.npy"
        np.save(str(out_path), arr)
        print(f"[try78] wrote {out_path}  shape={arr.shape}  dtype={arr.dtype}")

    print("[try78] done.")


if __name__ == "__main__":
    main()
