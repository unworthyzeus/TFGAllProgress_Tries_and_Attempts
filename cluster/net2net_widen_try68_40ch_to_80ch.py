#!/usr/bin/env python3
"""Net2Net function-preserving widening: Try 68 experts 40ch → 80ch.

For each of the 6 experts:
  1. Download best_model.pt from cluster (if not already local).
  2. Load 40ch state dict (generator + generator_ema).
  3. Build an 80ch PMHHNetResidualRegressor using the local YAML.
  4. Map every parameter via Net2Net tiling rules (output dim: tile; input dim: tile + ×0.5).
  5. Save the widened checkpoint locally to cluster_outputs/.../best_model.pt.
  6. Upload to cluster.

Usage (from TFGpractice/):
    set SSH_PASSWORD=Slenderman,2004
    python cluster/net2net_widen_try68_40ch_to_80ch.py [--dry-run] [--expert EXPERT_ID]

After this runs, the cluster best_model.pt files are widened 80ch checkpoints.
Then resubmit the 80ch jobs with resume_checkpoint pointing to these files.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import paramiko
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko

# Make Try68 importable
T68_DIR = Path(__file__).resolve().parent.parent / "TFGSixtyEighthTry68"
sys.path.insert(0, str(T68_DIR))

import torch
import torch.nn as nn
from model_pmhhnet import PMHHNetResidualRegressor

HOST = "sert.ac.upc.edu"
USER = "gmoreno"
REMOTE_BASE = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68"
LOCAL_OUTPUTS = Path(__file__).resolve().parent.parent / "cluster_outputs" / "TFGSixtyEighthTry68"
REG_PATH = T68_DIR / "experiments" / "sixtyeighth_try68_experts" / "try68_expert_registry.yaml"

EXPERTS = [
    "open_sparse_lowrise",
    "open_sparse_vertical",
    "mixed_compact_lowrise",
    "mixed_compact_midrise",
    "dense_block_midrise",
    "dense_block_highrise",
]


# ---------------------------------------------------------------------------
# Net2Net widening
# ---------------------------------------------------------------------------

def net2net_widen_tensor(old_t: torch.Tensor, new_shape: tuple) -> torch.Tensor:
    """Widen old_t to new_shape using Net2Net function-preserving rules (exact 2x only).

    - dim 0 doubles: tile output neurons, no scaling.
    - dim >= 1 doubles for ndim >= 2: tile input channels and multiply by 0.5.
    - 1D tensors (bias, GroupNorm weight/bias): only dim 0 tile, no halving.

    Raises ValueError if any dimension change is not exactly 2x.
    """
    if old_t.shape == new_shape:
        return old_t.clone()
    assert len(old_t.shape) == len(new_shape), (
        f"ndim mismatch: {old_t.shape} vs {new_shape}"
    )
    result = old_t.clone()
    for d, (old_sz, new_sz) in enumerate(zip(old_t.shape, new_shape)):
        if old_sz == new_sz:
            continue
        if new_sz != 2 * old_sz:
            raise ValueError(
                f"dim {d}: not 2x ({old_sz}->{new_sz})"
            )
        result = torch.cat([result, result], dim=d)
        # Input dimensions (dim >= 1) in a weight tensor require halving
        if d >= 1 and result.ndim >= 2:
            result = result * 0.5
    return result


def partial_copy_tensor(old_t: torch.Tensor, new_shape: tuple) -> torch.Tensor:
    """Fallback: copy old weights into zero-padded new tensor (for non-2x ratios like 48->80).

    Preserves learned filters up to min(old, new) along each dimension.
    New channels are zero-initialised (they will be trained from scratch).
    """
    result = torch.zeros(new_shape, dtype=old_t.dtype)
    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_t.shape, new_shape))
    result[slices] = old_t[slices]
    return result


def widen_state_dict(
    old_sd: dict[str, torch.Tensor],
    new_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a new state dict compatible with new_sd, using Net2Net from old_sd.

    - Exact 2x channel change: Net2Net tiling (function-preserving).
    - Other ratio change: partial copy + zero-pad (e.g. 48->80 for old 48ch checkpoints).
    - Same shape: direct copy.
    - Key missing in old: use new model random init.
    """
    out: dict[str, torch.Tensor] = {}
    n_net2net = 0
    n_partial = 0
    n_copy = 0
    n_new = 0
    for key, new_param in new_sd.items():
        old_param = old_sd.get(key)
        if old_param is None:
            out[key] = new_param.clone()
            n_new += 1
        elif old_param.shape == new_param.shape:
            out[key] = old_param.clone()
            n_copy += 1
        else:
            try:
                out[key] = net2net_widen_tensor(old_param, tuple(new_param.shape))
                n_net2net += 1
            except ValueError:
                out[key] = partial_copy_tensor(old_param, tuple(new_param.shape))
                n_partial += 1
    print(f"  Params: {n_copy} copied, {n_net2net} net2net-widened, {n_partial} partial-copy, {n_new} new-init")
    return out


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_80ch_model(cfg: dict[str, Any], in_channels: int) -> PMHHNetResidualRegressor:
    m = cfg["model"]
    scalar_hidden_dim = int(m.get("scalar_hidden_dim", max(32, int(m["base_channels"]) * 2)))
    model = PMHHNetResidualRegressor(
        in_channels=in_channels,
        out_channels=int(m.get("out_channels", 1)),
        base_channels=int(m["base_channels"]),
        encoder_blocks=list(m.get("encoder_blocks", [2, 2, 2, 2])),
        context_dilations=list(m.get("context_dilations", [1, 2, 4, 8])),
        norm_type=str(m.get("norm_type", "group")),
        dropout=float(m.get("dropout", 0.0)),
        gradient_checkpointing=False,
        hf_channels=int(m.get("hf_channels", max(8, int(m["base_channels"]) // 2))),
        scalar_dim=1,
        scalar_hidden_dim=scalar_hidden_dim,
    )
    return model


def compute_in_channels(cfg: dict[str, Any]) -> int:
    """Replicate data_utils.compute_input_channels without importing the full module."""
    d = cfg.get("data", {})
    n = 1  # topology_map always
    if d.get("los_input_column"):
        n += 1
    if d.get("distance_map_channel", False):
        n += 1
    pf = d.get("path_loss_formula_input", {})
    if pf.get("enabled", False):
        n += 1  # formula channel
        if pf.get("include_confidence_channel", False):
            n += 1
    obs = d.get("path_loss_obstruction_features", {})
    if obs.get("enabled", False):
        if obs.get("include_shadow_depth", False):
            n += 1
        if obs.get("include_distance_since_los_break", False):
            n += 1
        if obs.get("include_max_blocker_height", False):
            n += 1
        if obs.get("include_blocker_count", False):
            n += 1
    if d.get("tx_depth_map_channel", False):
        n += 1
    if d.get("elevation_angle_map_channel", False):
        n += 1
    if d.get("building_mask_channel", False):
        n += 1
    # scalar feature channels injected as raster channels (use_scalar_channels=False path)
    if not cfg.get("model", {}).get("use_scalar_channels", True):
        n += len(list(d.get("scalar_feature_columns", [])))
        n += len(dict(d.get("constant_scalar_features", {})))
    return n


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def remote_exec(c: paramiko.SSHClient, cmd: str, *, check: bool = False) -> tuple[str, str]:
    print("REMOTE>", cmd)
    _, o, e = c.exec_command(cmd)
    out = o.read().decode().strip()
    err = e.read().decode().strip()
    if out:
        print("  ", out)
    if err:
        print("  STDERR:", err)
    return out, err


def download_if_needed(sftp: paramiko.SFTPClient, expert_id: str) -> Path:
    local_dir = LOCAL_OUTPUTS / f"try68_expert_{expert_id}"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_pt = local_dir / "best_model.pt"
    if local_pt.exists():
        print(f"  [{expert_id}] best_model.pt already local ({local_pt.stat().st_size // 1_000_000} MB)")
        return local_pt
    remote_pt = f"{REMOTE_BASE}/outputs/try68_expert_{expert_id}/best_model.pt"
    print(f"  [{expert_id}] Downloading {remote_pt} -> {local_pt} ...")
    sftp.get(remote_pt, str(local_pt))
    print(f"  [{expert_id}] Downloaded ({local_pt.stat().st_size // 1_000_000} MB)")
    return local_pt


def upload(sftp: paramiko.SFTPClient, local_pt: Path, expert_id: str) -> str:
    remote_path = f"{REMOTE_BASE}/outputs/try68_expert_{expert_id}/best_model.pt"
    print(f"  [{expert_id}] Uploading {local_pt} -> {remote_path} ...")
    sftp.put(str(local_pt), remote_path)
    print(f"  [{expert_id}] Upload done")
    return remote_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Skip upload, only print transforms")
    ap.add_argument("--expert", help="Process only this expert_id (default: all)")
    args = ap.parse_args()

    password = os.environ.get("SSH_PASSWORD", "")
    if not password:
        import getpass
        password = getpass.getpass(f"SSH password for {USER}@{HOST}: ")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=password, timeout=30)
    sftp = client.open_sftp()

    experts = [args.expert] if args.expert else EXPERTS

    with REG_PATH.open("r") as f:
        registry = yaml.safe_load(f)

    for expert_id in experts:
        print(f"\n{'='*60}")
        print(f"Expert: {expert_id}")
        print('='*60)

        # Find config
        cfg_rel = None
        for row in registry.get("experts", []):
            if row["expert_id"] == expert_id:
                cfg_rel = row["config"]
                break
        if cfg_rel is None:
            print(f"  ERROR: {expert_id} not found in registry")
            continue
        cfg_path = T68_DIR / cfg_rel
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f)

        base_ch = int(cfg["model"]["base_channels"])
        if base_ch != 80:
            print(f"  SKIP: base_channels={base_ch} (expected 80 in new YAML)")
            continue

        in_ch = compute_in_channels(cfg)
        print(f"  in_channels={in_ch}, base_channels={base_ch}")

        # Build 80ch model (to get target state dict shapes)
        model_80 = build_80ch_model(cfg, in_ch)
        new_sd = model_80.state_dict()

        # Download 40ch checkpoint
        local_pt = download_if_needed(sftp, expert_id)

        # Load old checkpoint
        ckpt = torch.load(str(local_pt), map_location="cpu", weights_only=False)
        old_gen_sd = ckpt.get("generator", ckpt.get("model"))
        old_ema_sd = ckpt.get("generator_ema")
        if old_gen_sd is None:
            print(f"  ERROR: no generator/model key in checkpoint")
            continue

        # Check old base_channels from a known weight (key uses Sequential integer indices)
        sample_key = "stem.0.block.0.weight"
        already_widened = False
        if sample_key in old_gen_sd:
            old_out = old_gen_sd[sample_key].shape[0]
            print(f"  Old base_channels (from stem conv out): {old_out}")
            if old_out == 80:
                print(f"  Already 80ch - will upload without re-widening")
                already_widened = True
            elif old_out != 40:
                print(f"  WARNING: expected old 40ch, got {old_out}")

        if already_widened:
            # Checkpoint is already widened; just upload the local file
            if not args.dry_run:
                upload(sftp, local_pt, expert_id)
            continue

        # Widen generator
        print("  Widening generator weights ...")
        new_gen_sd = widen_state_dict(old_gen_sd, new_sd)

        # Widen EMA if present
        new_ema_sd = None
        if old_ema_sd is not None:
            print("  Widening EMA weights ...")
            new_ema_sd = widen_state_dict(old_ema_sd, new_sd)
        else:
            print("  No EMA in checkpoint - EMA will be initialised from generator on resume")

        # Verify shapes
        print("  Verifying shapes ...")
        ok = True
        for k, v in new_gen_sd.items():
            expected = new_sd[k].shape
            if v.shape != expected:
                print(f"  MISMATCH {k}: {v.shape} != {expected}")
                ok = False
        if ok:
            print("  All shapes match new model OK")

        if args.dry_run:
            print("  [dry-run] skipping save/upload")
            continue

        # Build new checkpoint (weights only — trainer uses resume_weights_only)
        new_ckpt: dict[str, Any] = {
            "generator": new_gen_sd,
            "generator_ema": new_ema_sd,
            # Preserve metadata so logs show the widening origin
            "epoch": ckpt.get("epoch", 0),
            "best_epoch": ckpt.get("best_epoch", 0),
            "best_score": ckpt.get("best_score", None),
            "widened_from": "40ch -> 80ch via Net2Net (net2net_widen_try68_40ch_to_80ch.py)",
        }

        # Overwrite local best_model.pt
        torch.save(new_ckpt, str(local_pt))
        print(f"  Saved widened checkpoint ({local_pt.stat().st_size // 1_000_000} MB)")

        # Upload to cluster
        upload(sftp, local_pt, expert_id)

    sftp.close()
    client.close()
    print("\nAll done.")


if __name__ == "__main__":
    main()
