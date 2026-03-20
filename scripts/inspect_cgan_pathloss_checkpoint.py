#!/usr/bin/env python3
"""
Inspect NinthTry-style CGAN generator checkpoint + one forward sanity check.

Use when path_loss preds explode in dB on DirectML (e.g. mean ~2000 vs GT ~30):
  - Verifies state_dict has no NaN/Inf
  - Shows last conv (outc) and a few BatchNorm stats
  - Runs the same batch on CPU vs optional DirectML and prints raw + denorm ranges

Run from TFGpractice/:

  python scripts/inspect_cgan_pathloss_checkpoint.py ^
    --ninth-root TFGNinthTry9 ^
    --config TFGNinthTry9/configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9.yaml ^
    --checkpoint cluster_outputs/TFGNinthTry9/.../best_cgan.pt ^
    --hdf5 Datasets/CKM_Dataset_180326_antenna_height.h5 ^
    --sample-index 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent


def _insert_try_path(try_root: Path) -> None:
    p = str(try_root.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _summarize_tensor(name: str, t: Any, max_lines: int = 200) -> List[str]:
    import torch

    if not isinstance(t, torch.Tensor):
        return [f"  {name}: (not a tensor) {type(t)}"]
    x = t.detach().float().cpu().reshape(-1)
    n = int(x.numel())
    if n == 0:
        return [f"  {name}: empty"]
    nan_c = int(torch.isnan(x).sum())
    inf_c = int(torch.isinf(x).sum())
    finite = x[torch.isfinite(x)]
    lines = [
        f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"nan={nan_c} inf={inf_c} n={n}"
    ]
    if finite.numel() == 0:
        lines.append("    (no finite values)")
        return lines
    lines.append(
        f"    finite: min={float(finite.min()):.6g} max={float(finite.max()):.6g} "
        f"mean={float(finite.mean()):.6g} std={float(finite.std(unbiased=False)):.6g}"
    )
    return lines


def summarize_generator_state_dict(sd: Dict[str, Any], verbose: bool) -> None:
    import torch

    print("\n=== Generator state_dict ===")
    keys = sorted(sd.keys())
    print(f"num_keys: {len(keys)}")
    bad: List[str] = []
    interesting_suffixes = ("outc.weight", "outc.bias", "inc.block.0.weight", "scalar_film_mlp.0.weight")
    for k in keys:
        v = sd[k]
        if not isinstance(v, torch.Tensor):
            continue
        x = v.detach().float().cpu().reshape(-1)
        if torch.isnan(x).any() or torch.isinf(x).any():
            bad.append(k)
    if bad:
        print("TENSORS WITH NaN/Inf:", bad)
    else:
        print("No NaN/Inf in generator tensors.")

    for k in keys:
        if not verbose and not any(k.endswith(s) for s in interesting_suffixes):
            if "running_var" not in k and "running_mean" not in k:
                continue
        if k.endswith("running_mean") or k.endswith("running_var"):
            if not verbose and "outc" not in k and "down4" not in k:
                continue
        v = sd[k]
        if isinstance(v, torch.Tensor):
            for line in _summarize_tensor(k, v):
                print(line)

    if verbose:
        print("\n--- all keys (first 80) ---")
        for k in keys[:80]:
            print(" ", k)
        if len(keys) > 80:
            print(f"  ... and {len(keys) - 80} more")


def build_generator_and_cfg(
    try_root: Path,
    config_path: Path,
    hdf5_path: Path,
    scalar_csv: Optional[Path],
) -> Tuple[Any, Dict[str, Any], Any]:
    _insert_try_path(try_root)
    from config_utils import anchor_data_paths_to_config_file, load_config
    from data_utils import (
        build_dataset_splits_from_config,
        compute_input_channels,
        compute_scalar_cond_dim,
        uses_scalar_film_conditioning,
    )
    from model_cgan import UNetGenerator

    cfg_path_s = str(config_path.resolve())
    cfg = load_config(cfg_path_s)
    cfg["data"]["hdf5_path"] = str(hdf5_path.resolve())
    if scalar_csv is not None:
        cfg["data"]["scalar_table_csv"] = str(scalar_csv.resolve())
    else:
        cfg["data"]["scalar_table_csv"] = None
    anchor_data_paths_to_config_file(cfg, cfg_path_s)

    hybrid_enabled = bool(dict(cfg.get("path_loss_hybrid", {})).get("enabled", False))
    sc_dim = int(compute_scalar_cond_dim(cfg)) if uses_scalar_film_conditioning(cfg) else 0
    film_h = int(cfg["model"].get("scalar_film_hidden", 128))
    gen = UNetGenerator(
        in_channels=compute_input_channels(cfg),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        path_loss_hybrid=hybrid_enabled,
        norm_type=str(cfg["model"].get("norm_type", "batch")),
        scalar_cond_dim=sc_dim,
        scalar_film_hidden=film_h,
    )
    return gen, cfg, build_dataset_splits_from_config(cfg)


def pick_device(name: str) -> Any:
    n = name.lower().strip()
    if n == "cpu":
        return "cpu"
    if n == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return "cuda"
    if n in ("directml", "dml", "amd"):
        try:
            import torch_directml
        except ImportError as e:
            raise SystemExit("Install torch-directml for DirectML device.") from e
        return torch_directml.device()
    raise SystemExit(f"Unknown device: {name!r}")


def forward_one(
    gen: Any,
    cfg: Dict[str, Any],
    ds: Any,
    sample_index: int,
    device: Any,
    target_columns: List[str],
) -> None:
    import torch
    from config_utils import is_cuda_device
    from data_utils import forward_cgan_generator, unpack_cgan_batch
    from evaluate_cgan import denormalize_channel

    target_metadata = dict(cfg.get("target_metadata", {}))
    idx = int(sample_index)
    batch = ds[idx]
    inputs = [t.unsqueeze(0) for t in batch[:3]]
    if len(batch) == 4:
        pack = (inputs[0], inputs[1], inputs[2], batch[3].unsqueeze(0))
    else:
        pack = (inputs[0], inputs[1], inputs[2])

    gen_dev = gen.to(device)
    if not is_cuda_device(device):
        gen_dev = gen_dev.float()
    gen_dev.eval()

    with torch.no_grad():
        x, y, m, sc_tensor = unpack_cgan_batch(pack, device)
        x = x.float()
        if sc_tensor is not None:
            sc_tensor = sc_tensor.float()
        pred = forward_cgan_generator(gen_dev, x, sc_tensor)
        pred = pred.float()

    pi = target_columns.index("path_loss")
    raw = pred[0, pi : pi + 1].detach().cpu()
    meta = target_metadata.get("path_loss", {})
    denorm_t = denormalize_channel(pred[0, pi : pi + 1].to("cpu"), meta)
    denorm_np = denorm_t.numpy().reshape(-1)

    y_sub = y[0, pi : pi + 1].float().cpu()
    y_denorm = denormalize_channel(y_sub, meta).numpy().reshape(-1)

    city, sample = ds.sample_refs[idx]
    print(f"\n--- forward sample_index={idx} {city}/{sample} device={device!r} ---")
    print(
        f"pred RAW (model output, normalized training space): "
        f"min={float(raw.min()):.6g} max={float(raw.max()):.6g} mean={float(raw.mean()):.6g}"
    )
    print(
        f"pred denorm (dB): min={float(np.nanmin(denorm_np)):.6g} max={float(np.nanmax(denorm_np)):.6g} "
        f"mean={float(np.nanmean(denorm_np)):.6g}"
    )
    print(
        f"gt   denorm (dB): min={float(np.nanmin(y_denorm)):.6g} max={float(np.nanmax(y_denorm)):.6g} "
        f"mean={float(np.nanmean(y_denorm)):.6g}"
    )
    hint = []
    if float(raw.abs().max()) > 5.0:
        hint.append(
            "RAW |max|>>1 vs GT ~0..1: often wrong scalar input scaling (antenna_height_m norm=1 vs training CSV max). "
            "Export/infer without scalar_table_csv now infers scalar_feature_norms from HDF5; or pass --scalar-csv."
        )
    if float(np.nanmean(denorm_np)) > 300:
        hint.append("Denorm mean huge: fix scalar norms first; CPU vs DirectML should match if inputs match.")
    if hint:
        print("  |  " + " ".join(hint))


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect CGAN generator checkpoint + forward sanity.")
    p.add_argument("--ninth-root", type=str, default=str(PRACTICE_ROOT / "TFGNinthTry9"))
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--hdf5", type=str, required=True)
    p.add_argument("--scalar-csv", type=str, default=None)
    p.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--verbose-keys", action="store_true", help="Print all state_dict tensor summaries (long).")
    p.add_argument(
        "--compare-devices",
        type=str,
        default="cpu,directml",
        help="Comma-separated: cpu, cuda, directml. Example: cpu only or cpu,directml",
    )
    args = p.parse_args()

    try_root = Path(args.ninth_root)
    config_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    hdf5_path = Path(args.hdf5)
    scalar_csv = Path(args.scalar_csv) if args.scalar_csv else None

    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    if not hdf5_path.is_file():
        raise SystemExit(f"HDF5 not found: {hdf5_path}")

    _insert_try_path(try_root)
    import torch
    from config_utils import load_torch_checkpoint

    print(f"Loading checkpoint (CPU map): {ckpt_path}")
    state = load_torch_checkpoint(str(ckpt_path), "cpu")
    print("Top-level keys:", sorted(state.keys()))
    if "generator" not in state:
        raise SystemExit("Expected 'generator' key in checkpoint (CGAN train_cgan.py format).")
    g_sd = state["generator"]
    summarize_generator_state_dict(g_sd, verbose=bool(args.verbose_keys))

    gen, cfg, splits = build_generator_and_cfg(try_root, config_path, hdf5_path, scalar_csv)
    gen.load_state_dict(g_sd, strict=True)
    print("\nload_state_dict(generator): strict OK")

    if args.split not in splits:
        raise SystemExit(f"Split {args.split!r} missing; have {list(splits)}")
    ds = splits[args.split]
    target_columns = list(cfg["target_columns"])

    devices = [s.strip() for s in args.compare_devices.split(",") if s.strip()]
    for dname in devices:
        try:
            dev = pick_device(dname)
        except SystemExit as e:
            print(f"Skip device {dname!r}: {e}")
            continue
        forward_one(gen, cfg, ds, args.sample_index, dev, target_columns)

    print("\nDone.")


if __name__ == "__main__":
    main()
