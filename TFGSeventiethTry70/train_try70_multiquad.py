#!/usr/bin/env python3
"""Try 70 — PMHHNet + multi-scale quad auxiliary heads (optional slim trainer).

**Prefer** ``train_partitioned_pathloss_expert.py`` with ``model.arch: pmhhnet_try70`` and a
``try70:`` block in the YAML (full cluster recipe, EMA, early stopping, etc.).

This script remains a minimal smoke-test entry point.

Loss: main 513 Huber (same as Try 68 full_map path) + weighted mean auxiliary Huber
across 257 / 129 / 65 branches (native + bilinear-up global branches).

Validation: RMSE on full-res prediction; optional ``try70.blend_report_first_batch`` runs
:func:`model_try70_multiquad.try70_blend_search_rmse_physical` on the first val batch.

Example::

  python train_try70_multiquad.py \\
    --config experiments/seventieth_try70_experts/try70_expert_open_sparse_lowrise.yaml \\
    --init-checkpoint path/to/try68_expert_open_sparse_lowrise/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_utils import anchor_data_paths_to_config_file, ensure_output_dir, load_config, load_torch_checkpoint, resolve_device
from data_utils import build_dataset_splits_from_config, compute_input_channels, unpack_cgan_batch
from model_try70_multiquad import PMHHNetTry70MultiQuad, try70_auxiliary_loss
from train_partitioned_pathloss_expert import (
    clip_to_target_range,
    compute_multiscale_path_loss_loss,
    compute_scalar_cond_dim,
    effective_huber_delta,
    extract_formula_prior_or_zero,
    masked_huber_loss,
    return_scalar_cond_from_config,
    set_seed,
)


def _build_try70_generator(cfg: Dict[str, Any], in_channels: int) -> PMHHNetTry70MultiQuad:
    m = cfg["model"]
    scalar_dim = int(compute_scalar_cond_dim(cfg)) if return_scalar_cond_from_config(cfg) else 0
    return PMHHNetTry70MultiQuad(
        in_channels,
        int(m["out_channels"]),
        base_channels=int(m.get("base_channels", 64)),
        encoder_blocks=tuple(m.get("encoder_blocks", (2, 2, 2, 2))),
        context_dilations=tuple(m.get("context_dilations", (1, 2, 4, 8))),
        norm_type=str(m.get("norm_type", "group")),
        dropout=float(m.get("dropout", 0.0)),
        gradient_checkpointing=bool(m.get("gradient_checkpointing", False)),
        hf_channels=int(m.get("hf_channels", max(8, int(m.get("base_channels", 64)) // 2))),
        scalar_dim=max(1, scalar_dim),
        scalar_hidden_dim=int(m.get("scalar_hidden_dim", max(32, int(m.get("base_channels", 64)) * 2))),
        sinusoidal_embed_dim=int(m.get("sinusoidal_embed_dim", 64)),
        sinusoidal_max_period=float(m.get("sinusoidal_max_period", 1000.0)),
        use_se_attention=bool(m.get("use_se_attention", False)),
        se_reduction=int(m.get("se_reduction", 4)),
    )


def _load_init_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> Dict[str, Any]:
    state = load_torch_checkpoint(path, device)
    raw = state.get("model", state.get("generator", state))
    if hasattr(raw, "state_dict"):
        raw = raw.state_dict()
    missing, unexpected = model.load_state_dict(raw, strict=False)
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def _is_cuda(device: object) -> bool:
    return getattr(device, "type", str(device)) == "cuda"


def _train_one_epoch(
    model: PMHHNetTry70MultiQuad,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    meta: Dict[str, Any],
    loss_cfg: Dict[str, Any],
    try70_cfg: Dict[str, Any],
) -> float:
    model.train()
    huber_delta_cfg = float(loss_cfg.get("huber_delta", 6.0))
    huber_delta_eff = effective_huber_delta(huber_delta_cfg, meta, loss_cfg)
    loss_type = str(loss_cfg.get("loss_type", "huber")).lower()
    aux_w = float(try70_cfg.get("aux_loss_weight", 0.12))
    amp_enabled = bool(cfg["training"].get("amp", True)) and _is_cuda(device)
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x, y, m, sc = unpack_cgan_batch(batch, device)
        target = y[:, :1]
        mask = m[:, :1]
        prior = extract_formula_prior_or_zero(x, cfg, target)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=amp_enabled):
            residual = model(x, sc)
            pred = prior + residual
            pred = clip_to_target_range(pred, meta)
            if loss_type == "huber":
                g_main = masked_huber_loss(pred, target, mask, delta=huber_delta_eff)
            else:
                diff = (pred - target) * mask
                g_main = (diff**2).sum() / mask.sum().clamp_min(1.0)
            ms = compute_multiscale_path_loss_loss(pred, target, mask, meta, cfg)
            aux = model.pop_last_aux()
            g_aux = (
                try70_auxiliary_loss(aux, target, mask, huber_delta=huber_delta_eff, weight=aux_w, prior=prior)
                if aux is not None
                else torch.zeros((), device=device)
            )
            loss = g_main + ms + g_aux
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().item())
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def _validate_rmse(
    model: PMHHNetTry70MultiQuad,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    try70_cfg: Dict[str, Any],
) -> tuple[float, Optional[Dict[str, Any]]]:
    """Validate RMSE and accumulate per-channel blend stats across the full val set.

    For each aux channel c and blend weight alpha in [0.0, 0.1, ..., 1.0]:
        pred_blend = prior + (1-alpha)*res_main + alpha*res_aux_up
    SSE is accumulated over all validation samples; final RMSE is reported per (channel, alpha).
    The best (channel, alpha) pair is returned in the blend report.
    """
    model.eval()
    amp_enabled = bool(cfg["training"].get("amp", True)) and _is_cuda(device)
    scale = float(meta.get("scale", 180.0))
    run_blend = bool(try70_cfg.get("blend_report_first_batch", True))
    alpha_steps = int(try70_cfg.get("blend_alpha_steps", 11))
    alphas = [i / max(alpha_steps - 1, 1) for i in range(alpha_steps)]

    # Accumulators for main RMSE
    sse_main = 0.0
    cnt_main = 0.0

    # Accumulators for blend search: {channel_label: [sse_per_alpha, cnt]}
    # Populated on first batch where aux is available; pre-allocated per-alpha as list of floats.
    blend_sse: Dict[str, list] = {}   # label -> [sse_a0, sse_a1, ...]
    blend_cnt: float = 0.0
    aux_schema_ready = False

    for batch in tqdm(loader, desc="val", leave=False):
        x, y, m, sc = unpack_cgan_batch(batch, device)
        target = y[:, :1]
        mask = m[:, :1]
        prior = extract_formula_prior_or_zero(x, cfg, target)
        with amp.autocast(device_type="cuda", enabled=amp_enabled):
            residual = model(x, sc)
            pred = prior + residual
            pred = clip_to_target_range(pred, meta)
        aux = model.pop_last_aux()

        # Main head RMSE accumulation
        diff = (pred - target) * mask
        sse_main += float((diff**2).sum().item())
        cnt_main += float(mask.sum().item())

        if not run_blend or aux is None:
            continue

        res_main = pred - prior  # [B,1,H,W] normalized residual from main head
        h, w = pred.shape[-2:]
        valid_px = float(mask.sum().item())
        blend_cnt += valid_px

        for ten_name, ten in aux.items():
            _, c, _, _ = ten.shape
            for i in range(c):
                label = f"{ten_name}_ch{i}"
                res_up = F.interpolate(ten[:, i : i + 1], size=(h, w), mode="bilinear", align_corners=False)
                if not aux_schema_ready:
                    blend_sse[label] = [0.0] * alpha_steps
                for ai, a in enumerate(alphas):
                    blended = prior + (1.0 - a) * res_main + a * res_up
                    blended = clip_to_target_range(blended, meta)
                    d = (blended - target) * mask
                    blend_sse[label][ai] += float((d**2).sum().item())
        aux_schema_ready = True

    # Compute main RMSE
    mse_main = sse_main / max(cnt_main, 1.0)
    rmse_phys = float(math.sqrt(mse_main)) * scale

    blend_report: Optional[Dict[str, Any]] = None
    if run_blend and blend_sse:
        baseline_rmse = rmse_phys
        rows = []
        best_label, best_alpha, best_rmse = "pred_513_only", None, baseline_rmse
        for label, sse_list in blend_sse.items():
            rmse_by_alpha = {}
            ch_best_r, ch_best_a = baseline_rmse, 0.0
            for ai, a in enumerate(alphas):
                r = math.sqrt(sse_list[ai] / max(blend_cnt, 1.0)) * scale
                rmse_by_alpha[f"a{a:.2f}"] = round(r, 4)
                if r < ch_best_r:
                    ch_best_r, ch_best_a = r, a
            rows.append({"name": label, "best_alpha": ch_best_a, "best_rmse_phys": ch_best_r,
                         "rmse_by_alpha": rmse_by_alpha})
            if ch_best_r < best_rmse:
                best_rmse, best_label, best_alpha = ch_best_r, label, ch_best_a

        rows.sort(key=lambda r: r["best_rmse_phys"])
        blend_report = {
            "rmse_phys_baseline_pred513": baseline_rmse,
            "best_rmse_phys_blend": best_rmse,
            "best_blend_label": best_label,
            "best_blend_alpha": best_alpha,
            "total_aux_channels": len(blend_sse),
            "per_component": rows,
        }

    return rmse_phys, blend_report


def main() -> None:
    p = argparse.ArgumentParser(description="Try 70 multi-quad PMHHNet trainer")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--epochs-override", type=int, default=None)
    p.add_argument("--init-checkpoint", type=Path, default=None, help="Try 68 (or compatible) best_model.pt")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.epochs_override is not None:
        cfg.setdefault("training", {})["epochs"] = max(1, int(args.epochs_override))
    anchor_data_paths_to_config_file(cfg, args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg["runtime"]["device"])
    try70_cfg = dict(cfg.get("try70", {}))

    splits = build_dataset_splits_from_config(cfg)
    train_ds = splits["train"]
    val_ds = splits["val"]
    pin = _is_cuda(device)
    nw = int(cfg["data"].get("num_workers", 0))
    loader_kw: Dict[str, Any] = {"num_workers": nw, "pin_memory": pin}
    if nw > 0:
        loader_kw["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"].get("val_batch_size", 1)),
        shuffle=False,
        **loader_kw,
    )

    in_ch = int(compute_input_channels(cfg))
    model = _build_try70_generator(cfg, in_ch).to(device)
    if args.init_checkpoint is not None:
        info = _load_init_checkpoint(model, args.init_checkpoint, torch.device("cpu"))
        print(json.dumps({"init_checkpoint": str(args.init_checkpoint), **info}, indent=2))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.1)),
    )
    scaler = amp.GradScaler(enabled=bool(cfg["training"].get("amp", True)) and _is_cuda(device))

    out_dir = ensure_output_dir(cfg["runtime"]["output_dir"])
    meta = dict(cfg["target_metadata"]["path_loss"])
    loss_cfg = dict(cfg.get("loss", {}))
    epochs = int(cfg["training"]["epochs"])
    best = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = _train_one_epoch(model, train_loader, device, cfg, opt, scaler, meta, loss_cfg, try70_cfg)
        va, blend = _validate_rmse(model, val_loader, device, cfg, meta, try70_cfg)
        dt = time.perf_counter() - t0
        row = {"epoch": epoch, "train_loss": tr, "val_rmse_physical_db": va, "seconds": round(dt, 2)}
        if blend is not None:
            row["blend_first_batch"] = blend
        print(json.dumps(row))
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": args.config}, out_dir / "best_model.pt")
        with (out_dir / "try70_metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
