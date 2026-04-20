"""Try 77 — evaluation on the held-out test split (spreads).

Reports:
    - RMSE / MAE in native units (ns or deg) masked by ground ∩ valid target
    - Wasserstein-1 / KL between per-image target and prediction histograms
    - Per-image CSV with pred / target histograms in ``n_bins`` bins across
      the expert's clamp range.

Usage:
    python evaluate.py --config .../config.yaml --checkpoint .../best_model.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Try77Cfg
from data_utils import HeightEmbedding, Try77Config, build_expert_datasets
from metrics import (
    kl_from_counts,
    masked_mae,
    masked_rmse,
    per_image_histogram,
    wasserstein1_from_counts,
)
from model import Try77Model, Try77ModelConfig


N_BINS = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    cfg = Try77Cfg.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = Try77Config(
        hdf5_path=cfg.data.hdf5_path,
        topology_class=cfg.data.topology_class,
        metric=cfg.data.metric,
        image_size=cfg.data.image_size,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        split_seed=cfg.data.split_seed,
    )
    _, val_ds, test_ds = build_expert_datasets(data_cfg)
    ds = val_ds if args.split == "val" else test_ds
    loader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=False)

    model_cfg = Try77ModelConfig(
        in_channels=4,
        cond_dim=64,
        height_embed_dim=32,
        base_width=cfg.model.base_width,
        K=cfg.model.K,
        clamp_lo=cfg.model.clamp_lo,
        clamp_hi=cfg.model.clamp_hi,
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        spike_mu_max=cfg.model.spike_mu_max,
        spike_sigma_min=cfg.model.spike_sigma_min,
        spike_sigma_max=cfg.model.spike_sigma_max,
    )
    model = Try77Model(model_cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    height_embed = HeightEmbedding()

    out_dir = cfg.runtime.output_dir / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_per_sample: List[Dict] = []
    hist_rows: List[List] = []
    metric_totals = {"rmse": 0.0, "mae": 0.0, "w1": 0.0, "kl_pred_vs_target": 0.0, "n": 0}

    clamp_lo = float(cfg.model.clamp_lo)
    clamp_hi = float(cfg.model.clamp_hi)
    bin_width = (clamp_hi - clamp_lo) / N_BINS
    bin_headers = [f"b{i}" for i in range(N_BINS)]
    hist_header = ["city", "sample", "altitude_m", "try_dir", "expert_id", "metric", "kind", "total_pixels"] + bin_headers
    hist_rows.append(hist_header)

    expert_id = f"{cfg.data.topology_class}_{cfg.data.metric}"

    for batch in loader:
        inputs = batch["inputs"].to(device)
        target = batch["target"].to(device)
        mask = batch["loss_mask"].to(device)
        h = batch["antenna_height_m"].to(device)
        h_emb = height_embed(h)

        with torch.no_grad():
            out = model(inputs, h_emb)
        pred = out["pred"]

        rmse = masked_rmse(pred, target, mask)
        mae = masked_mae(pred, target, mask)

        c_pred = per_image_histogram(pred, mask, clamp_lo, clamp_hi, N_BINS)
        c_tgt = per_image_histogram(target, mask, clamp_lo, clamp_hi, N_BINS)
        w1 = wasserstein1_from_counts(c_pred, c_tgt, bin_width=bin_width)
        kl = kl_from_counts(c_pred, c_tgt)

        metric_totals["rmse"] += rmse
        metric_totals["mae"] += mae
        if not np.isnan(w1):
            metric_totals["w1"] += w1
        if not np.isnan(kl):
            metric_totals["kl_pred_vs_target"] += kl
        metric_totals["n"] += 1

        rows_per_sample.append({
            "city": batch["city"][0],
            "sample": batch["sample"][0],
            "uav_height_m": float(h.item()),
            "rmse": rmse,
            "mae": mae,
            "w1": w1,
            "kl_pred_vs_target": kl,
            "gmm_pi": out["gmm"]["pi"].squeeze(0).detach().cpu().tolist(),
            "gmm_mu": out["gmm"]["mu"].squeeze(0).detach().cpu().tolist(),
            "gmm_sigma": out["gmm"]["sigma"].squeeze(0).detach().cpu().tolist(),
        })
        hist_rows.append([batch["city"][0], batch["sample"][0], float(h.item()),
                          "TFGSeventyEighthTry78", expert_id, cfg.data.metric,
                          "pred", int(c_pred.sum())] + c_pred.tolist())
        hist_rows.append([batch["city"][0], batch["sample"][0], float(h.item()),
                          "TFGSeventyEighthTry78", expert_id, cfg.data.metric,
                          "target", int(c_tgt.sum())] + c_tgt.tolist())

    n = max(metric_totals["n"], 1)
    agg = {
        "rmse": metric_totals["rmse"] / n,
        "mae": metric_totals["mae"] / n,
        "w1": metric_totals["w1"] / n,
        "kl_pred_vs_target": metric_totals["kl_pred_vs_target"] / n,
        "n_samples": n,
        "split": args.split,
        "expert_id": expert_id,
        "metric": cfg.data.metric,
        "clamp_range": [clamp_lo, clamp_hi],
    }
    (out_dir / "summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out_dir / "per_sample.json").write_text(json.dumps(rows_per_sample, indent=2), encoding="utf-8")
    with open(out_dir / "histograms.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(hist_rows)

    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
