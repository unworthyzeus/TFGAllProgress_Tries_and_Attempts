"""Per-sample + per-zone evaluation of Try 74 LoS / NLoS experts on local CUDA.

Runs EMA weights from best_model.pt over the val split and produces:
  - {out}.csv : per-sample row (rmse, los/nlos rmse split, city, antenna)
  - {out}.zones.csv : per-zone aggregated RMSE/MAE by scene characteristic
  - {out}.summary.json : overall stats

Zone definitions (derived from the input channels of the model, which are
already normalized 0..1 for 513x513 at 1 m/px; max_radius ~ 362 m):
  dist_since_break:  knife_edge(<5m), moderate(5-30m), deep(30-100m), very_deep(100m+)
  blocker_count:     1, 2, 3plus
  dist_from_tx:      near(<100m), mid(100-200m), far(200m+)
  elev_angle:        low(<10deg), mid(10-30deg), high(30deg+)
  city_type:         dense_highrise, mixed_midrise, open_lowrise  (from scene stats)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

TRY74 = Path(r"c:/TFG/TFGPractice/TFGSeventyFourthTry74")
sys.path.insert(0, str(TRY74))
os.chdir(TRY74)

from config_utils import anchor_data_paths_to_config_file, load_config, resolve_device  # noqa: E402
from data_utils import build_dataset_splits_from_config, compute_input_channels  # noqa: E402
from train_partitioned_pathloss_expert import (  # noqa: E402
    _build_pmnet_from_cfg,
    _compose_residual_prediction_with_aux,
    build_validation_loader,
    clip_to_target_range,
    denormalize,
    extract_formula_prior_or_zero,
    sanitize_masked_target,
    unpack_cgan_batch,
    uses_absolute_path_loss_prediction,
    uses_formula_prior,
    uses_los_input_channel,
)


MAX_RADIUS_M = 362.0  # 513x513 at 1 m/px, max diagonal radius
MAX_BLOCKERS_SCALE = 8.0  # blocker_count normalized by /8 in data_utils


def _gaussian_blur_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on (H,W) or (B,C,H,W). sigma in pixels."""
    if sigma <= 0:
        return img
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    ks = 2 * radius + 1
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=img.device)
    k1 = torch.exp(-0.5 * (xs / sigma) ** 2)
    k1 = k1 / k1.sum()
    reshaped = False
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        reshaped = True
    elif img.dim() == 3:
        img = img.unsqueeze(0)
        reshaped = True
    import torch.nn.functional as F
    kx = k1.view(1, 1, 1, ks)
    ky = k1.view(1, 1, ks, 1)
    out = F.conv2d(img, kx.to(img.dtype), padding=(0, radius))
    out = F.conv2d(out, ky.to(img.dtype), padding=(radius, 0))
    if reshaped:
        out = out.squeeze()
    return out


def _azimuthal_mean(values: torch.Tensor, mask: torch.Tensor, r_int: torch.Tensor, max_r: int) -> torch.Tensor:
    """Azimuthal mean restricted to pixels where mask==1, returns vector of length max_r+1 with nan where empty."""
    flat_v = values[mask]
    flat_r = r_int[mask]
    if flat_v.numel() == 0:
        return torch.full((max_r + 1,), float("nan"), dtype=values.dtype, device=values.device)
    sums = torch.zeros(max_r + 1, dtype=values.dtype, device=values.device)
    counts = torch.zeros(max_r + 1, dtype=values.dtype, device=values.device)
    sums.scatter_add_(0, flat_r, flat_v)
    counts.scatter_add_(0, flat_r, torch.ones_like(flat_v))
    out = torch.where(counts > 0, sums / counts.clamp_min(1.0), torch.full_like(sums, float("nan")))
    return out


def compute_ring_metrics_db(
    pred_db: torch.Tensor,     # (H,W)
    target_db: torch.Tensor,   # (H,W)
    los_mask: torch.Tensor,    # (H,W) bool, LoS pixels only
    valid_mask: torch.Tensor,  # (H,W) bool, pixels with valid supervision
    sigma_px: float = 3.0,
) -> dict:
    """Metrics probing how well the model reproduces the 2-ray ring pattern (LoS only)."""
    H, W = pred_db.shape
    device = pred_db.device
    mask_full = los_mask & valid_mask
    if mask_full.sum() < 32:
        return {"pixels_los": int(mask_full.sum().item())}

    # Fill outside-LoS with LoS mean so the blur isn't corrupted by building / NLoS regions.
    pred_fill = torch.where(mask_full, pred_db, pred_db[mask_full].mean())
    target_fill = torch.where(mask_full, target_db, target_db[mask_full].mean())

    pred_smooth = _gaussian_blur_2d(pred_fill, sigma_px)
    target_smooth = _gaussian_blur_2d(target_fill, sigma_px)
    pred_hp = pred_db - pred_smooth
    target_hp = target_db - target_smooth

    diff_hp = (pred_hp - target_hp)[mask_full]
    diff_sm = (pred_smooth - target_smooth)[mask_full]
    rmse_hp = float(torch.sqrt((diff_hp ** 2).mean()).item())
    rmse_sm = float(torch.sqrt((diff_sm ** 2).mean()).item())

    # Ring amplitude (std of the high-pass residual) — how strongly oscillates.
    tgt_hp_std = float(target_hp[mask_full].std().item())
    pred_hp_std = float(pred_hp[mask_full].std().item())
    amp_ratio = pred_hp_std / tgt_hp_std if tgt_hp_std > 1e-6 else float("nan")

    # Azimuthal radial profile (1D). Compute integer radius from center.
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    y = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0)
    r_f = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_int = r_f.long().clamp_max(int(math.ceil(math.sqrt((H / 2) ** 2 + (W / 2) ** 2))))
    max_r = int(r_int.max().item())

    prof_pred = _azimuthal_mean(pred_db.float(), mask_full, r_int, max_r)
    prof_tgt = _azimuthal_mean(target_db.float(), mask_full, r_int, max_r)
    both = ~(torch.isnan(prof_pred) | torch.isnan(prof_tgt))
    if both.sum() > 2:
        rmse_profile = float(torch.sqrt(((prof_pred[both] - prof_tgt[both]) ** 2).mean()).item())
    else:
        rmse_profile = float("nan")

    # Radial derivative fidelity: finite difference of radial profile.
    if both.sum() > 4:
        dp = prof_pred[both][1:] - prof_pred[both][:-1]
        dt = prof_tgt[both][1:] - prof_tgt[both][:-1]
        rmse_dderiv = float(torch.sqrt(((dp - dt) ** 2).mean()).item())
    else:
        rmse_dderiv = float("nan")

    return {
        "pixels_los": int(mask_full.sum().item()),
        "rmse_highpass_db": rmse_hp,
        "rmse_smooth_db": rmse_sm,
        "ring_amp_ratio_pred_over_target": amp_ratio,
        "target_highpass_std_db": tgt_hp_std,
        "pred_highpass_std_db": pred_hp_std,
        "rmse_radial_profile_db": rmse_profile,
        "rmse_radial_deriv_db": rmse_dderiv,
    }


def load_ema_state(ckpt_path: Path, device):
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    for k in ("generator_ema", "model", "generator"):
        if k in state:
            return state[k]
    return state


def compute_rmse(sse: float, count: float) -> float:
    if count <= 0:
        return float("nan")
    return float(math.sqrt(sse / count))


def compute_mae(sae: float, count: float) -> float:
    if count <= 0:
        return float("nan")
    return float(sae / count)


def channel_indices(cfg) -> dict:
    """Return {name: channel_index} for the input tensor of this config."""
    idx = {}
    c = 0
    idx["topo"] = c; c += 1
    if bool(cfg["data"].get("los_input_column")) and bool(cfg["data"].get("use_los_as_input", True)):
        idx["los"] = c; c += 1
    if cfg["data"].get("distance_map_channel", False):
        idx["distance"] = c; c += 1
    if bool(cfg["data"].get("path_loss_formula_input", {}).get("enabled", False)):
        idx["formula"] = c; c += 1
        if bool(cfg["data"].get("path_loss_formula_input", {}).get("include_confidence_channel", False)):
            idx["formula_conf"] = c; c += 1
    obs = dict(cfg["data"].get("path_loss_obstruction_features", {}))
    if bool(obs.get("enabled", False)):
        if bool(obs.get("include_shadow_depth", True)):
            idx["shadow_depth"] = c; c += 1
        if bool(obs.get("include_distance_since_los_break", True)):
            idx["dist_since_break"] = c; c += 1
        if bool(obs.get("include_max_blocker_height", True)):
            idx["max_blocker_h"] = c; c += 1
        if bool(obs.get("include_blocker_count", True)):
            idx["blocker_count"] = c; c += 1
    if bool(cfg["data"].get("tx_depth_map_channel", False)):
        idx["tx_depth"] = c; c += 1
    if bool(cfg["data"].get("elevation_angle_map_channel", False)):
        idx["elev_angle"] = c; c += 1
    if bool(cfg["data"].get("building_mask_channel", False)):
        idx["building_mask"] = c; c += 1
    return idx


def infer_city_type(topo_meters: np.ndarray, non_ground_threshold: float = 0.0) -> str:
    """Mirror _infer_city_type_simple from data_utils for a scene."""
    bm = topo_meters != non_ground_threshold
    density = float(np.mean(bm)) if bm.size else 0.0
    heights = topo_meters[bm]
    mean_h = float(np.mean(heights)) if heights.size else 0.0
    if density >= 0.34 or mean_h >= 90.0:
        return "dense_highrise"
    if density <= 0.18 and mean_h <= 30.0:
        return "open_lowrise"
    return "mixed_midrise"


def bucket_distance_since_break(x: torch.Tensor) -> torch.Tensor:
    """Returns integer bucket 0..3 for knife/moderate/deep/very_deep."""
    # x is normalized: 1.0 == MAX_RADIUS_M (~362 m).
    b = torch.full_like(x, -1, dtype=torch.int8)
    b[(x >= 0) & (x < 5.0 / MAX_RADIUS_M)] = 0
    b[(x >= 5.0 / MAX_RADIUS_M) & (x < 30.0 / MAX_RADIUS_M)] = 1
    b[(x >= 30.0 / MAX_RADIUS_M) & (x < 100.0 / MAX_RADIUS_M)] = 2
    b[x >= 100.0 / MAX_RADIUS_M] = 3
    return b


def bucket_blocker_count(x: torch.Tensor) -> torch.Tensor:
    # normalized by /8 in data_utils; so "1 blocker" ~ 0.125.
    b = torch.full_like(x, -1, dtype=torch.int8)
    # Threshold midpoints:
    b[(x > 0) & (x < 1.5 / MAX_BLOCKERS_SCALE)] = 0   # count=1
    b[(x >= 1.5 / MAX_BLOCKERS_SCALE) & (x < 2.5 / MAX_BLOCKERS_SCALE)] = 1  # count=2
    b[x >= 2.5 / MAX_BLOCKERS_SCALE] = 2             # count>=3
    return b


def bucket_distance_from_tx(x: torch.Tensor) -> torch.Tensor:
    # distance_map is [0,1] with 1 ~= 362 m.
    b = torch.full_like(x, -1, dtype=torch.int8)
    b[(x >= 0) & (x < 100.0 / MAX_RADIUS_M)] = 0
    b[(x >= 100.0 / MAX_RADIUS_M) & (x < 200.0 / MAX_RADIUS_M)] = 1
    b[x >= 200.0 / MAX_RADIUS_M] = 2
    return b


def bucket_elev_angle(x: torch.Tensor) -> torch.Tensor:
    # elevation_angle is [0,1], 1 = 90 deg.
    b = torch.full_like(x, -1, dtype=torch.int8)
    b[(x >= 0) & (x < 10.0 / 90.0)] = 0
    b[(x >= 10.0 / 90.0) & (x < 30.0 / 90.0)] = 1
    b[x >= 30.0 / 90.0] = 2
    return b


DIST_BREAK_NAMES = ["knife_edge<5m", "moderate_5_30m", "deep_30_100m", "very_deep_100m+"]
BLOCKER_COUNT_NAMES = ["count_1", "count_2", "count_3plus"]
DIST_TX_NAMES = ["tx_near<100m", "tx_mid_100_200m", "tx_far>200m"]
ELEV_NAMES = ["elev<10deg", "elev_10_30deg", "elev>30deg"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-prefix", required=True, type=str)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    cfg.setdefault("training", {})["batch_size"] = 1
    cfg["data"]["val_batch_size"] = 1
    cfg["data"]["num_workers"] = 0
    cfg["data"]["val_num_workers"] = 0
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["val_persistent_workers"] = False
    cfg.setdefault("model", {})["gradient_checkpointing"] = False

    device = resolve_device(cfg["runtime"]["device"])
    print(f"device={device}")

    splits = build_dataset_splits_from_config(cfg)
    dataset = splits[args.split]
    print(f"{args.split} samples: {len(dataset)}")

    in_ch = int(compute_input_channels(cfg))
    print(f"in_channels: {in_ch}")
    ch = channel_indices(cfg)
    print(f"channel map: {ch}")

    model = _build_pmnet_from_cfg(cfg, in_ch).to(device)
    model.load_state_dict(load_ema_state(Path(args.checkpoint), device))
    model.eval()

    loader, sample_indices = build_validation_loader(dataset, device, cfg, distributed=False)
    sample_indices = list(sample_indices)

    meta = dict(cfg["target_metadata"]["path_loss"])
    absolute_prediction = uses_absolute_path_loss_prediction(cfg)
    clamp_final = bool(cfg.get("prior_residual_path_loss", {}).get("clamp_final_output", True))
    amp_enabled = bool(args.amp) and (getattr(device, "type", str(device)) == "cuda")

    rows = []
    zone_totals = defaultdict(lambda: {"sse": 0.0, "sae": 0.0, "count": 0})
    processed = 0
    cursor = 0

    with torch.no_grad():
        for batch in loader:
            x, y, m, scalar_cond = unpack_cgan_batch(batch, device)
            prior = extract_formula_prior_or_zero(x, cfg, y[:, :1])
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                residual_pred, _, _, _ = _compose_residual_prediction_with_aux(
                    model, x, scalar_cond, prior,
                    separated_mode=False, base_generator=None,
                    use_gate=False, absolute_prediction=absolute_prediction,
                )
            pred = residual_pred if absolute_prediction else (prior + residual_pred)
            if clamp_final:
                pred = clip_to_target_range(pred, meta)

            valid_mask = m[:, :1] > 0.0  # (B,1,H,W)
            safe_target = sanitize_masked_target(y[:, :1], m[:, :1])
            pred_phys = denormalize(pred, meta).float()
            target_phys = denormalize(safe_target, meta).float()
            diff = pred_phys - target_phys  # (B,1,H,W)
            diff_sq = diff * diff
            diff_abs = diff.abs()

            B = x.shape[0]
            for off in range(B):
                idx = int(sample_indices[cursor + off])
                vm = valid_mask[off, 0]  # (H,W) bool
                d_sq = diff_sq[off, 0]
                d_ab = diff_abs[off, 0]
                d = diff[off, 0]

                n_valid = int(vm.sum().item())
                sse_total = float(d_sq[vm].sum().item()) if n_valid > 0 else 0.0
                sae_total = float(d_ab[vm].sum().item()) if n_valid > 0 else 0.0

                # Fetch sample refs + antenna + city type + los_mask from HDF5.
                city = sample_ref = ""
                ant_h = float("nan")
                if hasattr(dataset, "sample_refs") and idx < len(dataset.sample_refs):
                    city, sample_ref = dataset.sample_refs[idx]
                if scalar_cond is not None and scalar_cond.numel() > 0:
                    cols = list(cfg["data"].get("scalar_feature_columns", []))
                    if "antenna_height_m" in cols:
                        j = cols.index("antenna_height_m")
                        norm = float(cfg["data"].get("scalar_feature_norms", {}).get("antenna_height_m", 1.0))
                        ant_h = float(scalar_cond[off, j].item()) * norm
                los_map = None
                city_type = ""
                try:
                    h = dataset._get_handle()
                    grp = h[city][sample_ref]
                    if not np.isfinite(ant_h) and "uav_height" in grp:
                        arr = np.asarray(grp["uav_height"][...], dtype=np.float64).reshape(-1)
                        if arr.size:
                            ant_h = float(arr[0])
                    if "los_mask" in grp:
                        los_arr = np.asarray(grp["los_mask"][...], dtype=np.float32)
                        if los_arr.shape == tuple(vm.shape):
                            los_map = torch.from_numpy(los_arr).to(vm.device)
                    if "topology_map" in grp:
                        topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
                        city_type = infer_city_type(topo)
                except Exception:
                    pass

                # Per-sample LoS/NLoS split using los_mask from HDF5 (reliable even if expert masks out).
                los_rmse_s = nlos_rmse_s = float("nan")
                los_frac = float("nan")
                if los_map is not None:
                    los_m = vm & (los_map > 0.5)
                    nlos_m = vm & (los_map <= 0.5)
                    lc, nc = int(los_m.sum().item()), int(nlos_m.sum().item())
                    if lc > 0:
                        los_rmse_s = float(torch.sqrt((d_sq[los_m]).mean()).item())
                    if nc > 0:
                        nlos_rmse_s = float(torch.sqrt((d_sq[nlos_m]).mean()).item())
                    if n_valid > 0:
                        los_frac = lc / n_valid

                # Ring metrics on LoS pixels (cheap; skips if too few LoS px).
                ring = {}
                if los_map is not None:
                    ring = compute_ring_metrics_db(
                        pred_phys[off, 0].float(),
                        target_phys[off, 0].float(),
                        los_mask=(los_map > 0.5),
                        valid_mask=vm,
                        sigma_px=3.0,
                    )

                rows.append({
                    "idx": idx,
                    "city": city,
                    "sample": sample_ref,
                    "city_type": city_type,
                    "antenna_height_m": ant_h,
                    "valid_pixels": n_valid,
                    "overall_rmse_db": compute_rmse(sse_total, n_valid),
                    "overall_mae_db": compute_mae(sae_total, n_valid),
                    "los_rmse_db": los_rmse_s,
                    "nlos_rmse_db": nlos_rmse_s,
                    "los_fraction": los_frac,
                    "ring_los_px": ring.get("pixels_los", 0),
                    "ring_rmse_highpass_db": ring.get("rmse_highpass_db", float("nan")),
                    "ring_rmse_smooth_db": ring.get("rmse_smooth_db", float("nan")),
                    "ring_amp_ratio": ring.get("ring_amp_ratio_pred_over_target", float("nan")),
                    "ring_tgt_std_db": ring.get("target_highpass_std_db", float("nan")),
                    "ring_pred_std_db": ring.get("pred_highpass_std_db", float("nan")),
                    "ring_rmse_radial_profile_db": ring.get("rmse_radial_profile_db", float("nan")),
                    "ring_rmse_radial_deriv_db": ring.get("rmse_radial_deriv_db", float("nan")),
                })

                # --- Zone accumulation ---
                def accumulate(zone_name: str, mask_zone: torch.Tensor) -> None:
                    mm = mask_zone & vm
                    cnt = int(mm.sum().item())
                    if cnt <= 0:
                        return
                    zone_totals[zone_name]["sse"] += float(d_sq[mm].sum().item())
                    zone_totals[zone_name]["sae"] += float(d_ab[mm].sum().item())
                    zone_totals[zone_name]["count"] += cnt

                # overall + city_type
                accumulate("overall", torch.ones_like(vm, dtype=torch.bool))
                if city_type:
                    accumulate(f"city_type/{city_type}", torch.ones_like(vm, dtype=torch.bool))

                # LoS/NLoS side (if los_map available)
                has_los = "los" in ch
                if los_map is not None:
                    los_m = los_map > 0.5
                    nlos_m = ~los_m
                    accumulate("los_region/LoS", los_m)
                    accumulate("los_region/NLoS", nlos_m)

                # Zones from obstruction features (only meaningful on NLoS; but we emit them unconditionally
                # and masks naturally restrict to pixels with valid obstruction > 0).
                inp = x[off]  # (C,H,W)
                if "dist_since_break" in ch:
                    bkt = bucket_distance_since_break(inp[ch["dist_since_break"]])
                    for i, name in enumerate(DIST_BREAK_NAMES):
                        accumulate(f"dist_since_break/{name}", bkt == i)
                        if los_map is not None:
                            accumulate(f"nlos_dist_since_break/{name}", (bkt == i) & (los_map <= 0.5))

                if "blocker_count" in ch:
                    bkt = bucket_blocker_count(inp[ch["blocker_count"]])
                    for i, name in enumerate(BLOCKER_COUNT_NAMES):
                        accumulate(f"blocker_count/{name}", bkt == i)
                        if los_map is not None:
                            accumulate(f"nlos_blocker_count/{name}", (bkt == i) & (los_map <= 0.5))

                if "distance" in ch:
                    bkt = bucket_distance_from_tx(inp[ch["distance"]])
                    for i, name in enumerate(DIST_TX_NAMES):
                        accumulate(f"dist_from_tx/{name}", bkt == i)
                        if los_map is not None:
                            accumulate(f"nlos_dist_from_tx/{name}", (bkt == i) & (los_map <= 0.5))
                            accumulate(f"los_dist_from_tx/{name}", (bkt == i) & (los_map > 0.5))

                if "elev_angle" in ch:
                    bkt = bucket_elev_angle(inp[ch["elev_angle"]])
                    for i, name in enumerate(ELEV_NAMES):
                        accumulate(f"elev_angle/{name}", bkt == i)
                        if los_map is not None:
                            accumulate(f"nlos_elev_angle/{name}", (bkt == i) & (los_map <= 0.5))

                processed += 1
                if args.max_samples and processed >= args.max_samples:
                    break
            cursor += B
            if args.max_samples and processed >= args.max_samples:
                break
            if processed % 20 == 0:
                print(f"  processed {processed}/{len(dataset)}")

    # Write per-sample CSV
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    samples_csv = out.with_suffix(".samples.csv")
    with samples_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Zones CSV
    zones_csv = out.with_suffix(".zones.csv")
    with zones_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zone", "pixels", "rmse_db", "mae_db"])
        for key in sorted(zone_totals):
            t = zone_totals[key]
            w.writerow([key, t["count"], f"{compute_rmse(t['sse'], t['count']):.3f}", f"{compute_mae(t['sae'], t['count']):.3f}"])

    # Summary
    def agg_mean(key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    overall_t = zone_totals["overall"]
    summary = {
        "samples": len(rows),
        "overall_rmse_db_pooled": compute_rmse(overall_t["sse"], overall_t["count"]),
        "overall_mae_db_pooled": compute_mae(overall_t["sae"], overall_t["count"]),
        "mean_overall_rmse_db": agg_mean("overall_rmse_db"),
        "mean_los_rmse_db": agg_mean("los_rmse_db"),
        "mean_nlos_rmse_db": agg_mean("nlos_rmse_db"),
        "ring_mean_rmse_highpass_db": agg_mean("ring_rmse_highpass_db"),
        "ring_mean_rmse_smooth_db": agg_mean("ring_rmse_smooth_db"),
        "ring_mean_amp_ratio": agg_mean("ring_amp_ratio"),
        "ring_mean_target_std_db": agg_mean("ring_tgt_std_db"),
        "ring_mean_pred_std_db": agg_mean("ring_pred_std_db"),
        "ring_mean_rmse_radial_profile_db": agg_mean("ring_rmse_radial_profile_db"),
        "ring_mean_rmse_radial_deriv_db": agg_mean("ring_rmse_radial_deriv_db"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split": args.split,
    }
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))

    print("\n=== summary ===")
    print(f"samples: {len(rows)}")
    print(f"overall RMSE (pooled): {summary['overall_rmse_db_pooled']:.2f}")
    print(f"mean per-sample overall RMSE: {summary['mean_overall_rmse_db']:.2f}")
    print(f"mean per-sample LoS RMSE: {summary['mean_los_rmse_db']:.2f}")
    print(f"mean per-sample NLoS RMSE: {summary['mean_nlos_rmse_db']:.2f}")
    print("\n--- LoS ring (2-ray) metrics (mean per-sample) ---")
    print(f"  rmse_highpass (ring amp error): {summary['ring_mean_rmse_highpass_db']:.2f} dB")
    print(f"  rmse_smooth   (envelope error): {summary['ring_mean_rmse_smooth_db']:.2f} dB")
    print(f"  target highpass std:             {summary['ring_mean_target_std_db']:.2f} dB")
    print(f"  pred   highpass std:             {summary['ring_mean_pred_std_db']:.2f} dB")
    print(f"  amp ratio (pred/tgt):            {summary['ring_mean_amp_ratio']:.2f}")
    print(f"  radial profile RMSE:             {summary['ring_mean_rmse_radial_profile_db']:.2f} dB")
    print(f"  radial derivative RMSE:          {summary['ring_mean_rmse_radial_deriv_db']:.2f} dB")

    print("\n--- zones (key, pixels, rmse_db, mae_db) ---")
    for key in sorted(zone_totals):
        t = zone_totals[key]
        print(f"  {key:<40s}  n={t['count']:>10d}  rmse={compute_rmse(t['sse'], t['count']):>6.2f}  mae={compute_mae(t['sae'], t['count']):>6.2f}")

    print(f"\nwrote {samples_csv}\nwrote {zones_csv}\nwrote {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
