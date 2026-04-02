#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent


@dataclass
class LinearStats:
    dim: int

    def __post_init__(self) -> None:
        self.xty = np.zeros(self.dim, dtype=np.float64)
        self.xtx = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.size == 0:
            return
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.count += int(x.shape[0])

    def solve(self, ridge: float) -> np.ndarray:
        if self.count < max(self.dim * 4, 128):
            return np.zeros(self.dim, dtype=np.float64)
        reg = np.eye(self.dim, dtype=np.float64) * float(ridge)
        reg[-1, -1] = 0.0
        try:
            return np.linalg.solve(self.xtx + reg, self.xty)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(self.xtx + reg, self.xty, rcond=None)[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit a train-only obstruction-aware formula calibration for path_loss.")
    p.add_argument("--try-dir", required=True, help="Path to a try folder whose config/data pipeline should be reused")
    p.add_argument("--config", required=True, help="Config path relative to try-dir or absolute")
    p.add_argument("--dataset", default="", help="Optional HDF5 override")
    p.add_argument(
        "--base-calibration-json",
        default=str(PRACTICE_ROOT / "TFGFortySecondTry42" / "prior_calibration" / "regime_quadratic_train_only.json"),
        help="Existing train-only regime calibration used for city-type and antenna thresholds",
    )
    p.add_argument(
        "--out-json",
        default=str(PRACTICE_ROOT / "TFGFortyFifthTry45" / "prior_calibration" / "regime_obstruction_train_only.json"),
    )
    p.add_argument(
        "--out-md",
        default=str(PRACTICE_ROOT / "TFGFortyFifthTry45" / "prior_calibration" / "regime_obstruction_train_only.md"),
    )
    p.add_argument("--local-kernel-sizes", type=int, nargs="+", default=[15, 41])
    p.add_argument("--sample-prob", type=float, default=0.02)
    p.add_argument("--train-sample-frac", type=float, default=1.0, help="Fraction of train samples to scan")
    p.add_argument("--val-sample-frac", type=float, default=1.0, help="Fraction of val samples to scan")
    p.add_argument("--log-every", type=int, default=100, help="Progress log interval in samples")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="cpu/cuda/directml")
    p.add_argument("--formula", default="hybrid_two_ray_cost231_a2g_nlos")
    p.add_argument("--a2g-params-json", default="")
    return p.parse_args()


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _formula_channel_index(cfg: Dict[str, Any]) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):
        idx += 1
    if cfg["data"].get("distance_map_channel", False):
        idx += 1
    return idx


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_device(name: str) -> torch.device | object:
    dev = str(name).lower()
    if dev == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for calibration but is not available.")
        return torch.device("cuda")
    if dev == "directml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    return torch.device("cpu")


def _denormalize(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    return values * scale + offset


def _city_type_from_thresholds(density: float, height: float, thresholds: Dict[str, float]) -> str:
    density_q1 = float(thresholds.get("density_q1", 0.0))
    density_q2 = float(thresholds.get("density_q2", 1.0))
    height_q1 = float(thresholds.get("height_q1", 0.0))
    height_q2 = float(thresholds.get("height_q2", 1.0))
    if density >= density_q2 or height >= height_q2:
        return "dense_highrise"
    if density <= density_q1 and height <= height_q1:
        return "open_lowrise"
    return "mixed_midrise"


def _antenna_height_bin(antenna_height_m: float, thresholds: Dict[str, float]) -> str:
    q1 = float(thresholds.get("q1", 0.0))
    q2 = float(thresholds.get("q2", q1))
    if antenna_height_m <= q1:
        return "low_ant"
    if antenna_height_m <= q2:
        return "mid_ant"
    return "high_ant"


def _compute_local_features(topology_norm: torch.Tensor, kernel_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if topology_norm.ndim != 2:
        raise ValueError(f"Expected [H,W], got {tuple(topology_norm.shape)}")
    topo = topology_norm.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    building = (topo > 0.0).to(dtype=torch.float32)
    kernel = max(int(kernel_size), 1)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    density = F.avg_pool2d(building, kernel, stride=1, padding=pad).squeeze(0).squeeze(0)
    mean_height = F.avg_pool2d(topo * 255.0, kernel, stride=1, padding=pad).squeeze(0).squeeze(0)
    return density, mean_height


def _feature_matrix(
    prior_db: torch.Tensor,
    distance_norm: torch.Tensor,
    local_density_small: torch.Tensor,
    local_density_large: torch.Tensor,
    local_height_small: torch.Tensor,
    local_height_large: torch.Tensor,
    nlos_support_small: torch.Tensor,
    nlos_support_large: torch.Tensor,
    shadow_sigma_db: torch.Tensor,
    theta_norm: torch.Tensor,
) -> torch.Tensor:
    distance_scale_m = 256.0 * math.sqrt(2.0)
    logd = torch.log1p(distance_norm * distance_scale_m)
    feats = torch.stack(
        [
            prior_db * prior_db,
            prior_db,
            logd,
            local_density_small,
            local_density_large,
            local_height_small / 255.0,
            local_height_large / 255.0,
            local_density_large * logd,
            nlos_support_small,
            nlos_support_large,
            nlos_support_large * logd,
            shadow_sigma_db,
            theta_norm,
            nlos_support_large * theta_norm,
            torch.ones_like(prior_db),
        ],
        dim=-1,
    )
    return feats


def _select_indices(num_items: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    if num_items <= 0:
        return np.zeros((0,), dtype=np.int64)
    frac = float(frac)
    if frac >= 1.0:
        return np.arange(num_items, dtype=np.int64)
    keep = max(1, int(round(num_items * max(frac, 0.0))))
    return np.sort(rng.choice(num_items, size=keep, replace=False).astype(np.int64))


def _sigma_map(distance_norm: torch.Tensor, antenna_height_m: float, los_map: torch.Tensor, receiver_height_m: float = 1.5) -> torch.Tensor:
    distance_scale_m = 256.0 * math.sqrt(2.0)
    ground_distance_m = distance_norm * distance_scale_m
    theta_deg = torch.rad2deg(
        torch.atan2((antenna_height_m - receiver_height_m) * torch.ones_like(ground_distance_m), ground_distance_m.clamp(min=1.0))
    )
    sigma_los = 0.0272 * torch.pow((90.0 - theta_deg).clamp(min=0.0), 0.7475)
    sigma_nlos = 2.3197 * torch.pow((90.0 - theta_deg).clamp(min=0.0), 0.2361)
    return torch.where(los_map > 0.5, sigma_los, sigma_nlos)


def _theta_norm_map(distance_norm: torch.Tensor, antenna_height_m: float, receiver_height_m: float = 1.5) -> torch.Tensor:
    distance_scale_m = 256.0 * math.sqrt(2.0)
    ground_distance_m = distance_norm * distance_scale_m
    theta_deg = torch.rad2deg(
        torch.atan2((antenna_height_m - receiver_height_m) * torch.ones_like(ground_distance_m), ground_distance_m.clamp(min=1.0))
    )
    return (theta_deg / 90.0).clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = _resolve_device(args.device)
    started = time.time()

    try_dir = _resolve(PRACTICE_ROOT, args.try_dir).resolve()
    config_path = _resolve(try_dir, args.config).resolve()
    sys.path.insert(0, str(try_dir))

    config_utils = importlib.import_module("config_utils")
    data_utils = importlib.import_module("data_utils")
    cfg = config_utils.load_config(str(config_path))
    if args.dataset:
        cfg["data"]["hdf5_path"] = str(Path(args.dataset).resolve())
    cfg["augmentation"] = dict(cfg.get("augmentation", {}))
    cfg["augmentation"]["enable"] = False
    cfg["data"] = dict(cfg.get("data", {}))
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    formula_cfg["enabled"] = True
    formula_cfg["formula"] = str(args.formula)
    formula_cfg["regime_calibration_json"] = None
    if args.a2g_params_json:
        formula_cfg["a2g_params"] = json.loads(Path(args.a2g_params_json).read_text(encoding="utf-8"))
    cfg["data"]["path_loss_formula_input"] = formula_cfg

    splits = data_utils.build_dataset_splits_from_config(cfg)
    train_ds = splits["train"]
    val_ds = splits["val"]
    target_meta = dict(cfg["target_metadata"]["path_loss"])
    formula_idx = _formula_channel_index(cfg)
    los_idx = 1 if cfg["data"].get("los_input_column") else None
    dist_idx = formula_idx - 1 if cfg["data"].get("distance_map_channel", False) else None

    base_cal = json.loads(Path(args.base_calibration_json).read_text(encoding="utf-8"))
    city_type_by_city = dict(base_cal.get("city_type_by_city", {}))
    city_type_thresholds = dict(base_cal.get("city_type_thresholds", {}))
    ant_thresholds = dict(base_cal.get("antenna_height_thresholds", {}))

    stats: Dict[str, LinearStats] = {}

    def key_for(city_type: str, los_label: str, ant_bin: str) -> str:
        return f"{city_type}|{los_label}|{ant_bin}"

    train_indices = _select_indices(len(train_ds.sample_refs), float(args.train_sample_frac), rng)
    val_indices = _select_indices(len(val_ds.sample_refs), float(args.val_sample_frac), rng)

    print(
        f"[calibration] device={args.device} train_samples={len(train_indices)}/{len(train_ds.sample_refs)} "
        f"val_samples={len(val_indices)}/{len(val_ds.sample_refs)} pixel_sample_prob={args.sample_prob}",
        flush=True,
    )

    def update_split(dataset: Any, train_mode: bool, selected_indices: np.ndarray, split_name: str) -> None:
        total = len(selected_indices)
        for pos, idx in enumerate(selected_indices, start=1):
            city, sample = dataset.sample_refs[int(idx)]
            item = dataset[idx]
            x = item[0].to(device)
            y = item[1].to(device)
            m = item[2].to(device)
            topology_norm = x[0]
            prior_db = _denormalize(x[formula_idx], target_meta)
            target_db = _denormalize(y[0], target_meta)
            valid = m[0] > 0.0
            if not torch.any(valid):
                continue
            los_map = x[los_idx] if los_idx is not None else torch.ones_like(prior_db)
            distance_map = x[dist_idx] if dist_idx is not None else data_utils._compute_distance_map_2d(int(prior_db.shape[-1])).squeeze(0)
            local_density_small, local_height_small = _compute_local_features(topology_norm, int(args.local_kernel_sizes[0]))
            local_density_large, local_height_large = _compute_local_features(topology_norm, int(args.local_kernel_sizes[-1]))
            nlos_support_small = _compute_local_features((los_map <= 0.5).to(torch.float32), int(args.local_kernel_sizes[0]))[0]
            nlos_support_large = _compute_local_features((los_map <= 0.5).to(torch.float32), int(args.local_kernel_sizes[-1]))[0]
            antenna_height_m = float(dataset._resolve_hdf5_scalar_value(city, sample, "antenna_height_m"))
            shadow_sigma_db = _sigma_map(distance_map, antenna_height_m, los_map)
            theta_norm = _theta_norm_map(distance_map, antenna_height_m)
            feature_map = _feature_matrix(
                prior_db,
                distance_map,
                local_density_small,
                local_density_large,
                local_height_small,
                local_height_large,
                nlos_support_small,
                nlos_support_large,
                shadow_sigma_db,
                theta_norm,
            )
            non_ground = topology_norm.cpu().numpy() > 0.0
            density = float(np.mean(non_ground))
            nz = topology_norm.cpu().numpy()[non_ground]
            mean_height = float(np.mean(nz) * 255.0) if nz.size else 0.0
            city_type = city_type_by_city.get(city)
            if city_type is None:
                city_type = _city_type_from_thresholds(density, mean_height, city_type_thresholds)
            ant_bin = _antenna_height_bin(antenna_height_m, ant_thresholds)

            for los_label, los_mask in [("LoS", los_map > 0.5), ("NLoS", los_map <= 0.5)]:
                pix_mask = valid & los_mask
                if not torch.any(pix_mask):
                    continue
                feats = feature_map[pix_mask].cpu().numpy().astype(np.float64, copy=False)
                tgt = target_db[pix_mask].cpu().numpy().astype(np.float64, copy=False)
                if train_mode:
                    keep = rng.random(feats.shape[0]) < float(args.sample_prob)
                    if not np.any(keep):
                        continue
                    feats = feats[keep]
                    tgt = tgt[keep]
                    key = key_for(city_type, los_label, ant_bin)
                    if key not in stats:
                        stats[key] = LinearStats(dim=feats.shape[1])
                    stats[key].update(feats, tgt)

            if pos == 1 or pos % max(int(args.log_every), 1) == 0 or pos == total:
                elapsed = time.time() - started
                print(
                    f"[calibration:{split_name}] {pos}/{total} samples processed "
                    f"elapsed={elapsed/60.0:.1f} min",
                    flush=True,
                )

    update_split(train_ds, True, train_indices, "train")

    coeffs: Dict[str, Dict[str, Any]] = {}
    for key, ls in stats.items():
        weights = ls.solve(float(args.ridge))
        coeffs[key] = {
            "weights": [float(v) for v in weights[:-1]],
            "bias": float(weights[-1]),
            "count": int(ls.count),
            "distance_scale_m": float(256.0 * math.sqrt(2.0)),
            "height_scale": 255.0,
            "meters_per_pixel": float(formula_cfg.get("meters_per_pixel", 1.0)),
            "receiver_height_m": float(formula_cfg.get("receiver_height_m", 1.5)),
        }

    # Train-only evaluation on val split with the fitted coefficients.
    results = {
        "overall": {"count": 0, "sq": 0.0, "abs": 0.0},
        "LoS": {"count": 0, "sq": 0.0, "abs": 0.0},
        "NLoS": {"count": 0, "sq": 0.0, "abs": 0.0},
    }

    def apply_coeff_map(city_type: str, ant_bin: str, los_map: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        pred_los = torch.zeros_like(los_map, dtype=torch.float32)
        pred_nlos = torch.zeros_like(los_map, dtype=torch.float32)
        for los_label, out in [("LoS", pred_los), ("NLoS", pred_nlos)]:
            key = key_for(city_type, los_label, ant_bin)
            payload = coeffs.get(key)
            if payload is None:
                fallback = base_cal["coefficients"].get(key)
                if fallback:
                    poly = [float(v) for v in fallback.get("poly2", [0.0, 1.0, 0.0])]
                    prior = feature_map[..., 1]
                    out.copy_(poly[0] * prior * prior + poly[1] * prior + poly[2])
                continue
            weights = torch.tensor(payload["weights"], dtype=torch.float32, device=feature_map.device)
            bias = float(payload["bias"])
            out.copy_(torch.tensordot(feature_map[..., : weights.numel()], weights, dims=([-1], [0])) + bias)
        return torch.where(los_map > 0.5, pred_los, pred_nlos)

    total_val = len(val_indices)
    for pos, idx in enumerate(val_indices, start=1):
        city, sample = val_ds.sample_refs[int(idx)]
        item = val_ds[idx]
        x = item[0].to(device)
        y = item[1].to(device)
        m = item[2].to(device)
        topology_norm = x[0]
        prior_db = _denormalize(x[formula_idx], target_meta)
        target_db = _denormalize(y[0], target_meta)
        valid = m[0] > 0.0
        if not torch.any(valid):
            continue
        los_map = x[los_idx] if los_idx is not None else torch.ones_like(prior_db)
        distance_map = x[dist_idx] if dist_idx is not None else data_utils._compute_distance_map_2d(int(prior_db.shape[-1])).squeeze(0)
        local_density_small, local_height_small = _compute_local_features(topology_norm, int(args.local_kernel_sizes[0]))
        local_density_large, local_height_large = _compute_local_features(topology_norm, int(args.local_kernel_sizes[-1]))
        nlos_support_small = _compute_local_features((los_map <= 0.5).to(torch.float32), int(args.local_kernel_sizes[0]))[0]
        nlos_support_large = _compute_local_features((los_map <= 0.5).to(torch.float32), int(args.local_kernel_sizes[-1]))[0]
        antenna_height_m = float(val_ds._resolve_hdf5_scalar_value(city, sample, "antenna_height_m"))
        shadow_sigma_db = _sigma_map(distance_map, antenna_height_m, los_map)
        theta_norm = _theta_norm_map(distance_map, antenna_height_m)
        feature_map = _feature_matrix(
            prior_db,
            distance_map,
            local_density_small,
            local_density_large,
            local_height_small,
            local_height_large,
            nlos_support_small,
            nlos_support_large,
            shadow_sigma_db,
            theta_norm,
        )
        non_ground = topology_norm.cpu().numpy() > 0.0
        density = float(np.mean(non_ground))
        nz = topology_norm.cpu().numpy()[non_ground]
        mean_height = float(np.mean(nz) * 255.0) if nz.size else 0.0
        city_type = city_type_by_city.get(city)
        if city_type is None:
            city_type = _city_type_from_thresholds(density, mean_height, city_type_thresholds)
        ant_bin = _antenna_height_bin(antenna_height_m, ant_thresholds)

        pred = apply_coeff_map(city_type, ant_bin, los_map, feature_map)
        diff = (pred - target_db)[valid]
        results["overall"]["count"] += int(diff.numel())
        results["overall"]["sq"] += float(torch.sum(diff * diff).item())
        results["overall"]["abs"] += float(torch.sum(torch.abs(diff)).item())
        for label, mask in [("LoS", valid & (los_map > 0.5)), ("NLoS", valid & (los_map <= 0.5))]:
            if torch.any(mask):
                sub = (pred - target_db)[mask]
                results[label]["count"] += int(sub.numel())
                results[label]["sq"] += float(torch.sum(sub * sub).item())
                results[label]["abs"] += float(torch.sum(torch.abs(sub)).item())

        if pos == 1 or pos % max(int(args.log_every), 1) == 0 or pos == total_val:
            elapsed = time.time() - started
            print(
                f"[calibration:val] {pos}/{total_val} samples processed "
                f"elapsed={elapsed/60.0:.1f} min",
                flush=True,
            )

    def finalize(stats_dict: Dict[str, float]) -> Dict[str, float]:
        count = max(int(stats_dict["count"]), 1)
        mse = float(stats_dict["sq"] / count)
        return {
            "count": int(stats_dict["count"]),
            "rmse_db": float(math.sqrt(mse)),
            "mae_db": float(stats_dict["abs"] / count),
        }

    payload = {
        "model_type": "regime_obstruction_multiscale_v1",
        "dataset": str(cfg["data"]["hdf5_path"]),
        "split_seed": int(cfg["data"].get("split_seed", 42)),
        "official_metric_mask": "topology == 0 and dataset mask > 0",
        "city_type_thresholds": city_type_thresholds,
        "antenna_height_thresholds": ant_thresholds,
        "city_type_by_city": city_type_by_city,
        "local_kernel_sizes": [int(v) for v in args.local_kernel_sizes],
        "train_sample_frac": float(args.train_sample_frac),
        "val_sample_frac": float(args.val_sample_frac),
        "pixel_sample_prob": float(args.sample_prob),
        "formula": str(args.formula),
        "a2g_params_json": str(args.a2g_params_json) if args.a2g_params_json else None,
        "feature_order": [
            "prior_db_squared",
            "prior_db",
            "log1p_distance_m",
            "local_building_density_small",
            "local_building_density_large",
            "local_mean_height_small_norm",
            "local_mean_height_large_norm",
            "local_building_density_large_x_log1p_distance_m",
            "nlos_support_small",
            "nlos_support_large",
            "nlos_support_large_x_log1p_distance_m",
            "shadow_sigma_db",
            "theta_norm",
            "nlos_support_large_x_theta_norm",
            "bias",
        ],
        "coefficients": coeffs,
        "val_results": {k: finalize(v) for k, v in results.items()},
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    _ensure_parent(out_json)
    _ensure_parent(out_md)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Try 45 prior-only obstruction-aware calibration",
        "",
        "This calibration is fitted only on the training split and evaluated only on the validation split.",
        "",
        "## Design",
        "",
        f"- Base formula: `{args.formula}`.",
        "- Regimes: `city_type × LoS/NLoS × antenna-height bin`.",
        "- Extra NLoS-aware pixel features:",
        "  - `log(1 + distance)`",
        "  - local building occupancy density at a near scale",
        "  - local building occupancy density at a broader scale",
        "  - local mean building-height proxy at both scales",
        "  - broad-scale occupancy-distance interaction",
        "  - local NLoS support at near and broad scales",
        "  - `Eq. 12`-style shadow-sigma feature",
        "",
        "The goal is to strengthen the prior specifically where Try 42 still fails: NLoS, lower antennas, and denser urban morphologies.",
        "",
        "## Validation RMSE",
        "",
    ]
    for key, value in payload["val_results"].items():
        md_lines.append(f"- `{key}`: RMSE `{value['rmse_db']:.4f} dB`, MAE `{value['mae_db']:.4f} dB`, count `{value['count']}`")
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append("- The official metric mask ignores buildings (`topology != 0`).")
    md_lines.append("- No city ID regression is used beyond the train-defined city-type grouping already present in the previous calibration.")
    md_lines.append("- This keeps the prior more structured while still aiming to generalize to unseen cities and a new dataset.")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["val_results"], indent=2))


if __name__ == "__main__":
    main()
