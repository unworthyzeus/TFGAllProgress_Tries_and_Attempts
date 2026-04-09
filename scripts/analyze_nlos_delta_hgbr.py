#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tabular NLoS delta regression on top of frozen LoS prior.")
    p.add_argument("--try-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default="")
    p.add_argument("--sample-frac", type=float, default=0.01)
    p.add_argument("--val-sample-frac", type=float, default=-1.0)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--train-pixels-per-sample", type=int, default=2048)
    p.add_argument("--specialist-min-rows", type=int, default=30000)
    p.add_argument("--delta-clip-max", type=float, default=30.0)
    p.add_argument("--baseline-mode", choices=["modern", "old_exact"], default="modern")
    p.add_argument("--use-torch-mlp", action="store_true")
    p.add_argument("--mlp-max-rows", type=int, default=300000)
    p.add_argument(
        "--old-calibration-json",
        default=str(PRACTICE_ROOT / "TFGFiftiethTry50" / "prior_calibration" / "regime_obstruction_train_only_from_try47.json"),
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _resolve_device(device_name: str) -> torch.device:
    name = str(device_name).lower()
    if name == "directml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _subset_sample_refs(sample_refs: List[Tuple[str, str]], sample_frac: float, seed: int) -> List[Tuple[str, str]]:
    frac = float(sample_frac)
    if frac >= 1.0:
        return list(sample_refs)
    if frac <= 0.0:
        raise ValueError("--sample-frac must be in (0,1].")
    count = max(1, int(round(len(sample_refs) * frac)))
    rng = random.Random(int(seed))
    subset = list(sample_refs)
    rng.shuffle(subset)
    subset = subset[:count]
    subset.sort()
    return subset


def _nearest_resize(arr: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def _ground_mask_from_hdf5(handle: h5py.File, city: str, sample: str, image_size: int) -> np.ndarray:
    topo = np.asarray(handle[city][sample]["topology_map"][...], dtype=np.float32)
    topo_resized = _nearest_resize(topo, image_size)
    return np.isclose(topo_resized, 0.0)


def _sample_metadata(handle: h5py.File, city: str, sample: str) -> Tuple[float, float, float]:
    grp = handle[city][sample]
    topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
    non_zero = topo[topo != 0]
    density = float(np.mean(topo != 0))
    height = float(np.mean(non_zero)) if non_zero.size else 0.0
    ant = float(np.asarray(grp["uav_height"][...], dtype=np.float32).reshape(-1)[0])
    return density, height, ant


def _city_type_for_stats(density: float, height: float, dens_q1: float, dens_q2: float, h_q1: float, h_q2: float) -> str:
    if density >= dens_q2 or height >= h_q2:
        return "dense_highrise"
    if density <= dens_q1 and height <= h_q1:
        return "open_lowrise"
    return "mixed_midrise"


def _ant_bin(ant: float, q1: float, q2: float) -> str:
    if ant <= q1:
        return "low_ant"
    if ant <= q2:
        return "mid_ant"
    return "high_ant"


def _quantile_edges(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, 1.0 / 3.0)), float(np.quantile(arr, 2.0 / 3.0))


def _formula_channel_index(cfg: Dict[str, Any]) -> int:
    idx = 1
    if cfg.get("data", {}).get("los_input_column"):
        idx += 1
    if bool(cfg.get("data", {}).get("distance_map_channel", False)):
        idx += 1
    return idx


def _input_channel_layout(cfg: Dict[str, Any]) -> Dict[str, int]:
    idx = 0
    layout: Dict[str, int] = {"topology": idx}
    idx += 1
    if cfg.get("data", {}).get("los_input_column"):
        layout["los"] = idx
        idx += 1
    if bool(cfg.get("data", {}).get("distance_map_channel", False)):
        layout["distance"] = idx
        idx += 1
    formula_cfg = dict(cfg.get("data", {}).get("path_loss_formula_input", {}))
    if bool(formula_cfg.get("enabled", False)):
        layout["formula"] = idx
        idx += 1
        if bool(formula_cfg.get("include_confidence_channel", False)):
            layout["formula_confidence"] = idx
            idx += 1
    obstruction_cfg = dict(cfg.get("data", {}).get("path_loss_obstruction_features", {}))
    if bool(obstruction_cfg.get("enabled", False)):
        if bool(obstruction_cfg.get("include_shadow_depth", True)):
            layout["shadow_depth"] = idx
            idx += 1
        if bool(obstruction_cfg.get("include_distance_since_los_break", True)):
            layout["distance_since_los_break"] = idx
            idx += 1
        if bool(obstruction_cfg.get("include_max_blocker_height", True)):
            layout["max_blocker_height"] = idx
            idx += 1
        if bool(obstruction_cfg.get("include_blocker_count", True)):
            layout["blocker_count"] = idx
            idx += 1
    return layout


def _denormalize_channel(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    if abs(scale) < 1e-12:
        scale = 1.0
    return values * scale + offset


def _ensure_batch_dims(
    x: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    sc: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if y.ndim == 3:
        y = y.unsqueeze(0)
    if m.ndim == 3:
        m = m.unsqueeze(0)
    if sc is not None and sc.ndim == 1:
        sc = sc.unsqueeze(0)
    return x, y, m, sc


def _compute_los_reference_db(
    data_utils: Any,
    *,
    image_size: int,
    antenna_height_m: float,
    receiver_height_m: float,
    frequency_ghz: float,
    meters_per_pixel: float,
    a2g_params: Dict[str, Any],
) -> np.ndarray:
    _ = meters_per_pixel
    d2d = data_utils._compute_distance_map_2d(image_size)
    distance_scale_m = 256.0 * math.sqrt(2.0)
    d2d_m = d2d * float(distance_scale_m)
    coherent_db, _ = data_utils._coherent_two_ray_components_db(
        d2d_m,
        float(antenna_height_m),
        float(receiver_height_m),
        float(frequency_ghz),
        eps_r=float(a2g_params.get("ground_eps_r", 5.0)),
        roughness_m=float(a2g_params.get("ground_roughness_m", 0.01)),
    )
    smooth_los_db = data_utils._damped_coherent_two_ray_path_loss_db(
        d2d_m,
        float(antenna_height_m),
        float(receiver_height_m),
        float(frequency_ghz),
        eps_r=float(a2g_params.get("ground_eps_r", 5.0)),
        roughness_m=float(a2g_params.get("ground_roughness_m", 0.01)),
        excess_limit_db=float(a2g_params.get("interference_excess_limit_db", 10.5)),
        interference_decay_m=float(a2g_params.get("interference_decay_m", 900.0)),
        min_interference_blend=float(a2g_params.get("min_interference_blend", 0.35)),
    )
    los_reference_db = smooth_los_db + data_utils._compute_shadowed_ripple_db(
        coherent_db,
        smooth_los_db,
        None,
        None,
        None,
        a2g_params={
            **dict(a2g_params),
            "ripple_gain_los": float(a2g_params.get("ripple_gain_los", 0.95)),
        },
    )
    return np.asarray(los_reference_db.detach().cpu().numpy()[0], dtype=np.float64)


def _write_markdown(out_path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# NLoS Delta HGBoost",
        "",
        f"- Try folder: `{payload['try_dir']}`",
        f"- Config: `{payload['config']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Train samples: `{payload['train_samples']}`",
        f"- Val samples: `{payload['val_samples']}`",
        f"- Train NLoS pixels used: `{payload['train_nlos_pixels']}`",
        "",
        "## Metrics",
        "",
        f"- Overall RMSE: `{payload['metrics']['overall']['rmse_db']:.4f} dB`",
        f"- LoS RMSE: `{payload['metrics']['LoS']['rmse_db']:.4f} dB`",
        f"- NLoS RMSE: `{payload['metrics']['NLoS']['rmse_db']:.4f} dB`",
        "",
        "## Model",
        "",
        f"- Train R^2: `{payload['model']['train_r2']:.4f}`",
        f"- Max depth: `{payload['model']['max_depth']}`",
        f"- Learning rate: `{payload['model']['learning_rate']}`",
        f"- Max iter: `{payload['model']['max_iter']}`",
        "",
        "## Features",
        "",
    ]
    for name, imp in payload["feature_importance"].items():
        lines.append(f"- `{name}`: `{imp:.6f}`")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_old_calibrated_prior_db(
    data_utils: Any,
    calibration: Dict[str, Any] | None,
    *,
    prior_db_np: np.ndarray,
    city: str,
    density: float,
    height: float,
    antenna_height_m: float,
    los_np: np.ndarray,
    topology_np: np.ndarray,
    distance_np: np.ndarray,
    non_ground_threshold: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    path_meta = {"clip_min": float(clip_min), "clip_max": float(clip_max)}
    old_formula_prior = data_utils._compute_formula_path_loss_db(
        image_size=int(prior_db_np.shape[-1]),
        antenna_height_m=float(antenna_height_m),
        receiver_height_m=1.5,
        frequency_ghz=7.125,
        meters_per_pixel=1.0,
        formula_mode="hybrid_two_ray_cost231",
        los_tensor=torch.from_numpy(np.asarray(los_np, dtype=np.float32)).unsqueeze(0),
        a2g_params={},
        clip_min=float(path_meta["clip_min"]),
        clip_max=float(path_meta["clip_max"]),
    )
    if calibration is None:
        return np.asarray(old_formula_prior.detach().cpu().numpy()[0], dtype=np.float64)
    prior_t = old_formula_prior.to(dtype=torch.float32)
    los_t = torch.from_numpy(np.asarray(los_np, dtype=np.float32)).unsqueeze(0)
    topo_t = torch.from_numpy(np.asarray(topology_np, dtype=np.float32)).unsqueeze(0)
    dist_t = torch.from_numpy(np.asarray(distance_np, dtype=np.float32)).unsqueeze(0)
    calibrated = data_utils._apply_formula_regime_calibration(
        prior_t,
        calibration,
        city=city,
        density=float(density),
        height=float(height),
        antenna_height_m=float(antenna_height_m),
        los_tensor=los_t,
        topology_tensor=topo_t,
        distance_map_tensor=dist_t,
        non_ground_threshold=float(non_ground_threshold),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
    )
    return np.asarray(calibrated.detach().cpu().numpy()[0], dtype=np.float64)


class _ResidualMLP(torch.nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_torch_residual_mlp(
    X: np.ndarray,
    y: np.ndarray,
    sample_idx: np.ndarray,
    *,
    device: torch.device,
    seed: int,
) -> tuple[_ResidualMLP, np.ndarray, np.ndarray, int]:
    X_sel = np.asarray(X[sample_idx], dtype=np.float32)
    y_sel = np.asarray(y[sample_idx], dtype=np.float32)
    rng = np.random.default_rng(int(seed) + 701)
    perm = rng.permutation(X_sel.shape[0])
    val_count = max(1, min(int(round(0.1 * X_sel.shape[0])), 50000))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:] if val_count < X_sel.shape[0] else perm

    mean = X_sel[train_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_sel[train_idx].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1.0e-6] = 1.0

    def _norm_rows(rows: np.ndarray) -> np.ndarray:
        return ((rows - mean) / std).astype(np.float32, copy=False)

    X_train = torch.from_numpy(_norm_rows(X_sel[train_idx])).to(device)
    y_train = torch.from_numpy(y_sel[train_idx]).to(device)
    X_val = torch.from_numpy(_norm_rows(X_sel[val_idx])).to(device)
    y_val = torch.from_numpy(y_sel[val_idx]).to(device)

    torch.manual_seed(int(seed))
    model = _ResidualMLP(X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    loss_fn = torch.nn.MSELoss()
    batch_size = 8192
    max_epochs = 60
    patience = 8
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stagnant = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_perm = torch.randperm(X_train.shape[0], device=device)
        for start in range(0, X_train.shape[0], batch_size):
            idx = epoch_perm[start : start + batch_size]
            pred = model(X_train[idx])
            loss = loss_fn(pred, y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = float(loss_fn(val_pred, y_val).detach().cpu().item())
        if epoch == 1 or epoch % 5 == 0:
            print(f"[nlos-hgbr] mlp epoch {epoch}/{max_epochs} val_mse={val_loss:.6f}", flush=True)
        if val_loss + 1.0e-6 < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, mean, std, best_epoch


def _predict_torch_residual_mlp(
    model: _ResidualMLP,
    mean: np.ndarray,
    std: np.ndarray,
    X: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 16384,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            chunk = ((X[start : start + batch_size] - mean) / std).astype(np.float32, copy=False)
            chunk_t = torch.from_numpy(chunk).to(device)
            pred = model(chunk_t).detach().cpu().numpy()
            outputs.append(np.asarray(pred, dtype=np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0,), dtype=np.float32)


def main() -> None:
    args = parse_args()
    try_dir = _resolve(PRACTICE_ROOT, args.try_dir).resolve()
    config_path = _resolve(try_dir, args.config).resolve()
    device = _resolve_device(args.device)

    sys.path.insert(0, str(try_dir))
    config_utils = importlib.import_module("config_utils")
    data_utils = importlib.import_module("data_utils")

    cfg = config_utils.load_config(str(config_path))
    if args.dataset:
        cfg["data"]["hdf5_path"] = str(Path(args.dataset).resolve())
    else:
        hdf5_path = Path(cfg["data"]["hdf5_path"])
        if not hdf5_path.is_absolute():
            cfg["data"]["hdf5_path"] = str((PRACTICE_ROOT / "Datasets" / hdf5_path.name).resolve())
    cfg["augmentation"] = dict(cfg.get("augmentation", {}))
    cfg["augmentation"]["enable"] = False
    cfg["data"] = dict(cfg.get("data", {}))
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    old_calibration_path = _resolve(PRACTICE_ROOT, args.old_calibration_json).resolve()
    old_calibration = json.loads(old_calibration_path.read_text(encoding="utf-8")) if old_calibration_path.exists() else None
    if formula_cfg:
        formula_cfg["regime_calibration_json"] = None
        cfg["data"]["path_loss_formula_input"] = formula_cfg

    splits = data_utils.build_dataset_splits_from_config(cfg)
    train_ds = splits["train"]
    val_ds = splits["val"]
    train_total = len(train_ds.sample_refs)
    val_total = len(val_ds.sample_refs)
    val_sample_frac = float(args.sample_frac if float(args.val_sample_frac) <= 0.0 else args.val_sample_frac)
    train_ds.sample_refs = _subset_sample_refs(list(train_ds.sample_refs), args.sample_frac, args.sample_seed)
    val_ds.sample_refs = _subset_sample_refs(list(val_ds.sample_refs), val_sample_frac, args.sample_seed + 1)

    target_meta = cfg["target_metadata"]["path_loss"]
    formula_idx = _formula_channel_index(cfg)
    channel_layout = _input_channel_layout(cfg)
    image_size = int(cfg["data"].get("image_size", 513))
    dataset_path = Path(cfg["data"]["hdf5_path"]).resolve()
    rng = np.random.default_rng(int(args.sample_seed))

    print(
        f"[nlos-hgbr] try={try_dir.name} dataset={dataset_path.name} sample_frac={float(args.sample_frac):.3f} "
        f"train_subset={len(train_ds.sample_refs)}/{train_total} val_subset={len(val_ds.sample_refs)}/{val_total} "
        f"baseline={args.baseline_mode}"
    )

    with h5py.File(str(dataset_path), "r") as handle:
        train_city_aggs: Dict[str, Dict[str, float]] = {}
        antenna_values: List[float] = []
        sample_cache: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        for city, sample in train_ds.sample_refs:
            density, height, ant = _sample_metadata(handle, city, sample)
            sample_cache[(city, sample)] = (density, height, ant)
            antenna_values.append(ant)
            agg = train_city_aggs.setdefault(city, {"count": 0.0, "density_sum": 0.0, "height_sum": 0.0})
            agg["count"] += 1.0
            agg["density_sum"] += density
            agg["height_sum"] += height

        city_density_values: List[float] = []
        city_height_values: List[float] = []
        city_type_map: Dict[str, str] = {}
        city_avg_stats: Dict[str, Tuple[float, float]] = {}
        for city, agg in train_city_aggs.items():
            count = max(agg["count"], 1.0)
            dens = agg["density_sum"] / count
            height = agg["height_sum"] / count
            city_avg_stats[city] = (dens, height)
            city_density_values.append(dens)
            city_height_values.append(height)
        density_q1, density_q2 = _quantile_edges(city_density_values)
        height_q1, height_q2 = _quantile_edges(city_height_values)
        ant_q1, ant_q2 = _quantile_edges(antenna_values)
        for city, (dens, height) in city_avg_stats.items():
            city_type_map[city] = _city_type_for_stats(dens, height, density_q1, density_q2, height_q1, height_q2)

        city_code = {"open_lowrise": 0.0, "mixed_midrise": 1.0, "dense_highrise": 2.0}
        ant_code = {"low_ant": 0.0, "mid_ant": 1.0, "high_ant": 2.0}

        feature_names = [
            "current_minus_old_prior",
            "shadow_depth",
            "distance_since_los_break",
            "max_blocker_height",
            "blocker_count",
            "distance_norm",
            "prior_confidence",
            "old_prior_abs",
            "antenna_height_norm",
            "city_density",
            "city_height_norm",
            "city_type_code",
            "ant_bin_code",
            "prior_x_shadow",
            "prior_x_break",
            "shadow_x_blocker",
            "dist_x_ant",
        ]
        X_rows: List[np.ndarray] = []
        y_rows: List[np.ndarray] = []
        regime_rows: List[np.ndarray] = []
        weight_rows: List[np.ndarray] = []

        for idx, (city, sample) in enumerate(train_ds.sample_refs):
            density, height, ant = sample_cache[(city, sample)]
            ctype = city_type_map.get(city, "mixed_midrise")
            abin = _ant_bin(ant, ant_q1, ant_q2)
            x, y, m, sc = data_utils.unpack_cgan_batch(train_ds[idx], device)
            x, y, m, sc = _ensure_batch_dims(x, y, m, sc)
            prior = _denormalize_channel(x[:, formula_idx : formula_idx + 1], target_meta).detach().cpu().numpy()[0, 0]
            target = _denormalize_channel(y[:, :1], target_meta).detach().cpu().numpy()[0, 0]
            base_mask = (m[:, :1].detach().cpu().numpy()[0, 0] > 0.0)
            ground_mask = _ground_mask_from_hdf5(handle, city, sample, image_size)
            mask = base_mask & ground_mask
            los = x[:, channel_layout["los"] : channel_layout["los"] + 1].detach().cpu().numpy()[0, 0] > 0.5
            raw_topology = np.asarray(handle[city][sample][cfg["data"]["input_column"]][...], dtype=np.float32)
            topology_resized = _nearest_resize(raw_topology, image_size)
            mm = mask & (~los)
            if not np.any(mm):
                continue
            old_prior = _compute_old_calibrated_prior_db(
                data_utils,
                old_calibration,
                prior_db_np=prior,
                city=city,
                density=density,
                height=height,
                antenna_height_m=ant,
                los_np=los.astype(np.float32),
                topology_np=topology_resized,
                distance_np=x[:, channel_layout["distance"] : channel_layout["distance"] + 1].detach().cpu().numpy()[0, 0],
                non_ground_threshold=float(cfg["data"].get("non_ground_threshold", 0.0)),
                clip_min=float(target_meta.get("clip_min", 0.0)),
                clip_max=float(target_meta.get("clip_max", 180.0)),
            )
            base_prior = old_prior if str(args.baseline_mode).lower() == "old_exact" else prior
            nlos_idx = np.flatnonzero(mm.reshape(-1))
            if nlos_idx.size > int(args.train_pixels_per_sample):
                nlos_idx = rng.choice(nlos_idx, size=int(args.train_pixels_per_sample), replace=False)
            current_minus_old = (prior - old_prior).reshape(-1)[nlos_idx]
            target_delta = np.clip((target - base_prior).reshape(-1)[nlos_idx], 0.0, float(args.delta_clip_max))
            shadow = x[:, channel_layout["shadow_depth"] : channel_layout["shadow_depth"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[nlos_idx]
            break_dist = x[:, channel_layout["distance_since_los_break"] : channel_layout["distance_since_los_break"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[nlos_idx]
            blocker_h = x[:, channel_layout["max_blocker_height"] : channel_layout["max_blocker_height"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[nlos_idx]
            blocker_c = x[:, channel_layout["blocker_count"] : channel_layout["blocker_count"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[nlos_idx]
            dist_norm = x[:, channel_layout["distance"] : channel_layout["distance"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[nlos_idx]
            conf_idx = channel_layout.get("formula_confidence", None)
            conf = (
                x[:, conf_idx : conf_idx + 1].detach().cpu().numpy()[0, 0].reshape(-1)[nlos_idx]
                if conf_idx is not None
                else np.ones_like(current_minus_old, dtype=np.float32)
            )
            rows = np.stack(
                [
                    current_minus_old,
                    shadow,
                    break_dist,
                    blocker_h,
                    blocker_c,
                    dist_norm,
                    conf,
                    base_prior.reshape(-1)[nlos_idx],
                    np.full_like(current_minus_old, float(ant / 120.0)),
                    np.full_like(current_minus_old, float(density)),
                    np.full_like(current_minus_old, float(height / 32.0)),
                    np.full_like(current_minus_old, city_code[ctype]),
                    np.full_like(current_minus_old, ant_code[abin]),
                    current_minus_old * shadow,
                    current_minus_old * break_dist,
                    shadow * (0.5 * blocker_h + 0.5 * blocker_c),
                    dist_norm * float(ant / 120.0),
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            X_rows.append(rows)
            y_rows.append(target_delta.astype(np.float32, copy=False))
            regime_code = int(city_code[ctype] * 3 + ant_code[abin])
            regime_rows.append(np.full((rows.shape[0],), regime_code, dtype=np.int32))
            weights = (
                1.0
                + 0.08 * np.clip(target_delta, 0.0, 60.0)
                + 1.5 * shadow
                + 1.0 * break_dist
                + 0.8 * (0.5 * blocker_h + 0.5 * blocker_c)
            ).astype(np.float32, copy=False)
            weight_rows.append(weights)
            if (idx + 1) % max(int(args.log_every), 1) == 0 or (idx + 1) == len(train_ds):
                print(f"[nlos-hgbr] train sample {idx + 1}/{len(train_ds)} rows={sum(r.shape[0] for r in X_rows)}")

        X_train = np.concatenate(X_rows, axis=0)
        y_train = np.concatenate(y_rows, axis=0)
        regime_train = np.concatenate(regime_rows, axis=0)
        w_train = np.concatenate(weight_rows, axis=0)
        print("[nlos-hgbr] fitting global hgbr", flush=True)
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=10,
            max_iter=420,
            min_samples_leaf=32,
            l2_regularization=0.01,
            random_state=int(args.sample_seed),
            categorical_features=[11, 12],
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        global_pred_train = model.predict(X_train)
        residual_train = y_train - global_pred_train
        mlp_rows = 0
        mlp_best_epoch = 0
        mlp_device = "disabled"
        mlp_model = None
        mlp_mean = None
        mlp_std = None
        if bool(args.use_torch_mlp):
            mlp_rows = min(int(args.mlp_max_rows), int(X_train.shape[0]))
            mlp_rng = np.random.default_rng(int(args.sample_seed) + 77)
            mlp_idx = mlp_rng.choice(X_train.shape[0], size=mlp_rows, replace=False) if mlp_rows < X_train.shape[0] else np.arange(X_train.shape[0])
            print(
                f"[nlos-hgbr] fitting mlp residual rows={mlp_rows}/{X_train.shape[0]} device={device}",
                flush=True,
            )
            mlp_model, mlp_mean, mlp_std, mlp_best_epoch = _fit_torch_residual_mlp(
                X_train,
                residual_train,
                mlp_idx,
                device=device,
                seed=int(args.sample_seed),
            )
            global_pred_train = global_pred_train + _predict_torch_residual_mlp(
                mlp_model,
                mlp_mean,
                mlp_std,
                X_train,
                device=device,
            )
            mlp_device = str(device)
        train_r2 = 1.0 - float(np.sum((y_train - global_pred_train) ** 2) / max(np.sum((y_train - float(np.mean(y_train))) ** 2), 1e-8))
        specialist_models: Dict[int, HistGradientBoostingRegressor] = {}
        specialist_codes = sorted(np.unique(regime_train).tolist())
        specialist_total = len(specialist_codes)
        for specialist_idx, regime_code in enumerate(specialist_codes, start=1):
            mm_regime = regime_train == regime_code
            rows_count = int(np.sum(mm_regime))
            if rows_count < int(args.specialist_min_rows):
                print(
                    f"[nlos-hgbr] skipping specialist {specialist_idx}/{specialist_total} regime={regime_code} rows={rows_count}",
                    flush=True,
                )
                continue
            print(
                f"[nlos-hgbr] fitting specialist {specialist_idx}/{specialist_total} regime={regime_code} rows={rows_count}",
                flush=True,
            )
            specialist = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.04,
                max_depth=8,
                max_iter=280,
                min_samples_leaf=32,
                l2_regularization=0.01,
                random_state=int(args.sample_seed) + int(regime_code) + 100,
                categorical_features=[11, 12],
            )
            specialist.fit(
                X_train[mm_regime],
                y_train[mm_regime] - global_pred_train[mm_regime],
                sample_weight=w_train[mm_regime],
            )
            specialist_models[int(regime_code)] = specialist

        sq_overall = abs_overall = cnt_overall = 0.0
        sq_los = abs_los = cnt_los = 0.0
        sq_nlos = abs_nlos = cnt_nlos = 0.0
        print(
            f"[nlos-hgbr] starting validation samples={len(val_ds.sample_refs)}",
            flush=True,
        )

        for idx, (city, sample) in enumerate(val_ds.sample_refs):
            density, height, ant = sample_cache.get((city, sample), _sample_metadata(handle, city, sample))
            ctype = city_type_map.get(city, _city_type_for_stats(density, height, density_q1, density_q2, height_q1, height_q2))
            abin = _ant_bin(ant, ant_q1, ant_q2)
            x, y, m, sc = data_utils.unpack_cgan_batch(val_ds[idx], device)
            x, y, m, sc = _ensure_batch_dims(x, y, m, sc)
            prior = _denormalize_channel(x[:, formula_idx : formula_idx + 1], target_meta).detach().cpu().numpy()[0, 0]
            target = _denormalize_channel(y[:, :1], target_meta).detach().cpu().numpy()[0, 0]
            base_mask = (m[:, :1].detach().cpu().numpy()[0, 0] > 0.0)
            ground_mask = _ground_mask_from_hdf5(handle, city, sample, image_size)
            mask = base_mask & ground_mask
            los = x[:, channel_layout["los"] : channel_layout["los"] + 1].detach().cpu().numpy()[0, 0] > 0.5
            raw_topology = np.asarray(handle[city][sample][cfg["data"]["input_column"]][...], dtype=np.float32)
            topology_resized = _nearest_resize(raw_topology, image_size)
            old_prior = _compute_old_calibrated_prior_db(
                data_utils,
                old_calibration,
                prior_db_np=prior,
                city=city,
                density=density,
                height=height,
                antenna_height_m=ant,
                los_np=los.astype(np.float32),
                topology_np=topology_resized,
                distance_np=x[:, channel_layout["distance"] : channel_layout["distance"] + 1].detach().cpu().numpy()[0, 0],
                non_ground_threshold=float(cfg["data"].get("non_ground_threshold", 0.0)),
                clip_min=float(target_meta.get("clip_min", 0.0)),
                clip_max=float(target_meta.get("clip_max", 180.0)),
            )
            base_prior = old_prior if str(args.baseline_mode).lower() == "old_exact" else prior
            pred = base_prior.copy()
            nlos_mask = mask & (~los)
            if np.any(nlos_mask):
                flat_idx = np.flatnonzero(nlos_mask.reshape(-1))
                current_minus_old = (prior - old_prior).reshape(-1)[flat_idx]
                shadow = x[:, channel_layout["shadow_depth"] : channel_layout["shadow_depth"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                break_dist = x[:, channel_layout["distance_since_los_break"] : channel_layout["distance_since_los_break"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                blocker_h = x[:, channel_layout["max_blocker_height"] : channel_layout["max_blocker_height"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                blocker_c = x[:, channel_layout["blocker_count"] : channel_layout["blocker_count"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                dist_norm = x[:, channel_layout["distance"] : channel_layout["distance"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                conf_idx = channel_layout.get("formula_confidence", None)
                conf = (
                    x[:, conf_idx : conf_idx + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                    if conf_idx is not None
                    else np.ones_like(current_minus_old, dtype=np.float32)
                )
                X_val = np.stack(
                    [
                        current_minus_old,
                        shadow,
                        break_dist,
                        blocker_h,
                        blocker_c,
                        dist_norm,
                        conf,
                        base_prior.reshape(-1)[flat_idx],
                        np.full_like(current_minus_old, float(ant / 120.0)),
                        np.full_like(current_minus_old, float(density)),
                        np.full_like(current_minus_old, float(height / 32.0)),
                        np.full_like(current_minus_old, city_code[ctype]),
                        np.full_like(current_minus_old, ant_code[abin]),
                        current_minus_old * shadow,
                        current_minus_old * break_dist,
                        shadow * (0.5 * blocker_h + 0.5 * blocker_c),
                        dist_norm * float(ant / 120.0),
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
                delta_pred = model.predict(X_val)
                if mlp_model is not None and mlp_mean is not None and mlp_std is not None:
                    delta_pred = delta_pred + _predict_torch_residual_mlp(
                        mlp_model,
                        mlp_mean,
                        mlp_std,
                        X_val,
                        device=device,
                    )
                regime_code = int(city_code[ctype] * 3 + ant_code[abin])
                specialist = specialist_models.get(regime_code)
                if specialist is not None:
                    delta_pred = delta_pred + specialist.predict(X_val)
                delta_pred = np.clip(delta_pred, 0.0, float(args.delta_clip_max))
                pred.reshape(-1)[flat_idx] = base_prior.reshape(-1)[flat_idx] + delta_pred

            err = pred[mask] - target[mask]
            sq_overall += float(np.sum(err * err))
            abs_overall += float(np.sum(np.abs(err)))
            cnt_overall += float(err.size)
            err_los = pred[mask & los] - target[mask & los]
            sq_los += float(np.sum(err_los * err_los))
            abs_los += float(np.sum(np.abs(err_los)))
            cnt_los += float(err_los.size)
            err_nlos = pred[nlos_mask] - target[nlos_mask]
            sq_nlos += float(np.sum(err_nlos * err_nlos))
            abs_nlos += float(np.sum(np.abs(err_nlos)))
            cnt_nlos += float(err_nlos.size)
            if (idx + 1) % max(int(args.log_every), 1) == 0 or (idx + 1) == len(val_ds):
                rmse_so_far = math.sqrt(sq_overall / max(cnt_overall, 1.0))
                print(f"[nlos-hgbr] val sample {idx + 1}/{len(val_ds)} overall_rmse={rmse_so_far:.4f}")

    feature_importance = {
        name: float(val) for name, val in zip(feature_names, np.zeros(len(feature_names), dtype=np.float64))
    }
    payload = {
        "try_dir": str(try_dir),
        "config": str(config_path),
        "dataset": str(dataset_path),
        "train_samples": len(train_ds.sample_refs),
        "val_samples": len(val_ds.sample_refs),
        "train_nlos_pixels": int(X_train.shape[0]),
        "metrics": {
            "overall": {
                "count": int(cnt_overall),
                "rmse_db": float(math.sqrt(sq_overall / max(cnt_overall, 1.0))),
                "mae_db": float(abs_overall / max(cnt_overall, 1.0)),
            },
            "LoS": {
                "count": int(cnt_los),
                "rmse_db": float(math.sqrt(sq_los / max(cnt_los, 1.0))),
                "mae_db": float(abs_los / max(cnt_los, 1.0)),
            },
            "NLoS": {
                "count": int(cnt_nlos),
                "rmse_db": float(math.sqrt(sq_nlos / max(cnt_nlos, 1.0))),
                "mae_db": float(abs_nlos / max(cnt_nlos, 1.0)),
            },
        },
        "model": {
            "baseline_mode": str(args.baseline_mode),
            "train_r2": train_r2,
            "max_depth": 10,
            "learning_rate": 0.05,
            "max_iter": 420,
            "delta_clip_max": float(args.delta_clip_max),
            "use_torch_mlp": bool(args.use_torch_mlp),
            "mlp_max_rows": int(mlp_rows),
            "mlp_best_epoch": int(mlp_best_epoch),
            "mlp_device": mlp_device,
            "specialist_models": int(len(specialist_models)),
            "specialist_min_rows": int(args.specialist_min_rows),
        },
        "feature_importance": feature_importance,
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(out_md, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
