#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_nlos_delta_hgbr import (  # noqa: E402
    _ant_bin,
    _city_type_for_stats,
    _compute_old_calibrated_prior_db,
    _denormalize_channel,
    _ensure_batch_dims,
    _formula_channel_index,
    _ground_mask_from_hdf5,
    _input_channel_layout,
    _quantile_edges,
    _resolve,
    _resolve_device,
    _sample_metadata,
    _subset_sample_refs,
)


REGIME_NAMES = (
    "shallow_transition",
    "deep_shadow_canyon",
    "high_blocker_rooftop",
    "dense_clutter",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heuristic NLoS regime experts on top of the frozen Try47 prior.")
    p.add_argument("--try-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default="")
    p.add_argument("--sample-frac", type=float, default=0.05)
    p.add_argument("--val-sample-frac", type=float, default=0.062)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--train-pixels-per-sample", type=int, default=2048)
    p.add_argument("--delta-clip-max", type=float, default=30.0)
    p.add_argument("--expert-min-rows", type=int, default=30000)
    p.add_argument(
        "--old-calibration-json",
        default=str(PRACTICE_ROOT / "TFGFiftiethTry50" / "prior_calibration" / "regime_obstruction_train_only_from_try47.json"),
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def _heuristic_regime(shadow: np.ndarray, break_dist: np.ndarray, blocker_h: np.ndarray, blocker_c: np.ndarray) -> np.ndarray:
    regimes = np.full(shadow.shape, 3, dtype=np.int32)  # dense_clutter default
    shallow = (shadow < 0.07) & (break_dist < 0.10) & (blocker_h < 0.10) & (blocker_c < 0.08)
    rooftop = (blocker_h >= 0.12) & (blocker_c < 0.18) & (shadow < 0.18) & (break_dist < 0.22)
    canyon = (shadow >= 0.18) | (break_dist >= 0.22)
    regimes[shallow] = 0
    regimes[canyon] = 1
    regimes[rooftop & (~canyon) & (~shallow)] = 2
    return regimes


def _build_rows(
    *,
    current_minus_old: np.ndarray,
    shadow: np.ndarray,
    break_dist: np.ndarray,
    blocker_h: np.ndarray,
    blocker_c: np.ndarray,
    dist_norm: np.ndarray,
    conf: np.ndarray,
    old_prior_abs: np.ndarray,
    ant_norm: float,
    density: float,
    height_norm: float,
    city_code: float,
    ant_code: float,
) -> np.ndarray:
    return np.stack(
        [
            current_minus_old,
            shadow,
            break_dist,
            blocker_h,
            blocker_c,
            dist_norm,
            conf,
            old_prior_abs,
            np.full_like(current_minus_old, ant_norm),
            np.full_like(current_minus_old, density),
            np.full_like(current_minus_old, height_norm),
            np.full_like(current_minus_old, city_code),
            np.full_like(current_minus_old, ant_code),
            current_minus_old * shadow,
            current_minus_old * break_dist,
            shadow * (0.5 * blocker_h + 0.5 * blocker_c),
            dist_norm * ant_norm,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _write_markdown(out_path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# NLoS Regime Experts",
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
        "## Regimes",
        "",
    ]
    for name in REGIME_NAMES:
        rows = payload["regime_rows"].get(name, 0)
        lines.append(f"- `{name}`: `{rows}` train rows")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    if formula_cfg:
        formula_cfg["regime_calibration_json"] = None
        cfg["data"]["path_loss_formula_input"] = formula_cfg

    old_calibration_path = _resolve(PRACTICE_ROOT, args.old_calibration_json).resolve()
    old_calibration = json.loads(old_calibration_path.read_text(encoding="utf-8")) if old_calibration_path.exists() else None

    splits = data_utils.build_dataset_splits_from_config(cfg)
    train_ds = splits["train"]
    val_ds = splits["val"]
    train_total = len(train_ds.sample_refs)
    val_total = len(val_ds.sample_refs)
    train_ds.sample_refs = _subset_sample_refs(list(train_ds.sample_refs), args.sample_frac, args.sample_seed)
    val_ds.sample_refs = _subset_sample_refs(list(val_ds.sample_refs), args.val_sample_frac, args.sample_seed + 1)

    target_meta = cfg["target_metadata"]["path_loss"]
    formula_idx = _formula_channel_index(cfg)
    channel_layout = _input_channel_layout(cfg)
    image_size = int(cfg["data"].get("image_size", 513))
    dataset_path = Path(cfg["data"]["hdf5_path"]).resolve()
    rng = np.random.default_rng(int(args.sample_seed))

    print(
        f"[nlos-experts] try={try_dir.name} dataset={dataset_path.name} sample_frac={float(args.sample_frac):.3f} "
        f"train_subset={len(train_ds.sample_refs)}/{train_total} val_subset={len(val_ds.sample_refs)}/{val_total}",
        flush=True,
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

        X_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []
        reg_all: List[np.ndarray] = []
        w_all: List[np.ndarray] = []
        regime_rows = {name: 0 for name in REGIME_NAMES}

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
            topology_resized = np.asarray(raw_topology, dtype=np.float32)
            if topology_resized.shape != (image_size, image_size):
                from analyze_nlos_delta_hgbr import _nearest_resize  # local import to keep script focused

                topology_resized = _nearest_resize(raw_topology, image_size)
            nlos_mask = mask & (~los)
            if not np.any(nlos_mask):
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
            flat_idx = np.flatnonzero(nlos_mask.reshape(-1))
            if flat_idx.size > int(args.train_pixels_per_sample):
                flat_idx = rng.choice(flat_idx, size=int(args.train_pixels_per_sample), replace=False)

            shadow = x[:, channel_layout["shadow_depth"] : channel_layout["shadow_depth"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
            break_dist = x[:, channel_layout["distance_since_los_break"] : channel_layout["distance_since_los_break"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
            blocker_h = x[:, channel_layout["max_blocker_height"] : channel_layout["max_blocker_height"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
            blocker_c = x[:, channel_layout["blocker_count"] : channel_layout["blocker_count"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
            dist_norm = x[:, channel_layout["distance"] : channel_layout["distance"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
            conf_idx = channel_layout.get("formula_confidence", None)
            conf = (
                x[:, conf_idx : conf_idx + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                if conf_idx is not None
                else np.ones((flat_idx.shape[0],), dtype=np.float32)
            )
            current_minus_old = (prior - old_prior).reshape(-1)[flat_idx]
            target_delta = np.clip((target - old_prior).reshape(-1)[flat_idx], 0.0, float(args.delta_clip_max))
            regimes = _heuristic_regime(shadow, break_dist, blocker_h, blocker_c)
            rows = _build_rows(
                current_minus_old=current_minus_old,
                shadow=shadow,
                break_dist=break_dist,
                blocker_h=blocker_h,
                blocker_c=blocker_c,
                dist_norm=dist_norm,
                conf=conf,
                old_prior_abs=old_prior.reshape(-1)[flat_idx],
                ant_norm=float(ant / 120.0),
                density=float(density),
                height_norm=float(height / 32.0),
                city_code=city_code[ctype],
                ant_code=ant_code[abin],
            )
            weights = (
                1.0
                + 0.08 * np.clip(target_delta, 0.0, float(args.delta_clip_max))
                + 1.5 * shadow
                + 1.0 * break_dist
                + 0.8 * (0.5 * blocker_h + 0.5 * blocker_c)
            ).astype(np.float32, copy=False)
            X_all.append(rows)
            y_all.append(target_delta.astype(np.float32, copy=False))
            reg_all.append(regimes)
            w_all.append(weights)
            for regime_id, regime_name in enumerate(REGIME_NAMES):
                regime_rows[regime_name] += int(np.sum(regimes == regime_id))
            if (idx + 1) % max(int(args.log_every), 1) == 0 or (idx + 1) == len(train_ds):
                print(f"[nlos-experts] train sample {idx + 1}/{len(train_ds)} rows={sum(r.shape[0] for r in X_all)}", flush=True)

        X_train = np.concatenate(X_all, axis=0)
        y_train = np.concatenate(y_all, axis=0)
        regime_train = np.concatenate(reg_all, axis=0)
        w_train = np.concatenate(w_all, axis=0)

        print("[nlos-experts] fitting global fallback hgbr", flush=True)
        global_model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=8,
            max_iter=300,
            min_samples_leaf=32,
            l2_regularization=0.01,
            random_state=int(args.sample_seed),
            categorical_features=[11, 12],
        )
        global_model.fit(X_train, y_train, sample_weight=w_train)

        expert_models: Dict[int, HistGradientBoostingRegressor] = {}
        for regime_id, regime_name in enumerate(REGIME_NAMES):
            mm = regime_train == regime_id
            rows_count = int(np.sum(mm))
            if rows_count < int(args.expert_min_rows):
                print(f"[nlos-experts] skipping expert {regime_name} rows={rows_count}", flush=True)
                continue
            print(f"[nlos-experts] fitting expert {regime_name} rows={rows_count}", flush=True)
            expert = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.04,
                max_depth=8,
                max_iter=260,
                min_samples_leaf=32,
                l2_regularization=0.01,
                random_state=int(args.sample_seed) + regime_id + 100,
                categorical_features=[11, 12],
            )
            expert.fit(X_train[mm], y_train[mm], sample_weight=w_train[mm])
            expert_models[regime_id] = expert

        sq_overall = abs_overall = cnt_overall = 0.0
        sq_los = abs_los = cnt_los = 0.0
        sq_nlos = abs_nlos = cnt_nlos = 0.0
        print(f"[nlos-experts] starting validation samples={len(val_ds.sample_refs)}", flush=True)

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
            topology_resized = np.asarray(raw_topology, dtype=np.float32)
            if topology_resized.shape != (image_size, image_size):
                from analyze_nlos_delta_hgbr import _nearest_resize

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
            pred = old_prior.copy()
            nlos_mask = mask & (~los)
            if np.any(nlos_mask):
                flat_idx = np.flatnonzero(nlos_mask.reshape(-1))
                shadow = x[:, channel_layout["shadow_depth"] : channel_layout["shadow_depth"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                break_dist = x[:, channel_layout["distance_since_los_break"] : channel_layout["distance_since_los_break"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                blocker_h = x[:, channel_layout["max_blocker_height"] : channel_layout["max_blocker_height"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                blocker_c = x[:, channel_layout["blocker_count"] : channel_layout["blocker_count"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                dist_norm = x[:, channel_layout["distance"] : channel_layout["distance"] + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                conf_idx = channel_layout.get("formula_confidence", None)
                conf = (
                    x[:, conf_idx : conf_idx + 1].detach().cpu().numpy()[0, 0].reshape(-1)[flat_idx]
                    if conf_idx is not None
                    else np.ones((flat_idx.shape[0],), dtype=np.float32)
                )
                current_minus_old = (prior - old_prior).reshape(-1)[flat_idx]
                regimes = _heuristic_regime(shadow, break_dist, blocker_h, blocker_c)
                X_val = _build_rows(
                    current_minus_old=current_minus_old,
                    shadow=shadow,
                    break_dist=break_dist,
                    blocker_h=blocker_h,
                    blocker_c=blocker_c,
                    dist_norm=dist_norm,
                    conf=conf,
                    old_prior_abs=old_prior.reshape(-1)[flat_idx],
                    ant_norm=float(ant / 120.0),
                    density=float(density),
                    height_norm=float(height / 32.0),
                    city_code=city_code[ctype],
                    ant_code=ant_code[abin],
                )
                delta_pred = global_model.predict(X_val)
                for regime_id, expert in expert_models.items():
                    mm = regimes == regime_id
                    if np.any(mm):
                        delta_pred[mm] = expert.predict(X_val[mm])
                delta_pred = np.clip(delta_pred, 0.0, float(args.delta_clip_max))
                pred.reshape(-1)[flat_idx] = old_prior.reshape(-1)[flat_idx] + delta_pred

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
                print(f"[nlos-experts] val sample {idx + 1}/{len(val_ds)} overall_rmse={rmse_so_far:.4f}", flush=True)

    payload = {
        "try_dir": str(try_dir),
        "config": str(config_path),
        "dataset": str(dataset_path),
        "train_samples": len(train_ds.sample_refs),
        "val_samples": len(val_ds.sample_refs),
        "train_nlos_pixels": int(X_train.shape[0]),
        "thresholds": {
            "shadow_shallow_max": 0.07,
            "break_shallow_max": 0.10,
            "blocker_h_shallow_max": 0.10,
            "blocker_c_shallow_max": 0.08,
            "blocker_h_rooftop_min": 0.12,
            "blocker_c_rooftop_max": 0.18,
            "shadow_rooftop_max": 0.18,
            "break_rooftop_max": 0.22,
            "shadow_canyon_min": 0.18,
            "break_canyon_min": 0.22,
            "delta_clip_max": float(args.delta_clip_max),
        },
        "regime_rows": regime_rows,
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
            "global_model": "HistGradientBoostingRegressor",
            "expert_model": "HistGradientBoostingRegressor",
            "expert_min_rows": int(args.expert_min_rows),
            "trained_experts": {REGIME_NAMES[k]: True for k in expert_models.keys()},
        },
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(out_md, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
