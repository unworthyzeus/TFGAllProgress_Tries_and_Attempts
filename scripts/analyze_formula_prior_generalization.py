#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent


@dataclass
class LinStats:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_xy: float = 0.0
    positive_count: int = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.size == 0:
            return
        x = x.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        self.count += int(x.size)
        self.sum_x += float(np.sum(x))
        self.sum_y += float(np.sum(y))
        self.sum_xx += float(np.sum(x * x))
        self.sum_xy += float(np.sum(x * y))
        self.positive_count += int(np.sum(y > 0.0))

    def coeffs(self) -> Tuple[float, float]:
        if self.count < 2:
            return 1.0, 0.0
        n = float(self.count)
        denom = n * self.sum_xx - self.sum_x * self.sum_x
        if abs(denom) < 1e-12:
            return 1.0, 0.0
        a = (n * self.sum_xy - self.sum_x * self.sum_y) / denom
        b = (self.sum_y - a * self.sum_x) / n
        return float(a), float(b)

    def positive_rate(self) -> float:
        if self.count <= 0:
            return 1.0
        return float(self.positive_count) / float(self.count)


@dataclass
class ErrStats:
    count: int = 0
    sq: float = 0.0
    abs: float = 0.0

    def update(self, err: np.ndarray) -> None:
        if err.size == 0:
            return
        err = err.astype(np.float64, copy=False)
        self.count += int(err.size)
        self.sq += float(np.sum(err * err))
        self.abs += float(np.sum(np.abs(err)))

    def summary(self) -> Dict[str, float]:
        if self.count <= 0:
            return {"count": 0, "rmse_db": float("nan"), "mae_db": float("nan")}
        return {
            "count": self.count,
            "rmse_db": float(math.sqrt(self.sq / self.count)),
            "mae_db": float(self.abs / self.count),
        }


def _empty_regime_errstats() -> Dict[str, ErrStats]:
    return {
        "overall": ErrStats(),
        "LoS": ErrStats(),
        "NLoS": ErrStats(),
    }


@dataclass
class QuadStats:
    count: int = 0
    sum_x: float = 0.0
    sum_x2: float = 0.0
    sum_x3: float = 0.0
    sum_x4: float = 0.0
    sum_y: float = 0.0
    sum_xy: float = 0.0
    sum_x2y: float = 0.0
    positive_count: int = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.size == 0:
            return
        x = x.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        x2 = x * x
        self.count += int(x.size)
        self.sum_x += float(np.sum(x))
        self.sum_x2 += float(np.sum(x2))
        self.sum_x3 += float(np.sum(x2 * x))
        self.sum_x4 += float(np.sum(x2 * x2))
        self.sum_y += float(np.sum(y))
        self.sum_xy += float(np.sum(x * y))
        self.sum_x2y += float(np.sum(x2 * y))
        self.positive_count += int(np.sum(y > 0.0))

    def coeffs(self) -> Tuple[float, float, float]:
        if self.count < 3:
            return 0.0, 1.0, 0.0
        a = np.array(
            [
                [self.sum_x4, self.sum_x3, self.sum_x2],
                [self.sum_x3, self.sum_x2, self.sum_x],
                [self.sum_x2, self.sum_x, float(self.count)],
            ],
            dtype=np.float64,
        )
        b = np.array([self.sum_x2y, self.sum_xy, self.sum_y], dtype=np.float64)
        try:
            coeffs = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(a, b, rcond=None)[0]
        return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    def positive_rate(self) -> float:
        if self.count <= 0:
            return 1.0
        return float(self.positive_count) / float(self.count)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train-only prior calibration and val-only evaluation without leakage.")
    p.add_argument("--try-dir", required=True, help="Path to the try folder")
    p.add_argument("--config", required=True, help="Config path relative to the try folder or absolute")
    p.add_argument("--device", default="cpu", help="cpu/cuda/directml")
    p.add_argument("--sample-frac", type=float, default=1.0, help="Fraction of train/val samples to use, in (0,1].")
    p.add_argument("--sample-seed", type=int, default=42, help="Seed for deterministic sample subsetting.")
    p.add_argument("--log-every", type=int, default=100, help="Progress log interval in samples")
    p.add_argument("--dataset", default="", help="Override HDF5 path")
    p.add_argument("--out-json", default=str(PRACTICE_ROOT / "analysis" / "formula_prior_generalization.json"))
    p.add_argument("--out-md", default=str(PRACTICE_ROOT / "analysis" / "formula_prior_generalization.md"))
    p.add_argument(
        "--out-calibration-json",
        default=str(PRACTICE_ROOT / "analysis" / "formula_prior_regime_quadratic_calibration.json"),
        help="Where to write the train-only regime-aware quadratic calibration JSON",
    )
    return p.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    name = str(device_name).lower()
    if name == "directml":
        try:
            import torch_directml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DirectML requested but torch_directml is not available") from exc
        return torch_directml.device()
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _quantile_edges(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, 1.0 / 3.0)), float(np.quantile(arr, 2.0 / 3.0))


def _severity_bin(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "sev_low"
    if value <= q2:
        return "sev_mid"
    return "sev_high"


def _sample_pixels(values: np.ndarray, rng: random.Random, cap: int = 2048) -> List[float]:
    if values.size == 0:
        return []
    flat = values.reshape(-1)
    if flat.size <= cap:
        return flat.astype(np.float64, copy=False).tolist()
    idx = np.asarray(rng.sample(range(int(flat.size)), cap), dtype=np.int64)
    return flat[idx].astype(np.float64, copy=False).tolist()


def _subset_sample_refs(sample_refs: List[Tuple[str, str]], sample_frac: float, seed: int) -> List[Tuple[str, str]]:
    frac = float(sample_frac)
    if frac >= 1.0:
        return list(sample_refs)
    if frac <= 0.0:
        raise ValueError("--sample-frac must be in (0, 1].")
    count = max(1, int(round(len(sample_refs) * frac)))
    rng = random.Random(int(seed))
    subset = list(sample_refs)
    rng.shuffle(subset)
    subset = subset[:count]
    subset.sort()
    return subset


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


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def _sample_metadata(handle: h5py.File, city: str, sample: str) -> Tuple[float, float, float]:
    grp = handle[city][sample]
    topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
    non_zero = topo[topo != 0]
    density = float(np.mean(topo != 0))
    height = float(np.mean(non_zero)) if non_zero.size else 0.0
    ant = float(np.asarray(grp["uav_height"][...], dtype=np.float32).reshape(-1)[0])
    return density, height, ant


def _nearest_resize(arr: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def _ground_mask_from_hdf5(handle: h5py.File, city: str, sample: str, image_size: int) -> np.ndarray:
    topo = np.asarray(handle[city][sample]["topology_map"][...], dtype=np.float32)
    topo_resized = _nearest_resize(topo, image_size)
    return np.isclose(topo_resized, 0.0)


def _fallback_key(key: Tuple[str, ...]) -> List[Tuple[str, ...]]:
    if len(key) == 3:
        return [key, key[:2], key[:1], ("all",)]
    if len(key) == 2:
        return [key, key[:1], ("all",)]
    if len(key) == 1:
        return [key, ("all",)]
    return [("all",)]


def _solve_for_key(stats_map: Dict[Tuple[str, ...], LinStats], key: Tuple[str, ...]) -> Tuple[float, float, float]:
    for candidate in _fallback_key(key):
        stats = stats_map.get(candidate)
        if stats and stats.count >= 128:
            a, b = stats.coeffs()
            return a, b, stats.positive_rate()
    a, b = 1.0, 0.0
    return a, b, 1.0


def _solve_quadratic_for_key(stats_map: Dict[Tuple[str, ...], QuadStats], key: Tuple[str, ...]) -> Tuple[float, float, float, float]:
    for candidate in _fallback_key(key):
        stats = stats_map.get(candidate)
        if stats and stats.count >= 128:
            a2, a1, a0 = stats.coeffs()
            return a2, a1, a0, stats.positive_rate()
    return 0.0, 1.0, 0.0, 1.0


def _key_for_mode(mode: str, city_type: str, los_label: str, ant_bin: str) -> Tuple[str, ...]:
    if mode == "global":
        return ("all",)
    if mode == "city_type":
        return (city_type,)
    if mode == "city_type_los":
        return (city_type, los_label)
    if mode == "city_type_los_ant":
        return (city_type, los_label, ant_bin)
    raise KeyError(mode)


def _write_markdown(out_path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Formula Prior Generalization Check")
    lines.append("")
    lines.append("This report fits every calibration only on the training split and evaluates only on the validation split.")
    lines.append("That avoids leaking validation information into the prior calibration step.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Try folder: `{payload['try_dir']}`")
    lines.append(f"- Config: `{payload['config']}`")
    lines.append(f"- Dataset: `{payload['dataset']}`")
    lines.append("")
    lines.append("## Ground-truth Support")
    lines.append("")
    lines.append(f"- Train valid pixels: `{payload['support']['train_valid_pixels']}`")
    lines.append(f"- Train zero-valued valid pixels: `{payload['support']['train_zero_pixels']}` ({payload['support']['train_zero_ratio']:.2%})")
    lines.append(f"- Val valid pixels: `{payload['support']['val_valid_pixels']}`")
    lines.append(f"- Val zero-valued valid pixels: `{payload['support']['val_zero_pixels']}` ({payload['support']['val_zero_ratio']:.2%})")
    lines.append("")
    lines.append("## Train-defined Regimes")
    lines.append("")
    lines.append(f"- Density tertiles: `{payload['regimes']['density_q1']:.4f}`, `{payload['regimes']['density_q2']:.4f}`")
    lines.append(f"- Height tertiles: `{payload['regimes']['height_q1']:.4f}`, `{payload['regimes']['height_q2']:.4f}`")
    lines.append(f"- Antenna-height tertiles: `{payload['regimes']['ant_q1']:.4f}`, `{payload['regimes']['ant_q2']:.4f}`")
    lines.append("")
    lines.append("## Validation Results")
    lines.append("")
    for name, stats in payload["results"].items():
        overall = stats["overall"]
        los = stats["LoS"]
        nlos = stats["NLoS"]
        lines.append(
            f"- `{name}` overall: RMSE `{overall['rmse_db']:.4f} dB`, MAE `{overall['mae_db']:.4f} dB`, count `{overall['count']}`"
        )
        lines.append(
            f"  `LoS`: RMSE `{los['rmse_db']:.4f} dB`, MAE `{los['mae_db']:.4f} dB`, count `{los['count']}`"
        )
        lines.append(
            f"  `NLoS`: RMSE `{nlos['rmse_db']:.4f} dB`, MAE `{nlos['mae_db']:.4f} dB`, count `{nlos['count']}`"
        )
    lines.append("")
    lines.append("## Recommended Prior-Only System")
    lines.append("")
    lines.append(f"- Best validation system: `{payload['best']['name']}`")
    lines.append(f"- Validation RMSE: `{payload['best']['overall']['rmse_db']:.4f} dB`")
    lines.append(f"- Validation `LoS` RMSE: `{payload['best']['LoS']['rmse_db']:.4f} dB`")
    lines.append(f"- Validation `NLoS` RMSE: `{payload['best']['NLoS']['rmse_db']:.4f} dB`")
    lines.append("")
    lines.append("The systems compared are:")
    lines.append("")
    lines.append("- `raw_prior`: direct formula map as-is")
    lines.append("- `global_affine`: one train-only affine calibration for all valid pixels")
    lines.append("- `city_type_affine`: one affine calibration per train-defined urban morphology class")
    lines.append("- `city_type_los_affine`: one affine calibration per urban morphology class and pixel LoS/NLoS")
    lines.append("- `city_type_los_ant_affine`: same as above, also split by antenna-height tertile")
    lines.append("- `*_support_scaled`: multiply the calibrated prediction by the train-only positive-support rate of that regime")
    lines.append("- `city_type_los_ant_quadratic`: train-only quadratic calibration per urban type, LoS/NLoS, and antenna-height tertile")
    lines.append("")
    lines.append("This separation matters because many valid ground pixels have a target value of exactly `0 dB`, which can dominate the global RMSE.")
    _ensure_parent(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _denormalize_channel(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    if abs(scale) < 1e-12:
        scale = 1.0
    return values * scale + offset


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

    # Fit/evaluate the physical prior itself, not a prior that has already
    # been post-calibrated through the try config. Otherwise we end up
    # calibrating a previously calibrated signal and the reported "raw prior"
    # becomes misleading.
    cfg["data"] = dict(cfg.get("data", {}))
    formula_cfg = dict(cfg["data"].get("path_loss_formula_input", {}))
    if formula_cfg:
        formula_cfg["regime_calibration_json"] = None
        cfg["data"]["path_loss_formula_input"] = formula_cfg

    splits = data_utils.build_dataset_splits_from_config(cfg)
    train_ds = splits["train"]
    val_ds = splits["val"]
    train_total = len(train_ds.sample_refs)
    val_total = len(val_ds.sample_refs)
    train_ds.sample_refs = _subset_sample_refs(list(train_ds.sample_refs), args.sample_frac, args.sample_seed)
    val_ds.sample_refs = _subset_sample_refs(list(val_ds.sample_refs), args.sample_frac, args.sample_seed + 1)
    target_meta = cfg["target_metadata"]["path_loss"]
    formula_idx = _formula_channel_index(cfg)
    channel_layout = _input_channel_layout(cfg)
    image_size = int(cfg["data"].get("image_size", 513))

    dataset_path = Path(cfg["data"]["hdf5_path"]).resolve()
    print(
        f"[prior-calibration] try={try_dir.name} device={args.device} dataset={dataset_path.name} "
        f"log_every={int(args.log_every)} sample_frac={float(args.sample_frac):.3f}"
    )
    print(
        f"[prior-calibration] train_subset={len(train_ds.sample_refs)}/{train_total} "
        f"val_subset={len(val_ds.sample_refs)}/{val_total}"
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

        modes = ("global", "city_type", "city_type_los", "city_type_los_ant")
        stats_by_mode: Dict[str, Dict[Tuple[str, ...], LinStats]] = {mode: {} for mode in modes}
        quad_stats_by_mode: Dict[str, Dict[Tuple[str, ...], QuadStats]] = {
            "city_type_los_ant": {},
        }
        delta_stats_by_mode: Dict[str, Dict[Tuple[str, ...], LinStats]] = {
            "city_type_ant": {},
        }
        delta_quad_stats_by_mode: Dict[str, Dict[Tuple[str, ...], QuadStats]] = {
            "city_type_ant": {},
        }
        severity_delta_stats_by_mode: Dict[str, Dict[Tuple[str, ...], LinStats]] = {
            "city_type_ant_severity": {},
        }
        severity_delta_quad_stats_by_mode: Dict[str, Dict[Tuple[str, ...], QuadStats]] = {
            "city_type_ant_severity": {},
        }
        severity_shadow_values: List[float] = []
        severity_break_values: List[float] = []
        severity_blocker_values: List[float] = []
        severity_rng = random.Random(args.sample_seed + 17)

        support_train_valid = 0
        support_train_zero = 0

        for idx in range(len(train_ds)):
            city, sample = train_ds.sample_refs[idx]
            density, height, ant = sample_cache[(city, sample)]
            city_type = city_type_map.get(city, _city_type_for_stats(density, height, density_q1, density_q2, height_q1, height_q2))
            ant_label = _ant_bin(ant, ant_q1, ant_q2)

            x, y, m, sc = data_utils.unpack_cgan_batch(train_ds[idx], device)
            x, y, m, sc = _ensure_batch_dims(x, y, m, sc)
            prior = _denormalize_channel(x[:, formula_idx : formula_idx + 1], target_meta).detach().cpu().numpy()[0, 0]
            target = _denormalize_channel(y[:, :1], target_meta).detach().cpu().numpy()[0, 0]
            base_mask = (m[:, :1].detach().cpu().numpy()[0, 0] > 0.0)
            ground_mask = _ground_mask_from_hdf5(handle, city, sample, image_size)
            mask = base_mask & ground_mask
            los = x[:, 1:2].detach().cpu().numpy()[0, 0] > 0.5
            shadow_depth = (
                x[:, channel_layout["shadow_depth"] : channel_layout["shadow_depth"] + 1].detach().cpu().numpy()[0, 0]
                if "shadow_depth" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            distance_since_break = (
                x[:, channel_layout["distance_since_los_break"] : channel_layout["distance_since_los_break"] + 1]
                .detach()
                .cpu()
                .numpy()[0, 0]
                if "distance_since_los_break" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            blocker_height = (
                x[:, channel_layout["max_blocker_height"] : channel_layout["max_blocker_height"] + 1].detach().cpu().numpy()[0, 0]
                if "max_blocker_height" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            blocker_count = (
                x[:, channel_layout["blocker_count"] : channel_layout["blocker_count"] + 1].detach().cpu().numpy()[0, 0]
                if "blocker_count" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            los_reference = _compute_los_reference_db(
                data_utils,
                image_size=image_size,
                antenna_height_m=ant,
                receiver_height_m=float(formula_cfg.get("receiver_height_m", 1.5)),
                frequency_ghz=float(formula_cfg.get("frequency_ghz", 7.125)),
                meters_per_pixel=float(formula_cfg.get("meters_per_pixel", 1.0)),
                a2g_params=dict(formula_cfg.get("a2g_params", {})),
            )

            support_train_valid += int(mask.sum())
            support_train_zero += int((mask & np.isclose(target, 0.0)).sum())

            for los_name, los_mask in (("LoS", los), ("NLoS", ~los)):
                mm = mask & los_mask
                if not np.any(mm):
                    continue
                xv = prior[mm]
                yv = target[mm]
                for mode in modes:
                    key = _key_for_mode(mode, city_type, los_name, ant_label)
                    stats_by_mode[mode].setdefault(key, LinStats()).update(xv, yv)
                quad_stats_by_mode["city_type_los_ant"].setdefault((city_type, los_name, ant_label), QuadStats()).update(xv, yv)
            mm_nlos = mask & (~los)
            if np.any(mm_nlos):
                delta_prior = prior[mm_nlos] - los_reference[mm_nlos]
                delta_target = target[mm_nlos] - los_reference[mm_nlos]
                delta_stats_by_mode["city_type_ant"].setdefault((city_type, ant_label), LinStats()).update(delta_prior, delta_target)
                delta_quad_stats_by_mode["city_type_ant"].setdefault((city_type, ant_label), QuadStats()).update(delta_prior, delta_target)
                shadow_vals = shadow_depth[mm_nlos]
                break_vals = distance_since_break[mm_nlos]
                blocker_vals = 0.5 * blocker_height[mm_nlos] + 0.5 * blocker_count[mm_nlos]
                severity_shadow_values.extend(_sample_pixels(shadow_vals, severity_rng))
                severity_break_values.extend(_sample_pixels(break_vals, severity_rng))
                severity_blocker_values.extend(_sample_pixels(blocker_vals, severity_rng))

            xv = prior[mask]
            yv = target[mask]
            stats_by_mode["global"].setdefault(("all",), LinStats()).update(xv, yv)
            stats_by_mode["city_type"].setdefault((city_type,), LinStats()).update(xv, yv)
            if (idx + 1) % max(int(args.log_every), 1) == 0 or (idx + 1) == len(train_ds):
                print(
                    f"[prior-calibration] train {idx + 1}/{len(train_ds)} "
                    f"valid_pixels={support_train_valid} zero_ratio="
                    f"{(support_train_zero / max(support_train_valid, 1)):.3f}"
                )

        shadow_q1, shadow_q2 = _quantile_edges(severity_shadow_values)
        break_q1, break_q2 = _quantile_edges(severity_break_values)
        blocker_q1, blocker_q2 = _quantile_edges(severity_blocker_values)

        for idx in range(len(train_ds)):
            city, sample = train_ds.sample_refs[idx]
            density, height, ant = sample_cache[(city, sample)]
            city_type = city_type_map.get(city, _city_type_for_stats(density, height, density_q1, density_q2, height_q1, height_q2))
            ant_label = _ant_bin(ant, ant_q1, ant_q2)

            x, y, m, sc = data_utils.unpack_cgan_batch(train_ds[idx], device)
            x, y, m, sc = _ensure_batch_dims(x, y, m, sc)
            prior = _denormalize_channel(x[:, formula_idx : formula_idx + 1], target_meta).detach().cpu().numpy()[0, 0]
            target = _denormalize_channel(y[:, :1], target_meta).detach().cpu().numpy()[0, 0]
            base_mask = (m[:, :1].detach().cpu().numpy()[0, 0] > 0.0)
            ground_mask = _ground_mask_from_hdf5(handle, city, sample, image_size)
            mask = base_mask & ground_mask
            los = x[:, 1:2].detach().cpu().numpy()[0, 0] > 0.5
            shadow_depth = (
                x[:, channel_layout["shadow_depth"] : channel_layout["shadow_depth"] + 1].detach().cpu().numpy()[0, 0]
                if "shadow_depth" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            distance_since_break = (
                x[:, channel_layout["distance_since_los_break"] : channel_layout["distance_since_los_break"] + 1]
                .detach()
                .cpu()
                .numpy()[0, 0]
                if "distance_since_los_break" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            blocker_height = (
                x[:, channel_layout["max_blocker_height"] : channel_layout["max_blocker_height"] + 1].detach().cpu().numpy()[0, 0]
                if "max_blocker_height" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            blocker_count = (
                x[:, channel_layout["blocker_count"] : channel_layout["blocker_count"] + 1].detach().cpu().numpy()[0, 0]
                if "blocker_count" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            los_reference = _compute_los_reference_db(
                data_utils,
                image_size=image_size,
                antenna_height_m=ant,
                receiver_height_m=float(formula_cfg.get("receiver_height_m", 1.5)),
                frequency_ghz=float(formula_cfg.get("frequency_ghz", 7.125)),
                meters_per_pixel=float(formula_cfg.get("meters_per_pixel", 1.0)),
                a2g_params=dict(formula_cfg.get("a2g_params", {})),
            )

            mm_nlos = mask & (~los)
            if not np.any(mm_nlos):
                continue
            delta_prior = prior[mm_nlos] - los_reference[mm_nlos]
            delta_target = target[mm_nlos] - los_reference[mm_nlos]
            severity_score = (
                0.45 * shadow_depth[mm_nlos]
                + 0.35 * distance_since_break[mm_nlos]
                + 0.20 * (0.5 * blocker_height[mm_nlos] + 0.5 * blocker_count[mm_nlos])
            )
            sev_q1 = 0.45 * shadow_q1 + 0.35 * break_q1 + 0.20 * blocker_q1
            sev_q2 = 0.45 * shadow_q2 + 0.35 * break_q2 + 0.20 * blocker_q2
            sev_bins = np.empty(severity_score.shape, dtype=object)
            sev_bins[severity_score <= sev_q1] = "sev_low"
            sev_bins[(severity_score > sev_q1) & (severity_score <= sev_q2)] = "sev_mid"
            sev_bins[severity_score > sev_q2] = "sev_high"
            for sev_name in ("sev_low", "sev_mid", "sev_high"):
                mm_sev = sev_bins == sev_name
                if not np.any(mm_sev):
                    continue
                key = (city_type, ant_label, sev_name)
                severity_delta_stats_by_mode["city_type_ant_severity"].setdefault(key, LinStats()).update(
                    delta_prior[mm_sev],
                    delta_target[mm_sev],
                )
                severity_delta_quad_stats_by_mode["city_type_ant_severity"].setdefault(key, QuadStats()).update(
                    delta_prior[mm_sev],
                    delta_target[mm_sev],
                )

        results: Dict[str, Dict[str, ErrStats]] = {
            "raw_prior": _empty_regime_errstats(),
            "global_affine": _empty_regime_errstats(),
            "city_type_affine": _empty_regime_errstats(),
            "city_type_los_affine": _empty_regime_errstats(),
            "city_type_los_ant_affine": _empty_regime_errstats(),
            "city_type_los_ant_quadratic": _empty_regime_errstats(),
            "delta_nlos_city_type_ant_affine": _empty_regime_errstats(),
            "delta_nlos_city_type_ant_quadratic": _empty_regime_errstats(),
            "delta_nlos_city_type_ant_severity_affine": _empty_regime_errstats(),
            "delta_nlos_city_type_ant_severity_quadratic": _empty_regime_errstats(),
            "global_affine_support_scaled": _empty_regime_errstats(),
            "city_type_los_affine_support_scaled": _empty_regime_errstats(),
            "city_type_los_ant_affine_support_scaled": _empty_regime_errstats(),
            "city_type_los_ant_quadratic_support_scaled": _empty_regime_errstats(),
        }
        support_val_valid = 0
        support_val_zero = 0

        for idx in range(len(val_ds)):
            city, sample = val_ds.sample_refs[idx]
            density, height, ant = sample_cache.get((city, sample), _sample_metadata(handle, city, sample))
            city_type = city_type_map.get(city, _city_type_for_stats(density, height, density_q1, density_q2, height_q1, height_q2))
            ant_label = _ant_bin(ant, ant_q1, ant_q2)

            x, y, m, sc = data_utils.unpack_cgan_batch(val_ds[idx], device)
            x, y, m, sc = _ensure_batch_dims(x, y, m, sc)
            prior = _denormalize_channel(x[:, formula_idx : formula_idx + 1], target_meta).detach().cpu().numpy()[0, 0]
            target = _denormalize_channel(y[:, :1], target_meta).detach().cpu().numpy()[0, 0]
            base_mask = (m[:, :1].detach().cpu().numpy()[0, 0] > 0.0)
            ground_mask = _ground_mask_from_hdf5(handle, city, sample, image_size)
            mask = base_mask & ground_mask
            los = x[:, 1:2].detach().cpu().numpy()[0, 0] > 0.5
            shadow_depth = (
                x[:, channel_layout["shadow_depth"] : channel_layout["shadow_depth"] + 1].detach().cpu().numpy()[0, 0]
                if "shadow_depth" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            distance_since_break = (
                x[:, channel_layout["distance_since_los_break"] : channel_layout["distance_since_los_break"] + 1]
                .detach()
                .cpu()
                .numpy()[0, 0]
                if "distance_since_los_break" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            blocker_height = (
                x[:, channel_layout["max_blocker_height"] : channel_layout["max_blocker_height"] + 1].detach().cpu().numpy()[0, 0]
                if "max_blocker_height" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            blocker_count = (
                x[:, channel_layout["blocker_count"] : channel_layout["blocker_count"] + 1].detach().cpu().numpy()[0, 0]
                if "blocker_count" in channel_layout
                else np.zeros_like(prior, dtype=np.float32)
            )
            los_reference = _compute_los_reference_db(
                data_utils,
                image_size=image_size,
                antenna_height_m=ant,
                receiver_height_m=float(formula_cfg.get("receiver_height_m", 1.5)),
                frequency_ghz=float(formula_cfg.get("frequency_ghz", 7.125)),
                meters_per_pixel=float(formula_cfg.get("meters_per_pixel", 1.0)),
                a2g_params=dict(formula_cfg.get("a2g_params", {})),
            )

            support_val_valid += int(mask.sum())
            support_val_zero += int((mask & np.isclose(target, 0.0)).sum())

            xv = prior[mask]
            yv = target[mask]
            raw_err = xv - yv
            results["raw_prior"]["overall"].update(raw_err)

            a_global, b_global, p_global = _solve_for_key(stats_by_mode["global"], ("all",))
            pred = a_global * xv + b_global
            global_err = pred - yv
            results["global_affine"]["overall"].update(global_err)
            results["global_affine_support_scaled"]["overall"].update((p_global * pred) - yv)

            a_city, b_city, _ = _solve_for_key(stats_by_mode["city_type"], (city_type,))
            city_err = (a_city * xv + b_city) - yv
            results["city_type_affine"]["overall"].update(city_err)

            for los_name, los_mask in (("LoS", los), ("NLoS", ~los)):
                mm = mask & los_mask
                if not np.any(mm):
                    continue
                xvl = prior[mm]
                yvl = target[mm]
                regime_name = los_name

                results["raw_prior"][regime_name].update(xvl - yvl)
                global_regime_pred = a_global * xvl + b_global
                results["global_affine"][regime_name].update(global_regime_pred - yvl)
                results["global_affine_support_scaled"][regime_name].update((p_global * global_regime_pred) - yvl)

                city_regime_pred = a_city * xvl + b_city
                results["city_type_affine"][regime_name].update(city_regime_pred - yvl)

                a, b, _ = _solve_for_key(stats_by_mode["city_type_los"], (city_type, los_name))
                pred_los = a * xvl + b
                results["city_type_los_affine"]["overall"].update(pred_los - yvl)
                results["city_type_los_affine"][regime_name].update(pred_los - yvl)

                a, b, p = _solve_for_key(stats_by_mode["city_type_los"], (city_type, los_name))
                pred_los_scaled_err = (p * pred_los) - yvl
                results["city_type_los_affine_support_scaled"]["overall"].update(pred_los_scaled_err)
                results["city_type_los_affine_support_scaled"][regime_name].update(pred_los_scaled_err)

                a, b, _ = _solve_for_key(stats_by_mode["city_type_los_ant"], (city_type, los_name, ant_label))
                pred_ant = a * xvl + b
                pred_ant_err = pred_ant - yvl
                results["city_type_los_ant_affine"]["overall"].update(pred_ant_err)
                results["city_type_los_ant_affine"][regime_name].update(pred_ant_err)

                a, b, p = _solve_for_key(stats_by_mode["city_type_los_ant"], (city_type, los_name, ant_label))
                pred_ant_scaled_err = (p * pred_ant) - yvl
                results["city_type_los_ant_affine_support_scaled"]["overall"].update(pred_ant_scaled_err)
                results["city_type_los_ant_affine_support_scaled"][regime_name].update(pred_ant_scaled_err)

                a2, a1, a0, p = _solve_quadratic_for_key(quad_stats_by_mode["city_type_los_ant"], (city_type, los_name, ant_label))
                pred_quad = (a2 * xvl * xvl) + (a1 * xvl) + a0
                pred_quad_err = pred_quad - yvl
                results["city_type_los_ant_quadratic"]["overall"].update(pred_quad_err)
                results["city_type_los_ant_quadratic"][regime_name].update(pred_quad_err)
                pred_quad_scaled_err = (p * pred_quad) - yvl
                results["city_type_los_ant_quadratic_support_scaled"]["overall"].update(pred_quad_scaled_err)
                results["city_type_los_ant_quadratic_support_scaled"][regime_name].update(pred_quad_scaled_err)

            los_regime_mask = mask & los
            if np.any(los_regime_mask):
                los_ref = los_reference[los_regime_mask]
                los_tgt = target[los_regime_mask]
                los_delta_err = los_ref - los_tgt
                results["delta_nlos_city_type_ant_affine"]["overall"].update(los_delta_err)
                results["delta_nlos_city_type_ant_affine"]["LoS"].update(los_delta_err)
                results["delta_nlos_city_type_ant_quadratic"]["overall"].update(los_delta_err)
                results["delta_nlos_city_type_ant_quadratic"]["LoS"].update(los_delta_err)
                results["delta_nlos_city_type_ant_severity_affine"]["overall"].update(los_delta_err)
                results["delta_nlos_city_type_ant_severity_affine"]["LoS"].update(los_delta_err)
                results["delta_nlos_city_type_ant_severity_quadratic"]["overall"].update(los_delta_err)
                results["delta_nlos_city_type_ant_severity_quadratic"]["LoS"].update(los_delta_err)

            nlos_regime_mask = mask & (~los)
            if np.any(nlos_regime_mask):
                delta_prior = prior[nlos_regime_mask] - los_reference[nlos_regime_mask]
                delta_target = target[nlos_regime_mask] - los_reference[nlos_regime_mask]
                a_d, b_d, _ = _solve_for_key(delta_stats_by_mode["city_type_ant"], (city_type, ant_label))
                pred_delta_aff = a_d * delta_prior + b_d
                pred_path_aff = los_reference[nlos_regime_mask] + pred_delta_aff
                pred_aff_err = pred_path_aff - target[nlos_regime_mask]
                results["delta_nlos_city_type_ant_affine"]["overall"].update(pred_aff_err)
                results["delta_nlos_city_type_ant_affine"]["NLoS"].update(pred_aff_err)

                a2_d, a1_d, a0_d, _ = _solve_quadratic_for_key(delta_quad_stats_by_mode["city_type_ant"], (city_type, ant_label))
                pred_delta_quad = a2_d * delta_prior * delta_prior + a1_d * delta_prior + a0_d
                pred_path_quad = los_reference[nlos_regime_mask] + pred_delta_quad
                pred_quad_delta_err = pred_path_quad - target[nlos_regime_mask]
                results["delta_nlos_city_type_ant_quadratic"]["overall"].update(pred_quad_delta_err)
                results["delta_nlos_city_type_ant_quadratic"]["NLoS"].update(pred_quad_delta_err)

                severity_score = (
                    0.45 * shadow_depth[nlos_regime_mask]
                    + 0.35 * distance_since_break[nlos_regime_mask]
                    + 0.20 * (0.5 * blocker_height[nlos_regime_mask] + 0.5 * blocker_count[nlos_regime_mask])
                )
                sev_q1 = 0.45 * shadow_q1 + 0.35 * break_q1 + 0.20 * blocker_q1
                sev_q2 = 0.45 * shadow_q2 + 0.35 * break_q2 + 0.20 * blocker_q2
                sev_bins = np.empty(severity_score.shape, dtype=object)
                sev_bins[severity_score <= sev_q1] = "sev_low"
                sev_bins[(severity_score > sev_q1) & (severity_score <= sev_q2)] = "sev_mid"
                sev_bins[severity_score > sev_q2] = "sev_high"
                pred_aff_full = np.empty_like(delta_prior)
                pred_quad_full = np.empty_like(delta_prior)
                for sev_name in ("sev_low", "sev_mid", "sev_high"):
                    mm_sev = sev_bins == sev_name
                    if not np.any(mm_sev):
                        continue
                    key = (city_type, ant_label, sev_name)
                    a_s, b_s, _ = _solve_for_key(severity_delta_stats_by_mode["city_type_ant_severity"], key)
                    pred_aff_full[mm_sev] = a_s * delta_prior[mm_sev] + b_s
                    a2_s, a1_s, a0_s, _ = _solve_quadratic_for_key(
                        severity_delta_quad_stats_by_mode["city_type_ant_severity"],
                        key,
                    )
                    pred_quad_full[mm_sev] = (
                        a2_s * delta_prior[mm_sev] * delta_prior[mm_sev]
                        + a1_s * delta_prior[mm_sev]
                        + a0_s
                    )
                pred_path_aff_sev = los_reference[nlos_regime_mask] + pred_aff_full
                pred_path_quad_sev = los_reference[nlos_regime_mask] + pred_quad_full
                pred_aff_sev_err = pred_path_aff_sev - target[nlos_regime_mask]
                pred_quad_sev_err = pred_path_quad_sev - target[nlos_regime_mask]
                results["delta_nlos_city_type_ant_severity_affine"]["overall"].update(pred_aff_sev_err)
                results["delta_nlos_city_type_ant_severity_affine"]["NLoS"].update(pred_aff_sev_err)
                results["delta_nlos_city_type_ant_severity_quadratic"]["overall"].update(pred_quad_sev_err)
                results["delta_nlos_city_type_ant_severity_quadratic"]["NLoS"].update(pred_quad_sev_err)
            if (idx + 1) % max(int(args.log_every), 1) == 0 or (idx + 1) == len(val_ds):
                print(
                    f"[prior-calibration] val {idx + 1}/{len(val_ds)} "
                    f"raw_rmse={results['raw_prior']['overall'].summary()['rmse_db']:.4f} "
                    f"best_so_far={min(v['overall'].summary()['rmse_db'] for v in results.values()):.4f}"
                )

    result_payload = {
        name: {regime: stats.summary() for regime, stats in regime_stats.items()}
        for name, regime_stats in results.items()
    }
    best_name, best_stats = min(result_payload.items(), key=lambda kv: kv[1]["overall"]["rmse_db"])

    payload: Dict[str, Any] = {
        "try_dir": str(try_dir),
        "config": str(config_path),
        "dataset": str(dataset_path),
        "support": {
            "train_valid_pixels": support_train_valid,
            "train_zero_pixels": support_train_zero,
            "train_zero_ratio": float(support_train_zero / max(support_train_valid, 1)),
            "val_valid_pixels": support_val_valid,
            "val_zero_pixels": support_val_zero,
            "val_zero_ratio": float(support_val_zero / max(support_val_valid, 1)),
        },
        "regimes": {
            "density_q1": density_q1,
            "density_q2": density_q2,
            "height_q1": height_q1,
            "height_q2": height_q2,
            "ant_q1": ant_q1,
            "ant_q2": ant_q2,
            "shadow_q1": shadow_q1,
            "shadow_q2": shadow_q2,
            "break_q1": break_q1,
            "break_q2": break_q2,
            "blocker_q1": blocker_q1,
            "blocker_q2": blocker_q2,
        },
        "results": result_payload,
        "best": {"name": best_name, **best_stats},
    }

    calibration_payload: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "split_seed": int(cfg["data"].get("split_seed", 42)),
        "val_ratio": float(cfg["data"].get("val_ratio", 0.15)),
        "test_ratio": float(cfg["data"].get("test_ratio", 0.15)),
        "official_metric_mask": "topology == 0 and dataset mask > 0",
        "city_type_thresholds": {
            "density_q1": density_q1,
            "density_q2": density_q2,
            "height_q1": height_q1,
            "height_q2": height_q2,
        },
        "antenna_height_thresholds": {
            "q1": ant_q1,
            "q2": ant_q2,
        },
        "city_type_by_city": city_type_map,
        "coefficients": {},
    }
    for key, stats in sorted(quad_stats_by_mode["city_type_los_ant"].items()):
        calibration_payload["coefficients"]["|".join(key)] = {
            "poly2": list(stats.coeffs()),
            "positive_rate": stats.positive_rate(),
            "count": stats.count,
        }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_calibration_json = Path(args.out_calibration_json)
    _ensure_parent(out_json)
    _ensure_parent(out_calibration_json)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(out_md, payload)
    out_calibration_json.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
