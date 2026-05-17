"""Recalibrate and evaluate the Try 78 path-loss prior on the Try 80 split.

The original Try 78 scripts used a two-way 70/30 city holdout.  Try 79 and
Try 80 use a three-way city holdout: train, validation, and final test.  This
script reruns the Try 78 calibration with the Try 80 training cities and then
evaluates the resulting hybrid prior on the requested Try 80 split.

The LoS branch reuses the original Try 78 two-ray calibration functions.  The
NLoS branch refits the deployed regime-wise linear map from the same feature
set used by the frozen Try 78/Try 80 evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np

import try78_hybrid_path_loss_reference as hybrid_ref
import try78_los_path_loss_prior as los_model


N_NONBIAS = hybrid_ref.N_FEAT - 1


def split_city_holdout_try80(
    refs: Sequence[los_model.SampleRef],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: int = 42,
) -> Tuple[List[los_model.SampleRef], List[los_model.SampleRef], List[los_model.SampleRef]]:
    """Copy of the Try 80 split contract, adapted to Try 78 SampleRef."""
    refs = list(refs)
    if len(refs) < 2:
        return refs, list(refs), []
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    total = len(refs)
    test_size = max(1, int(round(total * test_ratio))) if test_ratio > 0.0 else 0
    val_size = max(1, int(round(total * val_ratio))) if val_ratio > 0.0 else 0

    by_city: Dict[str, List[los_model.SampleRef]] = {}
    for ref in refs:
        by_city.setdefault(ref.city, []).append(ref)

    city_names = list(by_city.keys())
    rng = random.Random(split_seed)
    rng.shuffle(city_names)

    train_refs: List[los_model.SampleRef] = []
    val_refs: List[los_model.SampleRef] = []
    test_refs: List[los_model.SampleRef] = []
    test_city_count = 0
    val_city_count = 0

    for city in city_names:
        remaining = len(city_names) - test_city_count - val_city_count
        city_refs = by_city[city]
        if len(test_refs) < test_size and remaining > 2:
            test_refs.extend(city_refs)
            test_city_count += 1
            continue
        if len(val_refs) < val_size and remaining > 1:
            val_refs.extend(city_refs)
            val_city_count += 1
            continue
        train_refs.extend(city_refs)

    if not train_refs:
        rng.shuffle(refs)
        test_refs = refs[:test_size]
        val_refs = refs[test_size : test_size + val_size]
        train_refs = refs[test_size + val_size :]
    return train_refs, val_refs, test_refs


def _cities(refs: Iterable[los_model.SampleRef]) -> List[str]:
    return sorted({ref.city for ref in refs})


def _stable_sample_seed(seed: int, city: str, sample: str, label: str) -> int:
    token = f"{seed}|{city}|{sample}|{label}".encode("utf-8")
    digest = hashlib.blake2b(token, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _select_flat_indices(
    region: np.ndarray,
    *,
    max_pixels: int,
    seed: int,
) -> np.ndarray:
    idx = np.flatnonzero(region.reshape(-1))
    if max_pixels > 0 and idx.size > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_pixels, replace=False)
    return idx.astype(np.int64, copy=False)


def _fit_progress(prefix: str, idx: int, total: int, started: float, log_every: int) -> None:
    if idx % max(1, log_every) == 0 or idx == total:
        elapsed = time.time() - started
        rate = idx / elapsed if elapsed > 0 else 0.0
        print(f"{prefix} [{idx}/{total}] elapsed={elapsed:.1f}s rate={rate:.2f} samples/s")


def _new_moment() -> Dict[str, np.ndarray | int]:
    return {
        "count": 0,
        "sum": np.zeros(N_NONBIAS, dtype=np.float64),
        "sum2": np.zeros(N_NONBIAS, dtype=np.float64),
    }


def _new_normal_eq() -> Dict[str, np.ndarray | int]:
    return {
        "count": 0,
        "xtx": np.zeros((hybrid_ref.N_FEAT, hybrid_ref.N_FEAT), dtype=np.float64),
        "xty": np.zeros(hybrid_ref.N_FEAT, dtype=np.float64),
        "sum_y": 0.0,
        "sum_y2": 0.0,
    }


def _sample_regions(sample: Dict[str, np.ndarray]) -> Tuple[Tuple[str, np.ndarray], ...]:
    valid = sample["valid"]
    return (
        ("LoS", valid & (sample["los_mask"] > 0)),
        ("NLoS", valid & (sample["los_mask"] == 0)),
    )


def fit_nlos_regime_calibration(
    hdf5_path: Path,
    fit_refs: Sequence[los_model.SampleRef],
    *,
    max_pixels_per_region_sample: int,
    ridge_lambda: float,
    seed: int,
    log_every: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Fit the deployed Try 78 regime-wise linear NLoS calibration.

    The fitted coefficient vector is stored in the same raw feature space as
    the frozen Try 78 JSON.  Internally, the solve uses standardized non-bias
    features to keep the normal equations stable, then converts back.
    """
    moments: Dict[str, Dict[str, np.ndarray | int]] = {}
    started = time.time()

    print("[try78-try80] NLoS calibration pass 1/2: feature moments")
    with h5py.File(str(hdf5_path), "r") as handle:
        for idx, ref in enumerate(fit_refs, start=1):
            sample = hybrid_ref.load_hybrid_sample(handle, ref)
            if not sample["valid"].any():
                _fit_progress("  pass1", idx, len(fit_refs), started, log_every)
                continue

            ct = hybrid_ref.sample_city_type(sample["topology"])
            ab = hybrid_ref.ant_bin(ref.uav_height_m)
            prior = hybrid_ref.compute_formula_prior(sample["los_mask"], ref.uav_height_m)
            x_all = hybrid_ref.compute_pixel_features(sample["topology"], sample["los_mask"], prior, ref.uav_height_m)
            x_flat = x_all.reshape(-1, hybrid_ref.N_FEAT).astype(np.float64, copy=False)

            for label, region in _sample_regions(sample):
                if not region.any():
                    continue
                key = hybrid_ref.regime_key(ct, label, ab)
                idx_flat = _select_flat_indices(
                    region,
                    max_pixels=max_pixels_per_region_sample,
                    seed=_stable_sample_seed(seed, ref.city, ref.sample, label),
                )
                if idx_flat.size == 0:
                    continue
                data = x_flat[idx_flat, :N_NONBIAS]
                item = moments.setdefault(key, _new_moment())
                item["count"] = int(item["count"]) + int(idx_flat.size)
                item["sum"] = item["sum"] + data.sum(axis=0)
                item["sum2"] = item["sum2"] + np.square(data).sum(axis=0)

            _fit_progress("  pass1", idx, len(fit_refs), started, log_every)

    scaler: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key, item in moments.items():
        count = max(int(item["count"]), 1)
        mean = np.asarray(item["sum"], dtype=np.float64) / count
        var = np.asarray(item["sum2"], dtype=np.float64) / count - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-8))
        std[std < 1e-4] = 1.0
        scaler[key] = (mean, std)

    normal_eq: Dict[str, Dict[str, np.ndarray | int | float]] = {}
    started = time.time()
    print("[try78-try80] NLoS calibration pass 2/2: normal equations")
    with h5py.File(str(hdf5_path), "r") as handle:
        for idx, ref in enumerate(fit_refs, start=1):
            sample = hybrid_ref.load_hybrid_sample(handle, ref)
            if not sample["valid"].any():
                _fit_progress("  pass2", idx, len(fit_refs), started, log_every)
                continue

            ct = hybrid_ref.sample_city_type(sample["topology"])
            ab = hybrid_ref.ant_bin(ref.uav_height_m)
            prior = hybrid_ref.compute_formula_prior(sample["los_mask"], ref.uav_height_m)
            x_all = hybrid_ref.compute_pixel_features(sample["topology"], sample["los_mask"], prior, ref.uav_height_m)
            x_flat = x_all.reshape(-1, hybrid_ref.N_FEAT).astype(np.float64, copy=False)
            y_flat = sample["path_loss"].reshape(-1).astype(np.float64, copy=False)

            for label, region in _sample_regions(sample):
                if not region.any():
                    continue
                key = hybrid_ref.regime_key(ct, label, ab)
                if key not in scaler:
                    continue
                idx_flat = _select_flat_indices(
                    region,
                    max_pixels=max_pixels_per_region_sample,
                    seed=_stable_sample_seed(seed, ref.city, ref.sample, label),
                )
                if idx_flat.size == 0:
                    continue
                mean, std = scaler[key]
                x = x_flat[idx_flat]
                x_std = np.empty((idx_flat.size, hybrid_ref.N_FEAT), dtype=np.float64)
                x_std[:, :N_NONBIAS] = (x[:, :N_NONBIAS] - mean) / std
                x_std[:, -1] = 1.0
                y = y_flat[idx_flat]

                item = normal_eq.setdefault(key, _new_normal_eq())
                item["count"] = int(item["count"]) + int(idx_flat.size)
                item["xtx"] = item["xtx"] + x_std.T @ x_std
                item["xty"] = item["xty"] + x_std.T @ y
                item["sum_y"] = float(item["sum_y"]) + float(y.sum())
                item["sum_y2"] = float(item["sum_y2"]) + float(np.square(y).sum())

            _fit_progress("  pass2", idx, len(fit_refs), started, log_every)

    coefs: Dict[str, np.ndarray] = {}
    fit_counts: Dict[str, int] = {}
    fit_rmse: Dict[str, float] = {}
    for key in sorted(normal_eq):
        item = normal_eq[key]
        count = int(item["count"])
        if count < hybrid_ref.N_FEAT:
            continue
        reg = ridge_lambda * np.eye(hybrid_ref.N_FEAT, dtype=np.float64)
        reg[-1, -1] = 0.0
        beta_std = np.linalg.solve(np.asarray(item["xtx"]) + reg, np.asarray(item["xty"]))
        mean, std = scaler[key]

        beta_raw = np.zeros(hybrid_ref.N_FEAT, dtype=np.float64)
        beta_raw[:N_NONBIAS] = beta_std[:N_NONBIAS] / std
        beta_raw[-1] = beta_std[-1] - float(np.sum(beta_std[:N_NONBIAS] * mean / std))
        coefs[key] = beta_raw
        fit_counts[key] = count

        sse = (
            float(item["sum_y2"])
            - 2.0 * float(beta_std @ np.asarray(item["xty"]))
            + float(beta_std @ np.asarray(item["xtx"]) @ beta_std)
        )
        fit_rmse[key] = math.sqrt(max(sse, 0.0) / max(count, 1))

    meta = {
        "n_fit_samples": len(fit_refs),
        "max_pixels_per_region_sample": max_pixels_per_region_sample,
        "ridge_lambda": ridge_lambda,
        "seed": seed,
        "fit_counts": fit_counts,
        "fit_rmse_db": fit_rmse,
    }
    return coefs, meta


def save_nlos_calibration(path: Path, coefs: Dict[str, np.ndarray], meta: Dict[str, object]) -> None:
    payload = {
        "model_type": "regime_obstruction_multiscale_try78",
        "feature_names": list(hybrid_ref.FEATURE_NAMES),
        "freq_ghz": hybrid_ref.FREQ_GHZ,
        "rx_height_m": hybrid_ref.RX_HEIGHT_M,
        "meters_per_pixel": hybrid_ref.METERS_PER_PIXEL,
        "height_scale": hybrid_ref.HEIGHT_SCALE,
        "kernel_sizes": list(hybrid_ref.KERNEL_SIZES),
        "city_type_thresholds": {
            "density_q1": hybrid_ref.DENSITY_Q1,
            "density_q2": hybrid_ref.DENSITY_Q2,
            "height_q1": hybrid_ref.HEIGHT_Q1,
            "height_q2": hybrid_ref.HEIGHT_Q2,
        },
        "antenna_height_thresholds": {
            "q1": hybrid_ref.ANT_Q1,
            "q2": hybrid_ref.ANT_Q2,
        },
        "meta": meta,
        "coefficients": {key: value.tolist() for key, value in coefs.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _accum(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Tuple[float, float, int]:
    if not mask.any():
        return 0.0, 0.0, 0
    err = pred[mask] - target[mask]
    return float(np.sum(err * err)), float(np.sum(np.abs(err))), int(mask.sum())


def _rmse(sse: float, n: int) -> float:
    return math.sqrt(sse / n) if n else float("nan")


def _mae(sae: float, n: int) -> float:
    return sae / n if n else float("nan")


def _metrics(prefix: str, sse: float, sae: float, n: int) -> Dict[str, float | int]:
    return {
        f"{prefix}_rmse_pw": _rmse(sse, n),
        f"{prefix}_mae_pw": _mae(sae, n),
        f"{prefix}_pixels": n,
    }


def evaluate_hybrid(
    hdf5_path: Path,
    eval_refs: Sequence[los_model.SampleRef],
    radial_calibration: Dict[str, np.ndarray],
    two_ray_calibration: Dict[str, np.ndarray],
    nlos_coefs: Dict[str, np.ndarray],
    *,
    log_every: int,
) -> Dict[str, object]:
    fspl_los_sse = radial_los_sse = two_los_sse = 0.0
    fspl_los_sae = radial_los_sae = two_los_sae = 0.0
    hybrid_all_sse = hybrid_los_sse = hybrid_nlos_sse = 0.0
    hybrid_all_sae = hybrid_los_sae = hybrid_nlos_sae = 0.0
    n_los = n_nlos = n_all = 0
    per_sample = []
    started = time.time()

    with h5py.File(str(hdf5_path), "r") as handle:
        for idx, ref in enumerate(eval_refs, start=1):
            sample = hybrid_ref.load_hybrid_sample(handle, ref)
            if not sample["valid"].any():
                _fit_progress("  eval", idx, len(eval_refs), started, log_every)
                continue

            ct = hybrid_ref.sample_city_type(sample["topology"])
            ab = hybrid_ref.ant_bin(ref.uav_height_m)

            fspl_map = los_model.fspl_db(ref.uav_height_m)
            radial_map = los_model.predict_radial_map(ref.uav_height_m, radial_calibration)
            two_ray_map = los_model.predict_two_ray_map(ref.uav_height_m, two_ray_calibration)

            formula_prior = hybrid_ref.compute_formula_prior(sample["los_mask"], ref.uav_height_m)
            x_all = hybrid_ref.compute_pixel_features(sample["topology"], sample["los_mask"], formula_prior, ref.uav_height_m)
            nlos_map = hybrid_ref.apply_calibration(formula_prior, x_all, sample["los_mask"], ct, ab, nlos_coefs)

            hybrid_map = two_ray_map.copy()
            hybrid_map[sample["los_mask"] == 0] = nlos_map[sample["los_mask"] == 0]

            target = sample["path_loss"]
            valid = sample["valid"]
            los_mask = valid & (sample["los_mask"] > 0)
            nlos_mask = valid & (sample["los_mask"] == 0)

            sse, sae, cnt = _accum(fspl_map, target, los_mask)
            fspl_los_sse += sse
            fspl_los_sae += sae

            sse, sae, _ = _accum(radial_map, target, los_mask)
            radial_los_sse += sse
            radial_los_sae += sae

            sse, sae, cnt_los = _accum(two_ray_map, target, los_mask)
            two_los_sse += sse
            two_los_sae += sae
            n_los += cnt_los

            sse, sae, cnt_all = _accum(hybrid_map, target, valid)
            hybrid_all_sse += sse
            hybrid_all_sae += sae
            n_all += cnt_all

            sse, sae, _ = _accum(hybrid_map, target, los_mask)
            hybrid_los_sse += sse
            hybrid_los_sae += sae

            sse, sae, cnt_nlos = _accum(hybrid_map, target, nlos_mask)
            hybrid_nlos_sse += sse
            hybrid_nlos_sae += sae
            n_nlos += cnt_nlos

            per_sample.append({
                "city": ref.city,
                "sample": ref.sample,
                "uav_height_m": ref.uav_height_m,
                "topology_class": ct,
                "antenna_bin": ab,
                "n_los": int(los_mask.sum()),
                "n_nlos": int(nlos_mask.sum()),
                "fspl_los_rmse": _rmse(*_accum(fspl_map, target, los_mask)[::2]),
                "radial_los_rmse": _rmse(*_accum(radial_map, target, los_mask)[::2]),
                "two_ray_los_rmse": _rmse(*_accum(two_ray_map, target, los_mask)[::2]),
                "hybrid_overall_rmse": _rmse(*_accum(hybrid_map, target, valid)[::2]),
                "hybrid_los_rmse": _rmse(*_accum(hybrid_map, target, los_mask)[::2]),
                "hybrid_nlos_rmse": _rmse(*_accum(hybrid_map, target, nlos_mask)[::2]),
                "hybrid_overall_mae": _mae(_accum(hybrid_map, target, valid)[1], int(valid.sum())),
                "hybrid_los_mae": _mae(_accum(hybrid_map, target, los_mask)[1], int(los_mask.sum())),
                "hybrid_nlos_mae": _mae(_accum(hybrid_map, target, nlos_mask)[1], int(nlos_mask.sum())),
            })

            if idx % max(1, log_every) == 0 or idx == len(eval_refs):
                print(
                    f"  eval [{idx}/{len(eval_refs)}] "
                    f"los_2ray={_rmse(two_los_sse, n_los):.4f} "
                    f"hybrid={_rmse(hybrid_all_sse, n_all):.4f} "
                    f"nlos={_rmse(hybrid_nlos_sse, n_nlos):.4f}"
                )

    aggregate = {}
    aggregate.update(_metrics("fspl_los", fspl_los_sse, fspl_los_sae, n_los))
    aggregate.update(_metrics("radial_los", radial_los_sse, radial_los_sae, n_los))
    aggregate.update(_metrics("two_ray_los", two_los_sse, two_los_sae, n_los))
    aggregate.update(_metrics("hybrid_overall", hybrid_all_sse, hybrid_all_sae, n_all))
    aggregate.update(_metrics("hybrid_los", hybrid_los_sse, hybrid_los_sae, n_los))
    aggregate.update(_metrics("hybrid_nlos", hybrid_nlos_sse, hybrid_nlos_sae, n_nlos))
    aggregate["total_valid_pixels"] = n_all
    aggregate["total_los_pixels"] = n_los
    aggregate["total_nlos_pixels"] = n_nlos
    return {"aggregate": aggregate, "per_sample": per_sample}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=Path("c:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5"))
    parser.add_argument("--out-dir", type=Path, default=Path("c:/TFG/TFGpractice/TFGSeventyEighthTry78/hybrid_out_try80_split"))
    parser.add_argument("--eval-split", choices=("val", "test", "valtest"), default="test")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--height-bin-m", type=float, default=5.0)
    parser.add_argument("--nlos-pixels-per-region-sample", type=int, default=1024)
    parser.add_argument("--nlos-ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--reuse-calibrations", action="store_true")
    parser.add_argument("--max-fit-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = args.out_dir / "calibrations"
    cal_dir.mkdir(parents=True, exist_ok=True)

    refs = los_model.enumerate_samples(args.hdf5)
    train_refs, val_refs, test_refs = split_city_holdout_try80(
        refs,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    if args.max_fit_samples is not None:
        train_refs = los_model.subsample_refs(train_refs, args.max_fit_samples, seed=args.seed)

    if args.eval_split == "val":
        eval_refs = val_refs
    elif args.eval_split == "test":
        eval_refs = test_refs
    else:
        eval_refs = list(val_refs) + list(test_refs)
    if args.max_eval_samples is not None:
        eval_refs = los_model.subsample_refs(eval_refs, args.max_eval_samples, seed=args.seed + 1)

    print(
        "[try78-try80] split "
        f"train={len(train_refs)} samples/{len(_cities(train_refs))} cities; "
        f"val={len(val_refs)} samples/{len(_cities(val_refs))} cities; "
        f"test={len(test_refs)} samples/{len(_cities(test_refs))} cities; "
        f"eval={args.eval_split}:{len(eval_refs)}"
    )

    los_cal_path = cal_dir / "try78_los_two_ray_calibration_try80split.json"
    nlos_cal_path = cal_dir / "try78_nlos_regime_calibration_try80split.json"

    if args.reuse_calibrations and los_cal_path.exists() and nlos_cal_path.exists():
        print("[try78-try80] reusing existing calibration JSONs")
        radial_calibration, two_ray_calibration = los_model.load_calibration(los_cal_path)
        nlos_payload = json.loads(nlos_cal_path.read_text(encoding="utf-8"))
        nlos_coefs = {key: np.asarray(value, dtype=np.float64) for key, value in nlos_payload["coefficients"].items()}
    else:
        print("[try78-try80] fitting LoS radial/two-ray calibration on Try80 train cities")
        radial_calibration = los_model.fit_radial_calibration(
            args.hdf5, train_refs, height_bin_m=args.height_bin_m, verbose=True
        )
        two_ray_calibration = los_model.fit_two_ray_calibration(
            args.hdf5, train_refs, height_bin_m=args.height_bin_m, seed=args.seed, verbose=True
        )
        two_ray_residual = los_model.fit_two_ray_residual_calibration(
            args.hdf5, train_refs, two_ray_calibration, height_bin_m=args.height_bin_m, verbose=True
        )
        two_ray_calibration.update(two_ray_residual)
        los_model.save_calibration(
            radial_calibration,
            two_ray_calibration,
            los_cal_path,
            meta={
                "split_contract": "try80_city_holdout",
                "n_fit_samples": len(train_refs),
                "n_eval_samples": len(eval_refs),
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "split_seed": args.split_seed,
                "fit_cities": _cities(train_refs),
                "eval_split": args.eval_split,
                "eval_cities": _cities(eval_refs),
            },
        )
        print(f"[try78-try80] wrote {los_cal_path}")

        print("[try78-try80] fitting NLoS regime calibration on Try80 train cities")
        nlos_coefs, nlos_meta = fit_nlos_regime_calibration(
            args.hdf5,
            train_refs,
            max_pixels_per_region_sample=args.nlos_pixels_per_region_sample,
            ridge_lambda=args.nlos_ridge_lambda,
            seed=args.seed,
            log_every=args.log_every,
        )
        nlos_meta.update({
            "split_contract": "try80_city_holdout",
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "split_seed": args.split_seed,
            "fit_cities": _cities(train_refs),
            "eval_split": args.eval_split,
            "eval_cities": _cities(eval_refs),
        })
        save_nlos_calibration(nlos_cal_path, nlos_coefs, nlos_meta)
        print(f"[try78-try80] wrote {nlos_cal_path}")

    print("[try78-try80] evaluating hybrid prior")
    eval_payload = evaluate_hybrid(
        args.hdf5,
        eval_refs,
        radial_calibration,
        two_ray_calibration,
        nlos_coefs,
        log_every=args.log_every,
    )

    summary = {
        "split_contract": {
            "name": "try80_city_holdout",
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "split_seed": args.split_seed,
            "train_samples": len(train_refs),
            "val_samples": len(val_refs),
            "test_samples": len(test_refs),
            "eval_split": args.eval_split,
            "eval_samples": len(eval_refs),
            "train_cities": _cities(train_refs),
            "val_cities": _cities(val_refs),
            "test_cities": _cities(test_refs),
            "eval_cities": _cities(eval_refs),
        },
        "calibrations": {
            "los": str(los_cal_path),
            "nlos": str(nlos_cal_path),
        },
        **eval_payload,
    }
    summary_path = args.out_dir / f"hybrid_eval_summary_try80_{args.eval_split}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[try78-try80] summary -> {summary_path}")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
