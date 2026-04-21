"""Try 78 — physics prior: regime-aware obstruction calibration (no DL).

Ports the ``regime_obstruction_multiscale_v1`` system from Try 47 to pure
numpy/scipy, evaluated on a city-holdout split matching the rest of the repo.

Algorithm
---------
1. Base formula: ``hybrid_two_ray_cost231_a2g_nlos`` — FSPL on LoS side,
   COST-231 Hata + A2G exponential term on NLoS side, blended by los_mask.
2. Per-pixel local features (scipy.ndimage avg-pool at 15 and 41 px):
   - local building density  (fraction of non-ground pixels in kernel)
   - local mean building height
   - local NLoS support  (fraction of NLoS pixels in kernel)
3. Elevation-angle features:
   - theta_norm = atan2(h_tx - 1.5, d_2D) / 90°
   - shadow_sigma_db from Al-Hourani/ITU empirical params
4. Design matrix (15 cols, last = bias):
   prior², prior, log1p(d2d_m), density_15, density_41,
   height_15/90, height_41/90, density_41*logd,
   nlos_15, nlos_41, nlos_41*logd, sigma_db, theta_norm,
   nlos_41*theta_norm, 1
5. Ridge regression fitted per regime (city_type × LoS/NLoS × ant_bin).
6. Evaluation on held-out cities (no GT from eval during fitting).

Usage
-----
  # Fit on 70 % of data (city holdout) and evaluate on the rest:
  python prior_try78.py \
      --hdf5 c:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5 \
      --out-dir prior_out

  # Smoke test (small sample cap):
  python prior_try78.py --max-samples 200 --skip-plots

  # Apply a pre-fitted calibration JSON to new data:
  python prior_try78.py --calibration-json prior_out/calibration.json --skip-fit
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.ndimage import uniform_filter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE: int = 513
TX_ROW: int = 256
TX_COL: int = 256
METERS_PER_PIXEL: float = 1.0
FREQ_GHZ: float = 7.125
RX_HEIGHT_M: float = 1.5
PATH_LOSS_MIN_DB: float = 20.0
HEIGHT_SCALE: float = 90.0           # building-height normalisation (matches Try 76)
KERNEL_SIZES: Tuple[int, int] = (15, 41)
RIDGE_LAMBDA: float = 1e-3
DEFAULT_HDF5 = Path("c:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5")

# Al-Hourani / ITU A2G shadow-sigma parameters (from Try 47 data_utils)
SIGMA_LOS_RHO: float = 0.0272
SIGMA_LOS_MU: float = 0.7475
SIGMA_NLOS_RHO: float = 2.3197
SIGMA_NLOS_MU: float = 0.2361

# COST-231 / A2G NLoS parameters (from Try 67 data_utils hybrid formula)
A2G_Nlos_BIAS: float = -16.16
A2G_NLoS_AMP: float = 12.0436
A2G_NLoS_TAU: float = 7.52
A2G_LOS_LOG_COEFF: float = -20.0
A2G_LOS_BIAS: float = 0.0

# City-type thresholds fitted on the full dataset (Try 41 calibration JSON)
DENSITY_Q1: float = 0.1957
DENSITY_Q2: float = 0.2549
HEIGHT_Q1: float = 10.91
HEIGHT_Q2: float = 15.95
ANT_Q1: float = 58.12   # height tertile boundaries (m)
ANT_Q2: float = 103.85

FEATURE_NAMES: Tuple[str, ...] = (
    "prior_sq", "prior", "log1p_d2d",
    "density_15", "density_41",
    "height_15", "height_41",
    "density_41_x_logd",
    "nlos_15", "nlos_41",
    "nlos_41_x_logd",
    "shadow_sigma",
    "theta_norm",
    "nlos_41_x_theta",
    "bias",
)
N_FEAT: int = len(FEATURE_NAMES)   # 15


# ---------------------------------------------------------------------------
# Geometry helpers (numpy, no torch)
# ---------------------------------------------------------------------------

def _d2d_map() -> np.ndarray:
    ii, jj = np.indices((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    dy = (ii - TX_ROW) * METERS_PER_PIXEL
    dx = (jj - TX_COL) * METERS_PER_PIXEL
    return np.sqrt(dy * dy + dx * dx).astype(np.float32)


def _d3d_map(d2d: np.ndarray, h_tx: float) -> np.ndarray:
    dz = float(h_tx) - RX_HEIGHT_M
    return np.sqrt(d2d * d2d + dz * dz).astype(np.float32)


# Pre-compute once globally
_D2D = _d2d_map()


# ---------------------------------------------------------------------------
# Base formula: hybrid_two_ray_cost231_a2g_nlos  (numpy port of Try 67/47)
# ---------------------------------------------------------------------------

def compute_formula_prior(
    los_mask: np.ndarray,
    h_tx: float,
    freq_ghz: float = FREQ_GHZ,
    rx_h: float = RX_HEIGHT_M,
) -> np.ndarray:
    """Per-pixel path-loss prior (dB) — numpy port of hybrid formula.

    On LoS pixels: blend of FSPL and A2G-LoS term.
    On NLoS pixels: max(COST-231 Hata, A2G-NLoS exponential).
    """
    h_tx_c = max(float(h_tx), 1.0)
    h_rx_c = max(float(rx_h), 0.5)
    d2d = _D2D.astype(np.float64)
    d3d = np.sqrt(d2d ** 2 + (h_tx_c - h_rx_c) ** 2)
    d2d = np.maximum(d2d, 1.0)
    d3d = np.maximum(d3d, 1.0)

    freq_mhz = freq_ghz * 1000.0
    log_f = math.log10(freq_mhz)

    # FSPL
    fspl = 32.45 + 20.0 * np.log10(d3d / 1000.0) + 20.0 * math.log10(freq_mhz)

    # COST-231 Hata (NLoS urban macro)
    a_hm = (1.1 * log_f - 0.7) * h_rx_c - (1.56 * log_f - 0.8)
    d_km = np.maximum(d2d / 1000.0, 0.001)
    hb_log = math.log10(max(h_tx_c, 1.0))
    cost231 = (
        46.3 + 33.9 * log_f - 13.82 * hb_log - a_hm
        + (44.9 - 6.55 * hb_log) * np.log10(d_km)
        + 3.0
    )

    # Two-ray crossover (incoherent) — stays at FSPL inside CKM scene at 7 GHz
    wavelength = 0.299792458 / freq_ghz
    crossover = max(4.0 * math.pi * h_tx_c * h_rx_c / wavelength, 1.0)
    two_ray = 40.0 * np.log10(d3d) - 20.0 * math.log10(h_tx_c) - 20.0 * math.log10(h_rx_c)
    los_path = np.where(d3d <= crossover, fspl, two_ray)

    # A2G terms
    theta_deg = np.degrees(np.arctan2(h_tx_c - h_rx_c, np.maximum(d2d, 1.0)))
    sin_theta = np.clip(np.sin(np.radians(theta_deg)), 1e-4, 1.0)
    lambda0_db = 20.0 * math.log10((4.0 * math.pi * h_tx_c * freq_ghz * 1e9) / 299792458.0)
    a2g_los = lambda0_db + A2G_LOS_BIAS + A2G_LOS_LOG_COEFF * np.log10(sin_theta)
    a2g_nlos = lambda0_db + (A2G_Nlos_BIAS + A2G_NLoS_AMP * np.exp(-(90.0 - theta_deg) / A2G_NLoS_TAU))
    nlos_path = np.maximum(cost231, a2g_nlos)

    los_prob = los_mask.astype(np.float64)
    los_blend = 0.7 * los_path + 0.3 * np.minimum(los_path, a2g_los)
    prior = los_prob * los_blend + (1.0 - los_prob) * nlos_path
    return np.clip(prior, 0.0, 180.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-pixel feature computation
# ---------------------------------------------------------------------------

def _avg_pool(arr: np.ndarray, k: int) -> np.ndarray:
    """scipy uniform_filter — equivalent to avg_pool2d with 'same' padding."""
    # uniform_filter size must be odd; if even, increment by 1
    size = k if k % 2 == 1 else k + 1
    return uniform_filter(arr.astype(np.float32), size=size, mode="reflect")


def compute_pixel_features(
    topology: np.ndarray,
    los_mask: np.ndarray,
    prior_db: np.ndarray,
    h_tx: float,
) -> np.ndarray:
    """Compute per-pixel design matrix (H, W, N_FEAT) from observable inputs.

    All inputs are numpy arrays; no ground truth (path_loss) is used.
    """
    ground = (topology == 0.0).astype(np.float32)
    building_mask = (1.0 - ground).astype(np.float32)
    nlos_mask_f = ((los_mask <= 0.5) & (ground > 0)).astype(np.float32)

    d2d = _D2D.astype(np.float32)
    logd = np.log1p(d2d)

    # Local building density and mean height at two kernel scales
    density_15 = _avg_pool(building_mask, KERNEL_SIZES[0])
    density_41 = _avg_pool(building_mask, KERNEL_SIZES[1])
    bh_sum_15  = _avg_pool(topology * building_mask, KERNEL_SIZES[0])
    bh_sum_41  = _avg_pool(topology * building_mask, KERNEL_SIZES[1])
    # mean height = avg(height) / normalization; where building_mask ~ 0, height ~ 0 too
    height_15  = np.clip(bh_sum_15 / HEIGHT_SCALE, 0.0, None)
    height_41  = np.clip(bh_sum_41 / HEIGHT_SCALE, 0.0, None)

    # Local NLoS support (fraction of NLoS ground pixels in kernel)
    nlos_15 = np.clip(_avg_pool(nlos_mask_f, KERNEL_SIZES[0]), 0.0, 1.0)
    nlos_41 = np.clip(_avg_pool(nlos_mask_f, KERNEL_SIZES[1]), 0.0, 1.0)

    # Elevation-angle shadow sigma (Al-Hourani empirical)
    h_tx_c = max(float(h_tx), 1.0)
    theta_deg = np.degrees(np.arctan2(h_tx_c - RX_HEIGHT_M, np.maximum(d2d, 1.0))).astype(np.float32)
    los_prob = los_mask.astype(np.float32)
    sigma_los  = SIGMA_LOS_RHO  * np.power(np.clip(90.0 - theta_deg, 0.0, None), SIGMA_LOS_MU)
    sigma_nlos = SIGMA_NLOS_RHO * np.power(np.clip(90.0 - theta_deg, 0.0, None), SIGMA_NLOS_MU)
    shadow_sigma = los_prob * sigma_los + (1.0 - los_prob) * sigma_nlos

    theta_norm = np.clip(theta_deg / 90.0, 0.0, 1.0)

    # Assemble design matrix (H, W, 15)
    bias = np.ones_like(prior_db)
    X = np.stack([
        prior_db * prior_db,   # 0  prior²
        prior_db,              # 1  prior
        logd,                  # 2  log1p(d2d_m)
        density_15,            # 3
        density_41,            # 4
        height_15,             # 5
        height_41,             # 6
        density_41 * logd,     # 7
        nlos_15,               # 8
        nlos_41,               # 9
        nlos_41 * logd,        # 10
        shadow_sigma,          # 11
        theta_norm,            # 12
        nlos_41 * theta_norm,  # 13
        bias,                  # 14  (unregularised)
    ], axis=-1)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Regime helpers
# ---------------------------------------------------------------------------

def city_type(density: float, mean_bh: float) -> str:
    if density >= DENSITY_Q2 or mean_bh >= HEIGHT_Q2:
        return "dense_highrise"
    if density <= DENSITY_Q1 and mean_bh <= HEIGHT_Q1:
        return "open_lowrise"
    return "mixed_midrise"


def ant_bin(h_tx: float) -> str:
    if h_tx <= ANT_Q1:
        return "low_ant"
    if h_tx <= ANT_Q2:
        return "mid_ant"
    return "high_ant"


def regime_key(ct: str, los_label: str, ab: str) -> str:
    return f"{ct}|{los_label}|{ab}"


def sample_city_type(topology: np.ndarray) -> str:
    ground = topology == 0.0
    non_ground = ~ground
    density = float(non_ground.mean())
    bh = topology[non_ground]
    mean_bh = float(bh.mean()) if bh.size else 0.0
    return city_type(density, mean_bh)


# ---------------------------------------------------------------------------
# Online ridge accumulator per regime
# ---------------------------------------------------------------------------

@dataclass
class RegimeAccum:
    xtx: np.ndarray = field(default_factory=lambda: np.zeros((N_FEAT, N_FEAT), np.float64))
    xty: np.ndarray = field(default_factory=lambda: np.zeros(N_FEAT, np.float64))
    count: int = 0

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (N, 15) float64, y: (N,) float64."""
        if X.shape[0] == 0:
            return
        self.xtx += X.T @ X
        self.xty += X.T @ y
        self.count += int(X.shape[0])

    def solve(self, ridge: float = RIDGE_LAMBDA) -> Optional[np.ndarray]:
        if self.count < N_FEAT * 4:
            return None
        reg = np.eye(N_FEAT, dtype=np.float64) * ridge
        reg[-1, -1] = 0.0   # bias column unregularised
        try:
            return np.linalg.solve(self.xtx + reg, self.xty)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(self.xtx + reg, self.xty, rcond=None)[0]


# ---------------------------------------------------------------------------
# HDF5 enumeration / split
# ---------------------------------------------------------------------------

@dataclass
class SampleRef:
    city: str
    sample: str
    uav_height_m: float


def enumerate_samples(hdf5_path: Path, max_samples: Optional[int] = None) -> List[SampleRef]:
    refs: List[SampleRef] = []
    with h5py.File(str(hdf5_path), "r") as f:
        for city in sorted(f.keys()):
            for sample in sorted(f[city].keys()):
                h = float(np.asarray(f[city][sample]["uav_height"][...]).reshape(-1)[0])
                refs.append(SampleRef(city=city, sample=sample, uav_height_m=h))
                if max_samples and len(refs) >= max_samples:
                    return refs
    return refs


def split_city_holdout(
    refs: Sequence[SampleRef],
    eval_ratio: float = 0.30,
    seed: int = 42,
) -> Tuple[List[SampleRef], List[SampleRef]]:
    by_city: Dict[str, List[SampleRef]] = {}
    for r in refs:
        by_city.setdefault(r.city, []).append(r)
    cities = list(by_city.keys())
    random.Random(seed).shuffle(cities)
    total = len(refs)
    target = int(round(total * eval_ratio))
    fit_refs: List[SampleRef] = []
    eval_refs: List[SampleRef] = []
    for idx, c in enumerate(cities):
        remaining = len(cities) - idx
        if len(eval_refs) < target and remaining > 1:
            eval_refs.extend(by_city[c])
        else:
            fit_refs.extend(by_city[c])
    if not fit_refs or not eval_refs:
        # fallback: random 70/30 sample split (only with tiny --max-samples runs)
        all_refs = list(refs)
        random.Random(seed).shuffle(all_refs)
        cut = max(1, int(len(all_refs) * (1.0 - eval_ratio)))
        fit_refs, eval_refs = all_refs[:cut], all_refs[cut:]
    return fit_refs, eval_refs


# ---------------------------------------------------------------------------
# Load one sample from HDF5
# ---------------------------------------------------------------------------

def load_sample(
    handle: h5py.File,
    ref: SampleRef,
) -> Dict[str, np.ndarray]:
    grp = handle[ref.city][ref.sample]
    topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
    los = np.asarray(grp["los_mask"][...], dtype=np.uint8)
    pl = np.asarray(grp["path_loss"][...], dtype=np.float32)
    ground = (topology == 0.0)
    valid = ground & np.isfinite(pl) & (pl >= PATH_LOSS_MIN_DB)
    return {"topology": topology, "los_mask": los, "path_loss": pl,
            "ground": ground, "valid": valid}


# ---------------------------------------------------------------------------
# Fit pass
# ---------------------------------------------------------------------------

def fit_calibration(
    hdf5_path: Path,
    fit_refs: Sequence[SampleRef],
    pixel_subsample: float = 0.02,
    seed: int = 1234,
    ridge: float = RIDGE_LAMBDA,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Iterate fit-split samples, accumulate (XtX, Xty) per regime, solve."""
    accums: Dict[str, RegimeAccum] = {}
    rng = random.Random(seed)

    with h5py.File(str(hdf5_path), "r") as handle:
        for i, ref in enumerate(fit_refs):
            s = load_sample(handle, ref)
            ct = sample_city_type(s["topology"])
            ab = ant_bin(ref.uav_height_m)

            prior = compute_formula_prior(s["los_mask"], ref.uav_height_m)
            X_all = compute_pixel_features(s["topology"], s["los_mask"], prior, ref.uav_height_m)

            for los_label in ("LoS", "NLoS"):
                if los_label == "LoS":
                    mask = s["valid"] & (s["los_mask"] > 0)
                else:
                    mask = s["valid"] & (s["los_mask"] == 0)
                if not mask.any():
                    continue
                idx_flat = np.where(mask.ravel())[0]
                if pixel_subsample < 1.0:
                    n_keep = max(1, int(len(idx_flat) * pixel_subsample))
                    chosen = rng.sample(range(len(idx_flat)), min(n_keep, len(idx_flat)))
                    idx_flat = idx_flat[np.asarray(chosen)]

                X = X_all.reshape(-1, N_FEAT)[idx_flat].astype(np.float64)
                y = s["path_loss"].ravel()[idx_flat].astype(np.float64)

                key = regime_key(ct, los_label, ab)
                if key not in accums:
                    accums[key] = RegimeAccum()
                accums[key].update(X, y)

            if verbose and (i + 1) % 50 == 0:
                print(f"  fit [{i+1}/{len(fit_refs)}] regimes seen: {len(accums)}")

    coefs: Dict[str, np.ndarray] = {}
    for key, acc in accums.items():
        w = acc.solve(ridge=ridge)
        if w is not None:
            coefs[key] = w
            if verbose:
                print(f"  regime [{key}] n_pixels={acc.count:,}  solved OK")
        else:
            if verbose:
                print(f"  regime [{key}] n_pixels={acc.count} — too few, skipped")
    return coefs


# ---------------------------------------------------------------------------
# Inference: apply calibrated prior to one sample
# ---------------------------------------------------------------------------

def apply_calibration(
    prior: np.ndarray,
    X_all: np.ndarray,
    los_label_map: np.ndarray,
    ct: str,
    ab: str,
    coefs: Dict[str, np.ndarray],
) -> np.ndarray:
    """Apply per-regime calibration to a full 513×513 map."""
    out = prior.copy()
    for los_label, los_flag in (("LoS", True), ("NLoS", False)):
        region = (los_label_map > 0) if los_flag else (los_label_map == 0)
        if not region.any():
            continue
        # Fallback key hierarchy: exact -> mid_ant -> low_ant -> high_ant
        for ab_try in (ab, "mid_ant", "low_ant", "high_ant"):
            key = regime_key(ct, los_label, ab_try)
            if key in coefs:
                w = coefs[key]
                pred_flat = (X_all.reshape(-1, N_FEAT) @ w).astype(np.float32)
                pred = pred_flat.reshape(prior.shape)
                out[region] = np.clip(pred[region], PATH_LOSS_MIN_DB, 180.0)
                break
    return out


# ---------------------------------------------------------------------------
# Evaluation pass
# ---------------------------------------------------------------------------

@dataclass
class SampleEvalResult:
    city: str
    sample: str
    uav_height_m: float
    n_los: int
    n_nlos: int
    prior_rmse_los: float
    prior_rmse_nlos: float
    prior_rmse_overall: float
    calib_rmse_los: float
    calib_rmse_nlos: float
    calib_rmse_overall: float


def evaluate(
    hdf5_path: Path,
    eval_refs: Sequence[SampleRef],
    coefs: Dict[str, np.ndarray],
    verbose: bool = True,
    log_every: int = 50,
) -> List[SampleEvalResult]:
    results: List[SampleEvalResult] = []
    with h5py.File(str(hdf5_path), "r") as handle:
        for i, ref in enumerate(eval_refs):
            s = load_sample(handle, ref)
            ct = sample_city_type(s["topology"])
            ab_k = ant_bin(ref.uav_height_m)

            prior = compute_formula_prior(s["los_mask"], ref.uav_height_m)
            X_all = compute_pixel_features(s["topology"], s["los_mask"], prior, ref.uav_height_m)
            calib = apply_calibration(prior, X_all, s["los_mask"], ct, ab_k, coefs)

            pl = s["path_loss"]
            valid = s["valid"]
            los_m = valid & (s["los_mask"] > 0)
            nlos_m = valid & (s["los_mask"] == 0)

            def rmse(pred: np.ndarray, mask: np.ndarray) -> float:
                if not mask.any():
                    return float("nan")
                return float(np.sqrt(np.mean((pred[mask] - pl[mask]) ** 2)))

            results.append(SampleEvalResult(
                city=ref.city,
                sample=ref.sample,
                uav_height_m=ref.uav_height_m,
                n_los=int(los_m.sum()),
                n_nlos=int(nlos_m.sum()),
                prior_rmse_los=rmse(prior, los_m),
                prior_rmse_nlos=rmse(prior, nlos_m),
                prior_rmse_overall=rmse(prior, valid),
                calib_rmse_los=rmse(calib, los_m),
                calib_rmse_nlos=rmse(calib, nlos_m),
                calib_rmse_overall=rmse(calib, valid),
            ))
            if verbose and (i + 1) % log_every == 0:
                r = results[-1]
                print(f"  eval [{i+1}/{len(eval_refs)}] {ref.city} h={ref.uav_height_m:.0f}m  "
                      f"prior LoS={r.prior_rmse_los:.2f} NLoS={r.prior_rmse_nlos:.2f}  "
                      f"calib LoS={r.calib_rmse_los:.2f} NLoS={r.calib_rmse_nlos:.2f}")
    return results


def aggregate(results: Sequence[SampleEvalResult]) -> Dict[str, float]:
    """Pixel-weighted aggregate RMSE."""
    if not results:
        return {}

    def pw_rmse(rmse_field: str, n_field: str) -> float:
        tot_sse = sum(
            getattr(r, n_field) * getattr(r, rmse_field) ** 2
            for r in results
            if math.isfinite(getattr(r, rmse_field))
        )
        tot_n = sum(getattr(r, n_field) for r in results if math.isfinite(getattr(r, rmse_field)))
        return math.sqrt(tot_sse / tot_n) if tot_n else float("nan")

    return {
        "n_samples": len(results),
        "prior_rmse_los_pw": pw_rmse("prior_rmse_los", "n_los"),
        "prior_rmse_nlos_pw": pw_rmse("prior_rmse_nlos", "n_nlos"),
        "prior_rmse_overall_pw": pw_rmse("prior_rmse_overall", "n_los"),
        "calib_rmse_los_pw": pw_rmse("calib_rmse_los", "n_los"),
        "calib_rmse_nlos_pw": pw_rmse("calib_rmse_nlos", "n_nlos"),
        "calib_rmse_overall_pw": pw_rmse("calib_rmse_overall", "n_los"),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(
    results: Sequence[SampleEvalResult],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = np.array([r.uav_height_m for r in results])
    fields = [
        ("prior_rmse_los",   "calib_rmse_los",   "LoS RMSE vs UAV height"),
        ("prior_rmse_nlos",  "calib_rmse_nlos",  "NLoS RMSE vs UAV height"),
        ("prior_rmse_overall", "calib_rmse_overall", "Overall RMSE vs UAV height"),
    ]
    for prior_f, calib_f, title in fields:
        prior_v = np.array([getattr(r, prior_f) for r in results])
        calib_v = np.array([getattr(r, calib_f) for r in results])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(hs, prior_v, s=8, alpha=0.5, label="formula prior (raw)")
        ax.scatter(hs, calib_v, s=8, alpha=0.5, label="calibrated prior")
        ax.set_xlabel("UAV height (m)")
        ax.set_ylabel("RMSE (dB)")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fname = out_dir / (calib_f + "_vs_height.png")
        fig.savefig(fname, dpi=120)
        plt.close(fig)

    # LoS vs NLoS scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([r.prior_rmse_los for r in results], [r.prior_rmse_nlos for r in results],
               s=8, alpha=0.5, label="prior")
    ax.scatter([r.calib_rmse_los for r in results], [r.calib_rmse_nlos for r in results],
               s=8, alpha=0.5, label="calibrated")
    ax.set_xlabel("LoS RMSE (dB)")
    ax.set_ylabel("NLoS RMSE (dB)")
    ax.set_title("LoS vs NLoS RMSE per sample")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "los_vs_nlos_rmse.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_calibration(
    coefs: Dict[str, np.ndarray],
    path: Path,
    meta: Optional[Dict] = None,
) -> None:
    payload = {
        "model_type": "regime_obstruction_multiscale_try78",
        "feature_names": list(FEATURE_NAMES),
        "freq_ghz": FREQ_GHZ,
        "rx_height_m": RX_HEIGHT_M,
        "meters_per_pixel": METERS_PER_PIXEL,
        "height_scale": HEIGHT_SCALE,
        "kernel_sizes": list(KERNEL_SIZES),
        "city_type_thresholds": {
            "density_q1": DENSITY_Q1, "density_q2": DENSITY_Q2,
            "height_q1": HEIGHT_Q1, "height_q2": HEIGHT_Q2,
        },
        "antenna_height_thresholds": {"q1": ANT_Q1, "q2": ANT_Q2},
        "meta": meta or {},
        "coefficients": {k: v.tolist() for k, v in coefs.items()},
    }
    path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text())
    return {k: np.asarray(v, dtype=np.float64) for k, v in payload["coefficients"].items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "prior_out")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.30)
    parser.add_argument("--pixel-subsample", type=float, default=0.02,
                        help="Fraction of valid pixels sampled per sample during fit (saves RAM).")
    parser.add_argument("--ridge", type=float, default=RIDGE_LAMBDA)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--calibration-json", type=Path, default=None,
                        help="Load pre-fitted coefficients instead of fitting from scratch.")
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[try78-prior] HDF5={args.hdf5}")
    refs = enumerate_samples(args.hdf5, max_samples=args.max_samples)
    print(f"[try78-prior] enumerated {len(refs)} samples")

    fit_refs, eval_refs = split_city_holdout(refs, eval_ratio=args.eval_ratio, seed=args.split_seed)
    print(f"[try78-prior] split seed={args.split_seed}: "
          f"fit {len({r.city for r in fit_refs})} cities / {len(fit_refs)} samples  |  "
          f"eval {len({r.city for r in eval_refs})} cities / {len(eval_refs)} samples")

    # Fit
    coefs: Dict[str, np.ndarray] = {}
    if args.calibration_json:
        print(f"[try78-prior] loading calibration from {args.calibration_json}")
        coefs = load_calibration(args.calibration_json)
    elif not args.skip_fit:
        print(f"[try78-prior] fitting  (pixel_subsample={args.pixel_subsample})")
        coefs = fit_calibration(
            args.hdf5, fit_refs,
            pixel_subsample=args.pixel_subsample,
            seed=args.seed,
            ridge=args.ridge,
        )
        cal_path = args.out_dir / "calibration.json"
        save_calibration(coefs, cal_path, meta={"n_fit_samples": len(fit_refs)})
        print(f"[try78-prior] calibration saved -> {cal_path}")

    # Evaluate
    eval_results: List[SampleEvalResult] = []
    if not args.skip_eval and eval_refs:
        print("[try78-prior] evaluating on eval split ...")
        eval_results = evaluate(args.hdf5, eval_refs, coefs, log_every=args.log_every)
        agg = aggregate(eval_results)
        print("\n[try78-prior] === EVAL AGGREGATE (pixel-weighted) ===")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save JSON summary
    summary = {
        "split_seed": args.split_seed,
        "eval_ratio": args.eval_ratio,
        "n_fit_samples": len(fit_refs),
        "n_eval_samples": len(eval_refs),
        "n_regimes_fitted": len(coefs),
        "aggregate": aggregate(eval_results),
        "per_sample": [
            {
                "city": r.city, "sample": r.sample, "h": r.uav_height_m,
                "prior_los": r.prior_rmse_los, "prior_nlos": r.prior_rmse_nlos,
                "calib_los": r.calib_rmse_los, "calib_nlos": r.calib_rmse_nlos,
            }
            for r in eval_results
        ],
    }
    (args.out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[try78-prior] summary -> {args.out_dir / 'eval_summary.json'}")

    if not args.skip_plots and eval_results:
        print("[try78-prior] generating plots ...")
        make_plots(eval_results, args.out_dir)
        print(f"[try78-prior] plots in {args.out_dir}")

    print("[try78-prior] done")


if __name__ == "__main__":
    main()
