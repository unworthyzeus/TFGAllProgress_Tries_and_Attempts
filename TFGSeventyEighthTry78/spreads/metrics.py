"""Try 77 — metrics (RMSE/MAE in native units, W1, KL, histograms)."""
from __future__ import annotations

import numpy as np
import torch


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (pred - target) ** 2
    denom = mask.sum().clamp_min(1.0)
    return float(((diff * mask).sum() / denom).sqrt().item())


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (pred - target).abs()
    denom = mask.sum().clamp_min(1.0)
    return float(((diff * mask).sum() / denom).item())


def per_image_histogram(
    values: torch.Tensor, mask: torch.Tensor, lo: float, hi: float, n_bins: int = 64
) -> np.ndarray:
    """Integer histogram over ``n_bins`` equal-width bins in ``[lo, hi)``."""
    v = values.squeeze(0).squeeze(0).detach().cpu().numpy()
    m = mask.squeeze(0).squeeze(0).detach().cpu().numpy() > 0.5
    if not m.any():
        return np.zeros(n_bins, dtype=np.int64)
    vals = np.clip(v[m], lo, hi - 1e-6)
    binwidth = (hi - lo) / max(n_bins, 1)
    idx = np.floor((vals - lo) / binwidth).astype(np.int64)
    idx = np.clip(idx, 0, n_bins - 1)
    return np.bincount(idx, minlength=n_bins).astype(np.int64)


def wasserstein1_from_counts(c_pred: np.ndarray, c_tgt: np.ndarray, bin_width: float = 1.0) -> float:
    if c_pred.sum() == 0 or c_tgt.sum() == 0:
        return float("nan")
    p = c_pred / c_pred.sum()
    q = c_tgt / c_tgt.sum()
    cdf_diff = np.cumsum(p - q)
    return float(np.abs(cdf_diff).sum() * bin_width)


def kl_from_counts(c_emp: np.ndarray, c_ref: np.ndarray) -> float:
    if c_emp.sum() == 0 or c_ref.sum() == 0:
        return float("nan")
    p = np.clip(c_emp / c_emp.sum(), 1e-12, None)
    q = np.clip(c_ref / c_ref.sum(), 1e-12, None)
    return float((p * (np.log(p) - np.log(q))).sum())
