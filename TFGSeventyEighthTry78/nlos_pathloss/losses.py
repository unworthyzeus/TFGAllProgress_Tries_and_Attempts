"""Try 76 — losses.

Everything here is masked by ``loss_mask`` (ground pixels within the expert's
LoS/NLoS region). No inheritance from Tries 67–75.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


EPS = 1e-6
LOG_2PI = math.log(2 * math.pi)
LOG_SIGMA_TILDE_MIN = math.log(1.0e-3)
LOG_SIGMA_TILDE_MAX = math.log(30.0)
# Caps squared-z (diff/σ)^2 so a bad target pixel cannot overflow log-pdf into ±inf.
# 100 ≈ (10σ)^2, which is already deep in the tail (exp(-50) ≈ 2e-22).
MAX_SQUARED_Z = 100.0


@dataclass
class LossWeights:
    map_nll: float = 1.0
    dist_kl: float = 0.5
    moment_match: float = 0.1
    outlier_budget: float = 0.1
    rmse_db: float = 0.5
    outlier_budget_threshold: float = 0.25


def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask > 0
    if not torch.any(active):
        return t.new_zeros(())
    return t.masked_select(active).mean()


def _masked_tensor(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask > 0, t, torch.zeros_like(t))


def map_nll_loss(
    target: torch.Tensor,           # (B, 1, H, W)
    p: torch.Tensor,                # (B, K, H, W)
    mu: torch.Tensor,               # (B, K, 1, 1)
    sigma: torch.Tensor,            # (B, K, 1, 1)
    log_sigma_tilde: torch.Tensor,  # (B, 1, H, W)
    loss_mask: torch.Tensor,        # (B, 1, H, W) float
) -> torch.Tensor:
    """Per-pixel NLL under a K-component mixture with heteroscedastic widening."""
    sigma_tilde = torch.exp(log_sigma_tilde.clamp(min=LOG_SIGMA_TILDE_MIN, max=LOG_SIGMA_TILDE_MAX)).clamp_min(EPS)
    var_total = sigma.pow(2) + sigma_tilde.pow(2)
    diff = target - mu  # (B, K, H, W)
    squared_z = (diff.pow(2) / var_total).clamp(max=MAX_SQUARED_Z)
    log_pdf = -0.5 * (squared_z + var_total.log() + LOG_2PI)
    log_weighted = log_pdf + (p.clamp_min(EPS)).log()
    log_mix = torch.logsumexp(log_weighted, dim=1, keepdim=True)  # (B, 1, H, W)
    per_pixel_nll = -log_mix
    return _masked_mean(per_pixel_nll, loss_mask)


def _soft_bin(
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    clamp_lo: float,
    clamp_hi: float,
    n_bins: int = 64,
    kernel_sigma_bins: float = 0.6,
) -> torch.Tensor:
    """Differentiable soft histogram -> (B, n_bins) pmf.

    Each valid pixel contributes a Gaussian of std ``kernel_sigma_bins`` to the
    bin grid. Normalised per image. Invariant to image size through the mask.
    """
    B = target.shape[0]
    device = target.device
    centers = torch.linspace(clamp_lo, clamp_hi, n_bins, device=device)  # (n_bins,)
    binwidth = (clamp_hi - clamp_lo) / max(n_bins - 1, 1)

    safe_target = _masked_tensor(target, loss_mask)
    diffs = (safe_target.unsqueeze(-1) - centers) / (kernel_sigma_bins * binwidth + EPS)  # (B, 1, H, W, n_bins)
    weights = torch.exp(-0.5 * diffs.pow(2))
    mask = loss_mask.unsqueeze(-1)
    weighted = weights * mask
    counts = weighted.flatten(1, 3).sum(dim=1)  # (B, n_bins)
    total = counts.sum(dim=1, keepdim=True).clamp_min(EPS)
    return counts / total


def _mixture_pmf(
    pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
    clamp_lo: float, clamp_hi: float, n_bins: int,
) -> torch.Tensor:
    device = pi.device
    centers = torch.linspace(clamp_lo, clamp_hi, n_bins, device=device)   # (n_bins,)
    binwidth = (clamp_hi - clamp_lo) / max(n_bins - 1, 1)
    mu_e = mu.unsqueeze(-1)                                                # (B, K, 1)
    sigma_e = sigma.unsqueeze(-1).clamp_min(EPS)                           # (B, K, 1)
    diffs = (centers.view(1, 1, -1) - mu_e) / sigma_e                      # (B, K, n_bins)
    pdf = torch.exp(-0.5 * diffs.pow(2)) / (sigma_e * math.sqrt(2 * math.pi))
    pmf = (pi.unsqueeze(-1) * pdf * binwidth).sum(dim=1)                   # (B, n_bins)
    total = pmf.sum(dim=1, keepdim=True).clamp_min(EPS)
    return pmf / total


def dist_kl_loss(
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    gmm: Dict[str, torch.Tensor],
    clamp_lo: float,
    clamp_hi: float,
    n_bins: int = 64,
) -> torch.Tensor:
    """KL( empirical || Stage-A mixture ) as a scalar."""
    emp = _soft_bin(target, loss_mask, clamp_lo, clamp_hi, n_bins)  # (B, n_bins)
    mix = _mixture_pmf(gmm["pi"], gmm["mu"].squeeze(-1).squeeze(-1) if gmm["mu"].dim() == 4 else gmm["mu"],
                       gmm["sigma"].squeeze(-1).squeeze(-1) if gmm["sigma"].dim() == 4 else gmm["sigma"],
                       clamp_lo, clamp_hi, n_bins)
    emp = emp.clamp_min(EPS)
    mix = mix.clamp_min(EPS)
    return (emp * (emp.log() - mix.log())).sum(dim=1).mean()


def moment_match_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    safe_target = _masked_tensor(target, loss_mask)
    safe_pred = _masked_tensor(pred, loss_mask)
    denom = loss_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    tgt_sum = (safe_target * loss_mask).sum(dim=(1, 2, 3))
    pred_sum = (safe_pred * loss_mask).sum(dim=(1, 2, 3))
    tgt_mean = tgt_sum / denom
    pred_mean = pred_sum / denom
    tgt_var = ((safe_target - tgt_mean.view(-1, 1, 1, 1)) ** 2 * loss_mask).sum(dim=(1, 2, 3)) / denom
    pred_var = ((safe_pred - pred_mean.view(-1, 1, 1, 1)) ** 2 * loss_mask).sum(dim=(1, 2, 3)) / denom
    return ((pred_mean - tgt_mean) ** 2).mean() + ((pred_var.sqrt() - tgt_var.sqrt()) ** 2).mean()


def rmse_db_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked RMSE in dB units between the reconstructed map and the target.

    Global objective on overall path-loss is RMSE < 5 dB (LoS + NLoS combined),
    so we include this as a direct training signal alongside the distribution
    losses.
    """
    safe_target = _masked_tensor(target, loss_mask)
    safe_pred = _masked_tensor(pred, loss_mask)
    sq = ((safe_pred - safe_target) ** 2) * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    return torch.sqrt((sq.sum() / denom).clamp_min(EPS))


def outlier_budget_loss(
    p: torch.Tensor,
    loss_mask: torch.Tensor,
    threshold: float = 0.25,
) -> torch.Tensor:
    """Cap the average usage of the last (outlier) mixture component."""
    outlier_p = p[:, -1:]  # (B, 1, H, W)
    mean_used = _masked_mean(outlier_p, loss_mask)
    return F.relu(mean_used - threshold)


def combined_loss(
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    clamp_lo: float,
    clamp_hi: float,
    weights: LossWeights = LossWeights(),
) -> Dict[str, torch.Tensor]:
    gmm = outputs["gmm"]
    mu = gmm["mu"].unsqueeze(-1).unsqueeze(-1)
    sigma = gmm["sigma"].unsqueeze(-1).unsqueeze(-1)

    l_map = map_nll_loss(target, outputs["p"], mu, sigma, outputs["log_sigma_tilde"], loss_mask)
    l_dist = dist_kl_loss(target, loss_mask, gmm, clamp_lo, clamp_hi)
    l_mom = moment_match_loss(target, outputs["pred"], loss_mask)
    l_out = outlier_budget_loss(outputs["p"], loss_mask, weights.outlier_budget_threshold)
    l_rmse = rmse_db_loss(target, outputs["pred"], loss_mask)

    total = (
        weights.map_nll * l_map
        + weights.dist_kl * l_dist
        + weights.moment_match * l_mom
        + weights.outlier_budget * l_out
        + weights.rmse_db * l_rmse
    )
    # Final guard: if any individual term is non-finite, drop it from the sum
    # rather than propagate NaN/inf into backprop and blow up a whole run.
    total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=-1e6)
    return {
        "total": total,
        "map_nll": l_map.detach(),
        "dist_kl": l_dist.detach(),
        "moment_match": l_mom.detach(),
        "outlier_budget": l_out.detach(),
        "rmse_db": l_rmse.detach(),
    }
