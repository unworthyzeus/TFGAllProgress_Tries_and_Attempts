"""Try 77 — losses for spread prediction (delay_spread, angular_spread).

Differences vs Try 76:
- Target is in native units (ns / deg), non-negative, heavy-tailed.
- Spike + GMM5 mixture (K+1 = 6 components), no forced outlier-sigma floor.
- Higher outlier budget threshold (more freedom for the tail).
- RMSE is reported in native units.

All losses are masked by ``loss_mask`` (ground pixels with finite, non-negative
target). No imports across tries.
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


@dataclass
class LossWeights:
    map_nll: float = 1.0
    dist_kl: float = 0.5
    moment_match: float = 0.1
    outlier_budget: float = 0.05
    rmse: float = 0.5
    mae: float = 0.1
    outlier_budget_threshold: float = 0.5   # relaxed vs Try 76 (0.25) — spreads need tail freedom


def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask > 0
    if not torch.any(active):
        return t.new_zeros(())
    return t.masked_select(active).mean()


def map_nll_loss(
    target: torch.Tensor,
    p: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    log_sigma_tilde: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    sigma_tilde = torch.exp(log_sigma_tilde.clamp(min=LOG_SIGMA_TILDE_MIN, max=LOG_SIGMA_TILDE_MAX)).clamp_min(EPS)
    var_total = sigma.pow(2) + sigma_tilde.pow(2)
    diff = target - mu
    log_pdf = -0.5 * (diff.pow(2) / var_total + var_total.log() + LOG_2PI)
    log_weighted = log_pdf + (p.clamp_min(EPS)).log()
    log_mix = torch.logsumexp(log_weighted, dim=1, keepdim=True)
    return _masked_mean(-log_mix, loss_mask)


def _soft_bin(
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    clamp_lo: float,
    clamp_hi: float,
    n_bins: int = 64,
    kernel_sigma_bins: float = 0.6,
) -> torch.Tensor:
    device = target.device
    centers = torch.linspace(clamp_lo, clamp_hi, n_bins, device=device)
    binwidth = (clamp_hi - clamp_lo) / max(n_bins - 1, 1)
    diffs = (target.unsqueeze(-1) - centers) / (kernel_sigma_bins * binwidth + EPS)
    weights = torch.exp(-0.5 * diffs.pow(2))
    mask = loss_mask.unsqueeze(-1)
    weighted = weights * mask
    counts = weighted.flatten(1, 3).sum(dim=1)
    total = counts.sum(dim=1, keepdim=True).clamp_min(EPS)
    return counts / total


def _mixture_pmf(
    pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
    clamp_lo: float, clamp_hi: float, n_bins: int,
) -> torch.Tensor:
    device = pi.device
    centers = torch.linspace(clamp_lo, clamp_hi, n_bins, device=device)
    binwidth = (clamp_hi - clamp_lo) / max(n_bins - 1, 1)
    mu_e = mu.unsqueeze(-1)                                  # (B, K+1, 1)
    sigma_e = sigma.unsqueeze(-1).clamp_min(EPS)             # (B, K+1, 1)
    diffs = (centers.view(1, 1, -1) - mu_e) / sigma_e
    pdf = torch.exp(-0.5 * diffs.pow(2)) / (sigma_e * math.sqrt(2 * math.pi))
    pmf = (pi.unsqueeze(-1) * pdf * binwidth).sum(dim=1)
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
    emp = _soft_bin(target, loss_mask, clamp_lo, clamp_hi, n_bins)
    mu = gmm["mu"]
    sigma = gmm["sigma"]
    if mu.dim() == 4:
        mu = mu.squeeze(-1).squeeze(-1)
        sigma = sigma.squeeze(-1).squeeze(-1)
    mix = _mixture_pmf(gmm["pi"], mu, sigma, clamp_lo, clamp_hi, n_bins)
    emp = emp.clamp_min(EPS)
    mix = mix.clamp_min(EPS)
    return (emp * (emp.log() - mix.log())).sum(dim=1).mean()


def moment_match_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    denom = loss_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    tgt_mean = (target * loss_mask).sum(dim=(1, 2, 3)) / denom
    pred_mean = (pred * loss_mask).sum(dim=(1, 2, 3)) / denom
    tgt_var = ((target - tgt_mean.view(-1, 1, 1, 1)) ** 2 * loss_mask).sum(dim=(1, 2, 3)) / denom
    pred_var = ((pred - pred_mean.view(-1, 1, 1, 1)) ** 2 * loss_mask).sum(dim=(1, 2, 3)) / denom
    return ((pred_mean - tgt_mean) ** 2).mean() + ((pred_var.sqrt() - tgt_var.sqrt()) ** 2).mean()


def rmse_loss(target: torch.Tensor, pred: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Masked RMSE between reconstructed map and target, in native units.

    Thesis global objective for spreads: RMSE < 50 ns (delay) and < 20 deg
    (angular). We include RMSE as a direct training signal.
    """
    sq = ((pred - target) ** 2) * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    return torch.sqrt((sq.sum() / denom).clamp_min(EPS))


def mae_loss(target: torch.Tensor, pred: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def outlier_budget_loss(
    p: torch.Tensor,
    loss_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Cap the average usage of the last (outlier) mixture component."""
    outlier_p = p[:, -1:]
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
    l_rmse = rmse_loss(target, outputs["pred"], loss_mask)
    l_mae = mae_loss(target, outputs["pred"], loss_mask)

    total = (
        weights.map_nll * l_map
        + weights.dist_kl * l_dist
        + weights.moment_match * l_mom
        + weights.outlier_budget * l_out
        + weights.rmse * l_rmse
        + weights.mae * l_mae
    )
    return {
        "total": total,
        "map_nll": l_map.detach(),
        "dist_kl": l_dist.detach(),
        "moment_match": l_mom.detach(),
        "outlier_budget": l_out.detach(),
        "rmse": l_rmse.detach(),
        "mae": l_mae.detach(),
    }
