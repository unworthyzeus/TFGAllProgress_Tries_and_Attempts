"""Try 76 — two-stage distribution-first model (from scratch).

Architecture summary (see DESIGN_TRY76.md for the full rationale):

    Stage-A  : context encoder -> GAP -> 3-component GMM parameters (π, μ, σ)
    Stage-B  : U-Net decoder conditioned on Stage-A + FiLM(height)
               -> per-pixel soft assignment p(i,j), z(i,j), log σ̃(i,j)

Output reconstruction:
    ŷ(i,j) = Σ_k p_k(i,j) * (μ_k + z(i,j) * σ_k)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    g = min(groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(max(g, 1), channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.gn1 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = _group_norm(out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return F.silu(y + self.skip(x))


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class StageAEncoder(nn.Module):
    """Shallow conv trunk used by both Stage-A and Stage-B."""

    def __init__(self, in_ch: int, base: int = 48, cond_dim: int = 32) -> None:
        super().__init__()
        chs = [base, base * 2, base * 2, base * 4, base * 4]
        self.stem = ConvBlock(in_ch, chs[0])
        self.down1 = ConvBlock(chs[0], chs[1], stride=2)
        self.down2 = ConvBlock(chs[1], chs[2], stride=2)
        self.down3 = ConvBlock(chs[2], chs[3], stride=2)
        self.down4 = ConvBlock(chs[3], chs[4], stride=2)
        self.film_mid = FiLM(cond_dim, chs[2])
        self.film_deep = FiLM(cond_dim, chs[4])
        self.out_channels = chs

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.film_mid(self.down2(x1), cond)
        x3 = self.down3(x2)
        x4 = self.film_deep(self.down4(x3), cond)
        return x0, x1, x2, x3, x4


class GMMHead(nn.Module):
    """Stage-A head: global 3-component Gaussian mixture on normalized PL."""

    def __init__(
        self,
        feat_dim: int,
        K: int = 3,
        clamp_lo: float = 30.0,
        clamp_hi: float = 178.0,
        sigma_min: float = 1.0,
        sigma_max: float = 30.0,
        outlier_sigma_floor: float = 15.0,
    ) -> None:
        super().__init__()
        self.K = K
        self.clamp_lo = float(clamp_lo)
        self.clamp_hi = float(clamp_hi)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.outlier_sigma_floor = float(outlier_sigma_floor)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 3 * K),  # logit_pi, mu_norm, log_sigma
        )

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.mlp(feat)  # (B, 3K)
        logit_pi, mu_raw, log_sigma_raw = raw.chunk(3, dim=-1)
        pi = F.softmax(logit_pi, dim=-1)
        mu_norm = torch.sigmoid(mu_raw)
        mu = self.clamp_lo + mu_norm * (self.clamp_hi - self.clamp_lo)
        sigma = F.softplus(log_sigma_raw) + self.sigma_min
        sigma = torch.clamp(sigma, max=self.sigma_max)
        # The "outlier" (highest-variance) component has its floor pushed up so
        # the head cannot avoid producing a heavy-tail branch.
        outlier_floor = torch.full_like(sigma[:, -1:], self.outlier_sigma_floor)
        sigma = torch.cat([sigma[:, :-1], torch.maximum(sigma[:, -1:], outlier_floor)], dim=-1)
        return {"pi": pi, "mu": mu, "sigma": sigma}


class StageBDecoder(nn.Module):
    """U-Net-ish decoder that outputs per-pixel (p, z, log_sigma_tilde)."""

    Z_ABS_MAX = 3.0
    SIGMA_TILDE_MIN = 1.0e-3
    SIGMA_TILDE_MAX = 30.0

    def __init__(self, enc_channels: Tuple[int, ...], cond_dim: int, K: int) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = enc_channels
        self.up4 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)
        self.film3 = FiLM(cond_dim, c3)
        self.up3 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)
        self.film1 = FiLM(cond_dim, c1)
        self.up1 = nn.ConvTranspose2d(c1, c0, 2, stride=2)
        self.dec0 = ConvBlock(c0 + c0, c0)
        self.head = nn.Conv2d(c0, K + 2, 1)  # K logits + z + log_sigma_tilde
        self.K = K
        self.dropout = nn.Dropout2d(0.1)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        u3 = self.up4(x4)
        u3 = self._crop_or_pad_to(u3, x3)
        d3 = self.film3(self.dec3(torch.cat([u3, x3], dim=1)), cond)
        u2 = self.up3(d3)
        u2 = self._crop_or_pad_to(u2, x2)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))
        u1 = self.up2(d2)
        u1 = self._crop_or_pad_to(u1, x1)
        d1 = self.film1(self.dec1(torch.cat([u1, x1], dim=1)), cond)
        u0 = self.up1(d1)
        u0 = self._crop_or_pad_to(u0, x0)
        d0 = self.dec0(torch.cat([u0, x0], dim=1))
        d0 = self.dropout(d0)
        raw = self.head(d0)
        p_logits = raw[:, : self.K]
        z_raw = raw[:, self.K : self.K + 1]
        log_sigma_tilde_raw = raw[:, self.K + 1 : self.K + 2]
        p = F.softmax(p_logits, dim=1)
        # Keep the per-pixel residual and heteroscedastic widening bounded so
        # hard NLoS samples cannot push the decoder into non-finite states.
        z = self.Z_ABS_MAX * torch.tanh(z_raw)
        sigma_tilde = self.SIGMA_TILDE_MIN + (self.SIGMA_TILDE_MAX - self.SIGMA_TILDE_MIN) * torch.sigmoid(log_sigma_tilde_raw)
        log_sigma_tilde = torch.log(sigma_tilde)
        return {"p": p, "z": z, "log_sigma_tilde": log_sigma_tilde}

    @staticmethod
    def _crop_or_pad_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # Handle odd spatial sizes (513) where ConvTranspose2d may drift by 1px.
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        dh = ref.shape[-2] - x.shape[-2]
        dw = ref.shape[-1] - x.shape[-1]
        pad = [
            dw // 2, dw - dw // 2,
            dh // 2, dh - dh // 2,
        ]
        if any(p < 0 for p in pad):
            # crop
            top = max(0, -pad[2]); bot = x.shape[-2] - max(0, -pad[3])
            left = max(0, -pad[0]); right = x.shape[-1] - max(0, -pad[1])
            return x[..., top:bot, left:right]
        return F.pad(x, pad)


@dataclass
class Try76ModelConfig:
    in_channels: int = 4
    cond_dim: int = 64          # FiLM conditioning dim (fused height emb + GMM)
    height_embed_dim: int = 32  # = 2 * n_freq (16 freqs -> 32 dims)
    base_width: int = 48
    K: int = 5
    clamp_lo: float = 30.0
    clamp_hi: float = 178.0
    outlier_sigma_floor: float = 15.0


class Try76Model(nn.Module):
    def __init__(self, cfg: Try76ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.height_mlp = nn.Sequential(
            nn.Linear(cfg.height_embed_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        # The Stage-A conditioning only uses the height embedding.
        self.encoder = StageAEncoder(cfg.in_channels, base=cfg.base_width, cond_dim=cfg.cond_dim)
        feat_dim = self.encoder.out_channels[-1]
        self.gmm_head = GMMHead(
            feat_dim,
            K=cfg.K,
            clamp_lo=cfg.clamp_lo,
            clamp_hi=cfg.clamp_hi,
            outlier_sigma_floor=cfg.outlier_sigma_floor,
        )

        # Stage-B conditioning = fused(height, GMM summary).
        self.gmm_embed = nn.Sequential(
            nn.Linear(3 * cfg.K, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(2 * cfg.cond_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        self.decoder = StageBDecoder(self.encoder.out_channels, cond_dim=cfg.cond_dim, K=cfg.K)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-1, -2))

    def forward(self, inputs: torch.Tensor, height_embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_cond = self.height_mlp(height_embed)
        x0, x1, x2, x3, x4 = self.encoder(inputs, h_cond)

        gmm = self.gmm_head(self._pool(x4))
        gmm_vec = torch.cat([gmm["pi"], gmm["mu"], gmm["sigma"]], dim=-1)
        gmm_emb = self.gmm_embed(gmm_vec)
        cond = self.cond_fuse(torch.cat([h_cond, gmm_emb], dim=-1))

        dec = self.decoder(x0, x1, x2, x3, x4, cond)

        p = dec["p"]                        # (B, K, H, W)
        z = dec["z"]                        # (B, 1, H, W)
        log_sigma_tilde = dec["log_sigma_tilde"]  # (B, 1, H, W)

        mu = gmm["mu"].unsqueeze(-1).unsqueeze(-1)         # (B, K, 1, 1)
        sigma = gmm["sigma"].unsqueeze(-1).unsqueeze(-1)   # (B, K, 1, 1)
        per_comp_pred = mu + z * sigma                     # (B, K, H, W)
        pred = (p * per_comp_pred).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        pred = torch.clamp(pred, self.cfg.clamp_lo, self.cfg.clamp_hi)

        return {
            "pred": pred,
            "p": p,
            "z": z,
            "log_sigma_tilde": log_sigma_tilde,
            "gmm": gmm,
        }
