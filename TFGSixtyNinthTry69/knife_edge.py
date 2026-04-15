"""Knife-edge diffraction channel (ITU-R P.526-15, §4.5.1).

Implements a Bullington-style single-dominant-edge diffraction loss map that
is appended to the network input so the model does not have to "invent" wave
diffraction from raw geometry (geometry-assisted DL, 2024).

Assumptions (consistent with the dataset's data_utils):

  * The TX is at the **center pixel** of the H x W map at altitude
    ``antenna_height_m`` above the local ground (topology is relative).
  * The RX is assumed at a fixed height ``rx_height_m`` (default 1.5 m)
    above ground at every pixel. This is the standard UE height in 3GPP
    TR 38.901 and matches what data_utils uses for elevation-angle channels.
  * The ground is flat at z = 0; building heights are stored in the raw
    topology map (already in meters after multiplying by the input scale).
  * Carrier frequency and meters-per-pixel are scene constants taken from
    ``data.path_loss_formula_input``.

Physics (ITU-R P.526, Lee 1985 closed form):

    v   = h * sqrt(2 * (d1 + d2) / (lambda * d1 * d2))
    J(v) = 6.9 + 20 * log10( sqrt((v - 0.1)^2 + 1) + v - 0.1 )     (v > -0.78)
         = 0                                                        (otherwise)

where ``h`` is the excess height of the dominant blocker above the straight
TX→RX line, ``d1`` the TX→edge distance, ``d2`` the edge→RX distance, and
``lambda`` the wavelength.

The dominant edge is approximated by sampling ``num_ray_samples`` points
along each TX→RX ray and taking the position of maximum excess height
(Bullington 1947). Multi-edge Deygout is O(rays^2) and we do not need it
here: the goal is a cheap physics-motivated prior, not a substitute for
ray tracing.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


_MIN_DISTANCE_M = 1.0


def _to_topo_1hw(topo_m: torch.Tensor) -> torch.Tensor:
    """Coerce topology to (1, H, W) in meters.

    ``precompute_augmented_batch`` passes ``(1, 1, H, W)`` (batch, channel); the dataset
    path uses ``(1, H, W)``. Both are valid.
    """
    t = topo_m.float()
    while t.ndim > 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(f"topo_m: expected batch 1 for knife-edge, got shape {tuple(topo_m.shape)}")
        t = t[:, 0, :, :].contiguous()
    if t.ndim != 3 or t.shape[0] != 1:
        raise ValueError(f"topo_m must reduce to (1, H, W), got {tuple(topo_m.shape)}")
    return t


def _bullington_edge(
    topo_m: torch.Tensor,
    num_ray_samples: int,
    h_tx_m: float,
    h_rx_m: float,
    meters_per_pixel: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the dominant building edge for every pixel.

    Returns (excess_height_m, d1_m, d2_m), each of shape (H, W).

    The TX sits at the map centre; the RX is the pixel itself. We bilinear-
    sample the topology along each ray with ``num_ray_samples`` points, then
    take argmax of (building_height - ray_height) clipped to >= 0.
    """
    topo_m = _to_topo_1hw(topo_m)
    device = topo_m.device
    dtype = torch.float32
    topo = topo_m.to(dtype=dtype)
    _, H, W = topo.shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    yy = torch.arange(H, device=device, dtype=dtype)
    xx = torch.arange(W, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(yy, xx, indexing="ij")  # (H, W)

    # Parametric ray: t=0 at TX (centre), t=1 at RX (pixel).
    # grid_sample expects normalised coordinates in [-1, 1].
    # Sample positions along each ray, excluding the two endpoints so that
    # the TX pixel itself (always at centre) and the RX pixel do not dominate.
    t = torch.linspace(1.0 / num_ray_samples, 1.0 - 1.0 / num_ray_samples,
                       steps=num_ray_samples, device=device, dtype=dtype)  # (K,)

    sy = cy + t.view(-1, 1, 1) * (y_grid - cy)  # (K, H, W) pixel y
    sx = cx + t.view(-1, 1, 1) * (x_grid - cx)  # (K, H, W) pixel x

    # Normalised grid_sample coords (align_corners=True mapping).
    gy = sy / max(H - 1, 1) * 2.0 - 1.0
    gx = sx / max(W - 1, 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (K, H, W, 2)

    K = num_ray_samples
    topo_rep = topo.unsqueeze(0).expand(K, 1, H, W)
    sampled = F.grid_sample(
        topo_rep, grid, mode="bilinear", padding_mode="border", align_corners=True
    ).squeeze(1)  # (K, H, W)

    # Ray altitude at each sample: linear blend between TX and RX altitudes
    # (topology is relative to ground z=0, so h_tx_m is TX altitude AGL).
    h_ray = (1.0 - t).view(-1, 1, 1) * h_tx_m + t.view(-1, 1, 1) * h_rx_m  # (K, H, W)
    excess = (sampled - h_ray).clamp_min(0.0)  # (K, H, W)

    # Dominant-edge position along the ray (argmax excess height).
    t_star_idx = excess.argmax(dim=0, keepdim=True)  # (1, H, W)
    excess_star = excess.gather(0, t_star_idx).squeeze(0)  # (H, W)
    t_star = t.gather(0, t_star_idx.flatten()).reshape(H, W)

    # Distances along the ray in meters.
    dy = (y_grid - cy)
    dx = (x_grid - cx)
    ray_len_px = torch.sqrt(dx * dx + dy * dy).clamp_min(1.0)
    ray_len_m = ray_len_px * float(meters_per_pixel)
    d1 = (t_star * ray_len_m).clamp_min(_MIN_DISTANCE_M)
    d2 = ((1.0 - t_star) * ray_len_m).clamp_min(_MIN_DISTANCE_M)
    return excess_star, d1, d2


def compute_knife_edge_loss_map(
    topo_m: torch.Tensor,
    *,
    antenna_height_m: float,
    frequency_ghz: float,
    meters_per_pixel: float,
    rx_height_m: float = 1.5,
    num_ray_samples: int = 48,
) -> torch.Tensor:
    """Return a (1, H, W) tensor of knife-edge diffraction loss in dB.

    ``topo_m`` is the building heightmap in meters ``(1, H, W)`` or ``(1, 1, H, W)``.
    """
    excess_m, d1_m, d2_m = _bullington_edge(
        topo_m=topo_m,
        num_ray_samples=int(num_ray_samples),
        h_tx_m=float(antenna_height_m),
        h_rx_m=float(rx_height_m),
        meters_per_pixel=float(meters_per_pixel),
    )
    wavelength_m = 0.299792458 / max(float(frequency_ghz), 0.1)
    # Fresnel parameter v = h * sqrt( 2 * (d1 + d2) / (lambda * d1 * d2) )
    denom = (wavelength_m * d1_m * d2_m).clamp_min(1e-6)
    v = excess_m * torch.sqrt(2.0 * (d1_m + d2_m) / denom)
    # Lee 1985 / ITU-R P.526 closed form, valid for v > -0.78.
    shifted = v - 0.1
    j_db = 6.9 + 20.0 * torch.log10(torch.sqrt(shifted * shifted + 1.0) + shifted)
    # No edge detected (excess clamped to 0) -> no diffraction loss. This
    # matters for pure-LoS pixels: the ITU formula returns ~6 dB at v=0,
    # which corresponds to an edge *grazing* the ray, not the absence of
    # one. A ~1 cm threshold prevents sampling noise from triggering it.
    has_edge = excess_m > 1e-2
    j_db = torch.where(has_edge & (v > -0.78), j_db, torch.zeros_like(j_db))
    j_db = j_db.clamp_min(0.0)
    return j_db.unsqueeze(0)


def normalize_knife_edge_db(loss_db: torch.Tensor, *, scale_db: float = 40.0) -> torch.Tensor:
    """Map a knife-edge loss map in dB to roughly [0, 1] for network input.

    40 dB is the expected physical ceiling for a single knife-edge at our
    geometry (deep shadow); values beyond are clipped.
    """
    return (loss_db / max(float(scale_db), 1.0)).clamp(0.0, 1.0)
