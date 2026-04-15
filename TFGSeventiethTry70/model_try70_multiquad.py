"""Try 70 — PMHHNet backbone with multi-scale quad-tree auxiliary heads.

Primary output: same 513×513 residual as Try 68 ``PMHHNetResidualRegressor``.

Additional predictions (all in **normalized** path-loss space, same as the main head):
  * ``out_257``: [B, 5, 257, 257] — 4 quadrant crops of fused features + 1 full-map low-res branch.
  * ``out_129``: [B, 17, 129, 129] — 4×4 spatial tiles + 1 full low-res branch.
  * ``out_65``: [B, 65, 65, 65] — 8×8 spatial windows + 1 full low-res branch.

``forward`` returns only ``[B, 1, 513, 513]`` for compatibility with the existing
CGAN-style training loop; auxiliary maps are exposed via ``pop_last_aux()`` after
each forward (same step as loss computation).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pmhhnet import ConvNormAct, PMHHNetResidualRegressor, ResidualBlock


def _quad_257_boxes() -> List[Tuple[int, int, int, int]]:
    return [(0, 257, 0, 257), (0, 257, 256, 513), (256, 513, 0, 257), (256, 513, 256, 513)]


def _tile_129_boxes() -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(4):
        for j in range(4):
            r0 = i * 128
            c0 = j * 128
            boxes.append((r0, r0 + 129, c0, c0 + 129))
    return boxes


def _tile_65_boxes() -> List[Tuple[int, int, int, int]]:
    starts = [int(round(k * (513 - 65) / 7)) for k in range(8)]
    boxes: List[Tuple[int, int, int, int]] = []
    for r0 in starts:
        for c0 in starts:
            boxes.append((r0, r0 + 65, c0, c0 + 65))
    return boxes


def _fused_channels(base_channels: int) -> int:
    b = int(base_channels)
    return b + 2 * b + 4 * b + b


class _TinyMapHead(nn.Module):
    def __init__(self, in_ch: int, *, norm_type: str, mid: int = 96) -> None:
        super().__init__()
        m = min(int(mid), max(in_ch * 2, 32))
        self.net = nn.Sequential(
            ConvNormAct(in_ch, m, kernel_size=3, norm_type=norm_type),
            ResidualBlock(m, m, norm_type=norm_type, dropout=0.0),
            nn.Conv2d(m, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PMHHNetTry70MultiQuad(PMHHNetResidualRegressor):
    """Try 68-compatible backbone + multi-resolution auxiliary heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        *,
        base_channels: int = 64,
        norm_type: str = "group",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            norm_type=norm_type,
            **kwargs,
        )
        base = int(base_channels)
        norm_type = str(norm_type)
        fc = _fused_channels(base)
        self.head_257_q = _TinyMapHead(fc, norm_type=norm_type)
        self.head_257_global = _TinyMapHead(fc, norm_type=norm_type)
        self.head_129_tile = _TinyMapHead(fc, norm_type=norm_type)
        self.head_129_global = _TinyMapHead(fc, norm_type=norm_type)
        self.head_65_tile = _TinyMapHead(fc, norm_type=norm_type)
        self.head_65_global = _TinyMapHead(fc, norm_type=norm_type)
        self._try70_boxes_257 = _quad_257_boxes()
        self._try70_boxes_129 = _tile_129_boxes()
        self._try70_boxes_65 = _tile_65_boxes()
        self._last_aux: Optional[Dict[str, torch.Tensor]] = None

    def pop_last_aux(self) -> Optional[Dict[str, torch.Tensor]]:
        out = self._last_aux
        self._last_aux = None
        return out

    def forward(self, x: torch.Tensor, scalar_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = self._resolve_scalar_cond(x, scalar_cond)

        hf = self._run(self.hf_project, self._high_frequency_map(x))
        x0 = self._run(self.stem, x) + hf
        x0 = self.film_stem(x0, cond)
        e1 = self.se1(self.film_e1(self._run(self.stage1, x0), cond))
        e2 = self.se2(self.film_e2(self._run(self.stage2, e1), cond))
        e3 = self.se3(self.film_e3(self._run(self.stage3, e2), cond))
        e4 = self.se4(self.film_e4(self._run(self.stage4, e3), cond))
        c4 = self._run(self.context, e4)
        c4 = self.film_context(c4, cond)

        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(e3) + self._upsample_like(p4, e3))
        p2 = self.smooth2(self.lat2(e2) + self._upsample_like(self.top3_to_2(p3), e2))
        p1 = self.smooth1(self.lat1(e1) + self._upsample_like(self.top2_to_1(p2), e1))

        hf = self.film_hf(hf, cond)
        fused = torch.cat(
            [
                p1,
                self._upsample_like(p2, p1),
                self._upsample_like(p3, p1),
                self._upsample_like(hf, p1),
            ],
            dim=1,
        )
        fused = self.film_fused(fused, cond)

        residual = self.head(fused)
        if residual.shape[-2:] != x.shape[-2:]:
            residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # --- Try 70 auxiliary maps (normalized residual space) ---
        quads_257: List[torch.Tensor] = []
        for r0, r1, c0, c1 in self._try70_boxes_257:
            sl = fused[:, :, r0:r1, c0:c1]
            sl = F.interpolate(sl, size=(257, 257), mode="bilinear", align_corners=False)
            quads_257.append(self.head_257_q(sl))
        g257 = F.interpolate(fused, size=(257, 257), mode="bilinear", align_corners=False)
        quads_257.append(self.head_257_global(g257))
        out_257 = torch.cat(quads_257, dim=1)

        tiles_129: List[torch.Tensor] = []
        for r0, r1, c0, c1 in self._try70_boxes_129:
            sl = fused[:, :, r0:r1, c0:c1]
            sl = F.interpolate(sl, size=(129, 129), mode="bilinear", align_corners=False)
            tiles_129.append(self.head_129_tile(sl))
        g129 = F.interpolate(fused, size=(129, 129), mode="bilinear", align_corners=False)
        tiles_129.append(self.head_129_global(g129))
        out_129 = torch.cat(tiles_129, dim=1)

        tiles_65: List[torch.Tensor] = []
        for r0, r1, c0, c1 in self._try70_boxes_65:
            sl = fused[:, :, r0:r1, c0:c1]
            sl = F.interpolate(sl, size=(65, 65), mode="bilinear", align_corners=False)
            tiles_65.append(self.head_65_tile(sl))
        g65 = F.interpolate(fused, size=(65, 65), mode="bilinear", align_corners=False)
        tiles_65.append(self.head_65_global(g65))
        out_65 = torch.cat(tiles_65, dim=1)

        self._last_aux = {"out_257": out_257, "out_129": out_129, "out_65": out_65}
        return residual


def try70_auxiliary_loss(
    aux: Dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    huber_delta: float,
    weight: float,
    prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE/Huber on pooled native resolutions + bilinear-up coarse branches vs residual target.

    Aux outputs are **residuals** (same space as the main head: normalized pred - prior).
    The loss is therefore computed against ``target - prior`` (residual target), not the
    full path-loss target. Pass ``prior=None`` only if the model is not using a formula prior.
    """
    if weight <= 0.0:
        return torch.zeros((), device=target.device, dtype=target.dtype)
    dev, dt = target.device, target.dtype
    # Aux heads predict the residual (pred - prior), so supervise against (target - prior).
    res_target = (target - prior) if prior is not None else target
    total = torch.zeros((), device=dev, dtype=dt)
    count = 0

    def _hub(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        d = float(huber_delta)
        err = a - b
        abs_e = err.abs()
        quad = torch.minimum(abs_e, torch.tensor(d, device=dev, dtype=dt))
        hub = 0.5 * (quad**2) + d * (abs_e - quad)
        msum = m.sum().clamp_min(1.0)
        return (hub * m).sum() / msum

    def _down_t_m(sz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Downsample the residual target (not the full target)
        t = F.interpolate(res_target, size=(sz, sz), mode="bilinear", align_corners=False)
        m = F.interpolate(mask.float(), size=(sz, sz), mode="nearest")
        m = (m > 0.5).to(dtype=dt)
        return t, m

    # 257 level: channels 0–3 = quadrants, channel 4 = global (handled separately below)
    o257 = aux["out_257"]
    for k in range(4):
        tk, mk = _down_t_m(257)
        total = total + _hub(o257[:, k : k + 1], tk, mk)
        count += 1
    up_g257 = F.interpolate(o257[:, 4:5], size=res_target.shape[-2:], mode="bilinear", align_corners=False)
    total = total + _hub(up_g257, res_target, mask)
    count += 1

    # 129 level: channels 0–15 = 4×4 tiles, channel 16 = global
    o129 = aux["out_129"]
    for k in range(16):
        tk, mk = _down_t_m(129)
        total = total + _hub(o129[:, k : k + 1], tk, mk)
        count += 1
    up_g129 = F.interpolate(o129[:, 16:17], size=res_target.shape[-2:], mode="bilinear", align_corners=False)
    total = total + _hub(up_g129, res_target, mask)
    count += 1

    # 65 level: channels 0–63 = 8×8 windows, channel 64 = global
    o65 = aux["out_65"]
    for k in range(64):
        tk, mk = _down_t_m(65)
        total = total + _hub(o65[:, k : k + 1], tk, mk)
        count += 1
    up_g65 = F.interpolate(o65[:, 64:65], size=res_target.shape[-2:], mode="bilinear", align_corners=False)
    total = total + _hub(up_g65, res_target, mask)
    count += 1

    mean_aux = total / max(count, 1)
    return float(weight) * mean_aux


def _denormalize_torch(values: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    return values * scale + offset


@torch.no_grad()
def try70_blend_search_rmse_physical(
    aux: Dict[str, torch.Tensor],
    pred_513_norm: torch.Tensor,
    prior_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    meta: Dict[str, Any],
    alpha_steps: int = 11,
) -> Dict[str, Any]:
    """Per-channel alpha sweep to find which upsampled aux residual blends best with the main head.

    For every aux channel the blend is:
        pred_blend = prior + (1-alpha)*res_main + alpha*res_aux_up

    alpha is swept from 0.0 (main only) to 1.0 (aux only) in ``alpha_steps`` steps.
    The best alpha and its RMSE are recorded for each channel.

    Aux channels are **residuals** (same normalized space as the main head).
    """
    tgt_phys = _denormalize_torch(target_norm, meta)
    valid = mask > 0.0

    def _rmse(pn: torch.Tensor) -> float:
        pp = _denormalize_torch(pn, meta)
        d = (pp - tgt_phys)[valid]
        if d.numel() == 0:
            return float("nan")
        return float(math.sqrt(float((d**2).mean().item())))

    alphas = [i / max(alpha_steps - 1, 1) for i in range(alpha_steps)]
    base_rmse = _rmse(pred_513_norm)
    res_main = pred_513_norm - prior_norm
    h, w = pred_513_norm.shape[-2:]

    best_name = "pred_513_only"
    best_rmse = base_rmse
    best_alpha = None
    rows: List[Dict[str, Any]] = []

    for name, ten in aux.items():
        _, c, _, _ = ten.shape
        for i in range(c):
            res_up = F.interpolate(ten[:, i : i + 1], size=(h, w), mode="bilinear", align_corners=False)
            # alpha=0.0 → pure main, alpha=1.0 → pure aux channel
            best_a, best_r = 0.0, base_rmse
            rmse_by_alpha: Dict[str, float] = {}
            for a in alphas:
                blended = prior_norm + (1.0 - a) * res_main + a * res_up
                r = _rmse(blended)
                rmse_by_alpha[f"a{a:.2f}"] = round(r, 4) if math.isfinite(r) else None
                if math.isfinite(r) and r < best_r:
                    best_r, best_a = r, a
            label = f"{name}_ch{i}"
            rows.append(
                {
                    "name": label,
                    "best_alpha": best_a,
                    "best_rmse_phys": best_r,
                    "rmse_by_alpha": rmse_by_alpha,
                }
            )
            if math.isfinite(best_r) and best_r < best_rmse:
                best_rmse = best_r
                best_name = label
                best_alpha = best_a

    # Sort by best RMSE so the most useful channels appear first
    rows.sort(key=lambda r: r["best_rmse_phys"] if math.isfinite(r["best_rmse_phys"]) else float("inf"))

    return {
        "rmse_phys_baseline_pred513": base_rmse,
        "best_rmse_phys_blend": best_rmse,
        "best_blend_label": best_name,
        "best_blend_alpha": best_alpha,
        "total_aux_channels": sum(ten.shape[1] for ten in aux.values()),
        "per_component": rows,
    }
