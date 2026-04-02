from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_unet import CKMUNet, DoubleConv, make_norm_2d


class ExpertHead(nn.Module):
    def __init__(self, channels: int, *, norm_type: str = "group", dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DoubleConv(channels, channels, dropout=dropout, norm_type=norm_type),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPriorResidualMoE(nn.Module):
    """
    Try 47:
    - shared U-Net backbone inherited from the strong Try 22 family
    - explicit LoS base residual head
    - NLoS delta head as a small MoE on top of the shared decoder feature map
    - final residual = LoS base for LoS pixels, LoS base + NLoS delta for NLoS pixels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        *,
        base_channels: int = 112,
        gradient_checkpointing: bool = False,
        norm_type: str = "group",
        scalar_cond_dim: int = 0,
        scalar_film_hidden: int = 192,
        upsample_mode: str = "bilinear",
        num_experts: int = 4,
        expert_channels: int = 96,
        los_channel_index: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(out_channels) != 1:
            raise ValueError("UNetPriorResidualMoE expects out_channels == 1")
        self.los_channel_index = int(los_channel_index)
        self.num_experts = max(int(num_experts), 2)

        self.backbone = CKMUNet(
            in_channels=in_channels,
            out_channels=base_channels,
            base_channels=base_channels,
            gradient_checkpointing=gradient_checkpointing,
            path_loss_hybrid=False,
            norm_type=norm_type,
            scalar_cond_dim=scalar_cond_dim,
            scalar_film_hidden=scalar_film_hidden,
            upsample_mode=upsample_mode,
        )

        self.shared = nn.Sequential(
            nn.Conv2d(base_channels, expert_channels, kernel_size=1, bias=False),
            make_norm_2d(expert_channels, norm_type),
            nn.SiLU(inplace=True),
            DoubleConv(expert_channels, expert_channels, dropout=dropout, norm_type=norm_type),
        )
        los_mid = max(expert_channels // 2, 16)
        self.los_base_head = nn.Sequential(
            nn.Conv2d(expert_channels, los_mid, kernel_size=1, bias=False),
            make_norm_2d(los_mid, norm_type),
            nn.SiLU(inplace=True),
            nn.Conv2d(los_mid, 1, kernel_size=1),
        )
        self.nlos_prep = nn.Sequential(
            nn.Conv2d(expert_channels + in_channels, expert_channels, kernel_size=1, bias=False),
            make_norm_2d(expert_channels, norm_type),
            nn.SiLU(inplace=True),
            DoubleConv(expert_channels, expert_channels, dropout=dropout, norm_type=norm_type),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(expert_channels + in_channels, expert_channels, kernel_size=1, bias=False),
            make_norm_2d(expert_channels, norm_type),
            nn.SiLU(inplace=True),
            nn.Conv2d(expert_channels, self.num_experts, kernel_size=1),
        )
        self.experts = nn.ModuleList(
            [ExpertHead(expert_channels, norm_type=norm_type, dropout=dropout) for _ in range(self.num_experts)]
        )

        self._last_gate_maps: Optional[torch.Tensor] = None
        self._last_los_base_residual: Optional[torch.Tensor] = None
        self._last_nlos_delta_residual: Optional[torch.Tensor] = None
        self._last_nlos_full_residual: Optional[torch.Tensor] = None
        self._last_residual: Optional[torch.Tensor] = None

    def moe_balance_loss(self, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._last_gate_maps is None:
            device = valid_mask.device if valid_mask is not None else "cpu"
            return torch.tensor(0.0, device=device)
        gates = self._last_gate_maps
        if valid_mask is None:
            mean_gate = gates.mean(dim=(0, 2, 3))
        else:
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.to(dtype=gates.dtype)
            denom = valid_mask.sum().clamp_min(1.0)
            mean_gate = (gates * valid_mask).sum(dim=(0, 2, 3)) / denom
        return float(self.num_experts) * torch.sum(mean_gate * mean_gate) - 1.0

    def last_branch_outputs(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._last_los_base_residual, self._last_nlos_delta_residual, self._last_nlos_full_residual

    def forward(self, x: torch.Tensor, scalar_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        decoder_features = self.backbone(x, scalar_cond)
        shared = self.shared(decoder_features)
        x_hint = x
        if x_hint.shape[-2:] != shared.shape[-2:]:
            x_hint = F.interpolate(x_hint, size=shared.shape[-2:], mode="bilinear", align_corners=False)

        los_base = self.los_base_head(shared)

        nlos_features = self.nlos_prep(torch.cat([shared, x_hint], dim=1))
        gate_logits = self.gate(torch.cat([shared, x_hint], dim=1))
        gate_maps = torch.softmax(gate_logits, dim=1)
        self._last_gate_maps = gate_maps

        nlos_delta = None
        for expert_idx, expert in enumerate(self.experts):
            expert_out = expert(nlos_features)
            weighted = gate_maps[:, expert_idx : expert_idx + 1] * expert_out
            nlos_delta = weighted if nlos_delta is None else (nlos_delta + weighted)
        if nlos_delta is None:
            raise RuntimeError("MoE branch produced no expert outputs")

        if los_base.shape[-2:] != x.shape[-2:]:
            los_base = F.interpolate(los_base, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if nlos_delta.shape[-2:] != x.shape[-2:]:
            nlos_delta = F.interpolate(nlos_delta, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if 0 <= self.los_channel_index < x.shape[1]:
            los_prob = x[:, self.los_channel_index : self.los_channel_index + 1].clamp(0.0, 1.0)
        else:
            los_prob = torch.full_like(los_base, 0.5)

        nlos_full = los_base + nlos_delta
        residual = los_prob * los_base + (1.0 - los_prob) * nlos_full

        self._last_los_base_residual = los_base
        self._last_nlos_delta_residual = nlos_delta
        self._last_nlos_full_residual = nlos_full
        self._last_residual = residual
        return residual
