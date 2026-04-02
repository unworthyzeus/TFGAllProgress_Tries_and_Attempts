from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _resolve_group_norm_groups(channels: int) -> int:
    target_groups = min(32, max(channels // 8, 1))
    groups = max(target_groups, 1)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def make_norm_2d(channels: int, norm_type: str = "group") -> nn.Module:
    norm = str(norm_type).lower()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        return nn.GroupNorm(_resolve_group_norm_groups(channels), channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        norm_type: str = "group",
        act: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            make_norm_2d(out_channels, norm_type),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        norm_type: str = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(
            in_channels,
            out_channels,
            stride=stride,
            dilation=dilation,
            norm_type=norm_type,
        )
        self.conv2 = ConvNormAct(
            out_channels,
            out_channels,
            dilation=dilation,
            norm_type=norm_type,
            act=False,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                norm_type=norm_type,
                act=False,
            )
        else:
            self.skip = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.act(x + residual)


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_blocks: int,
        stride: int,
        norm_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = [
            ResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                norm_type=norm_type,
                dropout=dropout,
            )
        ]
        for _ in range(max(0, num_blocks - 1)):
            blocks.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    norm_type=norm_type,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DilatedContextBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        dilations: Sequence[int],
        norm_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(channels, channels, dilation=int(d), norm_type=norm_type),
                    nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
                    ConvNormAct(channels, channels, dilation=int(d), norm_type=norm_type),
                )
                for d in dilations
            ]
        )
        self.project = nn.Sequential(
            ConvNormAct(channels * len(self.branches), channels, kernel_size=1, norm_type=norm_type),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        fused = self.project(torch.cat(feats, dim=1))
        return fused + x


class ExpertHead(nn.Module):
    def __init__(self, channels: int, *, norm_type: str, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock(channels, channels, norm_type=norm_type, dropout=dropout),
            ConvNormAct(channels, channels, norm_type=norm_type),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PMNetResidualRegressor(nn.Module):
    """
    Try 46:
    - PMNet-inspired shared trunk over the calibrated physical prior
    - lightweight LoS-specific residual head
    - stronger NLoS-only mixture-of-experts residual head
    - final residual blended by the explicit LoS map channel
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        *,
        base_channels: int = 64,
        encoder_blocks: Sequence[int] = (2, 2, 2, 2),
        context_dilations: Sequence[int] = (1, 2, 4, 8),
        norm_type: str = "group",
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        num_experts: int = 4,
        expert_channels: int = 64,
        los_channel_index: int = 1,
    ) -> None:
        super().__init__()
        if int(out_channels) != 1:
            raise ValueError("PMNetResidualRegressor currently expects out_channels == 1")
        widths = [
            int(base_channels),
            int(base_channels * 2),
            int(base_channels * 4),
            int(base_channels * 8),
        ]
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.num_experts = max(int(num_experts), 2)
        expert_channels = int(expert_channels)
        self.los_channel_index = int(los_channel_index)

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, widths[0], kernel_size=5, norm_type=norm_type),
            ResidualBlock(widths[0], widths[0], norm_type=norm_type, dropout=dropout),
        )
        self.stage1 = EncoderStage(
            widths[0], widths[0], num_blocks=int(encoder_blocks[0]), stride=1, norm_type=norm_type, dropout=dropout
        )
        self.stage2 = EncoderStage(
            widths[0], widths[1], num_blocks=int(encoder_blocks[1]), stride=2, norm_type=norm_type, dropout=dropout
        )
        self.stage3 = EncoderStage(
            widths[1], widths[2], num_blocks=int(encoder_blocks[2]), stride=2, norm_type=norm_type, dropout=dropout
        )
        self.stage4 = EncoderStage(
            widths[2], widths[3], num_blocks=int(encoder_blocks[3]), stride=2, norm_type=norm_type, dropout=dropout
        )
        self.context = DilatedContextBlock(
            widths[3],
            dilations=context_dilations,
            norm_type=norm_type,
            dropout=dropout,
        )

        self.lat4 = ConvNormAct(widths[3], widths[2], kernel_size=1, norm_type=norm_type)
        self.lat3 = ConvNormAct(widths[2], widths[2], kernel_size=1, norm_type=norm_type)
        self.lat2 = ConvNormAct(widths[1], widths[1], kernel_size=1, norm_type=norm_type)
        self.lat1 = ConvNormAct(widths[0], widths[0], kernel_size=1, norm_type=norm_type)
        self.top3_to_2 = ConvNormAct(widths[2], widths[1], kernel_size=1, norm_type=norm_type)
        self.top2_to_1 = ConvNormAct(widths[1], widths[0], kernel_size=1, norm_type=norm_type)

        self.smooth3 = ResidualBlock(widths[2], widths[2], norm_type=norm_type, dropout=dropout)
        self.smooth2 = ResidualBlock(widths[1], widths[1], norm_type=norm_type, dropout=dropout)
        self.smooth1 = ResidualBlock(widths[0], widths[0], norm_type=norm_type, dropout=dropout)

        fusion_channels = widths[0] + widths[1] + widths[2]
        self.shared = nn.Sequential(
            ConvNormAct(fusion_channels, expert_channels, norm_type=norm_type),
            ResidualBlock(expert_channels, expert_channels, norm_type=norm_type, dropout=dropout),
        )
        self.los_head = nn.Sequential(
            ResidualBlock(expert_channels, max(expert_channels // 2, 16), norm_type=norm_type, dropout=dropout),
            ConvNormAct(max(expert_channels // 2, 16), max(expert_channels // 2, 16), norm_type=norm_type),
            nn.Conv2d(max(expert_channels // 2, 16), 1, kernel_size=1),
        )
        self.nlos_prep = nn.Sequential(
            ConvNormAct(expert_channels + in_channels, expert_channels, kernel_size=1, norm_type=norm_type),
            ResidualBlock(expert_channels, expert_channels, norm_type=norm_type, dropout=dropout),
        )
        self.gate = nn.Sequential(
            ConvNormAct(expert_channels + in_channels, expert_channels, kernel_size=1, norm_type=norm_type),
            nn.Conv2d(expert_channels, self.num_experts, kernel_size=1),
        )
        self.experts = nn.ModuleList(
            [ExpertHead(expert_channels, norm_type=norm_type, dropout=dropout) for _ in range(self.num_experts)]
        )
        self._last_gate_maps: torch.Tensor | None = None
        self._last_los_residual: torch.Tensor | None = None
        self._last_nlos_residual: torch.Tensor | None = None
        self._last_residual: torch.Tensor | None = None

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    @staticmethod
    def _upsample_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def moe_balance_loss(self, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
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

    def last_branch_outputs(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._last_los_residual, self._last_nlos_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self._run(self.stem, x)
        e1 = self._run(self.stage1, x0)
        e2 = self._run(self.stage2, e1)
        e3 = self._run(self.stage3, e2)
        e4 = self._run(self.stage4, e3)
        c4 = self._run(self.context, e4)

        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(e3) + self._upsample_like(p4, e3))
        p2 = self.smooth2(self.lat2(e2) + self._upsample_like(self.top3_to_2(p3), e2))
        p1 = self.smooth1(self.lat1(e1) + self._upsample_like(self.top2_to_1(p2), e1))

        fused = torch.cat(
            [
                p1,
                self._upsample_like(p2, p1),
                self._upsample_like(p3, p1),
            ],
            dim=1,
        )
        shared = self.shared(fused)
        x_hint = self._upsample_like(x, shared)

        los_residual = self.los_head(shared)

        nlos_features = self.nlos_prep(torch.cat([shared, x_hint], dim=1))
        gate_logits = self.gate(torch.cat([shared, x_hint], dim=1))
        gate_maps = torch.softmax(gate_logits, dim=1)
        self._last_gate_maps = gate_maps

        expert_outputs = [expert(nlos_features) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=1).squeeze(2)
        nlos_residual = torch.sum(gate_maps * stacked, dim=1, keepdim=True)

        if los_residual.shape[-2:] != x.shape[-2:]:
            los_residual = F.interpolate(los_residual, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if nlos_residual.shape[-2:] != x.shape[-2:]:
            nlos_residual = F.interpolate(nlos_residual, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if 0 <= self.los_channel_index < x.shape[1]:
            los_prob = x[:, self.los_channel_index : self.los_channel_index + 1].clamp(0.0, 1.0)
        else:
            los_prob = torch.full_like(los_residual, 0.5)

        residual = los_prob * los_residual + (1.0 - los_prob) * nlos_residual
        self._last_los_residual = los_residual
        self._last_nlos_residual = nlos_residual
        self._last_residual = residual
        return residual
