from __future__ import annotations

from typing import Iterable, Sequence

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


class TokenMixingContextBlock(nn.Module):
    """Lightweight global-context mixer for small path-loss models."""

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 4,
        pooled_size: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pooled_size = max(int(pooled_size), 2)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=max(1, int(num_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        hidden = max(int(channels * float(mlp_ratio)), channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, output_size=(self.pooled_size, self.pooled_size))
        tokens = pooled.flatten(2).transpose(1, 2)
        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        mixed = tokens.transpose(1, 2).reshape_as(pooled)
        mixed = F.interpolate(mixed, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + mixed


class CityTypeRoutedNLoSMoERegressor(nn.Module):
    """
    Small PMNet-like regressor that predicts one NLoS residual map per city type.

    Intended usage:
    - keep LoS as hard passthrough from the physical prior / stage1 prediction;
    - select one expert map according to automatic city-type inference;
    - apply the learned correction only on NLoS pixels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        *,
        base_channels: int = 48,
        encoder_blocks: Sequence[int] = (1, 2, 2, 2),
        context_dilations: Sequence[int] = (1, 2, 4),
        norm_type: str = "group",
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        attention_heads: int = 4,
        attention_pool_size: int = 8,
    ) -> None:
        super().__init__()
        if int(out_channels) < 2:
            raise ValueError("CityTypeRoutedNLoSMoERegressor expects at least 2 expert outputs")
        widths = [
            int(base_channels),
            int(base_channels * 2),
            int(base_channels * 4),
            int(base_channels * 6),
        ]
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, widths[0], kernel_size=5, norm_type=norm_type),
            ResidualBlock(widths[0], widths[0], norm_type=norm_type, dropout=dropout),
        )
        self.stage1 = EncoderStage(
            widths[0],
            widths[0],
            num_blocks=int(encoder_blocks[0]),
            stride=1,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.stage2 = EncoderStage(
            widths[0],
            widths[1],
            num_blocks=int(encoder_blocks[1]),
            stride=2,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.stage3 = EncoderStage(
            widths[1],
            widths[2],
            num_blocks=int(encoder_blocks[2]),
            stride=2,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.stage4 = EncoderStage(
            widths[2],
            widths[3],
            num_blocks=int(encoder_blocks[3]),
            stride=2,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.context = DilatedContextBlock(
            widths[3],
            dilations=context_dilations,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.token_mixer = TokenMixingContextBlock(
            widths[3],
            num_heads=attention_heads,
            pooled_size=attention_pool_size,
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
        self.head = nn.Sequential(
            ConvNormAct(fusion_channels, widths[1], norm_type=norm_type),
            ResidualBlock(widths[1], widths[1], norm_type=norm_type, dropout=dropout),
            nn.Conv2d(widths[1], int(out_channels), kernel_size=1),
        )

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    @staticmethod
    def _upsample_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self._run(self.stem, x)
        e1 = self._run(self.stage1, x0)
        e2 = self._run(self.stage2, e1)
        e3 = self._run(self.stage3, e2)
        e4 = self._run(self.stage4, e3)
        c4 = self._run(self.context, e4)
        c4 = self._run(self.token_mixer, c4)

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
        out = self.head(fused)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


class PMNetResidualRegressor(nn.Module):
    """
    PMNet-inspired residual regressor:
    - residual encoder (ResNet-like)
    - multi-branch dilated context module
    - FPN-style top-down fusion instead of a U-Net decoder
    - predicts only the residual over the calibrated physical prior
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
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, widths[0], kernel_size=5, norm_type=norm_type),
            ResidualBlock(widths[0], widths[0], norm_type=norm_type, dropout=dropout),
        )
        self.stage1 = EncoderStage(
            widths[0],
            widths[0],
            num_blocks=int(encoder_blocks[0]),
            stride=1,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.stage2 = EncoderStage(
            widths[0],
            widths[1],
            num_blocks=int(encoder_blocks[1]),
            stride=2,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.stage3 = EncoderStage(
            widths[1],
            widths[2],
            num_blocks=int(encoder_blocks[2]),
            stride=2,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.stage4 = EncoderStage(
            widths[2],
            widths[3],
            num_blocks=int(encoder_blocks[3]),
            stride=2,
            norm_type=norm_type,
            dropout=dropout,
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
        self.head = nn.Sequential(
            ConvNormAct(fusion_channels, widths[1], norm_type=norm_type),
            ResidualBlock(widths[1], widths[1], norm_type=norm_type, dropout=dropout),
            nn.Conv2d(widths[1], out_channels, kernel_size=1),
        )

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    @staticmethod
    def _upsample_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

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
        residual = self.head(fused)
        if residual.shape[-2:] != x.shape[-2:]:
            residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return residual


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        target_channels: int,
        *,
        base_channels: int = 32,
        norm_type: str = "group",
        input_downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        self.input_downsample_factor = max(int(input_downsample_factor), 1)
        c = int(in_channels) + int(target_channels)
        self.model = nn.Sequential(
            nn.Conv2d(c, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            make_norm_2d(base_channels * 2, norm_type),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            make_norm_2d(base_channels * 4, norm_type),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            make_norm_2d(base_channels * 8, norm_type),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.input_downsample_factor > 1:
            kernel = self.input_downsample_factor
            x = F.avg_pool2d(x, kernel_size=kernel, stride=kernel)
            y = F.avg_pool2d(y, kernel_size=kernel, stride=kernel)
        return self.model(torch.cat([x, y], dim=1))


class UNetResidualRefiner(nn.Module):
    """Compact U-Net used as a stage-2 residual refiner."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        *,
        base_channels: int = 48,
        norm_type: str = "group",
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        c1 = int(base_channels)
        c2 = int(base_channels * 2)
        c3 = int(base_channels * 4)
        c4 = int(base_channels * 8)

        self.enc1 = nn.Sequential(
            ConvNormAct(in_channels, c1, norm_type=norm_type),
            ResidualBlock(c1, c1, norm_type=norm_type, dropout=dropout),
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2, norm_type=norm_type, dropout=dropout),
            ResidualBlock(c2, c2, norm_type=norm_type, dropout=dropout),
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2, norm_type=norm_type, dropout=dropout),
            ResidualBlock(c3, c3, norm_type=norm_type, dropout=dropout),
        )
        self.bottleneck = nn.Sequential(
            ResidualBlock(c3, c4, stride=2, norm_type=norm_type, dropout=dropout),
            ResidualBlock(c4, c4, norm_type=norm_type, dropout=dropout),
        )

        self.up3 = ConvNormAct(c4 + c3, c3, norm_type=norm_type)
        self.up2 = ConvNormAct(c3 + c2, c2, norm_type=norm_type)
        self.up1 = ConvNormAct(c2 + c1, c1, norm_type=norm_type)

        self.out_head = nn.Sequential(
            ResidualBlock(c1, c1, norm_type=norm_type, dropout=dropout),
            nn.Conv2d(c1, out_channels, kernel_size=1),
        )

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self._run(self.enc1, x)
        e2 = self._run(self.enc2, e1)
        e3 = self._run(self.enc3, e2)
        b = self._run(self.bottleneck, e3)

        d3 = self.up3(torch.cat([self._up_to(b, e3), e3], dim=1))
        d2 = self.up2(torch.cat([self._up_to(d3, e2), e2], dim=1))
        d1 = self.up1(torch.cat([self._up_to(d2, e1), e1], dim=1))

        out = self.out_head(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


class GlobalContextUNetRefiner(nn.Module):
    """Small refiner with a U-Net path plus a lightweight global-context mixer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        *,
        base_channels: int = 40,
        norm_type: str = "group",
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        attention_heads: int = 4,
        attention_pool_size: int = 8,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        c1 = int(base_channels)
        c2 = int(base_channels * 2)
        c3 = int(base_channels * 4)
        c4 = int(base_channels * 6)

        self.enc1 = nn.Sequential(
            ConvNormAct(in_channels, c1, norm_type=norm_type),
            ResidualBlock(c1, c1, norm_type=norm_type, dropout=dropout),
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2, norm_type=norm_type, dropout=dropout),
            ResidualBlock(c2, c2, norm_type=norm_type, dropout=dropout),
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2, norm_type=norm_type, dropout=dropout),
            ResidualBlock(c3, c3, norm_type=norm_type, dropout=dropout),
        )
        self.bottleneck = nn.Sequential(
            ResidualBlock(c3, c4, stride=2, norm_type=norm_type, dropout=dropout),
            ResidualBlock(c4, c4, norm_type=norm_type, dropout=dropout),
        )
        self.token_mixer = TokenMixingContextBlock(
            c4,
            num_heads=attention_heads,
            pooled_size=attention_pool_size,
            dropout=dropout,
        )

        self.up3 = ConvNormAct(c4 + c3, c3, norm_type=norm_type)
        self.up2 = ConvNormAct(c3 + c2, c2, norm_type=norm_type)
        self.up1 = ConvNormAct(c2 + c1, c1, norm_type=norm_type)
        self.out_head = nn.Sequential(
            ResidualBlock(c1, c1, norm_type=norm_type, dropout=dropout),
            nn.Conv2d(c1, out_channels, kernel_size=1),
        )

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self._run(self.enc1, x)
        e2 = self._run(self.enc2, e1)
        e3 = self._run(self.enc3, e2)
        b = self._run(self.bottleneck, e3)
        b = self._run(self.token_mixer, b)

        d3 = self.up3(torch.cat([self._up_to(b, e3), e3], dim=1))
        d2 = self.up2(torch.cat([self._up_to(d3, e2), e2], dim=1))
        d1 = self.up1(torch.cat([self._up_to(d2, e1), e1], dim=1))

        out = self.out_head(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out
