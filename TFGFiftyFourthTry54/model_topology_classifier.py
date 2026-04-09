from __future__ import annotations

import torch
import torch.nn as nn

from model_pmhhnet import ConvNormAct, ResidualBlock, make_norm_2d


class TinyTopologyClassifier(nn.Module):
    """Small CNN for routing topology maps into one of the Try54 expert classes."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        *,
        base_channels: int = 24,
        norm_type: str = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 3, base_channels * 4]
        self.stem = ConvNormAct(in_channels, widths[0], norm_type=norm_type)
        self.stage1 = nn.Sequential(
            ResidualBlock(widths[0], widths[0], norm_type=norm_type, dropout=dropout),
            ResidualBlock(widths[0], widths[1], stride=2, norm_type=norm_type, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(widths[1], widths[1], norm_type=norm_type, dropout=dropout),
            ResidualBlock(widths[1], widths[2], stride=2, norm_type=norm_type, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(widths[2], widths[2], norm_type=norm_type, dropout=dropout),
            ResidualBlock(widths[2], widths[3], stride=2, norm_type=norm_type, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(widths[3], widths[3]),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout if dropout > 0.0 else 0.1),
            nn.Linear(widths[3], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


class TinyAntennaAwareTopologyClassifier(nn.Module):
    """Optional classifier variant that fuses a scalar antenna-height embedding."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        *,
        base_channels: int = 24,
        norm_type: str = "group",
        dropout: float = 0.0,
        scalar_dim: int = 1,
    ) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 3, base_channels * 4]
        self.stem = ConvNormAct(in_channels, widths[0], norm_type=norm_type)
        self.stage1 = nn.Sequential(
            ResidualBlock(widths[0], widths[0], norm_type=norm_type, dropout=dropout),
            ResidualBlock(widths[0], widths[1], stride=2, norm_type=norm_type, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(widths[1], widths[1], norm_type=norm_type, dropout=dropout),
            ResidualBlock(widths[1], widths[2], stride=2, norm_type=norm_type, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(widths[2], widths[2], norm_type=norm_type, dropout=dropout),
            ResidualBlock(widths[2], widths[3], stride=2, norm_type=norm_type, dropout=dropout),
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(max(1, int(scalar_dim)), widths[1]),
            nn.SiLU(inplace=True),
            nn.Linear(widths[1], widths[2]),
            nn.SiLU(inplace=True),
        )
        self.norm = make_norm_2d(widths[3], norm_type)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(widths[3] + widths[2], widths[3]),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout if dropout > 0.0 else 0.1),
            nn.Linear(widths[3], num_classes),
        )

    def forward(self, x: torch.Tensor, scalar_cond: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.norm(x)
        image_feat = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        scalar_feat = self.scalar_proj(scalar_cond)
        return self.head(torch.cat([image_feat, scalar_feat], dim=1))
