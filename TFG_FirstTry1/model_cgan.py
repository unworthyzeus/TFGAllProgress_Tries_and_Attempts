from __future__ import annotations

import torch
import torch.nn as nn

from model_unet import CKMUNet


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64):
        super().__init__()
        self.generator = CKMUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, target_channels: int, base_channels: int = 64):
        super().__init__()
        c = in_channels + target_channels
        self.model = nn.Sequential(
            nn.Conv2d(c, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x, y], dim=1))
