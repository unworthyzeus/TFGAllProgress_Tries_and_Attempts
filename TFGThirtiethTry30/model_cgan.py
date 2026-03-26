from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_unet import CKMUNet, make_norm_2d


class UNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        gradient_checkpointing: bool = False,
        path_loss_hybrid: bool = False,
        norm_type: str = 'batch',
        scalar_cond_dim: int = 0,
        scalar_film_hidden: int = 128,
        upsample_mode: str = 'transpose',
    ):
        super().__init__()
        self.generator = CKMUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            gradient_checkpointing=gradient_checkpointing,
            path_loss_hybrid=path_loss_hybrid,
            norm_type=norm_type,
            scalar_cond_dim=scalar_cond_dim,
            scalar_film_hidden=scalar_film_hidden,
            upsample_mode=upsample_mode,
        )

    def forward(self, x: torch.Tensor, scalar_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.generator(x, scalar_cond)


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        target_channels: int,
        base_channels: int = 64,
        norm_type: str = 'batch',
        input_downsample_factor: int = 1,
    ):
        super().__init__()
        self.input_downsample_factor = max(int(input_downsample_factor), 1)
        c = in_channels + target_channels
        self.model = nn.Sequential(
            nn.Conv2d(c, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            make_norm_2d(base_channels * 2, norm_type),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            make_norm_2d(base_channels * 4, norm_type),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1),
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
