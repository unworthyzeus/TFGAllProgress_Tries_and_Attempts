from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model_unet import CKMUNet


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
        dropout_down3: float = 0.1,
        dropout_down4: float = 0.2,
        dropout_up1: float = 0.1,
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
            dropout_down3=dropout_down3,
            dropout_down4=dropout_down4,
            dropout_up1=dropout_up1,
        )

    def forward(self, x: torch.Tensor, scalar_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.generator(x, scalar_cond)
