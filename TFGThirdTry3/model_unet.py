from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CKMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 5,
        base_channels: int = 64,
        gradient_checkpointing: bool = False,
        path_loss_hybrid: bool = False,
    ):
        super().__init__()
        bc = base_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.path_loss_hybrid = path_loss_hybrid

        self.inc = DoubleConv(in_channels, bc)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc, bc * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc * 2, bc * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc * 4, bc * 8, dropout=0.1))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc * 8, bc * 16, dropout=0.2))

        self.up1 = nn.ConvTranspose2d(bc * 16, bc * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(bc * 16, bc * 8, dropout=0.1)
        self.up2 = nn.ConvTranspose2d(bc * 8, bc * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(bc * 8, bc * 4)
        self.up3 = nn.ConvTranspose2d(bc * 4, bc * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(bc * 4, bc * 2)
        self.up4 = nn.ConvTranspose2d(bc * 2, bc, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(bc * 2, bc)

        if self.path_loss_hybrid:
            if out_channels < 2:
                raise ValueError('path_loss_hybrid requires model.out_channels >= 2')
            self.path_loss_head = nn.Conv2d(bc, 1, kernel_size=1)
            self.confidence_head = nn.Conv2d(bc, 1, kernel_size=1)
            aux_channels = int(out_channels) - 2
            self.aux_head = nn.Conv2d(bc, aux_channels, kernel_size=1) if aux_channels > 0 else None
        else:
            self.outc = nn.Conv2d(bc, out_channels, kernel_size=1)

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    @staticmethod
    def _align_to_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y == 0 and diff_x == 0:
            return x

        if diff_y >= 0 and diff_x >= 0:
            return F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        start_y = max((-diff_y) // 2, 0)
        end_y = start_y + skip.size(2)
        start_x = max((-diff_x) // 2, 0)
        end_x = start_x + skip.size(3)
        return x[:, :, start_y:end_y, start_x:end_x]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._run_block(self.inc, x)
        x2 = self._run_block(self.down1, x1)
        x3 = self._run_block(self.down2, x2)
        x4 = self._run_block(self.down3, x3)
        x5 = self._run_block(self.down4, x4)

        x = self.up1(x5)
        x = self._align_to_skip(x, x4)
        x = torch.cat([x4, x], dim=1)
        x = self._run_block(self.conv1, x)

        x = self.up2(x)
        x = self._align_to_skip(x, x3)
        x = torch.cat([x3, x], dim=1)
        x = self._run_block(self.conv2, x)

        x = self.up3(x)
        x = self._align_to_skip(x, x2)
        x = torch.cat([x2, x], dim=1)
        x = self._run_block(self.conv3, x)

        x = self.up4(x)
        x = self._align_to_skip(x, x1)
        x = torch.cat([x1, x], dim=1)
        x = self._run_block(self.conv4, x)

        if not self.path_loss_hybrid:
            return self.outc(x)

        outputs = [
            self.path_loss_head(x),
            self.confidence_head(x),
        ]
        if self.aux_head is not None:
            outputs.append(self.aux_head(x))
        return torch.cat(outputs, dim=1)
