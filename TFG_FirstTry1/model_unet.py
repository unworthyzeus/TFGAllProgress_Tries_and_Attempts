from __future__ import annotations

import torch
import torch.nn as nn


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
    def __init__(self, in_channels: int = 3, out_channels: int = 5, base_channels: int = 64):
        super().__init__()
        bc = base_channels

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

        self.outc = nn.Conv2d(bc, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        return self.outc(x)
