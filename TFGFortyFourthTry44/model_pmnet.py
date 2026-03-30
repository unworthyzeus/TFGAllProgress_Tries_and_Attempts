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


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int = 1,
        *,
        norm_type: str = "group",
        relu: bool = True,
    ) -> None:
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
        if relu:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        dilation: int,
        downsample: bool,
        norm_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        mid_channels = out_channels // self.expansion
        self.reduce = ConvNormAct(in_channels, mid_channels, 1, stride, 0, 1, norm_type=norm_type)
        self.conv3x3 = ConvNormAct(
            mid_channels,
            mid_channels,
            3,
            1,
            dilation,
            dilation,
            norm_type=norm_type,
        )
        self.increase = ConvNormAct(mid_channels, out_channels, 1, 1, 0, 1, norm_type=norm_type, relu=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.shortcut = (
            ConvNormAct(in_channels, out_channels, 1, stride, 0, 1, norm_type=norm_type, relu=False)
            if downsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.dropout(h)
        h = self.increase(h)
        h = h + self.shortcut(x)
        return F.relu(h, inplace=True)


class ResLayer(nn.Sequential):
    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilation: int,
        *,
        norm_type: str,
        dropout: float,
        multi_grids: Sequence[int] | None = None,
    ) -> None:
        modules: list[nn.Module] = []
        if multi_grids is None:
            multi_grids = [1] * int(n_layers)
        for idx in range(int(n_layers)):
            modules.append(
                Bottleneck(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride if idx == 0 else 1,
                    dilation=int(dilation) * int(multi_grids[idx]),
                    downsample=(idx == 0),
                    norm_type=norm_type,
                    dropout=dropout,
                )
            )
        super().__init__(*modules)


class Stem(nn.Sequential):
    def __init__(self, out_channels: int, *, in_channels: int, norm_type: str) -> None:
        super().__init__(
            ConvNormAct(in_channels, out_channels, 7, 2, 3, 1, norm_type=norm_type),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )


class ImagePool(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, norm_type: str) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvNormAct(in_channels, out_channels, 1, 1, 0, 1, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pooled = self.pool(x)
        pooled = self.conv(pooled)
        return F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: Sequence[int], *, norm_type: str) -> None:
        super().__init__()
        self.stages = nn.ModuleList([ConvNormAct(in_channels, out_channels, 1, 1, 0, 1, norm_type=norm_type)])
        for rate in rates:
            self.stages.append(
                ConvNormAct(
                    in_channels,
                    out_channels,
                    3,
                    1,
                    int(rate),
                    int(rate),
                    norm_type=norm_type,
                )
            )
        self.stages.append(ImagePool(in_channels, out_channels, norm_type=norm_type))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([stage(x) for stage in self.stages], dim=1)


def conv_relu(in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(inplace=True),
    )


def conv_transpose_relu(in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
        nn.ReLU(inplace=True),
    )


class PMNetResidualRegressor(nn.Module):
    """
    PMNet-v3-style dense regressor adapted to the current pipeline.

    Key differences vs our previous PMNet-inspired backbone:
    - closer ResNet bottleneck encoder
    - ASPP context block as in the official repo
    - decoder closer to the original PMNet v3 head
    - still configurable norm_type so we can use group norm with batch_size=1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        *,
        base_channels: int = 32,
        encoder_blocks: Sequence[int] = (3, 4, 6, 3),
        context_dilations: Sequence[int] = (6, 12, 18),
        norm_type: str = "group",
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if int(out_channels) != 1:
            raise ValueError("PMNetResidualRegressor currently expects out_channels == 1")

        self.gradient_checkpointing = bool(gradient_checkpointing)

        c0 = int(base_channels)
        c2 = c0 * 4
        c3 = c0 * 8
        c4 = c0 * 8
        c5 = c0 * 16

        self.layer1 = Stem(c0, in_channels=in_channels, norm_type=norm_type)
        self.layer2 = ResLayer(
            int(encoder_blocks[0]),
            c0,
            c2,
            stride=1,
            dilation=1,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.layer3 = ResLayer(
            int(encoder_blocks[1]),
            c2,
            c3,
            stride=2,
            dilation=1,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.layer4 = ResLayer(
            int(encoder_blocks[2]),
            c3,
            c4,
            stride=2,
            dilation=1,
            norm_type=norm_type,
            dropout=dropout,
        )
        self.layer5 = ResLayer(
            int(encoder_blocks[3]),
            c4,
            c5,
            stride=1,
            dilation=2,
            norm_type=norm_type,
            dropout=dropout,
            multi_grids=(1, 2, 4)[: int(encoder_blocks[3])],
        )
        self.reduce = ConvNormAct(c2, c2, 1, 1, 0, 1, norm_type=norm_type)
        self.aspp = ASPP(c5, c2, context_dilations, norm_type=norm_type)
        concat_channels = c2 * (len(context_dilations) + 2)
        self.fc1 = ConvNormAct(concat_channels, c4, 1, 1, 0, 1, norm_type=norm_type)

        self.conv_up5 = conv_relu(c4, c4, 3, 1)
        self.conv_up4 = conv_relu(c4 + c4, c4, 3, 1)
        self.conv_up3 = conv_transpose_relu(c4 + c4, c3, 3, 1)
        self.conv_up2 = conv_relu(c3 + c2, c3, 3, 1)
        self.conv_up1 = conv_relu(c3 + c0, c3, 3, 1)
        self.conv_up0 = conv_relu(c3 + c0, c2, 3, 1)
        self.conv_up00 = nn.Sequential(
            nn.Conv2d(c2 + in_channels, c0, kernel_size=3, padding=1),
            make_norm_2d(c0, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, c0, kernel_size=3, padding=1),
            make_norm_2d(c0, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, out_channels, kernel_size=3, padding=1),
        )

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    @staticmethod
    def _align_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._run(self.layer1, x)
        x2 = self._run(self.layer2, x1)
        x3 = self.reduce(x2)
        x4 = self._run(self.layer3, x3)
        x5 = self._run(self.layer4, x4)
        x6 = self._run(self.layer5, x5)
        x7 = self._run(self.aspp, x6)
        x8 = self.fc1(x7)

        xup5 = self.conv_up5(x8)
        xup5 = self._align_like(xup5, x5)
        xup5 = torch.cat([xup5, x5], dim=1)
        xup4 = self.conv_up4(xup5)
        xup4 = self._align_like(xup4, x4)
        xup4 = torch.cat([xup4, x4], dim=1)
        xup3 = self.conv_up3(xup4)
        xup3 = self._align_like(xup3, x3)
        xup3 = torch.cat([xup3, x3], dim=1)
        xup2 = self.conv_up2(xup3)
        xup2 = self._align_like(xup2, x1)
        xup2 = torch.cat([xup2, x1], dim=1)
        xup1 = self.conv_up1(xup2)
        xup1 = self._align_like(xup1, x1)
        xup1 = torch.cat([xup1, x1], dim=1)
        xup0 = self.conv_up0(xup1)
        xup0 = self._align_like(xup0, x)
        xup0 = torch.cat([xup0, x], dim=1)
        return self.conv_up00(xup0)
