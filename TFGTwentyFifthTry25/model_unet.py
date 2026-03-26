from __future__ import annotations

from typing import Optional

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


def make_norm_2d(channels: int, norm_type: str = 'batch') -> nn.Module:
    norm = str(norm_type).lower()
    if norm == 'batch':
        return nn.BatchNorm2d(channels)
    if norm == 'group':
        return nn.GroupNorm(_resolve_group_norm_groups(channels), channels)
    if norm == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    if norm in {'none', 'identity'}:
        return nn.Identity()
    raise ValueError(f'Unsupported norm_type: {norm_type}')


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, norm_type: str = 'batch'):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            make_norm_2d(out_channels, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            make_norm_2d(out_channels, norm_type),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleProject(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = 'transpose'):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode == 'transpose':
            self.block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.proj = None
        elif self.mode in {'nearest', 'bilinear'}:
            self.block = None
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            raise ValueError(f'Unsupported upsample_mode: {mode}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'transpose':
            return self.block(x)
        if self.mode == 'nearest':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.proj(x)


class LightweightBottleneckAttention(nn.Module):
    def __init__(self, channels: int, attn_dim: int = 256, num_heads: int = 4, ff_mult: int = 2):
        super().__init__()
        attn_dim = max(32, int(attn_dim))
        num_heads = max(1, int(num_heads))
        while attn_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        ff_dim = max(attn_dim * int(ff_mult), attn_dim)
        self.in_proj = nn.Conv2d(channels, attn_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, attn_dim),
        )
        self.out_proj = nn.Conv2d(attn_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.in_proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_input = self.norm1(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(self.norm2(tokens))
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return residual + self.out_proj(x)


class CKMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 5,
        base_channels: int = 64,
        gradient_checkpointing: bool = False,
        path_loss_hybrid: bool = False,
        norm_type: str = 'batch',
        scalar_cond_dim: int = 0,
        scalar_film_hidden: int = 128,
        upsample_mode: str = 'transpose',
        bottleneck_attention: bool = False,
        bottleneck_attention_dim: int = 256,
        bottleneck_attention_heads: int = 4,
    ):
        super().__init__()
        bc = base_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.path_loss_hybrid = path_loss_hybrid
        self.norm_type = str(norm_type)
        self.upsample_mode = str(upsample_mode)
        self.scalar_cond_dim = max(0, int(scalar_cond_dim))
        self.bottleneck_attention = bool(bottleneck_attention)
        bot_c = bc * 16
        if self.scalar_cond_dim > 0:
            hid = max(8, int(scalar_film_hidden))
            self.scalar_film_mlp = nn.Sequential(
                nn.Linear(self.scalar_cond_dim, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, 2 * bot_c),
            )
            nn.init.zeros_(self.scalar_film_mlp[-1].weight)
            nn.init.zeros_(self.scalar_film_mlp[-1].bias)
        else:
            self.scalar_film_mlp = None

        self.inc = DoubleConv(in_channels, bc, norm_type=self.norm_type)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc, bc * 2, norm_type=self.norm_type))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc * 2, bc * 4, norm_type=self.norm_type))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc * 4, bc * 8, dropout=0.1, norm_type=self.norm_type))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc * 8, bc * 16, dropout=0.2, norm_type=self.norm_type))
        if self.bottleneck_attention:
            self.bottleneck_block = LightweightBottleneckAttention(
                bot_c,
                attn_dim=bottleneck_attention_dim,
                num_heads=bottleneck_attention_heads,
            )
        else:
            self.bottleneck_block = nn.Identity()

        self.up1 = UpsampleProject(bc * 16, bc * 8, mode=self.upsample_mode)
        self.conv1 = DoubleConv(bc * 16, bc * 8, dropout=0.1, norm_type=self.norm_type)
        self.up2 = UpsampleProject(bc * 8, bc * 4, mode=self.upsample_mode)
        self.conv2 = DoubleConv(bc * 8, bc * 4, norm_type=self.norm_type)
        self.up3 = UpsampleProject(bc * 4, bc * 2, mode=self.upsample_mode)
        self.conv3 = DoubleConv(bc * 4, bc * 2, norm_type=self.norm_type)
        self.up4 = UpsampleProject(bc * 2, bc, mode=self.upsample_mode)
        self.conv4 = DoubleConv(bc * 2, bc, norm_type=self.norm_type)

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

    def forward(self, x: torch.Tensor, scalar_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = self._run_block(self.inc, x)
        x2 = self._run_block(self.down1, x1)
        x3 = self._run_block(self.down2, x2)
        x4 = self._run_block(self.down3, x3)
        x5 = self._run_block(self.down4, x4)

        if self.scalar_film_mlp is not None:
            if scalar_cond is None:
                raise ValueError('scalar_cond is required when scalar_cond_dim > 0')
            if scalar_cond.dim() == 1:
                scalar_cond = scalar_cond.unsqueeze(0)
            gb = self.scalar_film_mlp(scalar_cond)
            gamma, beta = gb.chunk(2, dim=1)
            g = gamma.view(gamma.size(0), -1, 1, 1)
            b = beta.view(beta.size(0), -1, 1, 1)
            x5 = x5 * (1.0 + torch.tanh(g)) + b
        x5 = self._run_block(self.bottleneck_block, x5)

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
