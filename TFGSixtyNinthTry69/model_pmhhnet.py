from __future__ import annotations

import math
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


class SinusoidalScalarEmbedding(nn.Module):
    """Sinusoidal positional encoding for a continuous scalar (e.g. antenna height).

    Inspired by the timestep embedding in diffusion models (Ho et al., NeurIPS 2020;
    Dhariwal & Nichol, arXiv:2105.05233).  Maps a 1-D scalar to a rich frequency
    representation so the downstream MLP can resolve fine differences between
    nearby values (e.g. 30 m vs 35 m antenna height).

    The encoding is: [sin(h*f_0), cos(h*f_0), sin(h*f_1), cos(h*f_1), ...]
    where f_k = 1 / (max_period ^ (k / (D/2))) and D = embed_dim.
    """

    def __init__(self, embed_dim: int = 64, max_period: float = 1000.0) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        half = embed_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        """scalar: [B, 1] or [B] → output: [B, embed_dim]."""
        if scalar.ndim == 1:
            scalar = scalar.unsqueeze(-1)
        if scalar.shape[-1] > 1:
            scalar = scalar[:, :1]
        args = scalar.float() * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMAffine(nn.Module):
    def __init__(self, cond_dim: int, channels: int, *, hidden_dim: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mlp = nn.Sequential(
            nn.Linear(int(cond_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), int(channels) * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        affine = self.mlp(cond)
        gamma, beta = torch.chunk(affine, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., arXiv:1709.01507).

    Lightweight per-channel recalibration: global-average-pool → MLP bottleneck
    → sigmoid gating.  Costs < 0.1% extra FLOPs.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


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


class PMHNetResidualRegressor(PMNetResidualRegressor):
    """
    PMHNet:
    - PMNet backbone
    - plus a lightweight high-frequency branch driven by a fixed Laplacian filter
    - intended for experts where sharp local transitions matter more
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
        hf_channels: int | None = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            encoder_blocks=encoder_blocks,
            context_dilations=context_dilations,
            norm_type=norm_type,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        hf_width = int(hf_channels or max(8, base_channels // 2))
        self.hf_project = nn.Sequential(
            ConvNormAct(in_channels, hf_width, kernel_size=3, norm_type=norm_type),
            ResidualBlock(hf_width, hf_width, norm_type=norm_type, dropout=dropout),
            ConvNormAct(hf_width, base_channels, kernel_size=1, norm_type=norm_type),
        )
        fusion_channels = int(base_channels) + int(base_channels * 2) + int(base_channels * 4) + int(base_channels)
        self.head = nn.Sequential(
            ConvNormAct(fusion_channels, int(base_channels * 2), norm_type=norm_type),
            ResidualBlock(int(base_channels * 2), int(base_channels * 2), norm_type=norm_type, dropout=dropout),
            nn.Conv2d(int(base_channels * 2), out_channels, kernel_size=1),
        )
        laplacian = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", laplacian, persistent=False)

    def _high_frequency_map(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.laplacian_kernel.expand(x.shape[1], 1, 3, 3)
        hf = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
        return torch.abs(hf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf = self._run(self.hf_project, self._high_frequency_map(x))
        x0 = self._run(self.stem, x) + hf
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
                self._upsample_like(hf, p1),
            ],
            dim=1,
        )
        residual = self.head(fused)
        if residual.shape[-2:] != x.shape[-2:]:
            residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return residual


class PMHHNetResidualRegressor(PMHNetResidualRegressor):
    """PMHHNet — height-aware path-loss residual regressor.

    Core innovation: a single model that predicts path loss at **arbitrary
    continuous antenna heights** using multi-frequency FiLM conditioning,
    eliminating the need for separate height-specific models.

    Height conditioning pipeline (inspired by diffusion model practices):

    1. **Sinusoidal positional encoding** of the raw height scalar
       (Ho et al., NeurIPS 2020; Dhariwal & Nichol, arXiv:2105.05233).
       This maps the 1-D height to a multi-frequency representation so the
       network can resolve fine differences (e.g. 30 m vs 35 m).

    2. **Learned MLP projection** from sinusoidal embedding to a conditioning
       vector, following DDPM / ADM practice.

    3. **Per-layer FiLM modulation** at 7 points in the network:
       stem, encoder stages 1-4, context bottleneck, HF branch, and final
       fusion.  This follows the diffusion U-Net pattern where the time
       embedding modulates every residual block (Dhariwal & Nichol 2021).

    4. **SE channel attention** optionally applied after each encoder stage,
       letting the model learn height-dependent channel importance.

    The resulting architecture can interpolate between heights seen during
    training and partially extrapolate to unseen altitudes, because the
    sinusoidal encoding provides a smooth, structured representation of the
    continuous height variable.
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
        hf_channels: int | None = None,
        scalar_dim: int = 1,
        scalar_hidden_dim: int | None = None,
        sinusoidal_embed_dim: int = 64,
        sinusoidal_max_period: float = 1000.0,
        use_se_attention: bool = False,
        se_reduction: int = 4,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            encoder_blocks=encoder_blocks,
            context_dilations=context_dilations,
            norm_type=norm_type,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
            hf_channels=hf_channels,
        )
        hidden_dim = int(scalar_hidden_dim or max(32, base_channels * 2))
        fused_channels = int(base_channels) + int(base_channels * 2) + int(base_channels * 4) + int(base_channels)
        self.scalar_dim = int(max(1, scalar_dim))

        # --- Sinusoidal → MLP embedding (diffusion-model style) ---
        self.sinusoidal_pos_enc = SinusoidalScalarEmbedding(
            embed_dim=sinusoidal_embed_dim,
            max_period=sinusoidal_max_period,
        )
        self.scalar_embed = nn.Sequential(
            nn.Linear(sinusoidal_embed_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )

        # --- Per-layer FiLM: stem + 4 encoder stages + context + HF + fused ---
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.film_stem = FiLMAffine(hidden_dim, int(base_channels), hidden_dim=hidden_dim)
        self.film_e1 = FiLMAffine(hidden_dim, widths[0], hidden_dim=hidden_dim)
        self.film_e2 = FiLMAffine(hidden_dim, widths[1], hidden_dim=hidden_dim)
        self.film_e3 = FiLMAffine(hidden_dim, widths[2], hidden_dim=hidden_dim)
        self.film_e4 = FiLMAffine(hidden_dim, widths[3], hidden_dim=hidden_dim)
        self.film_context = FiLMAffine(hidden_dim, widths[3], hidden_dim=hidden_dim)
        self.film_hf = FiLMAffine(hidden_dim, int(base_channels), hidden_dim=hidden_dim)
        self.film_fused = FiLMAffine(hidden_dim, fused_channels, hidden_dim=hidden_dim)

        # --- SE attention after each encoder stage ---
        if use_se_attention:
            self.se1 = SEBlock(widths[0], reduction=se_reduction)
            self.se2 = SEBlock(widths[1], reduction=se_reduction)
            self.se3 = SEBlock(widths[2], reduction=se_reduction)
            self.se4 = SEBlock(widths[3], reduction=se_reduction)
        else:
            self.se1 = nn.Identity()
            self.se2 = nn.Identity()
            self.se3 = nn.Identity()
            self.se4 = nn.Identity()

    def _resolve_scalar_cond(self, x: torch.Tensor, scalar_cond: torch.Tensor | None) -> torch.Tensor:
        if scalar_cond is None:
            scalar_cond = torch.zeros((x.shape[0], self.scalar_dim), device=x.device, dtype=x.dtype)
        if scalar_cond.ndim == 1:
            scalar_cond = scalar_cond.unsqueeze(0)
        scalar_cond = scalar_cond.to(device=x.device, dtype=x.dtype)
        if scalar_cond.shape[1] != self.scalar_dim:
            raise ValueError(f"Expected scalar_cond dim {self.scalar_dim}, got {scalar_cond.shape[1]}")
        sinusoidal = self.sinusoidal_pos_enc(scalar_cond)
        return self.scalar_embed(sinusoidal)

    def forward(self, x: torch.Tensor, scalar_cond: torch.Tensor | None = None) -> torch.Tensor:
        cond = self._resolve_scalar_cond(x, scalar_cond)

        # Same as PMHNetResidualRegressor: stem(x) + hf_project(Laplacian|x|) so edges
        # reach the encoder; FiLM was added without this sum, leaving HF only at fusion.
        hf = self._run(self.hf_project, self._high_frequency_map(x))
        x0 = self._run(self.stem, x) + hf
        x0 = self.film_stem(x0, cond)
        e1 = self.se1(self.film_e1(self._run(self.stage1, x0), cond))
        e2 = self.se2(self.film_e2(self._run(self.stage2, e1), cond))
        e3 = self.se3(self.film_e3(self._run(self.stage3, e2), cond))
        e4 = self.se4(self.film_e4(self._run(self.stage4, e3), cond))
        c4 = self._run(self.context, e4)
        c4 = self.film_context(c4, cond)

        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(e3) + self._upsample_like(p4, e3))
        p2 = self.smooth2(self.lat2(e2) + self._upsample_like(self.top3_to_2(p3), e2))
        p1 = self.smooth1(self.lat1(e1) + self._upsample_like(self.top2_to_1(p2), e1))

        hf = self.film_hf(hf, cond)
        fused = torch.cat(
            [
                p1,
                self._upsample_like(p2, p1),
                self._upsample_like(p3, p1),
                self._upsample_like(hf, p1),
            ],
            dim=1,
        )
        fused = self.film_fused(fused, cond)
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

    def forward(self, x: torch.Tensor, scalar_cond: torch.Tensor | None = None) -> torch.Tensor:
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


class UNetResidualRefinerH(UNetResidualRefiner):
    """Height-conditioned Stage-2 refiner.

    Adds lightweight FiLM modulation at the bottleneck and output, so the
    full-resolution correction is altitude-aware.  This matters because edge
    sharpness and diffraction patterns are height-dependent: at low altitude,
    NLoS shadow edges are harder; at high altitude, the transition is smoother.

    Uses the same sinusoidal → MLP → FiLM pipeline as PMHHNet for consistency.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        *,
        base_channels: int = 48,
        norm_type: str = "group",
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        scalar_dim: int = 1,
        sinusoidal_embed_dim: int = 64,
        sinusoidal_max_period: float = 1000.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            norm_type=norm_type,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.scalar_dim = int(max(1, scalar_dim))
        hidden_dim = max(32, base_channels * 2)
        c1 = int(base_channels)
        c4 = int(base_channels * 8)
        self.sinusoidal_pos_enc = SinusoidalScalarEmbedding(
            embed_dim=sinusoidal_embed_dim,
            max_period=sinusoidal_max_period,
        )
        self.scalar_embed = nn.Sequential(
            nn.Linear(sinusoidal_embed_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.film_bottleneck = FiLMAffine(hidden_dim, c4, hidden_dim=hidden_dim)
        self.film_out = FiLMAffine(hidden_dim, c1, hidden_dim=hidden_dim)

    def _resolve_scalar_cond(self, x: torch.Tensor, scalar_cond: torch.Tensor | None) -> torch.Tensor:
        if scalar_cond is None:
            scalar_cond = torch.zeros((x.shape[0], self.scalar_dim), device=x.device, dtype=x.dtype)
        if scalar_cond.ndim == 1:
            scalar_cond = scalar_cond.unsqueeze(0)
        scalar_cond = scalar_cond.to(device=x.device, dtype=x.dtype)
        sinusoidal = self.sinusoidal_pos_enc(scalar_cond)
        return self.scalar_embed(sinusoidal)

    def forward(self, x: torch.Tensor, scalar_cond: torch.Tensor | None = None) -> torch.Tensor:
        cond = self._resolve_scalar_cond(x, scalar_cond)

        e1 = self._run(self.enc1, x)
        e2 = self._run(self.enc2, e1)
        e3 = self._run(self.enc3, e2)
        b = self._run(self.bottleneck, e3)
        b = self.film_bottleneck(b, cond)

        d3 = self.up3(torch.cat([self._up_to(b, e3), e3], dim=1))
        d2 = self.up2(torch.cat([self._up_to(d3, e2), e2], dim=1))
        d1 = self.up1(torch.cat([self._up_to(d2, e1), e1], dim=1))
        d1 = self.film_out(d1, cond)

        out = self.out_head(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out
