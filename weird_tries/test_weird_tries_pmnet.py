from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "Datasets" / "CKM_Dataset_270326.h5"
DEFAULT_CHECKPOINTS = [
    SCRIPT_DIR / "model_0.00008.pt",
    SCRIPT_DIR / "USC_16H_16W.pt",
]

MODEL_BLOCKS = (3, 3, 27, 3)
MODEL_ATROUS_RATES = (6, 12, 18)
MODEL_MULTI_GRIDS = (1, 2, 4)

TRY54_TOPOLOGY_CLASSES = [
    "open_sparse_lowrise",
    "open_sparse_vertical",
    "mixed_compact_lowrise",
    "mixed_compact_midrise",
    "dense_block_midrise",
    "dense_block_highrise",
]

TRY54_PARTITION_THRESHOLDS = {
    "density_q1": 0.12,
    "density_q2": 0.28,
    "height_q1": 12.0,
    "height_q2": 28.0,
}


@dataclass(frozen=True)
class SampleRecord:
    city: str
    sample: str
    height_m: float
    expert_id: str
    city_type: str
    building_density: float
    mean_height: float


@dataclass(frozen=True)
class PMNetVariant:
    output_stride: int
    conv_up4_transpose: bool


@dataclass(frozen=True)
class SampleMetrics:
    city: str
    sample: str
    expert_id: str
    city_type: str
    height_m: float
    ground_rmse_db: float
    los_rmse_db: float
    nlos_rmse_db: float
    ground_rmse_norm: float
    los_rmse_norm: float
    nlos_rmse_norm: float
    ground_fraction: float
    los_fraction: float
    nlos_fraction: float


def resolve_device(device_name: str) -> Any:
    name = str(device_name).strip().lower()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml

            return torch_directml.device()
        except Exception:
            return torch.device("cpu")

    if name in {"cpu", "cuda"}:
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return torch.device(name)

    if name in {"directml", "dml", "amd"}:
        try:
            import torch_directml

            return torch_directml.device()
        except Exception as exc:
            raise RuntimeError(
                "DirectML was requested but torch-directml is not installed. Install torch-directml and retry."
            ) from exc

    raise ValueError(f"Unsupported device value: {device_name}")


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    load_kwargs = {"map_location": "cpu"}
    try:
        loaded = torch.load(path, weights_only=True, **load_kwargs)
    except TypeError:
        loaded = torch.load(path, **load_kwargs)

    if isinstance(loaded, dict):
        for nested_key in ("state_dict", "model", "generator"):
            nested = loaded.get(nested_key)
            if isinstance(nested, dict) and nested and all(torch.is_tensor(value) for value in nested.values()):
                loaded = nested
                break

    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint {path} did not load as a state dict.")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in loaded.items():
        clean_key = key[7:] if key.startswith("module.") else key
        state_dict[clean_key] = value
    return state_dict


def detect_pmnet_variant(state_dict: dict[str, torch.Tensor]) -> PMNetVariant:
    conv_up4 = state_dict.get("conv_up4.0.weight")
    if conv_up4 is None:
        raise KeyError("Checkpoint is missing conv_up4.0.weight, so the PMNet variant cannot be inferred.")

    shape = tuple(conv_up4.shape)
    if shape == (512, 1024, 3, 3):
        return PMNetVariant(output_stride=8, conv_up4_transpose=False)
    if shape == (1024, 512, 3, 3):
        return PMNetVariant(output_stride=16, conv_up4_transpose=True)

    raise ValueError(f"Unsupported conv_up4.0.weight shape: {shape}")


def _infer_city_type_from_density_height(density: float, height: float) -> str:
    density_q1 = float(TRY54_PARTITION_THRESHOLDS["density_q1"])
    density_q2 = float(TRY54_PARTITION_THRESHOLDS["density_q2"])
    height_q1 = float(TRY54_PARTITION_THRESHOLDS["height_q1"])
    height_q2 = float(TRY54_PARTITION_THRESHOLDS["height_q2"])

    if density >= density_q2 or height >= height_q2:
        return "dense_highrise"
    if density <= density_q1 and height <= height_q1:
        return "open_lowrise"
    return "mixed_midrise"


def _infer_try54_topology_class(density: float, height: float) -> str:
    density_q1 = float(TRY54_PARTITION_THRESHOLDS["density_q1"])
    density_q2 = float(TRY54_PARTITION_THRESHOLDS["density_q2"])
    height_q1 = float(TRY54_PARTITION_THRESHOLDS["height_q1"])
    height_q2 = float(TRY54_PARTITION_THRESHOLDS["height_q2"])

    if density <= density_q1:
        if height <= height_q1:
            return "open_sparse_lowrise"
        return "open_sparse_vertical"
    if density >= density_q2:
        if height <= height_q2:
            return "dense_block_midrise"
        return "dense_block_highrise"
    if height <= height_q1:
        return "mixed_compact_lowrise"
    return "mixed_compact_midrise"


def infer_partition_metadata(topology_map: np.ndarray, antenna_height_m: float, non_ground_threshold: float = 0.0) -> dict[str, Any]:
    del antenna_height_m  # The current Try54/68/69/72 routing depends on topology only.
    non_ground = topology_map != float(non_ground_threshold)
    building_density = float(np.mean(non_ground))
    non_zero = topology_map[non_ground]
    mean_height = float(np.mean(non_zero)) if non_zero.size else 0.0
    expert_id = _infer_try54_topology_class(building_density, mean_height)
    city_type = _infer_city_type_from_density_height(building_density, mean_height)
    return {
        "expert_id": expert_id,
        "city_type": city_type,
        "building_density": building_density,
        "mean_height": mean_height,
    }


def normalize_expert_filters(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized = {str(value).strip() for value in values if str(value).strip()}
    if not normalized or normalized == {"all"}:
        return None
    invalid = sorted(normalized.difference(TRY54_TOPOLOGY_CLASSES))
    if invalid:
        raise ValueError(f"Unsupported expert filter(s): {', '.join(invalid)}")
    return normalized


def _empty_metric_bucket() -> dict[str, float | int]:
    return {"sse": 0.0, "count": 0, "sample_count": 0}


def _update_metric_bucket(bucket: dict[str, float | int], diff: torch.Tensor) -> None:
    if diff.numel() == 0:
        return
    bucket["sse"] = float(bucket["sse"]) + float(torch.sum(diff * diff).item())
    bucket["count"] = int(bucket["count"]) + int(diff.numel())
    bucket["sample_count"] = int(bucket["sample_count"]) + 1


def _finalize_metric_bucket(bucket: dict[str, float | int]) -> dict[str, float | int]:
    count = int(bucket["count"])
    rmse = float(np.sqrt(float(bucket["sse"]) / count)) if count > 0 else float("nan")
    return {
        "rmse": rmse,
        "sse": float(bucket["sse"]),
        "count": count,
        "sample_count": int(bucket["sample_count"]),
    }


def _new_metric_group() -> dict[str, dict[str, float | int]]:
    return {
        "ground": _empty_metric_bucket(),
        "los": _empty_metric_bucket(),
        "nlos": _empty_metric_bucket(),
    }


class _ConvBnReLU(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        relu: bool = True,
    ) -> None:
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=1 - 0.999))
        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, dilation: int, downsample: bool) -> None:
        super().__init__()
        mid_ch = out_ch // 4
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    def __init__(
        self,
        n_layers: int,
        in_ch: int,
        out_ch: int,
        stride: int,
        dilation: int,
        multi_grids: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        for index, multi_grid in enumerate(multi_grids):
            self.add_module(
                f"block{index + 1}",
                _Bottleneck(
                    in_ch=in_ch if index == 0 else out_ch,
                    out_ch=out_ch,
                    stride=stride if index == 0 else 1,
                    dilation=dilation * int(multi_grid),
                    downsample=index == 0,
                ),
            )


class _Stem(nn.Sequential):
    def __init__(self, out_ch: int, in_ch: int = 2) -> None:
        super().__init__()
        self.add_module("conv1", _ConvBnReLU(in_ch, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(2, 2, 1, ceil_mode=True))


class _ImagePool(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pooled = self.pool(x)
        pooled = self.conv(pooled)
        return F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)


class _ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: Iterable[int]) -> None:
        super().__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for index, rate in enumerate(rates):
            self.stages.add_module(
                f"c{index + 1}",
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=int(rate), dilation=int(rate)),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


def _conv_relu(in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding), nn.ReLU(inplace=True))


def _conv_transpose_relu(in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding),
        nn.ReLU(inplace=True),
    )


class PMNet(nn.Module):
    def __init__(
        self,
        n_blocks: Iterable[int],
        atrous_rates: Iterable[int],
        multi_grids: Iterable[int],
        output_stride: int,
        *,
        conv_up4_transpose: bool,
    ) -> None:
        super().__init__()

        if output_stride == 8:
            stage_strides = [1, 2, 1, 1]
            stage_dilations = [1, 1, 2, 4]
        elif output_stride == 16:
            stage_strides = [1, 2, 2, 1]
            stage_dilations = [1, 1, 1, 2]
        else:
            raise ValueError(f"Unsupported output_stride: {output_stride}")

        widths = [64 * (2**power) for power in range(6)]
        blocks = tuple(int(block) for block in n_blocks)
        atrous_rates = tuple(int(rate) for rate in atrous_rates)
        multi_grids = tuple(int(value) for value in multi_grids)

        self.layer1 = _Stem(widths[0])
        self.layer2 = _ResLayer(blocks[0], widths[0], widths[2], stage_strides[0], stage_dilations[0])
        self.reduce = _ConvBnReLU(widths[2], widths[2], 1, 1, 0, 1)
        self.layer3 = _ResLayer(blocks[1], widths[2], widths[3], stage_strides[1], stage_dilations[1])
        self.layer4 = _ResLayer(blocks[2], widths[3], widths[3], stage_strides[2], stage_dilations[2])
        self.layer5 = _ResLayer(blocks[3], widths[3], widths[4], stage_strides[3], stage_dilations[3], multi_grids=multi_grids)
        self.aspp = _ASPP(widths[4], 256, atrous_rates)
        concat_channels = 256 * (len(atrous_rates) + 2)
        self.fc1 = _ConvBnReLU(concat_channels, 512, 1, 1, 0, 1)
        self.conv_up5 = _conv_relu(512, 512, 3, 1)
        if conv_up4_transpose:
            self.conv_up4 = _conv_transpose_relu(512 + 512, 512, 3, 1)
        else:
            self.conv_up4 = _conv_relu(512 + 512, 512, 3, 1)
        self.conv_up3 = _conv_transpose_relu(512 + 512, 256, 3, 1)
        self.conv_up2 = _conv_relu(256 + 256, 256, 3, 1)
        self.conv_up1 = _conv_relu(256 + 256, 256, 3, 1)
        self.conv_up0 = _conv_relu(256 + 64, 128, 3, 1)
        self.conv_up00 = nn.Sequential(
            nn.Conv2d(128 + 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        xup5 = self.conv_up5(x8)
        xup5 = torch.cat([xup5, x5], dim=1)
        xup4 = self.conv_up4(xup5)
        xup4 = torch.cat([xup4, x4], dim=1)
        xup3 = self.conv_up3(xup4)
        xup3 = torch.cat([xup3, x3], dim=1)
        xup2 = self.conv_up2(xup3)
        xup2 = torch.cat([xup2, x2], dim=1)
        xup1 = self.conv_up1(xup2)
        xup1 = torch.cat([xup1, x1], dim=1)
        xup0 = self.conv_up0(xup1)
        xup0 = F.interpolate(xup0, size=x.shape[2:], mode="bilinear", align_corners=False)
        xup0 = torch.cat([xup0, x], dim=1)
        return self.conv_up00(xup0)


def build_model_from_checkpoint(state_dict: dict[str, torch.Tensor]) -> tuple[PMNet, PMNetVariant]:
    variant = detect_pmnet_variant(state_dict)
    model = PMNet(
        MODEL_BLOCKS,
        MODEL_ATROUS_RATES,
        MODEL_MULTI_GRIDS,
        variant.output_stride,
        conv_up4_transpose=variant.conv_up4_transpose,
    )
    model.load_state_dict(state_dict, strict=True)
    return model, variant


def scan_candidates(hdf5_path: Path, min_height: float, max_height: float) -> list[SampleRecord]:
    if min_height > max_height:
        raise ValueError("min_height must be less than or equal to max_height.")

    records: list[SampleRecord] = []
    with h5py.File(hdf5_path, "r") as handle:
        for city in sorted(handle.keys()):
            city_group = handle[city]
            if not isinstance(city_group, h5py.Group):
                continue
            for sample_name in sorted(city_group.keys()):
                if not sample_name.startswith("sample_"):
                    continue
                sample_group = city_group[sample_name]
                if not isinstance(sample_group, h5py.Group):
                    continue
                if "uav_height" not in sample_group:
                    continue
                height_value = np.asarray(sample_group["uav_height"][...], dtype=np.float32).reshape(-1)
                if height_value.size == 0:
                    continue
                height_m = float(height_value[0])
                if min_height <= height_m <= max_height:
                    topology = np.asarray(sample_group["topology_map"], dtype=np.float32)
                    topology = np.nan_to_num(topology, nan=0.0, posinf=0.0, neginf=0.0)
                    partition = infer_partition_metadata(topology, height_m)
                    records.append(
                        SampleRecord(
                            city=city,
                            sample=sample_name,
                            height_m=height_m,
                            expert_id=str(partition["expert_id"]),
                            city_type=str(partition["city_type"]),
                            building_density=float(partition["building_density"]),
                            mean_height=float(partition["mean_height"]),
                        )
                    )
    return records


def choose_records(records: list[SampleRecord], num_samples: int, seed: int) -> list[SampleRecord]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not records:
        return []

    count = min(num_samples, len(records))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=count, replace=False)
    return [records[int(index)] for index in indices.tolist()]


def load_sample_batch(
    hdf5_path: Path,
    records: list[SampleRecord],
    topology_scale: float,
    target_scale: float,
    ground_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    ground_masks: list[torch.Tensor] = []
    los_masks: list[torch.Tensor] = []

    with h5py.File(hdf5_path, "r") as handle:
        for record in records:
            sample_group = handle[record.city][record.sample]
            topology = np.asarray(sample_group["topology_map"], dtype=np.float32)
            target = np.asarray(sample_group["path_loss"], dtype=np.float32)
            los_mask = np.asarray(sample_group["los_mask"], dtype=np.float32)

            topology = np.nan_to_num(topology, nan=0.0, posinf=0.0, neginf=0.0)
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            los_mask = np.nan_to_num(los_mask, nan=0.0, posinf=0.0, neginf=0.0)

            height, width = topology.shape
            topology_norm = np.clip(topology, 0.0, topology_scale) / max(topology_scale, 1e-12)
            target_norm = np.clip(target, 0.0, target_scale) / max(target_scale, 1e-12)
            ground_mask = topology <= ground_threshold
            los_mask_bool = los_mask > 0.5

            topology_tensor = torch.from_numpy(topology_norm.astype(np.float32)).unsqueeze(0)
            tx_tensor = torch.zeros((1, height, width), dtype=torch.float32)
            tx_tensor[0, height // 2, width // 2] = 1.0
            inputs.append(torch.cat([topology_tensor, tx_tensor], dim=0))
            targets.append(torch.from_numpy(target_norm.astype(np.float32)).unsqueeze(0))
            ground_masks.append(torch.from_numpy(ground_mask.astype(np.bool_)).unsqueeze(0))
            los_masks.append(torch.from_numpy(los_mask_bool.astype(np.bool_)).unsqueeze(0))

    return (
        torch.stack(inputs, dim=0),
        torch.stack(targets, dim=0),
        torch.stack(ground_masks, dim=0),
        torch.stack(los_masks, dim=0),
    )


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    diff = pred - target
    if mask is not None:
        diff = diff[mask]
    if diff.numel() == 0:
        return float("nan")
    return float(torch.sqrt(torch.mean(diff * diff)).item())


def sanitize_name(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
    checkpoint_path: Path,
    device: Any,
    records: list[SampleRecord],
    hdf5_path: Path,
    *,
    batch_size: int,
    topology_scale: float,
    target_scale: float,
    ground_threshold: float,
    save_dir: Path | None,
    save_predictions: bool,
    use_quadrants: bool = False,
    auto_offset: bool = False,
) -> dict[str, Any]:
    """Evaluate a PMNet checkpoint on a list of CKM samples.

    Quadrant mode (``use_quadrants=True``):
        Each 513×513 sample is split into four 256×256 sub-images by dropping
        the centre row and column (pixel 256).  The TX indicator is placed at
        the corner of each quadrant nearest the original centre TX:

            TL [0:256,   0:256  ] → TX at bottom-right (255, 255)
            TR [0:256,   257:513] → TX at bottom-left  (255,   0)
            BL [257:513, 0:256  ] → TX at top-right    (  0, 255)
            BR [257:513, 257:513] → TX at top-left     (  0,   0)

        The four predictions are stitched back to 512×512.  Target / masks are
        trimmed identically (no centre cross).

    Auto-offset (``auto_offset=True``):
        After inference, a constant dB offset is estimated as
        ``mean(target_dB − pred_dB)`` over all ground pixels in the test set
        and applied to report an offset-corrected RMSE.  The offset is oracle
        (computed on the same samples used for evaluation).
    """
    state_dict = load_checkpoint_state(checkpoint_path)
    model, variant = build_model_from_checkpoint(state_dict)
    model.to(device)
    model.eval()

    per_sample_rows: list[dict[str, Any]] = []
    overall_stats = _new_metric_group()
    expert_stats: dict[str, dict[str, dict[str, float | int]]] = {}

    # Accumulated ground-pixel arrays for offset estimation.
    offset_preds_list: list[torch.Tensor] = []
    offset_targets_list: list[torch.Tensor] = []

    mode_str = "quadrant 4×256×256 (TX at corner)" if use_quadrants else f"full 513×513 (batch={batch_size})"
    print()
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"  variant  : output_stride={variant.output_stride}, conv_up4_transpose={variant.conv_up4_transpose}")
    print(f"  mode     : {mode_str}")
    if auto_offset:
        print("  offset   : auto (oracle, estimated on test samples)")

    # Quadrant slice info: (row_slice, col_slice, tx_row, tx_col)
    _QUAD_INFO = [
        (slice(0, 256),   slice(0, 256),   255, 255),  # TL
        (slice(0, 256),   slice(257, 513),  255, 0  ),  # TR
        (slice(257, 513), slice(0, 256),    0,   255),  # BL
        (slice(257, 513), slice(257, 513),  0,   0  ),  # BR
    ]

    def _trim_center_cross(t: torch.Tensor) -> torch.Tensor:
        """513×513 → 512×512: drop row 256 and col 256."""
        t = torch.cat([t[:256, :], t[257:, :]], dim=0)
        t = torch.cat([t[:, :256], t[:, 257:]], dim=1)
        return t

    def _infer_one(record: SampleRecord) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (pred, target, ground_mask, los_mask, topo_norm) for one record."""
        inputs_cpu, targets_cpu, gmasks_cpu, lmasks_cpu = load_sample_batch(
            hdf5_path, [record],
            topology_scale=topology_scale,
            target_scale=target_scale,
            ground_threshold=ground_threshold,
        )
        topo_norm = inputs_cpu[0, 0]  # (513, 513)

        if use_quadrants:
            quad_inputs: list[torch.Tensor] = []
            for rs, cs, txr, txc in _QUAD_INFO:
                tq = topo_norm[rs, cs]  # (256, 256)
                txq = torch.zeros(256, 256, dtype=torch.float32)
                txq[txr, txc] = 1.0
                quad_inputs.append(torch.stack([tq, txq], dim=0))
            qi = torch.stack(quad_inputs, dim=0).to(device)  # (4, 2, 256, 256)
            qp = model(qi).clamp(0.0, 1.0).cpu()[:, 0, :, :]  # (4, 256, 256)

            pred = torch.zeros(512, 512, dtype=torch.float32)
            pred[  :256,   :256] = qp[0]
            pred[  :256, 256:  ] = qp[1]
            pred[256:  ,   :256] = qp[2]
            pred[256:  , 256:  ] = qp[3]

            target     = _trim_center_cross(targets_cpu[0, 0])
            ground_mask = _trim_center_cross(gmasks_cpu[0, 0])
            los_mask    = _trim_center_cross(lmasks_cpu[0, 0])
        else:
            inp = inputs_cpu.to(device)
            pred       = model(inp).clamp(0.0, 1.0).cpu()[0, 0]
            target     = targets_cpu[0, 0]
            ground_mask = gmasks_cpu[0, 0]
            los_mask    = lmasks_cpu[0, 0]

        return pred, target, ground_mask, los_mask, topo_norm

    with torch.no_grad():
        for record in records:
            pred, target, ground_mask, los_mask, topo_norm = _infer_one(record)
            nlos_mask = ground_mask & (~los_mask)

            ground_rmse_norm = masked_rmse(pred, target, ground_mask)
            los_rmse_norm    = masked_rmse(pred, target, ground_mask & los_mask)
            nlos_rmse_norm   = masked_rmse(pred, target, nlos_mask)

            total_pixels = int(target.numel())
            valid_pixels = int(ground_mask.sum().item())
            los_pixels   = int((ground_mask & los_mask).sum().item())
            nlos_pixels  = int(nlos_mask.sum().item())

            ground_fraction = float(valid_pixels / total_pixels) if total_pixels > 0 else float("nan")
            los_fraction    = float(los_pixels  / valid_pixels)  if valid_pixels > 0 else float("nan")
            nlos_fraction   = float(nlos_pixels / valid_pixels)  if valid_pixels > 0 else float("nan")

            row = SampleMetrics(
                city=record.city, sample=record.sample,
                expert_id=record.expert_id, city_type=record.city_type, height_m=record.height_m,
                ground_rmse_db=ground_rmse_norm * target_scale,
                los_rmse_db=los_rmse_norm * target_scale,
                nlos_rmse_db=nlos_rmse_norm * target_scale,
                ground_rmse_norm=ground_rmse_norm,
                los_rmse_norm=los_rmse_norm,
                nlos_rmse_norm=nlos_rmse_norm,
                ground_fraction=ground_fraction,
                los_fraction=los_fraction,
                nlos_fraction=nlos_fraction,
            )
            per_sample_rows.append({
                "city": row.city, "sample": row.sample,
                "expert_id": row.expert_id, "city_type": row.city_type, "height_m": row.height_m,
                "ground_rmse_db": row.ground_rmse_db,
                "los_rmse_db": row.los_rmse_db,
                "nlos_rmse_db": row.nlos_rmse_db,
                "ground_rmse_norm": row.ground_rmse_norm,
                "los_rmse_norm": row.los_rmse_norm,
                "nlos_rmse_norm": row.nlos_rmse_norm,
                "ground_fraction": row.ground_fraction,
                "los_fraction": row.los_fraction,
                "nlos_fraction": row.nlos_fraction,
            })

            expert_bucket = expert_stats.setdefault(record.expert_id, _new_metric_group())
            full_diff = pred - target
            _update_metric_bucket(overall_stats["ground"],            full_diff[ground_mask])
            _update_metric_bucket(overall_stats["los"],               full_diff[ground_mask & los_mask])
            _update_metric_bucket(overall_stats["nlos"],              full_diff[nlos_mask])
            _update_metric_bucket(expert_bucket["ground"],            full_diff[ground_mask])
            _update_metric_bucket(expert_bucket["los"],               full_diff[ground_mask & los_mask])
            _update_metric_bucket(expert_bucket["nlos"],              full_diff[nlos_mask])

            print(
                f"  {record.city}/{record.sample} | expert={record.expert_id} | h={record.height_m:.2f} m | "
                f"RMSE(ground)={row.ground_rmse_db:.3f} dB | "
                f"RMSE(LoS)={row.los_rmse_db:.3f} dB | RMSE(NLoS)={row.nlos_rmse_db:.3f} dB"
            )

            if auto_offset:
                offset_preds_list.append(pred[ground_mask].clone())
                offset_targets_list.append(target[ground_mask].clone())

            if save_dir is not None and save_predictions:
                checkpoint_dir = save_dir / checkpoint_path.stem
                sample_dir = (
                    checkpoint_dir
                    / sanitize_name(record.expert_id)
                    / sanitize_name(f"{record.city}__{record.sample}")
                )
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.save(sample_dir / "prediction_db.npy",   (pred.numpy()       * target_scale).astype(np.float32))
                np.save(sample_dir / "target_db.npy",       (target.numpy()     * target_scale).astype(np.float32))
                np.save(sample_dir / "topology_norm.npy",   topo_norm.numpy().astype(np.float32))
                np.save(sample_dir / "ground_mask.npy",     ground_mask.numpy().astype(np.bool_))
                np.save(sample_dir / "los_mask.npy",        los_mask.numpy().astype(np.bool_))

    # --- Calibration (offset / linear) --------------------------------------
    offset_db: float = 0.0
    scale_factor: float = 1.0
    offset_corrected_ground_rmse_db: float = float("nan")
    pred_std_db: float = float("nan")

    if auto_offset and offset_preds_list:
        all_preds   = torch.cat(offset_preds_list).float()   # (N,) normalised
        all_targets = torch.cat(offset_targets_list).float()  # (N,) normalised

        pred_mean  = float(all_preds.mean())
        pred_std   = float(all_preds.std())
        tgt_mean   = float(all_targets.mean())
        tgt_std    = float(all_targets.std())

        pred_std_db = pred_std * target_scale
        tgt_std_db  = tgt_std  * target_scale

        # 1) Pure offset (mean alignment only)
        offset_norm = tgt_mean - pred_mean
        offset_db   = offset_norm * target_scale
        corrected_offset = (all_preds + offset_norm) - all_targets
        rmse_offset = float(torch.sqrt(torch.mean(corrected_offset ** 2)).item()) * target_scale

        # 2) Linear calibration: scale pred std to match target std, then align means
        #    scale = tgt_std / pred_std  (safe guard for near-zero pred_std)
        if pred_std > 1e-6:
            scale_factor = tgt_std / pred_std
            shift_norm   = tgt_mean - scale_factor * pred_mean
            corrected_lin = all_preds * scale_factor + shift_norm - all_targets
            rmse_linear  = float(torch.sqrt(torch.mean(corrected_lin ** 2)).item()) * target_scale
            offset_corrected_ground_rmse_db = rmse_linear  # report best (linear)
        else:
            scale_factor = 1.0
            rmse_linear  = float("nan")
            offset_corrected_ground_rmse_db = rmse_offset

        print(
            f"  calibration: pred_mean={pred_mean*target_scale:.1f} dB -> tgt_mean={tgt_mean*target_scale:.1f} dB | "
            f"pred_std={pred_std_db:.2f} dB | tgt_std={tgt_std_db:.2f} dB"
        )
        print(
            f"  offset-only RMSE(ground)  : {rmse_offset:.3f} dB  "
            f"(offset={offset_db:+.2f} dB)"
        )
        if not np.isnan(rmse_linear):
            print(
                f"  linear-calib RMSE(ground) : {rmse_linear:.3f} dB  "
                f"(scale={scale_factor:.4f}, offset={tgt_mean*target_scale - scale_factor*pred_mean*target_scale:+.2f} dB)"
            )

    # --- Summaries -----------------------------------------------------------
    overall_summary = {
        "ground": _finalize_metric_bucket(overall_stats["ground"]),
        "los":    _finalize_metric_bucket(overall_stats["los"]),
        "nlos":   _finalize_metric_bucket(overall_stats["nlos"]),
    }

    expert_summary: dict[str, dict[str, Any]] = {}
    for expert_id, stats in sorted(expert_stats.items()):
        expert_summary[expert_id] = {
            "expert_id": expert_id,
            "sample_count": int(stats["ground"]["sample_count"]),
            "metrics": {
                "ground": _finalize_metric_bucket(stats["ground"]),
                "los":    _finalize_metric_bucket(stats["los"]),
                "nlos":   _finalize_metric_bucket(stats["nlos"]),
            },
        }

    summary = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "output_stride": variant.output_stride,
        "conv_up4_transpose": variant.conv_up4_transpose,
        "use_quadrants": use_quadrants,
        "auto_offset_db": offset_db,
        "scale_factor": scale_factor,
        "offset_corrected_ground_rmse_db": offset_corrected_ground_rmse_db,
        "pred_std_db": pred_std_db,
        "sample_count": len(per_sample_rows),
        "metrics": {
            "ground_rmse_db":   overall_summary["ground"]["rmse"] * target_scale,
            "los_rmse_db":      overall_summary["los"  ]["rmse"] * target_scale,
            "nlos_rmse_db":     overall_summary["nlos" ]["rmse"] * target_scale,
            "ground_rmse_norm": overall_summary["ground"]["rmse"],
            "los_rmse_norm":    overall_summary["los"  ]["rmse"],
            "nlos_rmse_norm":   overall_summary["nlos" ]["rmse"],
            "counts": {
                "ground": overall_summary["ground"]["count"],
                "los":    overall_summary["los"  ]["count"],
                "nlos":   overall_summary["nlos" ]["count"],
            },
        },
        "expert_breakdown": expert_summary,
        "per_sample": per_sample_rows,
    }

    print(
        f"  summary: samples={len(per_sample_rows)} | "
        f"RMSE(ground)={summary['metrics']['ground_rmse_db']:.3f} dB "
        f"({summary['metrics']['ground_rmse_norm']:.5f} norm) | "
        f"RMSE(LoS)={summary['metrics']['los_rmse_db']:.3f} dB | "
        f"RMSE(NLoS)={summary['metrics']['nlos_rmse_db']:.3f} dB"
    )
    print("  expert breakdown:")
    for expert_id, erow in expert_summary.items():
        m = erow["metrics"]
        print(
            f"    {expert_id} | n={erow['sample_count']} | "
            f"RMSE(g)={m['ground']['rmse'] * target_scale:.3f} | "
            f"RMSE(L)={m['los'  ]['rmse'] * target_scale:.3f} | "
            f"RMSE(N)={m['nlos' ]['rmse'] * target_scale:.3f} dB"
        )

    if save_dir is not None:
        checkpoint_dir = save_dir / checkpoint_path.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        write_summary_csv(checkpoint_dir / "summary.csv", per_sample_rows)
        (checkpoint_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the two PMNet checkpoints in weird_tries on random CKM HDF5 samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the CKM HDF5 file.")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        default=DEFAULT_CHECKPOINTS,
        help="Checkpoint paths to evaluate. Defaults to the two weird_tries checkpoints.",
    )
    parser.add_argument("--num-samples", type=int, default=12, help="Number of random samples to test.")
    parser.add_argument("--min-height", type=float, default=0.0, help="Minimum UAV height to include.")
    parser.add_argument("--max-height", type=float, default=510.0, help="Maximum UAV height to include.")
    parser.add_argument(
        "--expert",
        "--topology-class",
        dest="experts",
        nargs="*",
        default=None,
        metavar="EXPERT",
        help="Optional expert topology classes to keep. Leave empty to test all experts.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for sample selection.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, directml, dml, or amd.",
    )
    parser.add_argument(
        "--topology-scale",
        type=float,
        default=255.0,
        help="Normalization scale for topology_map (matches the training input_metadata.scale default).",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=180.0,
        help="Normalization scale for path_loss (matches the training target_metadata.path_loss.scale default).",
    )
    parser.add_argument(
        "--ground-threshold",
        type=float,
        default=0.0,
        help="Pixels at or below this topology value are treated as ground.",
    )
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory for CSV/JSON outputs.")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample prediction, target, topology, and mask arrays under --save-dir.",
    )
    parser.add_argument(
        "--use-quadrants",
        action="store_true",
        help=(
            "Split each 513×513 map into 4×256×256 quadrants with TX at the nearest corner "
            "(drops centre row/col). Matches the 256×256 resolution PMNet was trained on."
        ),
    )
    parser.add_argument(
        "--auto-offset",
        action="store_true",
        help=(
            "Estimate a constant dB offset (mean bias) AND a linear scale correction (std matching) "
            "from the test samples and report both corrected RMSEs. Oracle calibration."
        ),
    )
    parser.add_argument(
        "--low-heights-only",
        action="store_true",
        help="Restrict to UAV heights ≤ 50 m (closest to terrestrial BS scenarios PMNet was trained on).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    expert_filters = normalize_expert_filters(args.experts)

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    checkpoints = [path.resolve() for path in args.checkpoints]
    for checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.low_heights_only:
        args.max_height = min(args.max_height, 50.0)
        print(f"--low-heights-only: max height clamped to {args.max_height:.1f} m")

    candidates = scan_candidates(dataset_path, args.min_height, args.max_height)
    if expert_filters is not None:
        candidates = [record for record in candidates if record.expert_id in expert_filters]
    if not candidates:
        filter_text = ""
        if expert_filters is not None:
            filter_text = f" and expert in {sorted(expert_filters)}"
        raise ValueError(
            f"No samples found in {dataset_path} with height in [{args.min_height}, {args.max_height}]"
            f"{filter_text}."
        )

    selected = choose_records(candidates, args.num_samples, args.seed)

    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print(
        f"Height filter: [{args.min_height:.2f}, {args.max_height:.2f}] m | "
        f"candidates={len(candidates)} | selected={len(selected)}"
    )
    if len(selected) < args.num_samples:
        print(f"Requested {args.num_samples} samples but only {len(selected)} matched the height range.")

    if expert_filters is not None:
        print(f"Expert filter: {', '.join(sorted(expert_filters))}")

    expert_pool_counts: dict[str, int] = {}
    for record in candidates:
        expert_pool_counts[record.expert_id] = expert_pool_counts.get(record.expert_id, 0) + 1
    print("Expert pool:")
    for expert_id in sorted(expert_pool_counts):
        print(f"  {expert_id}: {expert_pool_counts[expert_id]} candidates")

    print("Selected samples:")
    for record in selected:
        print(
            f"  {record.city}/{record.sample} | expert={record.expert_id} | city_type={record.city_type} | "
            f"density={record.building_density:.3f} | h={record.height_m:.2f} m"
        )

    save_dir = args.save_dir.resolve() if args.save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "selected_samples.json").write_text(
            json.dumps(
                [
                    {
                        "city": record.city,
                        "sample": record.sample,
                        "height_m": record.height_m,
                        "expert_id": record.expert_id,
                        "city_type": record.city_type,
                        "building_density": record.building_density,
                        "mean_height": record.mean_height,
                    }
                    for record in selected
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    summaries: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        summary = evaluate_checkpoint(
            checkpoint_path,
            device,
            selected,
            dataset_path,
            batch_size=args.batch_size,
            topology_scale=args.topology_scale,
            target_scale=args.target_scale,
            ground_threshold=args.ground_threshold,
            save_dir=save_dir,
            save_predictions=args.save_predictions,
            use_quadrants=args.use_quadrants,
            auto_offset=args.auto_offset,
        )
        summaries.append(summary)

    print()
    print("Comparison summary:")
    for summary in summaries:
        metrics = summary["metrics"]
        calib_str = ""
        corrected = summary.get("offset_corrected_ground_rmse_db", float("nan"))
        if not (isinstance(corrected, float) and np.isnan(corrected)):
            calib_str = f" | linear-calib RMSE(g)={corrected:.3f} dB (scale={summary.get('scale_factor',1.0):.4f})"
        print(
            f"  {summary['checkpoint']} | stride={summary['output_stride']} | quads={summary['use_quadrants']} | "
            f"RMSE(g)={metrics['ground_rmse_db']:.3f} dB | "
            f"RMSE(L)={metrics['los_rmse_db']:.3f} dB | RMSE(N)={metrics['nlos_rmse_db']:.3f} dB"
            f"{calib_str}"
        )

    if save_dir is not None:
        (save_dir / "comparison.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        write_summary_csv(
            save_dir / "comparison.csv",
            [
                {
                    "checkpoint": summary["checkpoint"],
                    "checkpoint_path": summary["checkpoint_path"],
                    "output_stride": summary["output_stride"],
                    "conv_up4_transpose": summary["conv_up4_transpose"],
                    "sample_count": summary["sample_count"],
                    "ground_rmse_db": summary["metrics"]["ground_rmse_db"],
                    "los_rmse_db": summary["metrics"]["los_rmse_db"],
                    "nlos_rmse_db": summary["metrics"]["nlos_rmse_db"],
                    "ground_rmse_norm": summary["metrics"]["ground_rmse_norm"],
                    "los_rmse_norm": summary["metrics"]["los_rmse_norm"],
                    "nlos_rmse_norm": summary["metrics"]["nlos_rmse_norm"],
                }
                for summary in summaries
            ],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())