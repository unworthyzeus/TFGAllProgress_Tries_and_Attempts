"""Try 76 — config loader.

YAML schema (minimal, no inheritance from Try 75):

    seed: 42
    data:
      hdf5_path: ../Datasets/CKM_Dataset_270326.h5
      split_mode: city_holdout
      val_ratio: 0.15
      test_ratio: 0.15
      split_seed: 42
      partition_filter:
        topology_class: open_sparse_lowrise
      region_mode: los_only        # or nlos_only
    model:
      clamp_lo: 30.0
      clamp_hi: 178.0
      base_width: 48
      K: 3
      outlier_sigma_floor: 15.0
    training:
      optimizer: adamw
      lr: 3.0e-4
      weight_decay: 0.01
      warmup_epochs: 1
      epochs: 120
      batch_size: 1
      grad_accum_steps: 8
      patience: 15
    runtime:
      device: cuda
      output_dir: outputs/try76_expert_open_sparse_lowrise_los
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataSection:
    hdf5_path: Path
    topology_class: str
    region_mode: str
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    image_size: int = 513
    path_loss_no_data_mask_column: Optional[str] = None
    derive_no_data_from_non_ground: bool = False


@dataclass
class ModelSection:
    clamp_lo: float = 30.0
    clamp_hi: float = 178.0
    base_width: int = 48
    K: int = 5
    outlier_sigma_floor: float = 15.0


@dataclass
class TrainingSection:
    optimizer: str = "adamw"
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 1
    epochs: int = 120
    batch_size: int = 1
    grad_accum_steps: int = 8
    patience: int = 15
    num_workers: int = 4


@dataclass
class RuntimeSection:
    device: str = "cuda"
    output_dir: Path = field(default_factory=lambda: Path("outputs/try76"))
    resume_checkpoint: Optional[Path] = None


@dataclass
class Try76Cfg:
    seed: int
    data: DataSection
    model: ModelSection
    training: TrainingSection
    runtime: RuntimeSection

    @classmethod
    def load(cls, path: Path) -> "Try76Cfg":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw, root=path.parent)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], root: Path) -> "Try76Cfg":
        d = raw.get("data", {}) or {}
        partition = d.get("partition_filter", {}) or {}
        data = DataSection(
            hdf5_path=(root / d["hdf5_path"]).resolve() if not Path(d["hdf5_path"]).is_absolute() else Path(d["hdf5_path"]),
            topology_class=str(partition.get("topology_class", "")),
            region_mode=str(d.get("los_region_mask_mode", d.get("region_mode", "los_only"))),
            val_ratio=float(d.get("val_ratio", 0.15)),
            test_ratio=float(d.get("test_ratio", 0.15)),
            split_seed=int(d.get("split_seed", 42)),
            image_size=int(d.get("image_size", 513)),
            path_loss_no_data_mask_column=(
                str(d.get("path_loss_no_data_mask_column")).strip()
                if d.get("path_loss_no_data_mask_column") is not None
                else None
            ),
            derive_no_data_from_non_ground=bool(d.get("derive_no_data_from_non_ground", False)),
        )
        if not data.topology_class:
            raise ValueError("data.partition_filter.topology_class is required")
        if data.region_mode not in {"los_only", "nlos_only"}:
            raise ValueError(f"Unsupported region_mode {data.region_mode!r}; expected los_only or nlos_only")

        m = raw.get("model", {}) or {}
        model = ModelSection(
            clamp_lo=float(m.get("clamp_lo", 30.0)),
            clamp_hi=float(m.get("clamp_hi", 178.0)),
            base_width=int(m.get("base_width", 48)),
            K=int(m.get("K", 5)),
            outlier_sigma_floor=float(m.get("outlier_sigma_floor", 15.0)),
        )

        t = raw.get("training", {}) or {}
        training = TrainingSection(
            optimizer=str(t.get("optimizer", "adamw")),
            lr=float(t.get("lr", 3.0e-4)),
            weight_decay=float(t.get("weight_decay", 0.01)),
            warmup_epochs=int(t.get("warmup_epochs", 1)),
            epochs=int(t.get("epochs", 120)),
            batch_size=int(t.get("batch_size", 1)),
            grad_accum_steps=int(t.get("grad_accum_steps", 8)),
            patience=int(t.get("patience", 15)),
            num_workers=int(t.get("num_workers", 4)),
        )

        r = raw.get("runtime", {}) or {}
        output_dir = r.get("output_dir", "outputs/try76")
        resume = r.get("resume_checkpoint", "") or None
        runtime = RuntimeSection(
            device=str(r.get("device", "cuda")),
            output_dir=(root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir),
            resume_checkpoint=((root / resume).resolve() if not Path(resume).is_absolute() else Path(resume)) if resume else None,
        )

        return cls(
            seed=int(raw.get("seed", 42)),
            data=data,
            model=model,
            training=training,
            runtime=runtime,
        )


def load_clamp_table(path: Path) -> Dict[str, Dict[str, float]]:
    """Load expert clamp table produced by ``scripts/build_expert_clamps.py``."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
