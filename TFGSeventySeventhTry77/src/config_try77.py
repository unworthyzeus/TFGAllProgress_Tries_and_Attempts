"""Try 77 — YAML config loader.

Schema (no inheritance from Try 75/76):

    seed: 42
    data:
      hdf5_path: ../Datasets/CKM_Dataset_270326.h5
      val_ratio: 0.15
      test_ratio: 0.15
      split_seed: 42
      partition_filter:
        topology_class: open_sparse_lowrise
      metric: delay_spread        # or angular_spread
    model:
      clamp_lo: 0.0
      clamp_hi: 400.0             # 400 for delay, 90 for angular
      base_width: 48
      K: 5
      sigma_min: 1.0
      sigma_max: 120.0
      spike_mu_max: 10.0
      spike_sigma_min: 0.3
      spike_sigma_max: 5.0
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
      output_dir: outputs/try77_expert_open_sparse_lowrise_delay_spread
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataSection:
    hdf5_path: Path
    topology_class: str
    metric: str                # "delay_spread" or "angular_spread"
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    image_size: int = 513


@dataclass
class ModelSection:
    clamp_lo: float = 0.0
    clamp_hi: float = 400.0
    base_width: int = 48
    K: int = 5
    sigma_min: float = 1.0
    sigma_max: float = 120.0
    spike_mu_max: float = 10.0
    spike_sigma_min: float = 0.3
    spike_sigma_max: float = 5.0


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
    output_dir: Path = field(default_factory=lambda: Path("outputs/try77"))
    resume_checkpoint: Optional[Path] = None


@dataclass
class Try77Cfg:
    seed: int
    data: DataSection
    model: ModelSection
    training: TrainingSection
    runtime: RuntimeSection

    @classmethod
    def load(cls, path: Path) -> "Try77Cfg":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw, root=path.parent)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], root: Path) -> "Try77Cfg":
        d = raw.get("data", {}) or {}
        partition = d.get("partition_filter", {}) or {}
        data = DataSection(
            hdf5_path=(root / d["hdf5_path"]).resolve() if not Path(d["hdf5_path"]).is_absolute() else Path(d["hdf5_path"]),
            topology_class=str(partition.get("topology_class", "")),
            metric=str(d.get("metric", "delay_spread")),
            val_ratio=float(d.get("val_ratio", 0.15)),
            test_ratio=float(d.get("test_ratio", 0.15)),
            split_seed=int(d.get("split_seed", 42)),
            image_size=int(d.get("image_size", 513)),
        )
        if not data.topology_class:
            raise ValueError("data.partition_filter.topology_class is required")
        if data.metric not in {"delay_spread", "angular_spread"}:
            raise ValueError(f"Unsupported metric {data.metric!r}; expected delay_spread or angular_spread")

        m = raw.get("model", {}) or {}
        model = ModelSection(
            clamp_lo=float(m.get("clamp_lo", 0.0)),
            clamp_hi=float(m.get("clamp_hi", 400.0)),
            base_width=int(m.get("base_width", 48)),
            K=int(m.get("K", 5)),
            sigma_min=float(m.get("sigma_min", 1.0)),
            sigma_max=float(m.get("sigma_max", 120.0)),
            spike_mu_max=float(m.get("spike_mu_max", 10.0)),
            spike_sigma_min=float(m.get("spike_sigma_min", 0.3)),
            spike_sigma_max=float(m.get("spike_sigma_max", 5.0)),
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
        output_dir = r.get("output_dir", "outputs/try77")
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
