from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config_utils import anchor_data_paths_to_config_file, ensure_output_dir, load_config, load_torch_checkpoint, resolve_device
from data_utils import (
    TRY54_TOPOLOGY_CLASSES,
    _compute_try54_partition_metadata,
    _list_hdf5_samples,
    _normalize_array,
    _resolve_hdf5_scalar_value_static,
    _resolve_try54_topology_thresholds,
    _split_hdf5_samples,
)
from model_topology_classifier import TinyAntennaAwareTopologyClassifier, TinyTopologyClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Try54TopologyClassificationDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        sample_refs: Sequence[Tuple[str, str]],
        *,
        image_size: int,
        input_column: str,
        input_metadata: Dict[str, Any],
        scalar_specs: Sequence[Dict[str, Any]],
        topology_thresholds: Dict[str, float],
        antenna_thresholds: Dict[str, float],
        non_ground_threshold: float,
        include_antenna_scalar: bool = False,
        antenna_norm: float = 120.0,
    ) -> None:
        self.hdf5_path = str(hdf5_path)
        self.sample_refs = list(sample_refs)
        self.image_size = int(image_size)
        self.input_column = str(input_column)
        self.input_metadata = dict(input_metadata)
        self.scalar_specs = list(scalar_specs)
        self.topology_thresholds = dict(topology_thresholds)
        self.antenna_thresholds = dict(antenna_thresholds)
        self.non_ground_threshold = float(non_ground_threshold)
        self.include_antenna_scalar = bool(include_antenna_scalar)
        self.antenna_norm = float(antenna_norm)
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.sample_refs)

    def _get_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_path, "r")
        return self._handle

    def __getitem__(self, idx: int):
        city, sample = self.sample_refs[idx]
        grp = self._get_handle()[city][sample]
        raw_topology = np.asarray(grp[self.input_column][...], dtype=np.float32)
        norm_topology = _normalize_array(raw_topology, self.input_metadata)
        topo = torch.from_numpy(norm_topology).unsqueeze(0)
        topo = F.interpolate(topo.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
        antenna_height_m = float(_resolve_hdf5_scalar_value_static(grp, "antenna_height_m", self.scalar_specs))
        meta = _compute_try54_partition_metadata(
            raw_topology,
            antenna_height_m,
            self.topology_thresholds,
            self.antenna_thresholds,
            non_ground_threshold=self.non_ground_threshold,
        )
        label = TRY54_TOPOLOGY_CLASSES.index(str(meta["topology_class"]))
        if self.include_antenna_scalar:
            scalar = torch.tensor([antenna_height_m / max(self.antenna_norm, 1e-12)], dtype=torch.float32)
            return topo, torch.tensor(label, dtype=torch.long), scalar
        return topo, torch.tensor(label, dtype=torch.long)


def _build_loaders(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    data_cfg = dict(cfg["data"])
    refs = _list_hdf5_samples(str(data_cfg["hdf5_path"]))
    train_refs, val_refs, test_refs = _split_hdf5_samples(
        refs,
        float(data_cfg.get("val_ratio", 0.15)),
        int(data_cfg.get("split_seed", cfg.get("seed", 42))),
        float(data_cfg.get("test_ratio", 0.15)),
        str(data_cfg.get("split_mode", "city_holdout")),
    )
    formula_cfg = dict(data_cfg.get("path_loss_formula_input", {}))
    topology_thresholds = _resolve_try54_topology_thresholds(data_cfg, formula_cfg)
    calibration = None
    calibration_path = formula_cfg.get("regime_calibration_json")
    if calibration_path:
        from data_utils import _load_formula_regime_calibration

        try:
            calibration = _load_formula_regime_calibration(str(calibration_path))
        except FileNotFoundError:
            calibration = None
    antenna_thresholds = dict((calibration or {}).get("antenna_height_thresholds", {}))
    common = dict(
        hdf5_path=str(data_cfg["hdf5_path"]),
        image_size=int(data_cfg.get("image_size", 513)),
        input_column=str(data_cfg.get("input_column", "topology_map")),
        input_metadata=dict(data_cfg.get("input_metadata", {})),
        scalar_specs=list(data_cfg.get("hdf5_scalar_specs", [])),
        topology_thresholds=topology_thresholds,
        antenna_thresholds=antenna_thresholds,
        non_ground_threshold=float(data_cfg.get("non_ground_threshold", 0.0)),
        include_antenna_scalar=bool(cfg.get("model", {}).get("use_antenna_scalar", False)),
        antenna_norm=float(data_cfg.get("scalar_feature_norms", {}).get("antenna_height_m", 120.0)),
    )
    train_ds = Try54TopologyClassificationDataset(sample_refs=train_refs, **common)
    val_ds = Try54TopologyClassificationDataset(sample_refs=val_refs, **common)
    test_ds = Try54TopologyClassificationDataset(sample_refs=test_refs, **common) if test_refs else None
    batch_size = int(cfg["training"].get("batch_size", 8))
    workers = int(data_cfg.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, workers // 2), pin_memory=True)
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, workers // 2), pin_memory=True)
    return train_loader, val_loader, test_loader


def _forward_model(model: nn.Module, batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        x, y, scalar = batch
        return model(x.to(device), scalar.to(device)), y.to(device)
    x, y = batch
    return model(x.to(device)), y.to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            logits, target = _forward_model(model, batch, device)
            loss = criterion(logits, target)
            total_loss += float(loss.item()) * int(target.shape[0])
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == target).sum().item())
            total += int(target.shape[0])
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "count": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Try54 topology classifier for expert routing.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    anchor_data_paths_to_config_file(cfg, args.config)
    set_seed(int(cfg.get("seed", 42)))
    device_raw = resolve_device(str(cfg["runtime"].get("device", "cuda")))
    device = torch.device("cuda" if str(device_raw) == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = ensure_output_dir(str(cfg["runtime"]["output_dir"]))

    train_loader, val_loader, test_loader = _build_loaders(cfg)

    model_cfg = dict(cfg.get("model", {}))
    use_antenna_scalar = bool(model_cfg.get("use_antenna_scalar", False))
    if use_antenna_scalar:
        model = TinyAntennaAwareTopologyClassifier(
            in_channels=1,
            num_classes=len(TRY54_TOPOLOGY_CLASSES),
            base_channels=int(model_cfg.get("base_channels", 24)),
            norm_type=str(model_cfg.get("norm_type", "group")),
            dropout=float(model_cfg.get("dropout", 0.05)),
            scalar_dim=1,
        )
    else:
        model = TinyTopologyClassifier(
            in_channels=1,
            num_classes=len(TRY54_TOPOLOGY_CLASSES),
            base_channels=int(model_cfg.get("base_channels", 24)),
            norm_type=str(model_cfg.get("norm_type", "group")),
            dropout=float(model_cfg.get("dropout", 0.05)),
        )
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 6e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg["training"].get("lr_scheduler_factor", 0.5)),
        patience=int(cfg["training"].get("lr_scheduler_patience", 4)),
        min_lr=float(cfg["training"].get("lr_scheduler_min_lr", 1e-5)),
    )

    resume_checkpoint = str(cfg["runtime"].get("resume_checkpoint", "") or "")
    start_epoch = 1
    best_acc = -1.0
    if resume_checkpoint:
        ckpt_path = Path(resume_checkpoint)
        if ckpt_path.exists():
            state = load_torch_checkpoint(ckpt_path, device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state.get("epoch", 0)) + 1
            best_acc = float(state.get("best_accuracy", -1.0))

    criterion = nn.CrossEntropyLoss()
    epochs = int(cfg["training"].get("epochs", 30))
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for batch in progress:
            logits, target = _forward_model(model, batch, device)
            loss = criterion(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg["training"].get("clip_grad_norm", 0.0)) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("clip_grad_norm", 0.0)))
            optimizer.step()
            running_loss += float(loss.item()) * int(target.shape[0])
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == target).sum().item())
            total += int(target.shape[0])
            progress.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["accuracy"])
        payload = {
            "_train": {
                "loss": running_loss / max(total, 1),
                "accuracy": correct / max(total, 1),
                "count": total,
            },
            "_val": val_metrics,
            "_checkpoint": {
                "epoch": epoch,
                "best_accuracy": max(best_acc, val_metrics["accuracy"]),
            },
        }
        latest_path = out_dir / "validate_metrics_classifier_latest.json"
        epoch_path = out_dir / f"validate_metrics_classifier_epoch_{epoch}.json"
        latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        epoch_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        state = {
            "epoch": epoch,
            "best_accuracy": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config_path": args.config,
        }
        torch.save(state, out_dir / f"epoch_{epoch}_classifier.pt")
        prev = out_dir / f"epoch_{epoch - 1}_classifier.pt"
        if prev.exists():
            prev.unlink()
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            state["best_accuracy"] = best_acc
            torch.save(state, out_dir / "best_classifier.pt")
        print(json.dumps({"epoch": epoch, "train_loss": running_loss / max(total, 1), "val_accuracy": val_metrics["accuracy"]}))

    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, device)
        (out_dir / "test_metrics_classifier.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
