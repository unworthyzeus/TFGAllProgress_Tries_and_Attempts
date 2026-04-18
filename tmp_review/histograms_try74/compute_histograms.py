"""Compute per-image dB histograms for path_loss (+ optional delay/angular spread)
for a chosen try's experts, and dump all bin counts to a single wide CSV.

Discovery order (unless --try is set):
    1) Try 75  (TFGSeventyFifthTry75)
    2) Try 74  (TFGSeventyFourthTry74)
    3) Try 68  (TFGSixtyEighthTry68) -- all 6 experts

For each try, only experts whose best_model.pt exists under
    cluster_outputs/<try_dir>/<expert_id>/best_model.pt
are included. Partition / height filters in the YAML are stripped in-memory
so the histogram run spans ALL samples in the split.

Supports CUDA and DirectML (Windows AMD/Intel). Pass --device {auto, cuda,
directml, cpu}. Default 'auto'.

CSV schema (one row per sample x kind):
    city, sample, altitude_m, city_type, city_type_3, city_type_6,
    try_dir, expert_mode, exclude_non_ground_targets,
        expert_id, expert_region, metric, kind, total_pixels, b<LO>..b<HI-1>

metric in {path_loss, delay_spread, angular_spread}

kind includes:
    - path_loss: target_los, target_nlos, pred_los, pred_nlos
    - delay_spread: target_delay_spread, pred_delay_spread
    - angular_spread: target_angular_spread, pred_angular_spread

Delay/angular prediction rows are produced from topology experts:
    - delay_spread   from Try58 (TFGFiftyEighthTry58)
    - angular_spread from Try59 (TFGFiftyNinthTry59)
using best checkpoint files (best_model.pt or best_cgan.pt).

expert_region in {los_only, nlos_only, full}  -- from cfg.data.los_region_mask_mode
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


ROOT = Path(r"c:/TFG/TFGPractice")
EVAL_DIR = ROOT / "tmp_review" / "try74_local_eval"

# (try_dir_name, experiments_subdir, registry_filename)
TRY_CANDIDATES = [
    ("TFGSeventyFifthTry75", "seventyfifth_try75_experts", "try75_expert_registry.yaml"),
    ("TFGSeventyFourthTry74", "seventyfourth_try74_experts", "try74_expert_registry.yaml"),
    ("TFGSixtyEighthTry68",   "sixtyeighth_try68_experts",   "try68_expert_registry.yaml"),
]

TRY54_PARTITION_THRESHOLDS = {
    "density_q1": 0.12,
    "density_q2": 0.28,
    "height_q1": 12.0,
    "height_q2": 28.0,
}

TRY_EXPERT_MODE = {
    "TFGSeventyFifthTry75": "3_experts",
    "TFGSeventyFourthTry74": "3_experts",
    "TFGSixtyEighthTry68": "6_experts",
}

SPREAD_TRIES = {
    "delay_spread": {
        "try_dir": "TFGFiftyEighthTry58",
        "registry_subdir": "fiftyeighthtry58_topology_experts",
        "registry_name": "fiftyeighthtry58_expert_registry.yaml",
    },
    "angular_spread": {
        "try_dir": "TFGFiftyNinthTry59",
        "registry_subdir": "fiftyninthtry59_topology_experts",
        "registry_name": "fiftyninthtry59_expert_registry.yaml",
    },
}


def histogram_db(values_db: np.ndarray, lo: int, hi: int) -> np.ndarray:
    if values_db.size == 0:
        return np.zeros(hi - lo, dtype=np.int64)
    v = np.clip(values_db, lo, hi - 1e-6)
    idx = np.floor(v - lo).astype(np.int64)
    idx = np.clip(idx, 0, hi - lo - 1)
    return np.bincount(idx, minlength=hi - lo).astype(np.int64)


def histogram_db_padded(values_db: np.ndarray, lo: int, hi_metric: int, hi_global: int) -> np.ndarray:
    """Histogram clipped to hi_metric, padded to a common hi_global range."""
    hi_metric = int(max(hi_metric, lo + 1))
    hi_global = int(max(hi_global, hi_metric))
    h = histogram_db(values_db, lo, hi_metric)
    pad = (hi_global - lo) - h.size
    if pad > 0:
        h = np.pad(h, (0, pad), mode="constant")
    return h


def round_metric_values(values: np.ndarray) -> np.ndarray:
    """Round float-valued metric maps to integer dB bins before histogramming."""
    return np.rint(values.astype(np.float32)).astype(np.float32)


def infer_city_type_3(density: float, height: float) -> str:
    if density >= 0.34 or height >= 90.0:
        return "dense_highrise"
    if density <= 0.18 and height <= 30.0:
        return "open_lowrise"
    return "mixed_midrise"


def infer_city_type_6(density: float, height: float) -> str:
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


def infer_city_types_from_topology(topo_meters: np.ndarray, non_ground_threshold: float = 0.0) -> tuple[str, str]:
    non_ground = topo_meters != float(non_ground_threshold)
    density = float(np.mean(non_ground)) if non_ground.size else 0.0
    heights = topo_meters[non_ground]
    mean_h = float(np.mean(heights)) if heights.size else 0.0
    return infer_city_type_3(density, mean_h), infer_city_type_6(density, mean_h)


def _resolve_spread_checkpoint_path(
    try_dir: str,
    topology_class: str,
    row_output_dir: str | None,
    checkpoint_name: str,
) -> Path | None:
    cluster_try_dir = ROOT / "cluster_outputs" / try_dir
    if not cluster_try_dir.exists():
        return None

    ckpt_candidates = [checkpoint_name] if checkpoint_name != "auto" else ["best_model.pt", "best_cgan.pt"]

    # Preferred: folder from registry output_dir.
    if row_output_dir:
        out_name = Path(str(row_output_dir)).name
        if out_name:
            for ck_name in ckpt_candidates:
                p = cluster_try_dir / out_name / ck_name
                if p.exists():
                    return p

    # Fallback: fuzzy search by topology class tokens.
    topo_tokens = _name_tokens(topology_class)
    best: tuple[int, Path] | None = None
    for d in cluster_try_dir.iterdir():
        if not d.is_dir():
            continue
        name_tokens = _name_tokens(d.name)
        if topo_tokens and not topo_tokens.issubset(name_tokens):
            continue
        for ck_name in ckpt_candidates:
            p = d / ck_name
            if not p.exists():
                continue
            overlap = len(topo_tokens & name_tokens)
            extras = len(name_tokens - topo_tokens)
            score = 20 * overlap - 3 * extras
            if best is None or score > best[0]:
                best = (score, p)
    return best[1] if best is not None else None


def discover_spread_experts(metric: str, checkpoint_name: str) -> list[dict]:
    spec = SPREAD_TRIES[metric]
    try_dir = str(spec["try_dir"])
    try_root = ROOT / try_dir
    reg_path = try_root / "experiments" / str(spec["registry_subdir"]) / str(spec["registry_name"])
    if not reg_path.exists():
        print(f"[warn] spread {metric}: registry not found: {reg_path}")
        return []

    with reg_path.open("r", encoding="utf-8") as f:
        reg = yaml.safe_load(f) or {}

    resolved: list[dict] = []
    for row in list(reg.get("experts", [])):
        topology_class = str(row.get("topology_class", "")).strip()
        cfg_rel = str(row.get("config", "")).strip()
        if not topology_class or not cfg_rel:
            continue
        cfg_path = (try_root / cfg_rel).resolve()
        if not cfg_path.exists():
            continue
        ckpt = _resolve_spread_checkpoint_path(
            try_dir=try_dir,
            topology_class=topology_class,
            row_output_dir=row.get("output_dir"),
            checkpoint_name=checkpoint_name,
        )
        if ckpt is None:
            print(f"[warn] spread {metric}: missing checkpoint for {topology_class} under {try_dir}")
            continue
        resolved.append(
            {
                "metric": metric,
                "topology_class": topology_class,
                "try_root": try_root,
                "config": cfg_path,
                "checkpoint": ckpt,
            }
        )

    print(f"[spread] {metric}: discovered {len(resolved)} expert checkpoint(s) from {try_dir}")
    return resolved


def run_spread_checkpoint(
    metric: str,
    try_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    device_str: str,
    max_samples: int,
    required_keys: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return sample_key -> predicted physical map for a spread expert checkpoint."""
    if str(try_root) not in sys.path:
        sys.path.insert(0, str(try_root))

    for mod in list(sys.modules):
        if mod in {
            "config_utils",
            "data_utils",
            "evaluate_topology_expert",
            "model_topology_expert",
            "topology_expert_heuristics",
        }:
            del sys.modules[mod]

    from config_utils import load_config, anchor_data_paths_to_config_file, resolve_device, load_torch_checkpoint  # type: ignore
    from data_utils import (  # type: ignore
        build_datasets_from_config,
        unpack_expert_batch,
        forward_expert_model,
        compute_input_channels,
        compute_scalar_cond_dim,
        uses_scalar_film_conditioning,
    )
    from evaluate_topology_expert import denormalize_channel  # type: ignore
    from model_topology_expert import UNetGenerator  # type: ignore

    cfg = load_config(str(config_path))
    anchor_data_paths_to_config_file(cfg, str(config_path))
    cfg.setdefault("runtime", {})["device"] = device_str
    cfg.setdefault("training", {})["batch_size"] = 1
    cfg.setdefault("augmentation", {})["enable"] = False
    cfg["data"]["num_workers"] = 0

    datasets_obj = build_datasets_from_config(cfg)
    if isinstance(datasets_obj, dict):
        datasets = datasets_obj
    elif isinstance(datasets_obj, (tuple, list)):
        datasets = {}
        if len(datasets_obj) >= 1:
            datasets["train"] = datasets_obj[0]
        if len(datasets_obj) >= 2:
            datasets["val"] = datasets_obj[1]
        if len(datasets_obj) >= 3 and datasets_obj[2] is not None:
            datasets["test"] = datasets_obj[2]
    else:
        print(f"[warn] spread {metric}: unsupported datasets type {type(datasets_obj)!r}")
        return {}

    if split not in datasets:
        return {}
    dataset = datasets[split]

    selected_keys_in_order: list[str] | None = None
    if required_keys:
        refs = getattr(dataset, "sample_refs", None)
        if refs is None:
            print(f"[warn] spread {metric}: dataset has no sample_refs; cannot filter required keys")
            return {}
        key_to_idx = {f"{city}/{sample}": i for i, (city, sample) in enumerate(refs)}
        selected: list[tuple[int, str]] = []
        for key in required_keys:
            idx = key_to_idx.get(key)
            if idx is not None:
                selected.append((idx, key))
        if not selected:
            return {}
        selected.sort(key=lambda t: t[0])
        selected_indices = [idx for idx, _ in selected]
        selected_keys_in_order = [key for _, key in selected]
        dataset_for_loader = Subset(dataset, selected_indices)
    else:
        dataset_for_loader = dataset

    loader = DataLoader(
        dataset_for_loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    device = resolve_device(device_str)
    in_channels = int(compute_input_channels(cfg))
    sc_dim = int(compute_scalar_cond_dim(cfg)) if uses_scalar_film_conditioning(cfg) else 0
    model = UNetGenerator(
        in_channels=in_channels,
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        path_loss_hybrid=bool(dict(cfg.get("path_loss_hybrid", {})).get("enabled", False)),
        norm_type=str(cfg["model"].get("norm_type", "batch")),
        scalar_cond_dim=sc_dim,
        scalar_film_hidden=int(cfg["model"].get("scalar_film_hidden", 128)),
        upsample_mode=str(cfg["model"].get("upsample_mode", "transpose")),
    ).to(device)

    state = load_torch_checkpoint(str(checkpoint_path), device)
    sd = state.get("generator") if isinstance(state, dict) and "generator" in state else state
    model.load_state_dict(sd, strict=False)
    model.eval()

    target_columns = list(cfg.get("target_columns", []))
    if metric not in target_columns:
        cand = [c for c in target_columns if c != "no_data"]
        if not cand:
            return {}
        metric = cand[0]
    metric_idx = target_columns.index(metric)
    metric_meta = dict(cfg.get("target_metadata", {}).get(metric, {}))

    out: dict[str, np.ndarray] = {}
    cursor = 0
    amp_enabled = (getattr(device, "type", str(device)) == "cuda")
    with torch.no_grad():
        for batch in loader:
            if selected_keys_in_order is None and max_samples and len(out) >= max_samples:
                break
            x, _y, _m, sc = unpack_expert_batch(batch, device)
            if amp_enabled:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    pred = forward_expert_model(model, x, sc)
            else:
                pred = forward_expert_model(model, x, sc)

            pred_metric = denormalize_channel(pred[:, metric_idx : metric_idx + 1], metric_meta).float()
            bsz = int(pred_metric.shape[0])
            for off in range(bsz):
                if selected_keys_in_order is not None:
                    key = selected_keys_in_order[cursor + off]
                else:
                    sample_idx = cursor + off
                    city, sample_ref = dataset.sample_refs[sample_idx]
                    key = f"{city}/{sample_ref}"
                out[key] = pred_metric[off, 0].detach().cpu().numpy().astype(np.float32)
            cursor += bsz
            if selected_keys_in_order is None and max_samples and len(out) >= max_samples:
                break

    del model
    if amp_enabled:
        torch.cuda.empty_cache()
    return out


def collect_spread_predictions(
    split: str,
    device_str: str,
    checkpoint_name: str,
    max_samples: int,
    required_keys_by_topology: dict[str, set[str]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect sample_key -> pred map for delay and angular from Try58/Try59 experts."""
    all_preds: dict[str, dict[str, np.ndarray]] = {"delay_spread": {}, "angular_spread": {}}
    for metric in ("delay_spread", "angular_spread"):
        entries = discover_spread_experts(metric, checkpoint_name=checkpoint_name)
        preds_metric: dict[str, np.ndarray] = {}
        needed_topologies = set(required_keys_by_topology.keys()) if required_keys_by_topology else None
        for entry in entries:
            topology_class = str(entry["topology_class"])
            if needed_topologies is not None and topology_class not in needed_topologies:
                continue
            chunk = run_spread_checkpoint(
                metric=metric,
                try_root=Path(entry["try_root"]),
                config_path=Path(entry["config"]),
                checkpoint_path=Path(entry["checkpoint"]),
                split=split,
                device_str=device_str,
                max_samples=max_samples,
                required_keys=(required_keys_by_topology or {}).get(topology_class),
            )
            preds_metric.update(chunk)
        all_preds[metric] = preds_metric
        print(f"[spread] {metric}: predictions for {len(preds_metric)} sample(s)")
    return all_preds


def _name_tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _resolve_checkpoint_path(try_dir: str, expert_id: str, row_checkpoint: str | None) -> Path | None:
    """Resolve best_model.pt for an expert allowing prefixed/suffixed folder names."""
    cluster_try_dir = ROOT / "cluster_outputs" / try_dir
    if not cluster_try_dir.exists():
        return None

    # Fast path: exact folder match.
    ckpt = cluster_try_dir / expert_id / "best_model.pt"
    if ckpt.exists():
        return ckpt

    # If registry checkpoint includes a folder name, try that under cluster_outputs/<try>/.
    if row_checkpoint:
        row_parent = Path(str(row_checkpoint)).parent.name
        if row_parent:
            ckpt = cluster_try_dir / row_parent / "best_model.pt"
            if ckpt.exists():
                return ckpt

    expert_id_l = expert_id.lower()
    expert_tokens = _name_tokens(expert_id_l)
    candidates: list[tuple[int, str, Path]] = []
    for d in cluster_try_dir.iterdir():
        if not d.is_dir():
            continue
        cand_ckpt = d / "best_model.pt"
        if not cand_ckpt.exists():
            continue
        name_l = d.name.lower()
        name_tokens = _name_tokens(name_l)

        # Eligibility: exact/substring or token superset.
        eligible = (
            name_l == expert_id_l
            or expert_id_l in name_l
            or name_l in expert_id_l
            or (expert_tokens and expert_tokens.issubset(name_tokens))
        )
        if not eligible:
            continue

        # Prefer exact > suffix/prefix > substring > token overlap with fewer extra tokens.
        score = 0
        if name_l == expert_id_l:
            score += 1000
        if name_l.endswith(expert_id_l) or name_l.startswith(expert_id_l):
            score += 400
        if expert_id_l in name_l:
            score += 300
        overlap = len(expert_tokens & name_tokens)
        extras = len(name_tokens - expert_tokens)
        score += 20 * overlap
        score -= 3 * extras
        candidates.append((score, d.name, cand_ckpt))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (-t[0], t[1]))
    return candidates[0][2]


def discover_try(cli_try: str | None) -> tuple[Path, list[dict]]:
    """Return (try_root, expert_entries)."""
    if cli_try:
        # match by substring
        cands = [t for t in TRY_CANDIDATES if cli_try.lower() in t[0].lower() or cli_try in t[0]]
        if not cands:
            raise SystemExit(f"--try {cli_try!r} did not match any known try.")
        order = cands
    else:
        order = TRY_CANDIDATES

    for try_dir, exp_subdir, registry_name in order:
        try_root = ROOT / try_dir
        reg_path = try_root / "experiments" / exp_subdir / registry_name
        if not reg_path.exists():
            continue
        with reg_path.open("r", encoding="utf-8") as f:
            reg = yaml.safe_load(f) or {}
        experts = list(reg.get("experts", []))
        resolved = []
        expert_mode = TRY_EXPERT_MODE.get(try_dir, f"{len(experts)}_experts")
        for row in experts:
            expert_id = str(row.get("expert_id"))
            cfg_path = (try_root / str(row.get("config"))).resolve()
            if not cfg_path.exists():
                continue
            ckpt = _resolve_checkpoint_path(
                try_dir=try_dir,
                expert_id=expert_id,
                row_checkpoint=row.get("checkpoint"),
            )
            if ckpt is None:
                expected = ROOT / "cluster_outputs" / try_dir / expert_id / "best_model.pt"
                print(f"[skip] {try_dir}/{expert_id}: no best_model.pt (expected {expected})")
                continue
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            region = str((cfg.get("data", {}) or {}).get("los_region_mask_mode", "full"))
            resolved.append({
                "try_dir": try_dir,
                "expert_mode": expert_mode,
                "expert_id": expert_id,
                "config": str(cfg_path),
                "checkpoint": str(ckpt),
                "region": region,
            })
        if resolved:
            print(f"[try] using {try_dir}: {len(resolved)} expert(s) with checkpoints")
            for r in resolved:
                print(f"    - {r['expert_id']}  region={r['region']}")
            return try_root, resolved
        else:
            print(f"[try] {try_dir}: registry found but no usable checkpoints")

    raise SystemExit("No try with available checkpoints in cluster_outputs/<try>/<expert>/best_model.pt")


def run_checkpoint(
    try_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    max_samples: int,
    device_str: str,
    all_samples: bool,
    force_full_region_mask: bool,
    exclude_building_pixels: bool,
) -> Iterable[tuple]:
    """Yield per-sample arrays for path_loss (+ delay/angular GT maps from HDF5)."""
    # Inject this try's root so we pick up its own train_partitioned_pathloss_expert.py.
    for p in (try_root, EVAL_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # IMPORTANT: different tries have distinct but same-named modules; reset cache between runs.
    for mod in list(sys.modules):
        if mod in {"train_partitioned_pathloss_expert", "data_utils", "config_utils", "eval_per_sample"}:
            del sys.modules[mod]

    from train_partitioned_pathloss_expert import (  # type: ignore
        load_config,
        anchor_data_paths_to_config_file,
        build_dataset_splits_from_config,
        build_validation_loader,
        compute_input_channels,
        unpack_cgan_batch,
        extract_formula_prior_or_zero,
        _compose_residual_prediction_with_aux,
        _build_pmnet_from_cfg,
        uses_absolute_path_loss_prediction,
        clip_to_target_range,
        denormalize,
        sanitize_masked_target,
    )
    from config_utils import resolve_device, load_torch_checkpoint  # type: ignore

    cfg = load_config(str(config_path))
    anchor_data_paths_to_config_file(cfg, str(config_path))
    cfg.setdefault("training", {})["batch_size"] = 1
    cfg["data"]["val_batch_size"] = 1
    cfg["data"]["num_workers"] = 0
    cfg["data"]["val_num_workers"] = 0
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["val_persistent_workers"] = False
    cfg.setdefault("model", {})["gradient_checkpointing"] = False
    # Override runtime device.
    cfg.setdefault("runtime", {})["device"] = device_str
    # Remove any partition filter / height bands: we want ALL samples.
    if all_samples:
        cfg["data"]["partition_filter"] = {}
    # For histogram accounting we need GT/valid masks over the full map;
    # expert region filtering is still applied when writing pred_* rows.
    if force_full_region_mask:
        # Use full-map validity by disabling LoS/NLoS region masking in dataset masks.
        cfg["data"]["los_region_mask_mode"] = None
    # Building filtering for histograms is applied explicitly from topology_map
    # so we keep dataset non-ground masking disabled here.
    if exclude_building_pixels:
        cfg["data"]["exclude_non_ground_targets"] = False

    device = resolve_device(device_str)
    print(f"[run] device={device} split={split} config={config_path.name}")

    splits = build_dataset_splits_from_config(cfg)
    dataset = splits[split]
    print(f"[run] {split} samples: {len(dataset)}")

    in_ch = int(compute_input_channels(cfg))
    model = _build_pmnet_from_cfg(cfg, in_ch).to(device)

    # Load checkpoint with device-agnostic map_location (DirectML isn't a torch map_location string).
    state = load_torch_checkpoint(str(checkpoint_path), device)
    sd = None
    if isinstance(state, dict):
        for key in ("generator_ema", "model_ema", "ema_state_dict", "generator", "model", "state_dict"):
            if key in state and isinstance(state[key], dict):
                sd = state[key]
                break
        if sd is None and all(isinstance(v, torch.Tensor) for v in state.values()):
            sd = state
    if sd is None:
        raise RuntimeError(f"Could not find state_dict in checkpoint {checkpoint_path}")
    # Strip DDP 'module.' prefix if present.
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()

    total_split_samples = len(dataset)
    if max_samples and max_samples > 0 and max_samples < total_split_samples:
        rng = np.random.default_rng(42)
        sample_indices = sorted(rng.choice(total_split_samples, size=max_samples, replace=False).tolist())
        subset = Subset(dataset, sample_indices)
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        print(f"[run] random sample subset: {len(sample_indices)} / {total_split_samples}")
    else:
        loader, sample_indices = build_validation_loader(dataset, device, cfg, distributed=False)
        sample_indices = list(sample_indices)

    meta = dict(cfg["target_metadata"]["path_loss"])
    absolute_prediction = uses_absolute_path_loss_prediction(cfg)
    clamp_final = bool(cfg.get("prior_residual_path_loss", {}).get("clamp_final_output", True))
    is_cuda = (getattr(device, "type", str(device)) == "cuda")
    amp_enabled = is_cuda  # AMP only on CUDA
    non_ground_threshold = float(cfg.get("data", {}).get("non_ground_threshold", 0.0))
    exclude_non_ground_targets = bool(cfg.get("data", {}).get("exclude_non_ground_targets", False))

    cursor = 0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if max_samples and seen >= max_samples:
                break
            x, y, m, scalar_cond = unpack_cgan_batch(batch, device)
            prior = extract_formula_prior_or_zero(x, cfg, y[:, :1])
            if amp_enabled:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    residual_pred, _, _, _ = _compose_residual_prediction_with_aux(
                        model, x, scalar_cond, prior,
                        separated_mode=False, base_generator=None,
                        use_gate=False, absolute_prediction=absolute_prediction,
                    )
            else:
                residual_pred, _, _, _ = _compose_residual_prediction_with_aux(
                    model, x, scalar_cond, prior,
                    separated_mode=False, base_generator=None,
                    use_gate=False, absolute_prediction=absolute_prediction,
                )
            pred = residual_pred if absolute_prediction else (prior + residual_pred)
            if clamp_final:
                pred = clip_to_target_range(pred, meta)
            safe_target = sanitize_masked_target(y[:, :1], m[:, :1])
            pred_phys = denormalize(pred, meta).float()
            target_phys = denormalize(safe_target, meta).float()
            valid_mask = m[:, :1] > 0.0

            B = x.shape[0]
            for off in range(B):
                idx = int(sample_indices[cursor + off])
                city = sample_ref = ""
                if hasattr(dataset, "sample_refs") and idx < len(dataset.sample_refs):
                    city, sample_ref = dataset.sample_refs[idx]
                ant_h = float("nan")
                if scalar_cond is not None and scalar_cond.numel() > 0:
                    cols = list(cfg["data"].get("scalar_feature_columns", []))
                    if "antenna_height_m" in cols:
                        j = cols.index("antenna_height_m")
                        norm = float(cfg["data"].get("scalar_feature_norms", {}).get("antenna_height_m", 1.0))
                        ant_h = float(scalar_cond[off, j].item()) * norm
                los_arr = None
                topo_arr = None
                city_type_3 = ""
                city_type_6 = ""
                delay_gt = None
                angular_gt = None
                try:
                    h = dataset._get_handle()
                    grp = h[city][sample_ref]
                    if (not np.isfinite(ant_h)) and "uav_height" in grp:
                        arr = np.asarray(grp["uav_height"][...], dtype=np.float64).reshape(-1)
                        if arr.size:
                            ant_h = float(arr[0])
                    if "los_mask" in grp:
                        los_arr = np.asarray(grp["los_mask"][...], dtype=np.float32)
                    if "topology_map" in grp:
                        topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
                        topo_arr = topo
                        city_type_3, city_type_6 = infer_city_types_from_topology(
                            topo,
                            non_ground_threshold=non_ground_threshold,
                        )
                    if "delay_spread" in grp:
                        delay_gt = np.asarray(grp["delay_spread"][...], dtype=np.float32)
                    if "angular_spread" in grp:
                        angular_gt = np.asarray(grp["angular_spread"][...], dtype=np.float32)
                except Exception:
                    pass

                pred_np = pred_phys[off, 0].detach().float().cpu().numpy().astype(np.float32)
                tgt_np = target_phys[off, 0].detach().float().cpu().numpy().astype(np.float32)
                vm_np = valid_mask[off, 0].detach().cpu().numpy().astype(bool)
                # Keep only physically valid GT pixels for histogram accounting.
                vm_np &= np.isfinite(tgt_np) & (tgt_np > 0.0)
                if exclude_building_pixels and topo_arr is not None and topo_arr.shape == vm_np.shape:
                    # User-requested mask: keep only pixels where topology == 0.
                    vm_np &= (topo_arr == 0.0)
                if los_arr is None or los_arr.shape != tgt_np.shape:
                    los_bool = np.zeros_like(tgt_np, dtype=bool)
                else:
                    los_bool = los_arr > 0.5

                yield (
                    f"{city}/{sample_ref}", pred_np, tgt_np, los_bool, vm_np,
                    city, sample_ref, ant_h, city_type_3, city_type_6, exclude_non_ground_targets,
                    delay_gt, angular_gt,
                )
                seen += 1
                if max_samples and seen >= max_samples:
                    break
            cursor += B

    # Release GPU memory between experts.
    del model
    if is_cuda:
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--try", dest="try_name", default=None,
                    help="Substring match against TFGSeventyFifthTry75 / ...Fourth... / ...Sixtyeighth... . "
                         "Default: auto (prefer Try75, then 74, then 68).")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 = all samples (default).")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "directml", "dml", "cpu"])
    ap.add_argument("--db-lo", type=int, default=0)
    ap.add_argument("--db-hi-path-loss", type=int, default=180)
    ap.add_argument("--db-hi-delay-spread", type=int, default=360)
    ap.add_argument("--db-hi-angular-spread", type=int, default=180)
    ap.add_argument(
        "--spread-checkpoint-name",
        default="auto",
        choices=["auto", "best_model.pt", "best_cgan.pt"],
        help="Checkpoint file name for Try58/59 spread experts (auto tries best_model.pt then best_cgan.pt).",
    )
    ap.add_argument("--skip-spread-predictions", action="store_true",
                    help="If set, only spread target rows are emitted (no spread prediction rows).")
    ap.add_argument("--out-csv", default=str(Path(__file__).parent / "histograms.csv"))
    args = ap.parse_args()

    try_root, experts = discover_try(args.try_name)

    db_lo = int(args.db_lo)
    db_hi_path = int(args.db_hi_path_loss)
    db_hi_delay = int(args.db_hi_delay_spread)
    db_hi_ang = int(args.db_hi_angular_spread)
    db_hi = max(db_hi_path, db_hi_delay, db_hi_ang)
    n_bins = db_hi - db_lo
    bin_cols = [f"b{db_lo + i}" for i in range(n_bins)]

    spread_preds: dict[str, dict[str, np.ndarray]] = {"delay_spread": {}, "angular_spread": {}}
    if args.skip_spread_predictions:
        print("[spread] --skip-spread-predictions enabled: spread pred rows will be skipped")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "city", "sample", "altitude_m", "city_type", "city_type_3", "city_type_6",
        "try_dir", "expert_mode", "exclude_non_ground_targets",
        "expert_id", "expert_region", "metric", "kind", "total_pixels", *bin_cols,
    ]

    # Write streaming so long runs don't lose data on crash.
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        f.flush()

        # Collect rows per sample first, then flush in sample order to keep groups contiguous.
        sample_rows: dict[str, list[list]] = {}
        sample_base: dict[str, list] = {}
        sample_valid_mask: dict[str, np.ndarray] = {}
        sample_city_type_6: dict[str, str] = {}
        sample_order: list[str] = []

        for exp in experts:
            print(f"\n=== Expert {exp['expert_id']} ({exp['region']}) ===")
            pred_los_pixels = 0
            pred_nlos_pixels = 0
            for (
                key,
                pred_np,
                tgt_np,
                los_bool,
                vm_np,
                city,
                sample_ref,
                ant_h,
                city_type_3,
                city_type_6,
                exclude_non_ground_targets,
                delay_gt,
                angular_gt,
            ) in run_checkpoint(
                try_root=try_root,
                config_path=Path(exp["config"]),
                checkpoint_path=Path(exp["checkpoint"]),
                split=args.split,
                max_samples=args.max_samples,
                device_str=args.device,
                all_samples=True,
                force_full_region_mask=True,
                exclude_building_pixels=True,
            ):
                los_valid = los_bool & vm_np
                nlos_valid = (~los_bool) & vm_np
                base = [
                    city,
                    sample_ref,
                    f"{ant_h:.3f}",
                    city_type_3,
                    city_type_3,
                    city_type_6,
                    exp["try_dir"],
                    exp["expert_mode"],
                    int(exclude_non_ground_targets),
                ]

                if key not in sample_rows:
                    sample_rows[key] = []
                    sample_order.append(key)
                    sample_base[key] = list(base)
                    sample_valid_mask[key] = vm_np.copy()
                    sample_city_type_6[key] = city_type_6

                    # Path-loss GT rows.
                    for kind, sel in (("target_los", los_valid), ("target_nlos", nlos_valid)):
                        h = histogram_db_padded(tgt_np[sel], db_lo, db_hi_path, db_hi)
                        sample_rows[key].append([
                            *base,
                            "ground_truth",
                            "full",
                            "path_loss",
                            kind,
                            int(sel.sum()),
                            *[int(v) for v in h],
                        ])

                    # Spread GT rows.
                    if delay_gt is not None and delay_gt.shape == vm_np.shape:
                        sel = vm_np & np.isfinite(delay_gt) & (delay_gt > 0.0)
                        vals = round_metric_values(delay_gt[sel])
                        h = histogram_db_padded(vals, db_lo, db_hi_delay, db_hi)
                        sample_rows[key].append([
                            *base,
                            "ground_truth",
                            "full",
                            "delay_spread",
                            "target_delay_spread",
                            int(sel.sum()),
                            *[int(v) for v in h],
                        ])

                    if angular_gt is not None and angular_gt.shape == vm_np.shape:
                        sel = vm_np & np.isfinite(angular_gt) & (angular_gt > 0.0)
                        vals = round_metric_values(angular_gt[sel])
                        h = histogram_db_padded(vals, db_lo, db_hi_ang, db_hi)
                        sample_rows[key].append([
                            *base,
                            "ground_truth",
                            "full",
                            "angular_spread",
                            "target_angular_spread",
                            int(sel.sum()),
                            *[int(v) for v in h],
                        ])

                # Path-loss prediction rows scoped by expert region.
                if exp["region"] in ("los_only", "full"):
                    los_count = int(los_valid.sum())
                    h = histogram_db_padded(pred_np[los_valid], db_lo, db_hi_path, db_hi)
                    sample_rows[key].append([
                        *base,
                        exp["expert_id"],
                        exp["region"],
                        "path_loss",
                        "pred_los",
                        los_count,
                        *[int(v) for v in h],
                    ])
                    pred_los_pixels += los_count
                if exp["region"] in ("nlos_only", "full"):
                    nlos_count = int(nlos_valid.sum())
                    h = histogram_db_padded(pred_np[nlos_valid], db_lo, db_hi_path, db_hi)
                    sample_rows[key].append([
                        *base,
                        exp["expert_id"],
                        exp["region"],
                        "path_loss",
                        "pred_nlos",
                        nlos_count,
                        *[int(v) for v in h],
                    ])
                    pred_nlos_pixels += nlos_count

            print(
                f"[expert-summary] {exp['expert_id']} "
                f"pred_los_pixels={pred_los_pixels} pred_nlos_pixels={pred_nlos_pixels}"
            )

        if not args.skip_spread_predictions and sample_order:
            required_keys_by_topology: dict[str, set[str]] = {}
            for key in sample_order:
                topo = str(sample_city_type_6.get(key, "")).strip()
                if topo:
                    required_keys_by_topology.setdefault(topo, set()).add(key)

            spread_preds = collect_spread_predictions(
                split=args.split,
                device_str=args.device,
                checkpoint_name=args.spread_checkpoint_name,
                max_samples=0,
                required_keys_by_topology=required_keys_by_topology,
            )

            for key in sample_order:
                if key not in sample_rows:
                    continue
                base = sample_base.get(key)
                vm_np = sample_valid_mask.get(key)
                if base is None or vm_np is None:
                    continue

                delay_pred = spread_preds.get("delay_spread", {}).get(key)
                if delay_pred is not None and delay_pred.shape == vm_np.shape:
                    sel = vm_np & np.isfinite(delay_pred) & (delay_pred > 0.0)
                    vals = round_metric_values(delay_pred[sel])
                    h = histogram_db_padded(vals, db_lo, db_hi_delay, db_hi)
                    sample_rows[key].append([
                        *base,
                        "spread_prediction",
                        "full",
                        "delay_spread",
                        "pred_delay_spread",
                        int(sel.sum()),
                        *[int(v) for v in h],
                    ])

                angular_pred = spread_preds.get("angular_spread", {}).get(key)
                if angular_pred is not None and angular_pred.shape == vm_np.shape:
                    sel = vm_np & np.isfinite(angular_pred) & (angular_pred > 0.0)
                    vals = round_metric_values(angular_pred[sel])
                    h = histogram_db_padded(vals, db_lo, db_hi_ang, db_hi)
                    sample_rows[key].append([
                        *base,
                        "spread_prediction",
                        "full",
                        "angular_spread",
                        "pred_angular_spread",
                        int(sel.sum()),
                        *[int(v) for v in h],
                    ])

        for key in sample_order:
            for row in sample_rows.get(key, []):
                w.writerow(row)
        f.flush()

    print(f"\nWrote histograms to {out_path}")
    print(
        f"Bins: {db_lo}..{db_hi} ({n_bins} cols, 1 dB each) "
        f"[path_loss_hi={db_hi_path}, delay_hi={db_hi_delay}, angular_hi={db_hi_ang}]"
    )


if __name__ == "__main__":
    main()
