#!/usr/bin/env python3
"""
1) Exporta el HDF5 a PNG en dos vistas: by_sample/<ciudad>/<muestra>/ y by_field/<tipo>/<ciudad>/<muestra>.png
   (tipos: topology_map, path_loss, delay_spread, angular_spread, los para LoS).
2) Path loss: importas el try con --ninth-root / --path-loss-try-root (TFGNinthTry9 un modelo, TFGTenthTry10 LoS+NLoS).
3) Delay / angular spread: modelo 3 salidas de First / Second / Third try; elige el mejor try
   según JSONs en cluster_outputs (suma rmse_physical delay + angular); si no hay métricas, usa Third.

Dataset por defecto: <repo>/Datasets/CKM_Dataset_180326.h5 y CKM_180326_antenna_height.csv
  (equivalente a C:\\TFG\\TFGpractice\\Datasets si el repo está ahí).

Requisitos: pip install h5py pillow torch torchvision pyyaml

Ejemplo (desde TFGpractice/):

  python scripts/export_dataset_and_predictions.py ^
    --dataset-out D:/Dataset_Imagenes ^
    --ninth-checkpoint D:/ckpt/ninth/best_cgan.pt ^
    --spread-checkpoint D:/ckpt/third/best_cgan.pt

  # Elegir try manualmente (sin escanear JSONs)
  python scripts/export_dataset_and_predictions.py --spread-try second --spread-checkpoint ...
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import importlib
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent
CLUSTER_OUTPUTS = PRACTICE_ROOT / "cluster_outputs"

# HDF5 con uav_height por muestra (no hace falta CSV para NinthTry9).
DEFAULT_HDF5 = PRACTICE_ROOT / "Datasets" / "CKM_Dataset_180326_antenna_height.h5"
# Si usas el .h5 sin sufijo _antenna_height, suele ir junto a este CSV:
DEFAULT_SCALAR_CSV = PRACTICE_ROOT / "Datasets" / "CKM_180326_antenna_height.csv"

HDF5_FIELDS: Tuple[str, ...] = (
    "topology_map",
    "path_loss",
    "delay_spread",
    "angular_spread",
    "los_mask",
)


class _ConcatDatasetWithSampleRefs:
    def __init__(self, datasets: Sequence[Any]) -> None:
        self.datasets = [ds for ds in datasets if ds is not None]
        self.cum_sizes: List[int] = []
        total = 0
        self.sample_refs: List[Tuple[str, str]] = []
        for ds in self.datasets:
            n = len(ds)
            total += n
            self.cum_sizes.append(total)
            refs = list(getattr(ds, "sample_refs", []))
            if refs:
                self.sample_refs.extend(refs)

    def __len__(self) -> int:
        return self.cum_sizes[-1] if self.cum_sizes else 0

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        prev = 0
        for ds, end in zip(self.datasets, self.cum_sizes):
            if idx < end:
                return ds[idx - prev]
            prev = end
        raise IndexError(idx)


def merge_splits_with_fallback(build_dataset_splits_fn: Any, cfg: Dict[str, Any]) -> Any:
    splits = build_dataset_splits_fn(cfg)
    ordered = [splits[k] for k in ("train", "val", "test") if k in splits]
    return _ConcatDatasetWithSampleRefs(ordered)

def pick_inference_device(device_pref: str, resolve_device_fn: Any) -> Any:
    """
    Pick a device that works with the installed PyTorch build.
    - auto: NVIDIA CUDA if available, else DirectML if torch-directml is installed, else CPU.
    - cuda: CUDA only if available; otherwise CPU + hint for AMD (DirectML).
    - cpu: CPU.
    - directml / dml / amd: delegated to resolve_device_fn (needs pip install torch-directml).
    """
    import torch

    pref = (device_pref or "auto").strip().lower()
    if pref in ("gpu", "cuda"):
        if torch.cuda.is_available():
            return "cuda"
        print(
            "[export] CUDA not available (CPU-only PyTorch or no NVIDIA GPU). Using CPU.\n"
            "         AMD (e.g. RX 7800 XT) on Windows: pip install torch-directml  then  --device directml"
        )
        return "cpu"
    if pref == "cpu":
        return "cpu"
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            import torch_directml

            print("[export] Using DirectML (AMD/Intel GPU via torch-directml).")
            return torch_directml.device()
        except Exception:
            print("[export] No CUDA or torch-directml; using CPU.")
            return "cpu"
    if pref in ("directml", "dml", "amd"):
        return resolve_device_fn(pref)
    return resolve_device_fn(pref)


def should_use_cuda_amp(device: Any) -> bool:
    """AMP autocast only makes sense on NVIDIA CUDA in this codebase."""
    import torch

    if not torch.cuda.is_available():
        return False
    if device == "cuda":
        return True
    if isinstance(device, torch.device) and device.type == "cuda":
        return True
    return False


SPREAD_TRY_LAYOUT: Dict[str, Tuple[str, Path, Path]] = {
    "first": (
        "firsttry1",
        PRACTICE_ROOT / "TFG_FirstTry1",
        PRACTICE_ROOT / "TFG_FirstTry1" / "configs" / "cgan_unet_hdf5.yaml",
    ),
    "second": (
        "secondtry2",
        PRACTICE_ROOT / "TFGSecondTry2",
        PRACTICE_ROOT / "TFGSecondTry2" / "configs" / "cgan_unet_hdf5_cuda_max.yaml",
    ),
    "third": (
        "thirdtry3",
        PRACTICE_ROOT / "TFGThirdTry3",
        PRACTICE_ROOT / "TFGThirdTry3" / "configs" / "cgan_unet_hdf5_cuda_max.yaml",
    ),
}


def _sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def field_type_folder(field: str) -> str:
    """Carpeta por tipo de mapa en by_field/ (los_mask → los)."""
    return {
        "topology_map": "topology_map",
        "path_loss": "path_loss",
        "delay_spread": "delay_spread",
        "angular_spread": "angular_spread",
        "los_mask": "los",
    }.get(field, _sanitize(field))


def array_to_rgb_u8(
    arr: np.ndarray,
    field_name: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    Grayscale → RGB uint8. Default: per-image min–max (good for standalone maps).

    For GT|Pred comparisons, pass vmin/vmax so both panels share one scale. If the prediction
    is almost flat, independent min–max amplifies float noise into full-range “TV static”.
    """
    a = np.asarray(arr, dtype=np.float32)
    if field_name == "los_mask":
        preview = (a > 0.5).astype(np.uint8) * 255
        rgb = np.stack([preview, preview, preview], axis=-1)
        return rgb
    if vmin is not None and vmax is not None:
        lo, hi = float(vmin), float(vmax)
    else:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if hi - lo < 1e-12:
        gray = np.zeros(a.shape, dtype=np.uint8)
    else:
        a = np.nan_to_num(a, nan=lo)
        gray = ((a - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _paired_map_viz_range(
    tgt_map: np.ndarray,
    pred_map: np.ndarray,
    mode: str,
) -> Tuple[float, float]:
    """Return (vmin, vmax) for GT|Pred PNG scaling: 'gt' = GT only, 'joint' = union of both."""
    m = str(mode).lower().strip()
    t_lo, t_hi = float(np.nanmin(tgt_map)), float(np.nanmax(tgt_map))
    p_lo, p_hi = float(np.nanmin(pred_map)), float(np.nanmax(pred_map))
    if m == "joint":
        return min(t_lo, p_lo), max(t_hi, p_hi)
    if m == "gt":
        return t_lo, t_hi
    raise ValueError(f"Unknown paired viz scale mode: {mode!r} (use gt or joint)")


def _multi_map_viz_range(maps: Sequence[np.ndarray], mode: str, gt_index: int = 0) -> Tuple[float, float]:
    """Same semantics as paired, but over many maps (e.g. GT + LoS + NLoS + combined)."""
    m = str(mode).lower().strip()
    if not maps:
        raise ValueError("maps must be non-empty")
    if m == "gt":
        t = maps[gt_index]
        return float(np.nanmin(t)), float(np.nanmax(t))
    if m == "joint":
        lo = min(float(np.nanmin(a)) for a in maps)
        hi = max(float(np.nanmax(a)) for a in maps)
        return lo, hi
    raise ValueError(f"Unknown multi viz scale mode: {mode!r} (use gt or joint)")


def hconcat_images(images: List[Image.Image], labels: List[str]) -> Image.Image:
    label_h = 26  # if changed, update GT_PRED_LABEL_BAR_PX in build_alltogether_panel.py
    max_h = max(im.height for im in images) + label_h
    total_w = sum(im.width for im in images)
    canvas = Image.new("RGB", (total_w, max_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    x = 0
    for im, lab in zip(images, labels):
        canvas.paste(im, (x, label_h))
        draw.text((x + 4, 4), lab[:48], fill=(0, 0, 0), font=font)
        x += im.width
    return canvas


def pick_best_spread_try_from_metrics(
    cluster_root: Path,
    practice_root: Path,
) -> Tuple[str, Optional[Path], Optional[int], Optional[float]]:
    """Elige first|second|third por menor (delay_rmse_physical + angular_rmse_physical) en validate_metrics*.json."""
    best_score = math.inf
    best_key = "third"
    best_json: Optional[Path] = None
    best_epoch: Optional[int] = None

    scan_dirs: List[Tuple[str, Path]] = [
        ("first", cluster_root / "TFG_FirstTry1"),
        ("second", cluster_root / "TFGSecondTry2"),
        ("third", cluster_root / "TFGThirdTry3"),
        ("first", practice_root / "TFG_FirstTry1" / "outputs"),
        ("second", practice_root / "TFGSecondTry2" / "outputs"),
        ("third", practice_root / "TFGThirdTry3" / "outputs"),
    ]
    for key, base in scan_dirs:
        if not base.is_dir():
            continue
        for jp in sorted(base.rglob("validate_metrics*.json")):
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            ds = data.get("delay_spread") or {}
            ang = data.get("angular_spread") or {}
            d = ds.get("rmse_physical")
            a = ang.get("rmse_physical")
            if d is None or a is None:
                continue
            score = float(d) + float(a)
            if score < best_score:
                best_score = score
                best_key = key
                best_json = jp
                ck = data.get("_checkpoint") or {}
                if isinstance(ck.get("epoch"), int):
                    best_epoch = int(ck["epoch"])
                elif isinstance(ck.get("epoch"), float):
                    best_epoch = int(ck["epoch"])

    if best_json is None:
        return "third", None, None, None
    return best_key, best_json, best_epoch, float(best_score)


def export_hdf5_dataset(hdf5_path: Path, out_root: Path) -> None:
    import h5py

    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root = out_root / "raw_hdf5"
    by_sample_root = raw_root / "by_sample"
    by_field_root = raw_root / "by_field"
    n_samples = 0
    with h5py.File(hdf5_path, "r") as handle:
        for city in sorted(handle.keys()):
            city_g = handle[city]
            city_s = _sanitize(str(city))
            for sample in sorted(city_g.keys()):
                sg = city_g[sample]
                sample_s = _sanitize(str(sample))
                # Por muestra: todas las modalidades en la misma carpeta
                base = by_sample_root / city_s / sample_s
                base.mkdir(parents=True, exist_ok=True)
                for field in HDF5_FIELDS:
                    if field not in sg:
                        continue
                    arr = np.asarray(sg[field][...])
                    rgb = array_to_rgb_u8(arr, field)
                    Image.fromarray(rgb).save(base / f"{field}.png")
                    # Por tipo: topology / path_loss / los / … → ciudad → muestra.png
                    field_dir = by_field_root / field_type_folder(field) / city_s
                    field_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(field_dir / f"{sample_s}.png")
                n_samples += 1
    print(
        f"[dataset] Exported {n_samples} samples: "
        f"{by_sample_root} (por muestra) y {by_field_root} (por tipo de mapa)"
    )


def _insert_try_path(try_root: Path) -> None:
    p = str(try_root.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _prepare_path_loss_cfg(
    config_path: Path,
    hdf5_override: Path,
    scalar_csv: Path | None,
) -> Tuple[Dict[str, Any], str]:
    from config_utils import anchor_data_paths_to_config_file, load_config

    cfg_path_s = str(config_path.resolve())
    cfg = load_config(cfg_path_s)
    cfg["data"]["hdf5_path"] = str(hdf5_override.resolve())
    if scalar_csv is not None:
        cfg["data"]["scalar_table_csv"] = str(scalar_csv.resolve())
    else:
        cfg["data"]["scalar_table_csv"] = None
    anchor_data_paths_to_config_file(cfg, cfg_path_s)
    return cfg, cfg_path_s


def _build_path_loss_generator(cfg: Dict[str, Any], device: Any, checkpoint: Path) -> Any:
    import torch
    from config_utils import is_cuda_device, load_torch_checkpoint
    from data_utils import compute_input_channels, compute_scalar_cond_dim, uses_scalar_film_conditioning
    from model_cgan import UNetGenerator

    hybrid_enabled = bool(dict(cfg.get("path_loss_hybrid", {})).get("enabled", False))
    sc_dim = int(compute_scalar_cond_dim(cfg)) if uses_scalar_film_conditioning(cfg) else 0
    film_h = int(cfg["model"].get("scalar_film_hidden", 128))
    gen_kw: Dict[str, Any] = dict(
        in_channels=compute_input_channels(cfg),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        path_loss_hybrid=hybrid_enabled,
        norm_type=str(cfg["model"].get("norm_type", "batch")),
        scalar_cond_dim=sc_dim,
        scalar_film_hidden=film_h,
    )
    sig = inspect.signature(UNetGenerator.__init__)
    if "upsample_mode" in sig.parameters:
        gen_kw["upsample_mode"] = str(cfg["model"].get("upsample_mode", "transpose"))
    if "bottleneck_attention" in sig.parameters:
        gen_kw["bottleneck_attention"] = bool(cfg["model"].get("bottleneck_attention", False))
    if "bottleneck_attention_dim" in sig.parameters:
        gen_kw["bottleneck_attention_dim"] = int(cfg["model"].get("bottleneck_attention_dim", 256))
    if "bottleneck_attention_heads" in sig.parameters:
        gen_kw["bottleneck_attention_heads"] = int(cfg["model"].get("bottleneck_attention_heads", 4))
    gen = UNetGenerator(**gen_kw).to(device)
    if not is_cuda_device(device):
        gen = gen.float()
    state = load_torch_checkpoint(str(checkpoint), device)
    gen.load_state_dict(state["generator"] if "generator" in state else state, strict=True)
    gen.eval()
    return gen


def predict_ninth_path_loss(
    try_root: Path,
    config_path: Path,
    checkpoint: Path,
    hdf5_override: Path,
    scalar_csv: Path | None,
    out_root: Path,
    split: str,
    limit: int | None,
    device_pref: str,
    output_label: str = "ninthtry9",
    path_loss_viz_scale: str = "gt",
    checkpoint_nlos: Optional[Path] = None,
    config_los: Optional[Path] = None,
    config_nlos: Optional[Path] = None,
) -> None:
    import torch
    from torch import amp

    _insert_try_path(try_root)
    from config_utils import resolve_device
    data_utils = importlib.import_module("data_utils")
    build_dataset_splits_from_config = getattr(data_utils, "build_dataset_splits_from_config")
    forward_cgan_generator = getattr(data_utils, "forward_cgan_generator")
    unpack_cgan_batch = getattr(data_utils, "unpack_cgan_batch")
    merge_hdf5_splits_for_inference = getattr(data_utils, "merge_hdf5_splits_for_inference", None)
    sample_is_los_dominant = getattr(data_utils, "sample_is_los_dominant", None)
    from evaluate_cgan import denormalize_channel

    device = pick_inference_device(device_pref, resolve_device)
    dual = checkpoint_nlos is not None
    if dual and not checkpoint.is_file():
        raise SystemExit("Dual LoS/NLoS: --ninth-checkpoint (LoS model) must exist.")
    if dual and not checkpoint_nlos.is_file():
        raise SystemExit("Dual LoS/NLoS: --ninth-checkpoint-nlos must exist.")

    path_los = config_los if config_los is not None else config_path
    path_nlos = config_nlos if config_nlos is not None else config_path

    cfg_los, _ = _prepare_path_loss_cfg(path_los, hdf5_override, scalar_csv)
    cfg_nlos, _ = _prepare_path_loss_cfg(path_nlos, hdf5_override, scalar_csv)

    if dual:
        cfg_ds = copy.deepcopy(cfg_los)
        cfg_ds.setdefault("data", {})["los_sample_filter"] = None
        active_cfg_for_ds = cfg_ds
        print(
            "[path_loss] Dual LoS/NLoS: LoS checkpoint + NLoS checkpoint; dataset uses **all** samples "
            "(los_sample_filter cleared). Classification from los_mask vs "
            f"{cfg_los.get('data', {}).get('los_classify_field', 'los_mask')}, "
            f"threshold={cfg_los.get('data', {}).get('los_classify_threshold', 0.5)}."
        )
        print(f"[path_loss]   LoS  config: {path_los}")
        print(f"[path_loss]   NLoS config: {path_nlos}")
        gen_los = _build_path_loss_generator(cfg_los, device, checkpoint)
        gen_nlos = _build_path_loss_generator(cfg_nlos, device, checkpoint_nlos)
    else:
        active_cfg_for_ds = cfg_los
        gen_los = _build_path_loss_generator(cfg_los, device, checkpoint)
        gen_nlos = None

    target_columns = list(cfg_los["target_columns"])
    if list(cfg_nlos["target_columns"]) != target_columns:
        warnings.warn("[path_loss] LoS vs NLoS target_columns differ; using LoS column list.")

    if split == "all":
        if callable(merge_hdf5_splits_for_inference):
            ds = merge_hdf5_splits_for_inference(active_cfg_for_ds)
        else:
            ds = merge_splits_with_fallback(build_dataset_splits_from_config, active_cfg_for_ds)
        split_dir = "all"
        print(f"[path_loss] split=all: {len(ds)} samples (train+val+test), augmentation off")
    else:
        splits = build_dataset_splits_from_config(active_cfg_for_ds)
        if split not in splits:
            raise SystemExit(
                f"Split {split!r} not in dataset splits {list(splits.keys())!r} "
                f"(e.g. test_ratio=0 removes 'test'). Use train, val, test, or all."
            )
        ds = splits[split]
        split_dir = split

    split_root = out_root / f"predictions_{output_label}_path_loss" / split_dir
    by_field_pl = split_root / "by_field" / "path_loss"
    by_field_pl.mkdir(parents=True, exist_ok=True)

    pl_meta = dict(cfg_los.get("target_metadata", {})).get("path_loss", {})
    print(
        "[path_loss] path_loss target_metadata (LoS yaml): "
        f"predict_linear={pl_meta.get('predict_linear', False)}, "
        f"scale={pl_meta.get('scale', 1.0)}, offset={pl_meta.get('offset', 0.0)}"
    )
    print(
        "[path_loss] Tip: use the **same** YAML as the Slurm/training run for each branch (--ninth-config / -los / -nlos)."
    )
    print(
        f"[path_loss] path_loss PNG scaling: {path_loss_viz_scale!r} — "
        "'gt'/'joint' use one dB range for both panels; 'independent' = per-panel min-max."
    )

    def denorm_path(t: "torch.Tensor", tmeta: Dict[str, Any]) -> np.ndarray:
        meta = tmeta.get("path_loss", {})
        x = denormalize_channel(t, meta)
        return np.asarray(x.detach().cpu().numpy(), dtype=np.float32)

    n = len(ds) if limit is None else min(len(ds), limit)
    amp_ok = bool(cfg_los["training"].get("amp", False)) and should_use_cuda_amp(device)
    h5p = str(hdf5_override.resolve())
    los_field = str(cfg_los.get("data", {}).get("los_classify_field", "los_mask"))
    los_th = float(cfg_los.get("data", {}).get("los_classify_threshold", 0.5))
    n_los_branch = 0
    n_nlos_branch = 0
    n_unknown = 0

    for idx in range(n):
        batch = ds[idx]
        inputs = [t.unsqueeze(0).to(device) for t in batch[:3]]
        if len(batch) == 4:
            sc = batch[3].unsqueeze(0).to(device)
            pack = (inputs[0], inputs[1], inputs[2], sc)
        else:
            pack = (inputs[0], inputs[1], inputs[2])
        city, sample = ds.sample_refs[idx]
        city_s, sample_s = _sanitize(city), _sanitize(sample)

        use_los_model = True
        if dual:
            if callable(sample_is_los_dominant):
                dom = sample_is_los_dominant(h5p, city, sample, los_field, los_th)
            else:
                dom = None
            if dom is None:
                n_unknown += 1
                if n_unknown == 1:
                    warnings.warn(
                        f"[path_loss] Could not classify LoS/NLoS for some samples (missing {los_field}?); "
                        "using LoS model.",
                        stacklevel=1,
                    )
                use_los_model = True
            else:
                use_los_model = bool(dom)
            pred_label_suffix = "LoS model" if use_los_model else "NLoS model"
            if use_los_model:
                n_los_branch += 1
            else:
                n_nlos_branch += 1
        else:
            pred_label_suffix = ""

        with torch.no_grad():
            x, y, m, sc_tensor = unpack_cgan_batch(pack, device)
            x = x.float()
            if sc_tensor is not None:
                sc_tensor = sc_tensor.float()
            with amp.autocast(device_type="cuda", enabled=amp_ok):
                if dual:
                    pred_los_t = forward_cgan_generator(gen_los, x, sc_tensor)
                    pred_nlos_t = forward_cgan_generator(gen_nlos, x, sc_tensor)
                else:
                    pred_los_t = forward_cgan_generator(gen_los, x, sc_tensor)
                    pred_nlos_t = None
            pred_los_t = pred_los_t.float()
            if pred_nlos_t is not None:
                pred_nlos_t = pred_nlos_t.float()

        pi = target_columns.index("path_loss")
        tmeta_gt = dict(cfg_los.get("target_metadata", {}))
        tgt_map = denorm_path(y[0, pi : pi + 1], tmeta_gt)[0]

        if dual:
            tmeta_los = dict(cfg_los.get("target_metadata", {}))
            tmeta_nlos = dict(cfg_nlos.get("target_metadata", {}))
            pred_los_map = denorm_path(pred_los_t[0, pi : pi + 1], tmeta_los)[0]
            pred_nlos_map = denorm_path(pred_nlos_t[0, pi : pi + 1], tmeta_nlos)[0]
            pred_map = pred_los_map if use_los_model else pred_nlos_map
        else:
            pred_map = denorm_path(pred_los_t[0, pi : pi + 1], dict(cfg_los.get("target_metadata", {})))[0]

        if idx == 0:
            print(
                "[path_loss] sample 0 denorm dB — "
                f"pred min/max/mean: {np.nanmin(pred_map):.3f}/{np.nanmax(pred_map):.3f}/{np.nanmean(pred_map):.3f}, "
                f"gt min/max/mean: {np.nanmin(tgt_map):.3f}/{np.nanmax(tgt_map):.3f}/{np.nanmean(tgt_map):.3f}"
            )

        viz = str(path_loss_viz_scale).lower().strip()
        dest = by_field_pl / city_s
        dest.mkdir(parents=True, exist_ok=True)

        if viz == "independent":
            pred_im = Image.fromarray(array_to_rgb_u8(pred_map, "path_loss"))
            tgt_im = Image.fromarray(array_to_rgb_u8(tgt_map, "path_loss"))
        else:
            lo, hi = _paired_map_viz_range(tgt_map, pred_map, viz)
            pred_im = Image.fromarray(array_to_rgb_u8(pred_map, "path_loss", lo, hi))
            tgt_im = Image.fromarray(array_to_rgb_u8(tgt_map, "path_loss", lo, hi))
        pred_title = "Pred path_loss (dB)" + (f" [{pred_label_suffix}]" if pred_label_suffix else "")
        hconcat_images([tgt_im, pred_im], ["GT path_loss (dB)", pred_title]).save(
            dest / f"{sample_s}_gt_pred.png"
        )

        if dual:
            # GT | NLoS | LoS | combined (same dB scale for fair comparison)
            quad_maps = (tgt_map, pred_nlos_map, pred_los_map, pred_map)
            if viz == "independent":
                q_ims = [Image.fromarray(array_to_rgb_u8(m, "path_loss")) for m in quad_maps]
            else:
                lo4, hi4 = _multi_map_viz_range(quad_maps, viz)
                q_ims = [
                    Image.fromarray(array_to_rgb_u8(m, "path_loss", lo4, hi4)) for m in quad_maps
                ]
            hconcat_images(
                q_ims,
                [
                    "GT path_loss (dB)",
                    "Pred NLoS model (dB)",
                    "Pred LoS model (dB)",
                    "Pred combined (routed)",
                ],
            ).save(dest / f"{sample_s}_gt_nlos_los_combined.png")
    extra = ""
    if dual:
        extra = f" (LoS model: {n_los_branch}, NLoS model: {n_nlos_branch}, unknown→LoS: {n_unknown})"
    quad_note = " Also *_gt_nlos_los_combined.png (4 cols: GT|NLoS|LoS|combined)." if dual else ""
    print(f"[path_loss] Saved {n} path_loss GT|Pred under {by_field_pl}/<city>/{extra}.{quad_note}")


def predict_spread_multichannel(
    try_root: Path,
    config_path: Path,
    checkpoint: Path,
    hdf5_override: Path,
    out_root: Path,
    split: str,
    limit: int | None,
    device_pref: str,
    run_label: str,
    output_label: Optional[str] = None,
    columns_only: Optional[Sequence[str]] = None,
) -> None:
    import torch
    from torch import amp

    _insert_try_path(try_root)
    from config_utils import (
        anchor_data_paths_to_config_file,
        is_cuda_device,
        load_config,
        load_torch_checkpoint,
        resolve_device,
    )
    data_utils = importlib.import_module("data_utils")
    build_dataset_splits_from_config = getattr(data_utils, "build_dataset_splits_from_config")
    compute_input_channels = getattr(data_utils, "compute_input_channels")
    compute_scalar_cond_dim = getattr(data_utils, "compute_scalar_cond_dim", None)
    forward_cgan_generator = getattr(data_utils, "forward_cgan_generator", None)
    unpack_cgan_batch = getattr(data_utils, "unpack_cgan_batch", None)
    uses_scalar_film_conditioning = getattr(data_utils, "uses_scalar_film_conditioning", None)
    merge_hdf5_splits_for_inference = getattr(data_utils, "merge_hdf5_splits_for_inference", None)
    from model_cgan import UNetGenerator

    cfg_path_s = str(config_path.resolve())
    cfg = load_config(cfg_path_s)
    cfg["data"]["hdf5_path"] = str(hdf5_override.resolve())
    anchor_data_paths_to_config_file(cfg, cfg_path_s)
    device = pick_inference_device(device_pref, resolve_device)

    target_columns = list(cfg["target_columns"])
    target_metadata = dict(cfg.get("target_metadata", {}))
    hybrid_enabled = bool(dict(cfg.get("path_loss_hybrid", {})).get("enabled", False))
    scalar_film_enabled = bool(callable(uses_scalar_film_conditioning) and uses_scalar_film_conditioning(cfg))
    scalar_cond_dim = int(compute_scalar_cond_dim(cfg)) if scalar_film_enabled and callable(compute_scalar_cond_dim) else 0
    gen_kw: Dict[str, Any] = dict(
        in_channels=compute_input_channels(cfg),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
    )
    sig = inspect.signature(UNetGenerator.__init__)
    if "path_loss_hybrid" in sig.parameters:
        gen_kw["path_loss_hybrid"] = hybrid_enabled
    if "norm_type" in sig.parameters:
        gen_kw["norm_type"] = str(cfg["model"].get("norm_type", "batch"))
    if "scalar_cond_dim" in sig.parameters:
        gen_kw["scalar_cond_dim"] = scalar_cond_dim
    if "scalar_film_hidden" in sig.parameters:
        gen_kw["scalar_film_hidden"] = int(cfg["model"].get("scalar_film_hidden", 128))
    if "upsample_mode" in sig.parameters:
        gen_kw["upsample_mode"] = str(cfg["model"].get("upsample_mode", "transpose"))
    gen = UNetGenerator(**gen_kw).to(device)
    if not is_cuda_device(device):
        gen = gen.float()
    state = load_torch_checkpoint(str(checkpoint), device)
    gen.load_state_dict(state["generator"] if "generator" in state else state, strict=True)
    gen.eval()

    if split == "all":
        if callable(merge_hdf5_splits_for_inference):
            ds = merge_hdf5_splits_for_inference(cfg)
        else:
            ds = merge_splits_with_fallback(build_dataset_splits_from_config, cfg)
        split_dir = "all"
        print(f"[{run_label}] split=all: {len(ds)} samples (train+val+test), augmentation off")
    else:
        splits = build_dataset_splits_from_config(cfg)
        if split not in splits:
            raise SystemExit(
                f"Split {split!r} not in dataset splits {list(splits.keys())!r}. Use train, val, test, or all."
            )
        ds = splits[split]
        split_dir = split
    export_label = str(output_label or run_label)
    split_root = out_root / f"predictions_{export_label}_delay_angular" / split_dir
    by_field_root = split_root / "by_field"
    by_field_root.mkdir(parents=True, exist_ok=True)

    export_cols = list(columns_only) if columns_only else list(target_columns)

    def denorm_channel(t: "torch.Tensor", col: str) -> np.ndarray:
        meta = target_metadata.get(col, {})
        scale = float(meta.get("scale", 1.0))
        offset = float(meta.get("offset", 0.0))
        return (t * scale + offset).detach().cpu().numpy()

    n = len(ds) if limit is None else min(len(ds), limit)
    amp_ok = bool(cfg["training"].get("amp", False)) and should_use_cuda_amp(device)
    for idx in range(n):
        batch = ds[idx]
        if callable(unpack_cgan_batch):
            items = list(batch) if isinstance(batch, (tuple, list)) else [batch]
            pack = tuple(t.unsqueeze(0).to(device) for t in items)
            x, y, m, sc_tensor = unpack_cgan_batch(pack, device)
        else:
            x = batch[0].unsqueeze(0).to(device)
            y = batch[1].unsqueeze(0).to(device)
            m = batch[2].unsqueeze(0).to(device) if len(batch) > 2 else None
            sc_tensor = batch[3].unsqueeze(0).to(device) if len(batch) > 3 else None
        city, sample = ds.sample_refs[idx]
        city_s, sample_s = _sanitize(city), _sanitize(sample)

        with torch.no_grad():
            x = x.float()
            if sc_tensor is not None:
                sc_tensor = sc_tensor.float()
            with amp.autocast(device_type="cuda", enabled=amp_ok):
                if callable(forward_cgan_generator):
                    pred = forward_cgan_generator(gen, x, sc_tensor)
                elif sc_tensor is not None:
                    pred = gen(x, sc_tensor)
                else:
                    pred = gen(x)
            pred = pred.float()

        for col in export_cols:
            if col not in target_columns:
                continue
            ci = target_columns.index(col)
            pred_map = np.asarray(denorm_channel(pred[0, ci : ci + 1], col)[0], dtype=np.float32)
            tgt_map = np.asarray(denorm_channel(y[0, ci : ci + 1], col)[0], dtype=np.float32)
            if col == "los_mask":
                pred_im = Image.fromarray(array_to_rgb_u8(pred_map, col))
                tgt_im = Image.fromarray(array_to_rgb_u8(tgt_map, col))
            else:
                lo, hi = _paired_map_viz_range(tgt_map, pred_map, "gt")
                pred_im = Image.fromarray(array_to_rgb_u8(pred_map, col, lo, hi))
                tgt_im = Image.fromarray(array_to_rgb_u8(tgt_map, col, lo, hi))
            unit = target_metadata.get(col, {}).get("unit", "")
            fdir = by_field_root / field_type_folder(col) / city_s
            fdir.mkdir(parents=True, exist_ok=True)
            hconcat_images([tgt_im, pred_im], [f"GT {col} ({unit})", f"Pred {col} ({unit})"]).save(
                fdir / f"{sample_s}_gt_pred.png"
            )
    print(f"[{run_label}] Saved GT|Pred by field under {by_field_root}/<field>/<city>/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export HDF5 maps to PNG + Ninth path loss + First/Second/Third delay/angular.")
    p.add_argument(
        "--hdf5",
        type=str,
        default=None,
        help=f"HDF5 (defecto: {DEFAULT_HDF5})",
    )
    p.add_argument("--dataset-out", type=str, default="D:/Dataset_Imagenes", help="Carpeta raíz de salida")
    p.add_argument(
        "--skip-dataset-export",
        action="store_true",
        help="Do not re-export raw_hdf5 from HDF5; only run model inference (GT|Pred PNGs). "
        "Requires existing dataset-out tree if you use build_alltogether_panel later.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test", "all"),
        help="Which HDF5 split to run inference on. "
        "'all' = train+val+test in one pass (same refs as training split, no augmentation). "
        "Outputs go under .../predictions_*/<split>/ (folder name 'all').",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (CUDA if available, else DirectML if torch-directml installed, else CPU) | cpu | cuda | directml",
    )

    p.add_argument(
        "--scalar-csv",
        type=str,
        default=None,
        help=(
            "Antenna height CSV (same as cluster Slurm SCALAR_CSV_PATH). "
            "Use with NinthTry9 path loss so scalar_feature_norms match training; "
            "still compatible with *_antenna_height.h5 (HDF5 uav_height per sample). "
            "If omitted and HDF5 filename contains 'antenna_height', CSV is not auto-used — "
            "norms are inferred from HDF5 maxima unless you pass this flag."
        ),
    )
    p.add_argument(
        "--ninth-root",
        "--path-loss-try-root",
        type=str,
        default=str(PRACTICE_ROOT / "TFGNinthTry9"),
        dest="ninth_root",
        help=(
            "Carpeta del try Python para path loss (sys.path: data_utils, model_cgan…). "
            "TFGNinthTry9 = un solo checkpoint predice todo el dataset; "
            "TFGTenthTry10 = típicamente dos runs (los_only / nlos_only) + --ninth-checkpoint-nlos. "
            "Alias: --path-loss-try-root."
        ),
    )
    p.add_argument(
        "--ninth-config",
        type=str,
        default=str(
            PRACTICE_ROOT
            / "TFGNinthTry9"
            / "configs"
            / "cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9.yaml"
        ),
    )
    p.add_argument("--ninth-checkpoint", type=str, default=None)
    p.add_argument(
        "--ninth-checkpoint-nlos",
        type=str,
        default=None,
        help="Second generator for NLoS-dominant samples (with --ninth-checkpoint for LoS). "
        "Enables dual inference: full split without los_sample_filter, each sample routed by los_mask. "
        "Use TenthTry10 los_film / nlos_film checkpoints + matching YAMLs via --ninth-config-los / --ninth-config-nlos.",
    )
    p.add_argument(
        "--ninth-config-los",
        type=str,
        default=None,
        help="YAML for LoS-trained model (default: --ninth-config). Only used with --ninth-checkpoint-nlos.",
    )
    p.add_argument(
        "--ninth-config-nlos",
        type=str,
        default=None,
        help="YAML for NLoS-trained model (default: --ninth-config). Only used with --ninth-checkpoint-nlos.",
    )
    p.add_argument(
        "--path-loss-viz-scale",
        type=str,
        default="gt",
        choices=("gt", "joint", "independent"),
        help="How to scale dB to gray for path_loss GT|Pred PNGs: "
        "'gt' = use GT min/max for both (default; fixes flat-pred + per-image min-max looking like static); "
        "'joint' = min/max over GT union Pred; 'independent' = old per-panel min-max.",
    )
    p.add_argument(
        "--path-loss-output-label",
        type=str,
        default="ninthtry9",
        help=(
            "Folder label for path loss exports: predictions_<label>_path_loss/<split>/... "
            "Default keeps the historical predictions_ninthtry9_path_loss layout."
        ),
    )

    p.add_argument(
        "--spread-try",
        type=str,
        default="auto",
        choices=("auto", "first", "second", "third"),
        help="Try para delay+angular: auto = mejor según JSONs en cluster_outputs; si no hay, third.",
    )
    p.add_argument(
        "--spread-root",
        type=str,
        default=None,
        help=(
            "Custom Python try root for delay/angular inference (overrides built-in first/second/third layout). "
            "Use together with --spread-config and optionally --spread-output-label."
        ),
    )
    p.add_argument("--spread-config", type=str, default=None, help="Sobrescribe el yaml del try spread.")
    p.add_argument("--spread-checkpoint", type=str, default=None)
    p.add_argument(
        "--spread-output-label",
        type=str,
        default=None,
        help=(
            "Folder label for delay/angular exports: predictions_<label>_delay_angular/<split>/... "
            "Default uses the built-in try label or the custom --spread-root folder name."
        ),
    )
    p.add_argument(
        "--spread-include-path-loss",
        action="store_true",
        help="También exporta path_loss del modelo 3-canales (por defecto solo delay_spread y angular_spread).",
    )
    p.add_argument(
        "--print-metric-hints",
        action="store_true",
        help="Imprime pistas desde validate_metrics (ninth + spread) y sale.",
    )
    return p.parse_args()


def ensure_torch_available() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "PyTorch (torch) is not available in this Python environment.\n"
            "  Activate the venv that has torch + torchvision, then rerun, e.g. PowerShell:\n"
            "    .\\.venv\\Scripts\\Activate.ps1\n"
            "  Or call the venv interpreter directly:\n"
            "    .venv\\Scripts\\python.exe scripts\\export_dataset_and_predictions.py ...\n"
            "  See scripts/run_export_local_assets.ps1 (activates .venv by default)."
        ) from e


def main() -> None:
    args = parse_args()

    hdf5_path = Path(args.hdf5) if args.hdf5 else DEFAULT_HDF5
    if not hdf5_path.is_file():
        raise SystemExit(f"HDF5 not found: {hdf5_path}")

    scalar_csv: Optional[Path]
    if args.scalar_csv:
        scalar_csv = Path(args.scalar_csv)
    elif "antenna_height" in hdf5_path.name.lower():
        scalar_csv = None  # NinthTry9 lee uav_height del H5
    else:
        scalar_csv = DEFAULT_SCALAR_CSV if DEFAULT_SCALAR_CSV.is_file() else None

    if args.print_metric_hints:
        ninth_best = CLUSTER_OUTPUTS / "TFGNinthTry9"
        nb = list(ninth_best.rglob("validate_metrics_cgan_best.json")) if ninth_best.is_dir() else []
        if nb:
            data = json.loads(nb[0].read_text(encoding="utf-8"))
            ep = (data.get("_checkpoint") or {}).get("epoch")
            pl = (data.get("path_loss") or {}).get("rmse_physical")
            print(f"[hints] NinthTry9 best JSON: {nb[0]}")
            print(f"        path_loss.rmse_physical (dB) ≈ {pl}, epoch ≈ {ep}")
        else:
            print("[hints] No validate_metrics_cgan_best.json under cluster_outputs/TFGNinthTry9")

        sk, sj, se, sc = pick_best_spread_try_from_metrics(CLUSTER_OUTPUTS, PRACTICE_ROOT)
        if sj is not None and sc is not None:
            print(f"[hints] Mejor try (delay+angular rmse sum): {sk} score={sc:.4f}")
            print(f"        JSON: {sj}")
            if se is not None:
                print(f"        época en ese JSON: {se} -> suele coincidir con epoch_{se}_cgan.pt o best_cgan.pt")
        else:
            print(
                "[hints] No hay JSONs con delay_spread + angular_spread en cluster_outputs ni en */outputs; "
                "usa --spread-try third (o first/second)."
            )
        return

    out_root = Path(args.dataset_out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_dataset_export:
        export_hdf5_dataset(hdf5_path, out_root)

    if args.ninth_checkpoint or args.ninth_checkpoint_nlos or args.spread_checkpoint:
        ensure_torch_available()

    if args.ninth_checkpoint_nlos and not args.ninth_checkpoint:
        raise SystemExit(
            "Dual LoS/NLoS: set --ninth-checkpoint (LoS model) together with --ninth-checkpoint-nlos."
        )

    if args.ninth_checkpoint:
        ck_nl = args.ninth_checkpoint_nlos
        predict_ninth_path_loss(
            Path(args.ninth_root),
            Path(args.ninth_config),
            Path(args.ninth_checkpoint),
            hdf5_path,
            scalar_csv.resolve() if scalar_csv is not None and scalar_csv.is_file() else None,
            out_root,
            args.split,
            args.limit,
            args.device,
            output_label=str(args.path_loss_output_label),
            path_loss_viz_scale=str(args.path_loss_viz_scale),
            checkpoint_nlos=Path(ck_nl) if ck_nl else None,
            config_los=Path(args.ninth_config_los) if args.ninth_config_los else None,
            config_nlos=Path(args.ninth_config_nlos) if args.ninth_config_nlos else None,
        )

    if args.spread_checkpoint:
        spread_key = args.spread_try
        if spread_key == "auto":
            auto_key, jpath, epoch, score = pick_best_spread_try_from_metrics(CLUSTER_OUTPUTS, PRACTICE_ROOT)
            if jpath is not None and score is not None:
                print(f"[spread] auto: elegido try={auto_key} (delay_rmse+angular_rmse={score:.4f}) según {jpath.name}")
            else:
                auto_key = "third"
                print("[spread] auto: sin métricas delay+angular en cluster_outputs -> fallback third try")
            spread_key = auto_key

        label, try_root, default_cfg = SPREAD_TRY_LAYOUT[spread_key]
        cfg_path = Path(args.spread_config) if args.spread_config else default_cfg
        cols: Optional[List[str]] = None
        if not args.spread_include_path_loss:
            cols = ["delay_spread", "angular_spread"]
        predict_spread_multichannel(
            try_root,
            cfg_path,
            Path(args.spread_checkpoint),
            hdf5_path,
            out_root,
            args.split,
            args.limit,
            args.device,
            label,
            columns_only=cols,
        )

    if not args.ninth_checkpoint and not args.spread_checkpoint:
        print(
            "[info] Solo export de dataset. Añade --ninth-checkpoint y/o --spread-checkpoint "
            "(y opcional --spread-try auto|first|second|third)."
        )


if __name__ == "__main__":
    main()
