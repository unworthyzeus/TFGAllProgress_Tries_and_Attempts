#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image, ImageDraw

from build_alltogether2_panel import _pick_font, _draw_multiline
from build_alltogether3 import error_heatmap, non_ground_mask_from_topology
from build_alltogether_panel import cell_with_label_bar, height_suffix_for_filename, prediction_map_from_gt_pred_png
from export_dataset_and_predictions import array_to_rgb_u8, _sanitize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Try 42 2x2 topology/GT/pred/error panels with antenna height.")
    p.add_argument("--hdf5", type=str, default="C:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5")
    p.add_argument("--pred-root", type=str, default="D:/Dataset_Imagenes/predictions_fortysecondtry42_path_loss/all/by_field/path_loss")
    p.add_argument("--output-dir", type=str, default="D:/Dataset_Imagenes/try42_topology_error_2x2")
    p.add_argument("--cell-w", type=int, default=320)
    p.add_argument("--cell-h", type=int, default=320)
    p.add_argument("--label-bar-h", type=int, default=56)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--city", type=str, default="")
    p.add_argument("--sample", type=str, default="")
    p.add_argument("--mask-dilate", type=int, default=1)
    return p.parse_args()


def _build_hdf5_index(hdf5_path: Path) -> Dict[Tuple[str, str], Tuple[str, str]]:
    idx: Dict[Tuple[str, str], Tuple[str, str]] = {}
    with h5py.File(hdf5_path, "r") as handle:
        for city in handle.keys():
            city_s = _sanitize(str(city))
            for sample in handle[city].keys():
                sample_s = _sanitize(str(sample))
                idx[(city_s, sample_s)] = (str(city), str(sample))
    return idx


def _iter_prediction_samples(pred_root: Path) -> List[Tuple[str, str, Path]]:
    out: List[Tuple[str, str, Path]] = []
    if not pred_root.is_dir():
        return out
    for city_dir in sorted(pred_root.iterdir()):
        if not city_dir.is_dir():
            continue
        for pred_png in sorted(city_dir.glob("*_gt_pred.png")):
            sample = pred_png.stem.replace("_gt_pred", "")
            out.append((city_dir.name, sample, pred_png))
    return out


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    out = mask.copy()
    for _ in range(radius):
        nbrs = [out]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nbrs.append(np.roll(np.roll(out, dy, axis=0), dx, axis=1))
        out = np.logical_or.reduce(nbrs)
    return out


def apply_visual_mask(im: Image.Image, mask: np.ndarray, fill_rgb: Tuple[int, int, int] = (18, 18, 18)) -> Image.Image:
    arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    arr[mask] = np.array(fill_rgb, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _load_gt_from_hdf5(hdf5_path: Path, city: str, sample: str) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    with h5py.File(hdf5_path, "r") as handle:
        grp = handle[city][sample]
        topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
        path_loss = np.asarray(grp["path_loss"][...], dtype=np.float32)
        uav_height = None
        if "uav_height" in grp:
            raw = np.asarray(grp["uav_height"][...], dtype=np.float32).reshape(-1)
            if raw.size:
                uav_height = float(raw[0])
    return topology, path_loss, uav_height


def _header(draw: ImageDraw.ImageDraw, width: int, city: str, sample: str, height_m: Optional[float]) -> None:
    if height_m is None or not np.isfinite(float(height_m)):
        htxt = "height: unknown"
    else:
        htxt = f"height: {float(height_m):.2f} m"
    title = f"Try 42 failure inspection\n{city}/{sample}\n{htxt}"
    font = _pick_font(title, width, 92, max_size=20)
    _draw_multiline(draw, width, 92, title, font, fill=(20, 24, 30))


def build_panel(topology: np.ndarray, gt_path_loss: np.ndarray, pred_gt_pred_png: Path, city_s: str, sample_s: str, uav_height_m: Optional[float], cell_w: int, cell_h: int, label_bar_h: int, mask_dilate: int) -> Image.Image:
    topology_im = Image.fromarray(array_to_rgb_u8(topology, "topology_map"), mode="RGB")
    gt_im = Image.fromarray(array_to_rgb_u8(gt_path_loss, "path_loss"), mode="RGB")
    pred_src = Image.open(pred_gt_pred_png).convert("RGB")
    pred_im = prediction_map_from_gt_pred_png(pred_src)
    masked = topology != 0
    masked = dilate_mask(masked, int(mask_dilate))
    gt_im = apply_visual_mask(gt_im, masked)
    pred_im = apply_visual_mask(pred_im, masked)
    err_im = error_heatmap(gt_im, pred_im, masked)

    tiles = [
        cell_with_label_bar(topology_im, "Topology", cell_w, cell_h, label_bar_h),
        cell_with_label_bar(gt_im, "Path loss GT", cell_w, cell_h, label_bar_h),
        cell_with_label_bar(pred_im, "Path loss prediction", cell_w, cell_h, label_bar_h),
        cell_with_label_bar(err_im, "Masked error", cell_w, cell_h, label_bar_h),
    ]

    cols = 2
    rows = 2
    header_h = 92
    panel_w = cols * cell_w
    panel_h = header_h + rows * (cell_h + label_bar_h)
    panel = Image.new("RGB", (panel_w, panel_h), (220, 222, 228))
    draw = ImageDraw.Draw(panel)
    _header(draw, panel_w, city_s, sample_s, uav_height_m)

    for i, tile in enumerate(tiles):
        x = (i % cols) * cell_w
        y = header_h + (i // cols) * (cell_h + label_bar_h)
        panel.paste(tile, (x, y))
    return panel


def main() -> None:
    args = parse_args()
    hdf5_path = Path(args.hdf5)
    pred_root = Path(args.pred_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not hdf5_path.is_file():
        raise SystemExit(f"HDF5 not found: {hdf5_path}")
    if not pred_root.is_dir():
        raise SystemExit(f"Prediction folder not found: {pred_root}")

    idx = _build_hdf5_index(hdf5_path)
    samples = _iter_prediction_samples(pred_root)
    if args.city:
        samples = [item for item in samples if item[0] == args.city]
    if args.sample:
        samples = [item for item in samples if item[1] == args.sample]
    if args.limit is not None:
        samples = samples[: int(args.limit)]

    written = 0
    for city_s, sample_s, pred_png in samples:
        key = (city_s, sample_s)
        if key not in idx:
            continue
        city_raw, sample_raw = idx[key]
        topology, gt_path_loss, uav_height_m = _load_gt_from_hdf5(hdf5_path, city_raw, sample_raw)
        panel = build_panel(
            topology,
            gt_path_loss,
            pred_png,
            city_s,
            sample_s,
            uav_height_m,
            int(args.cell_w),
            int(args.cell_h),
            int(args.label_bar_h),
            int(args.mask_dilate),
        )
        city_out = out_root / city_s
        city_out.mkdir(parents=True, exist_ok=True)
        htag = height_suffix_for_filename(uav_height_m)
        panel.save(city_out / f"{sample_s}_try42_topology_error_2x2_{htag}.png", quality=95)
        written += 1

    print(f"[try42-2x2] Wrote {written} panels under {out_root}")


if __name__ == "__main__":
    main()
