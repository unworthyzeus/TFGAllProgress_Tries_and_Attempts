#!/usr/bin/env python3
"""
Build a 3x5 panel (15 tiles) per sample under <data_root>/alltogether3/.

This version differs from alltogether2 in two ways:

- building pixels (topology != 0) are excluded from error visualization;
- the three-channel diagnostic composite is RGB, not YCbCr:
  - R = path_loss
  - G = angular_spread
  - B = delay_spread
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_alltogether2_panel import (
    DEFAULT_HDF5,
    DEFAULT_SCALAR_CSV,
    antenna_projection,
    build_antenna_height_from_csv,
    build_antenna_height_index_m,
    cell_with_label_bar,
    height_suffix_for_filename,
    iter_samples,
    load_rgb,
    lookup_uav_height_m,
    merge_csv_heights_into_index,
    prediction_map_from_gt_pred_png,
)


def _load_title_font(size: int):
    windir = os.environ.get("WINDIR", "C:/Windows")
    for name in ("segoeui.ttf", "arial.ttf", "calibri.ttf"):
        p = Path(windir) / "Fonts" / name
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _pick_font(text: str, width: int, height: int, max_size: int = 12):
    lines = text.split("\n")
    for sz in range(max_size, 7, -1):
        font = _load_title_font(sz)
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        max_w = 0
        total_h = 0
        for line in lines:
            box = probe.textbbox((0, 0), line, font=font)
            max_w = max(max_w, box[2] - box[0])
            total_h += box[3] - box[1]
        total_h += (len(lines) - 1) * 2
        if max_w <= width - 8 and total_h <= height - 6:
            return font
    return _load_title_font(8)


def _draw_multiline(draw: ImageDraw.ImageDraw, width: int, height: int, text: str, font, fill=(20, 24, 30)) -> None:
    lines = text.split("\n")
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    heights: List[int] = []
    for line in lines:
        box = probe.textbbox((0, 0), line, font=font)
        heights.append(box[3] - box[1])
    gap = 2
    total_h = sum(heights) + gap * max(0, len(lines) - 1)
    y = max(0, (height - total_h) // 2)
    for i, line in enumerate(lines):
        box = probe.textbbox((0, 0), line, font=font)
        tw = box[2] - box[0]
        x = max(0, (width - tw) // 2)
        draw.text((x, y), line, font=font, fill=fill)
        y += heights[i] + gap


def to_gray_array(im: Optional[Image.Image]) -> np.ndarray:
    if im is None:
        return np.zeros((513, 513), dtype=np.uint8)
    return np.asarray(im.convert("L"), dtype=np.uint8)


def non_ground_mask_from_topology(topology_im: Optional[Image.Image]) -> np.ndarray:
    topo = to_gray_array(topology_im)
    return topo != 0


def error_heatmap(gt_im: Optional[Image.Image], pred_im: Optional[Image.Image], non_ground_mask: Optional[np.ndarray]) -> Image.Image:
    gt = to_gray_array(gt_im).astype(np.float32)
    pred = to_gray_array(pred_im).astype(np.float32)
    err = np.abs(pred - gt)
    if non_ground_mask is not None:
        err = np.where(non_ground_mask, np.nan, err)
    valid = np.isfinite(err)
    max_err = float(np.nanmax(err)) if np.any(valid) else 0.0
    if max_err > 0.0:
        err = err / max_err
    err = np.nan_to_num(err, nan=0.0)
    stops = np.array(
        [
            [10, 12, 28],
            [48, 18, 59],
            [110, 34, 95],
            [197, 70, 63],
            [250, 166, 26],
            [255, 245, 214],
        ],
        dtype=np.float32,
    )
    pos = err * (len(stops) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    alpha = (pos - lo)[..., None]
    rgb = stops[lo] * (1.0 - alpha) + stops[hi] * alpha
    if non_ground_mask is not None:
        rgb[non_ground_mask] = np.array([18, 18, 18], dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def rgb_composite(path_loss_im: Optional[Image.Image], angular_im: Optional[Image.Image], delay_im: Optional[Image.Image], non_ground_mask: Optional[np.ndarray]) -> Image.Image:
    r = to_gray_array(path_loss_im)
    g = to_gray_array(angular_im)
    b = to_gray_array(delay_im)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    if non_ground_mask is not None:
        arr[non_ground_mask] = 0
    return Image.fromarray(arr, mode="RGB")


def build_panel(
    sample_dir: Path,
    city: str,
    sample: str,
    path_loss_root: Path,
    spread_root: Path,
    cell_w: int,
    cell_h: int,
    label_bar_h: int,
    uav_height_m: Optional[float],
) -> Image.Image:
    topology = load_rgb(sample_dir / "topology_map.png")
    los_mask = load_rgb(sample_dir / "los_mask.png")
    gt_path = load_rgb(sample_dir / "path_loss.png")
    gt_delay = load_rgb(sample_dir / "delay_spread.png")
    gt_ang = load_rgb(sample_dir / "angular_spread.png")
    non_ground_mask = non_ground_mask_from_topology(topology)

    pred_path_src = load_rgb(path_loss_root / city / f"{sample}_gt_pred.png")
    pred_delay_src = load_rgb(spread_root / "delay_spread" / city / f"{sample}_gt_pred.png")
    pred_ang_src = load_rgb(spread_root / "angular_spread" / city / f"{sample}_gt_pred.png")

    pred_path = prediction_map_from_gt_pred_png(pred_path_src) if pred_path_src is not None else None
    pred_delay = prediction_map_from_gt_pred_png(pred_delay_src) if pred_delay_src is not None else None
    pred_ang = prediction_map_from_gt_pred_png(pred_ang_src) if pred_ang_src is not None else None

    err_path = error_heatmap(gt_path, pred_path, non_ground_mask)
    err_delay = error_heatmap(gt_delay, pred_delay, non_ground_mask)
    err_ang = error_heatmap(gt_ang, pred_ang, non_ground_mask)
    gt_rgb = rgb_composite(gt_path, gt_ang, gt_delay, non_ground_mask)
    pred_rgb = rgb_composite(pred_path, pred_ang, pred_delay, non_ground_mask)
    err_rgb = rgb_composite(err_path, err_ang, err_delay, non_ground_mask)
    antenna = antenna_projection(topology, uav_height_m, cell_w, cell_h)

    titles = [
        "Topology\n(input map)",
        "LoS\n(mask)",
        "Antenna height\npseudo-3D view",
        "Path loss\nGT",
        "Path loss\nprediction",
        "Path loss\nmasked error",
        "Delay spread\nGT",
        "Delay spread\nprediction",
        "Delay spread\nmasked error",
        "Angular spread\nGT",
        "Angular spread\nprediction",
        "Angular spread\nmasked error",
        "RGB GT\nR:path G:ang B:delay",
        "RGB prediction\nR:path G:ang B:delay",
        "RGB error\nR:path G:ang B:delay",
    ]
    images = [
        topology,
        los_mask,
        antenna,
        gt_path,
        pred_path,
        err_path,
        gt_delay,
        pred_delay,
        err_delay,
        gt_ang,
        pred_ang,
        err_ang,
        gt_rgb,
        pred_rgb,
        err_rgb,
    ]

    cols = 5
    rows = 3
    row_h = label_bar_h + cell_h
    header_h = 116
    grid_w = cols * cell_w
    grid_h = header_h + rows * row_h
    grid = Image.new("RGB", (grid_w, grid_h), (220, 222, 228))

    draw = ImageDraw.Draw(grid)
    header = (
        "alltogether3: masked-building diagnostics with RGB composite\n"
        "3x5 panel with GT, prediction, masked error, RGB fusion, and antenna-height context\n"
        f"sample: {city}/{sample}"
    )
    if uav_height_m is not None and np.isfinite(float(uav_height_m)):
        header += f"\nUAV / antenna height: {float(uav_height_m):.4f} m"
    font = _pick_font(header, grid_w, header_h - 4, max_size=12)
    _draw_multiline(draw, grid_w, header_h, header, font)
    draw.line((0, header_h - 1, grid_w, header_h - 1), fill=(140, 145, 155), width=1)

    for idx, (title, im) in enumerate(zip(titles, images)):
        row = idx // cols
        col = idx % cols
        cell = cell_with_label_bar(im, title, cell_w, cell_h, label_bar_h)
        grid.paste(cell, (col * cell_w, header_h + row * row_h))

    return grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build alltogether3 masked-building RGB panels.")
    p.add_argument("--data-root", type=str, default="D:/Dataset_Imagenes")
    p.add_argument("--split", type=str, default="all", choices=("train", "val", "test", "all"))
    p.add_argument("--path-loss-label", type=str, default="thirtythirdtry33")
    p.add_argument("--spread-label", type=str, default="thirtysixthtry36")
    p.add_argument("--cell-w", type=int, default=256)
    p.add_argument("--cell-h", type=int, default=256)
    p.add_argument("--label-bar-h", type=int, default=54)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose-paths", action="store_true")
    p.add_argument("--hdf5", type=str, default=str(DEFAULT_HDF5))
    p.add_argument("--scalar-csv", type=str, default=str(DEFAULT_SCALAR_CSV))
    p.add_argument("--prefer-hdf5-uav-height", action="store_true")
    p.add_argument("--output-dir-name", type=str, default="alltogether3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).resolve()
    by_sample = root / "raw_hdf5" / "by_sample"
    path_loss_root = root / f"predictions_{args.path_loss_label}_path_loss" / args.split / "by_field" / "path_loss"
    spread_root = root / f"predictions_{args.spread_label}_delay_angular" / args.split / "by_field"
    out_root = root / args.output_dir_name

    if not by_sample.is_dir():
        raise SystemExit(f"Missing dataset export: {by_sample}")
    if not path_loss_root.is_dir():
        raise SystemExit(f"Missing path loss predictions: {path_loss_root}")
    if not spread_root.is_dir():
        raise SystemExit(f"Missing delay/angular predictions: {spread_root}")

    hdf5_path = Path(args.hdf5)
    height_idx = build_antenna_height_index_m(hdf5_path) if hdf5_path.is_file() else {}
    csv_path = Path(args.scalar_csv)
    if csv_path.is_file():
        if args.prefer_hdf5_uav_height:
            merge_csv_heights_into_index(height_idx, csv_path, fill_missing_only=True)
        else:
            height_idx.update(build_antenna_height_from_csv(csv_path))

    samples = iter_samples(by_sample)
    if args.limit is not None:
        samples = samples[: args.limit]

    for idx, (city, sample, sample_dir) in enumerate(samples):
        if args.verbose_paths and idx == 0:
            print(f"[alltogether3] path_loss: {path_loss_root / city / f'{sample}_gt_pred.png'}")
            print(f"[alltogether3] delay:     {spread_root / 'delay_spread' / city / f'{sample}_gt_pred.png'}")
            print(f"[alltogether3] angular:   {spread_root / 'angular_spread' / city / f'{sample}_gt_pred.png'}")

        uav_height_m = lookup_uav_height_m(height_idx, city, sample)
        panel = build_panel(
            sample_dir=sample_dir,
            city=city,
            sample=sample,
            path_loss_root=path_loss_root,
            spread_root=spread_root,
            cell_w=args.cell_w,
            cell_h=args.cell_h,
            label_bar_h=args.label_bar_h,
            uav_height_m=uav_height_m,
        )
        city_out = out_root / city
        city_out.mkdir(parents=True, exist_ok=True)
        suffix = height_suffix_for_filename(uav_height_m)
        panel.save(city_out / f"{sample}_3x5_{suffix}.png")

    print(f"[alltogether3] Saved {len(samples)} panels under {out_root}")


if __name__ == "__main__":
    main()
