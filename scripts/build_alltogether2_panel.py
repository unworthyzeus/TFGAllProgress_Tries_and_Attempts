#!/usr/bin/env python3
"""
Build a 3x5 panel (15 tiles) per sample under <data_root>/alltogether2/.

Tiles:
  1. topology
  2. LoS mask
  3. antenna height pseudo-3D projection
  4. path loss GT
  5. path loss prediction
  6. path loss error
  7. delay spread GT
  8. delay spread prediction
  9. delay spread error
 10. angular spread GT
 11. angular spread prediction
 12. angular spread error
 13. GT YCbCr composite (Y=path_loss, Cb=angular, Cr=delay)
 14. prediction YCbCr composite
 15. error YCbCr composite

It uses already exported raw_hdf5 PNGs and prediction GT|Pred PNGs. It does not rerun HDF5 export.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_alltogether_panel import (
    DEFAULT_HDF5,
    DEFAULT_SCALAR_CSV,
    build_antenna_height_from_csv,
    build_antenna_height_index_m,
    build_scene_stats_index,
    cell_with_label_bar,
    height_suffix_for_filename,
    iter_samples,
    load_rgb,
    lookup_scene_stats,
    lookup_uav_height_m,
    merge_csv_heights_into_index,
    prediction_map_from_gt_pred_png,
    resize_cover,
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


def error_heatmap(gt_im: Optional[Image.Image], pred_im: Optional[Image.Image]) -> Image.Image:
    gt = to_gray_array(gt_im).astype(np.float32)
    pred = to_gray_array(pred_im).astype(np.float32)
    err = np.abs(pred - gt)
    if float(err.max()) > 0.0:
        err = err / float(err.max())
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
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def ycbcr_composite(path_loss_im: Optional[Image.Image], angular_im: Optional[Image.Image], delay_im: Optional[Image.Image]) -> Image.Image:
    y = to_gray_array(path_loss_im)
    cb = to_gray_array(angular_im)
    cr = to_gray_array(delay_im)
    arr = np.stack([y, cb, cr], axis=-1).astype(np.uint8)
    return Image.fromarray(arr, mode="YCbCr").convert("RGB")


def antenna_projection(topology_im: Optional[Image.Image], height_m: Optional[float], cell_w: int, cell_h: int) -> Image.Image:
    canvas = Image.new("RGB", (cell_w, cell_h), (228, 232, 238))
    draw = ImageDraw.Draw(canvas)

    topo = topology_im if topology_im is not None else Image.new("RGB", (513, 513), (160, 160, 160))
    plane = resize_cover(topo, int(cell_w * 0.8), int(cell_h * 0.5)).convert("RGB")
    plane = plane.rotate(45, resample=Image.Resampling.BICUBIC, expand=True)
    plane = plane.resize((int(plane.width * 0.92), int(plane.height * 0.45)), Image.Resampling.BICUBIC)

    px = (cell_w - plane.width) // 2
    py = int(cell_h * 0.46)
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.ellipse(
        (cell_w * 0.38, cell_h * 0.73, cell_w * 0.62, cell_h * 0.84),
        fill=(0, 0, 0, 30),
    )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow).convert("RGB")
    canvas.paste(plane, (px, py))

    center_x = cell_w // 2
    center_y = py + plane.height // 2 - 12
    if height_m is None or not np.isfinite(float(height_m)):
        pillar_h = int(cell_h * 0.22)
        label = "height: unknown"
    else:
        pillar_h = int(np.interp(float(height_m), [10.0, 400.0], [int(cell_h * 0.16), int(cell_h * 0.42)]))
        label = f"height: {float(height_m):.2f} m"

    draw.line((center_x, center_y, center_x, center_y - pillar_h), fill=(22, 28, 36), width=4)
    drone_y = center_y - pillar_h
    draw.ellipse((center_x - 8, drone_y - 8, center_x + 8, drone_y + 8), fill=(220, 68, 55), outline=(255, 255, 255), width=2)
    draw.line((center_x - 15, drone_y, center_x + 15, drone_y), fill=(255, 255, 255), width=2)
    draw.line((center_x, drone_y - 15, center_x, drone_y + 15), fill=(255, 255, 255), width=2)

    font = _load_title_font(12)
    box = draw.textbbox((0, 0), label, font=font)
    tw = box[2] - box[0]
    draw.rounded_rectangle((cell_w - tw - 14, 10, cell_w - 8, 30), radius=8, fill=(52, 55, 65))
    draw.text((cell_w - tw - 11, 14), label, fill=(255, 255, 255), font=font)
    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build alltogether2 3x5 panels using Try 28 + Try 26 prediction PNGs.")
    p.add_argument("--data-root", type=str, default="D:/Dataset_Imagenes")
    p.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("train", "val", "test", "all"),
    )
    p.add_argument("--path-loss-label", type=str, default="twentyeighthtry28")
    p.add_argument("--spread-label", type=str, default="twentysixthtry26")
    p.add_argument("--cell-w", type=int, default=256)
    p.add_argument("--cell-h", type=int, default=256)
    p.add_argument("--label-bar-h", type=int, default=54)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose-paths", action="store_true")
    p.add_argument("--hdf5", type=str, default=str(DEFAULT_HDF5))
    p.add_argument("--scalar-csv", type=str, default=str(DEFAULT_SCALAR_CSV))
    p.add_argument("--prefer-hdf5-uav-height", action="store_true")
    p.add_argument("--output-dir-name", type=str, default="alltogether2")
    return p.parse_args()


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
    scene_stats: Optional[dict[str, float]],
) -> Image.Image:
    topology = load_rgb(sample_dir / "topology_map.png")
    los_mask = load_rgb(sample_dir / "los_mask.png")
    gt_path = load_rgb(sample_dir / "path_loss.png")
    gt_delay = load_rgb(sample_dir / "delay_spread.png")
    gt_ang = load_rgb(sample_dir / "angular_spread.png")

    pred_path_src = load_rgb(path_loss_root / city / f"{sample}_gt_pred.png")
    pred_delay_src = load_rgb(spread_root / "delay_spread" / city / f"{sample}_gt_pred.png")
    pred_ang_src = load_rgb(spread_root / "angular_spread" / city / f"{sample}_gt_pred.png")

    pred_path = prediction_map_from_gt_pred_png(pred_path_src) if pred_path_src is not None else None
    pred_delay = prediction_map_from_gt_pred_png(pred_delay_src) if pred_delay_src is not None else None
    pred_ang = prediction_map_from_gt_pred_png(pred_ang_src) if pred_ang_src is not None else None

    err_path = error_heatmap(gt_path, pred_path)
    err_delay = error_heatmap(gt_delay, pred_delay)
    err_ang = error_heatmap(gt_ang, pred_ang)
    gt_ycbcr = ycbcr_composite(gt_path, gt_ang, gt_delay)
    pred_ycbcr = ycbcr_composite(pred_path, pred_ang, pred_delay)
    err_ycbcr = ycbcr_composite(err_path, err_ang, err_delay)
    antenna = antenna_projection(topology, uav_height_m, cell_w, cell_h)

    titles = [
        "Topology\n(input map)",
        "LoS\n(mask)",
        "Antenna height\npseudo-3D view",
        "Path loss\nGT",
        "Path loss\nprediction",
        "Path loss\nabs. error",
        "Delay spread\nGT",
        "Delay spread\nprediction",
        "Delay spread\nabs. error",
        "Angular spread\nGT",
        "Angular spread\nprediction",
        "Angular spread\nabs. error",
        "YCbCr GT\nY:path Cb:ang Cr:delay",
        "YCbCr prediction\nY:path Cb:ang Cr:delay",
        "YCbCr error\nY:path Cb:ang Cr:delay",
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
        gt_ycbcr,
        pred_ycbcr,
        err_ycbcr,
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
        "alltogether2: Try 28 path loss + Try 26 delay/angular\n"
        "3x5 panel with GT, prediction, per-target error, YCbCr fusion, and antenna-height context\n"
        f"sample: {city}/{sample}"
    )
    if uav_height_m is not None and np.isfinite(float(uav_height_m)):
        header += f"\nUAV / antenna height: {float(uav_height_m):.4f} m"
    if scene_stats is not None:
        header += (
            "\nLoS/NLoS: "
            f"{scene_stats.get('los_pct', float('nan')):.2f}% / {scene_stats.get('nlos_pct', float('nan')):.2f}%"
            " | Buildings/non-buildings: "
            f"{scene_stats.get('building_pct', float('nan')):.2f}% / {scene_stats.get('non_building_pct', float('nan')):.2f}%"
            " | Mean building height: "
            f"{scene_stats.get('mean_building_height_m', float('nan')):.2f} m"
        )
    else:
        header += "\nLoS/NLoS + building stats: unavailable"
    font = _pick_font(header, grid_w, header_h - 4, max_size=12)
    _draw_multiline(draw, grid_w, header_h, header, font)
    draw.line((0, header_h - 1, grid_w, header_h - 1), fill=(140, 145, 155), width=1)

    for idx, (title, im) in enumerate(zip(titles, images)):
        row = idx // cols
        col = idx % cols
        cell = cell_with_label_bar(im, title, cell_w, cell_h, label_bar_h)
        grid.paste(cell, (col * cell_w, header_h + row * row_h))

    return grid


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
    scene_stats_idx = build_scene_stats_index(hdf5_path) if hdf5_path.is_file() else {}
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
            print(f"[alltogether2] path_loss: {path_loss_root / city / f'{sample}_gt_pred.png'}")
            print(f"[alltogether2] delay:     {spread_root / 'delay_spread' / city / f'{sample}_gt_pred.png'}")
            print(f"[alltogether2] angular:   {spread_root / 'angular_spread' / city / f'{sample}_gt_pred.png'}")

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
            scene_stats=lookup_scene_stats(scene_stats_idx, city, sample),
        )
        city_out = out_root / city
        city_out.mkdir(parents=True, exist_ok=True)
        suffix = height_suffix_for_filename(uav_height_m)
        panel.save(city_out / f"{sample}_3x5_{suffix}.png")

    print(f"[alltogether2] Saved {len(samples)} panels under {out_root}")


if __name__ == "__main__":
    main()
