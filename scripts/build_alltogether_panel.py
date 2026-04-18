#!/usr/bin/env python3
"""
Build a 2×4 panel (8 tiles) per sample under <data_root>/alltogether/.
Each tile has a top label bar naming the map (readable text; system font on Windows).

Row 1 — inputs / context (from dataset, raw_hdf5/by_sample):
  [topology_map] [los_mask] [path_loss GT] [delay_spread GT]

Row 2 — predictions from *_gt_pred.png: right half of the export, with the top label bar
  stripped so raster size matches dataset GT PNGs (see prediction_map_from_gt_pred_png).
  [angular_spread GT] [path_loss pred] [delay_spread pred] [angular_spread pred]

Prediction paths must match export_dataset_and_predictions.py output layout.

  python scripts/build_alltogether_panel.py --data-root D:/Dataset_Imagenes --split test

If predictions show "Missing image", usually:
  - **Wrong --split** (export used `--split all` but panel defaults to `test`, or the reverse).
  - **Wrong --spread-label** (folder is `predictions_<label>_delay_angular`, e.g. `secondtry2` from `--spread-try second`).
  - Path loss was exported but **delay/angular** was not (`--spread-checkpoint` missing).

Use `--spread-label auto` to pick a matching `predictions_*_delay_angular` folder under data-root.

Filename `sample_2x4_h56p6469m.png`: **2x4** = two rows × four tiles (not pixel size); **h56p6469m** = UAV height for the filename (HDF5 `uav_height` and/or CSV `antenna_height_m`).

  # optional: --hdf5 … --scalar-csv … (default CSV fills gaps if HDF5 has no uav_height for a sample)
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent
DEFAULT_HDF5 = PRACTICE_ROOT / "Datasets" / "CKM_Dataset_180326_antenna_height.h5"
DEFAULT_SCALAR_CSV = PRACTICE_ROOT / "Datasets" / "CKM_180326_antenna_height.csv"


def _sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def load_rgb(path: Path) -> Optional[Image.Image]:
    if not path.is_file():
        return None
    return Image.open(path).convert("RGB")


def resize_cover(im: Image.Image, w: int, h: int) -> Image.Image:
    """Scale preserving aspect ratio; center-crop to fill w×h."""
    im = im.copy()
    src_w, src_h = im.size
    scale = max(w / src_w, h / src_h)
    nw, nh = int(src_w * scale + 0.5), int(src_h * scale + 0.5)
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    return im.crop((left, top, left + w, top + h))


# Must match scripts/export_dataset_and_predictions.py hconcat_images(..., label_h=26)
GT_PRED_LABEL_BAR_PX = 26


def prediction_map_from_gt_pred_png(full: Image.Image) -> Image.Image:
    """
    export_dataset_and_predictions saves *_gt_pred.png as hconcat(GT, Pred) with a shared
    top label bar (GT_PRED_LABEL_BAR_PX). Dataset GT PNGs have no bar — without stripping,
    predictions look shorter/taller and show a gray band under the tile title.
    """
    w, h = full.size
    if w < 4:
        return full
    # Export uses two panels of equal width (same tensor -> same RGB size).
    half = w // 2
    right = full.crop((half, 0, w, h))
    lh = min(GT_PRED_LABEL_BAR_PX, max(0, right.height - 1))
    if right.height > lh:
        return right.crop((0, lh, right.width, right.height))
    return right


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


def _pick_font_for_bar(text: str, bar_w: int, bar_h: int, max_size: int = 16):
    """Shrink font until the text block fits in the bar."""
    lines = text.split("\n")
    for sz in range(max_size, 7, -1):
        font = _load_title_font(sz)
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        max_line_w = 0
        total_h = 0
        for line in lines:
            b = draw.textbbox((0, 0), line, font=font)
            max_line_w = max(max_line_w, b[2] - b[0])
            total_h += b[3] - b[1]
        total_h += (len(lines) - 1) * 2
        if max_line_w <= bar_w - 8 and total_h <= bar_h - 6:
            return font
    return _load_title_font(8)


def _draw_multiline_centered(
    draw: ImageDraw.ImageDraw,
    bar_w: int,
    bar_h: int,
    text: str,
    font,
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    lines = text.split("\n")
    heights: List[int] = []
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for line in lines:
        b = draw_probe.textbbox((0, 0), line, font=font)
        heights.append(b[3] - b[1])
    gap = 2
    total_h = sum(heights) + gap * max(0, len(lines) - 1)
    y0 = max(0, (bar_h - total_h) // 2)
    y = y0
    for i, line in enumerate(lines):
        b = draw.textbbox((0, 0), line, font=font)
        tw = b[2] - b[0]
        x = max(0, (bar_w - tw) // 2)
        draw.text((x, y), line, font=font, fill=fill)
        y += heights[i] + gap


def cell_with_label_bar(
    im: Optional[Image.Image],
    title: str,
    cell_w: int,
    cell_h: int,
    label_bar_h: int,
) -> Image.Image:
    """One tile: top label bar + image below."""
    total_h = label_bar_h + cell_h
    canvas = Image.new("RGB", (cell_w, total_h), (40, 42, 48))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, cell_w, label_bar_h), fill=(52, 55, 65))
    font = _pick_font_for_bar(title, cell_w, label_bar_h)
    _draw_multiline_centered(draw, cell_w, label_bar_h, title, font)

    y_img = label_bar_h
    if im is not None:
        tile = resize_cover(im, cell_w, cell_h)
        canvas.paste(tile, (0, y_img))
    else:
        draw.rectangle((0, y_img, cell_w - 1, total_h - 1), fill=(160, 160, 168))
        miss_font = _load_title_font(14)
        msg = "Missing image"
        b = draw.textbbox((0, 0), msg, font=miss_font)
        tw, th = b[2] - b[0], b[3] - b[1]
        draw.text(
            ((cell_w - tw) // 2, y_img + (cell_h - th) // 2),
            msg,
            font=miss_font,
            fill=(60, 60, 60),
        )
    return canvas


def iter_samples(by_sample_root: Path) -> List[Tuple[str, str, Path]]:
    """List (city, sample, sample_dir) for folders that contain topology_map.png."""
    out: List[Tuple[str, str, Path]] = []
    if not by_sample_root.is_dir():
        return out
    for city_dir in sorted(by_sample_root.iterdir()):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        for sample_dir in sorted(city_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            if (sample_dir / "topology_map.png").is_file():
                out.append((city, sample_dir.name, sample_dir))
    return out


def build_antenna_height_index_m(hdf5_path: Path) -> Dict[Tuple[str, str], float]:
    """
    Index (sanitized_city, sanitized_sample) -> antenna height in metres from uav_height in HDF5.
    """
    import h5py

    idx: Dict[Tuple[str, str], float] = {}
    if not hdf5_path.is_file():
        return idx
    with h5py.File(hdf5_path, "r") as handle:
        for city in handle.keys():
            cg = handle[city]
            if not isinstance(cg, h5py.Group):
                continue
            for sample in cg.keys():
                grp = cg[sample]
                if not isinstance(grp, h5py.Group):
                    continue
                if "uav_height" not in grp or not isinstance(grp["uav_height"], h5py.Dataset):
                    continue
                try:
                    raw = np.asarray(grp["uav_height"][...], dtype=np.float64)
                    arr = raw.reshape(-1)
                except (OSError, TypeError, ValueError):
                    continue
                if arr.size == 0:
                    continue
                # Scalar or 1D (or flattened map): one nominal UAV / antenna height per sample
                v = float(np.nanmean(arr))
                if not np.isfinite(v):
                    continue
                key = (_sanitize(str(city)), _sanitize(str(sample)))
                idx[key] = v
    return idx


def build_antenna_height_from_csv(csv_path: Path) -> Dict[Tuple[str, str], float]:
    """
    CKM-style CSV: columns city, sample, antenna_height_m (same as cluster SCALAR_CSV).
    Keys match export folders: sanitized city / sample names.
    """
    idx: Dict[Tuple[str, str], float] = {}
    if not csv_path.is_file():
        return idx
    with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return idx
        fields = {f.strip().lower(): f for f in reader.fieldnames if f}
        col_city = fields.get("city")
        col_sample = fields.get("sample")
        col_h = fields.get("antenna_height_m")
        if not col_city or not col_sample or not col_h:
            return idx
        for row in reader:
            try:
                city = str(row.get(col_city) or "").strip()
                sample = str(row.get(col_sample) or "").strip()
                h_raw = row.get(col_h)
                if not city or not sample or h_raw is None or str(h_raw).strip() == "":
                    continue
                v = float(h_raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            idx[(_sanitize(city), _sanitize(sample))] = v
    return idx


def merge_csv_heights_into_index(
    idx: Dict[Tuple[str, str], float],
    csv_path: Path,
    fill_missing_only: bool = True,
) -> int:
    """Merge CSV heights into idx. Returns number of keys added (or updated if fill_missing_only=False)."""
    merged = build_antenna_height_from_csv(csv_path)
    n = 0
    for k, v in merged.items():
        if fill_missing_only:
            if k not in idx:
                idx[k] = v
                n += 1
        else:
            if idx.get(k) != v:
                idx[k] = v
                n += 1
    return n


def build_scene_stats_index(hdf5_path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Index scene-level stats from los_mask/topology_map in HDF5."""
    import h5py

    idx: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not hdf5_path.is_file():
        return idx

    with h5py.File(hdf5_path, "r") as handle:
        for city in handle.keys():
            cg = handle[city]
            if not isinstance(cg, h5py.Group):
                continue
            for sample in cg.keys():
                grp = cg[sample]
                if not isinstance(grp, h5py.Group):
                    continue
                if "los_mask" not in grp or "topology_map" not in grp:
                    continue
                try:
                    los = np.asarray(grp["los_mask"][...], dtype=np.float32)
                    topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
                except (OSError, TypeError, ValueError):
                    continue
                if los.shape != topo.shape or los.size == 0:
                    continue

                valid = np.isfinite(los) & np.isfinite(topo)
                n_valid = int(valid.sum())
                if n_valid <= 0:
                    continue

                los_m = los > 0.5
                building_m = topo > 0.0

                los_pct = float(100.0 * np.mean(los_m[valid]))
                nlos_pct = float(100.0 - los_pct)
                building_pct = float(100.0 * np.mean(building_m[valid]))
                non_building_pct = float(100.0 - building_pct)

                b_valid = building_m & valid
                mean_building_h = float(np.mean(topo[b_valid])) if np.any(b_valid) else float("nan")

                key = (_sanitize(str(city)), _sanitize(str(sample)))
                idx[key] = {
                    "los_pct": los_pct,
                    "nlos_pct": nlos_pct,
                    "building_pct": building_pct,
                    "non_building_pct": non_building_pct,
                    "mean_building_height_m": mean_building_h,
                }
    return idx


def lookup_uav_height_m(
    height_idx: Dict[Tuple[str, str], float],
    city: str,
    sample: str,
) -> Optional[float]:
    """Resolve UAV / antenna height for by_sample folder names (sanitized)."""
    key = (city, sample)
    if key in height_idx:
        return height_idx[key]
    c_l, s_l = city.lower(), sample.lower()
    for (kc, ks), v in height_idx.items():
        if kc.lower() == c_l and ks.lower() == s_l:
            return v
    return None


def lookup_scene_stats(
    stats_idx: Dict[Tuple[str, str], Dict[str, float]],
    city: str,
    sample: str,
) -> Optional[Dict[str, float]]:
    key = (city, sample)
    if key in stats_idx:
        return stats_idx[key]
    c_l, s_l = city.lower(), sample.lower()
    for (kc, ks), stats in stats_idx.items():
        if kc.lower() == c_l and ks.lower() == s_l:
            return stats
    return None


def discover_spread_labels(data_root: Path, split: str) -> List[str]:
    """Labels `xyz` such that predictions_xyz_delay_angular/<split>/by_field exists."""
    out: List[str] = []
    for p in sorted(data_root.glob("predictions_*_delay_angular")):
        if not p.is_dir():
            continue
        if (p / split / "by_field").is_dir():
            name = p.name
            prefix, suffix = "predictions_", "_delay_angular"
            if name.startswith(prefix) and name.endswith(suffix):
                out.append(name[len(prefix) : -len(suffix)])
    return out


def resolve_spread_label(arg: str, data_root: Path, split: str) -> str:
    if arg != "auto":
        return arg
    labels = discover_spread_labels(data_root, split)
    if not labels:
        raise SystemExit(
            f"No predictions_*_delay_angular/{split}/by_field under {data_root}. "
            "Export with --spread-checkpoint (and same --split), or fix --split."
        )
    priority = ("secondtry2", "thirdtry3", "firsttry1")
    for pref in priority:
        if pref in labels:
            print(f"[alltogether] --spread-label auto -> {pref!r} (available: {labels})")
            return pref
    chosen = sorted(labels)[0]
    print(f"[alltogether] --spread-label auto -> {chosen!r} (available: {labels})")
    return chosen


def warn_path_loss_split_mismatch(root: Path, split: str, ninth_pl: Path) -> None:
    """If chosen split has no path_loss PNGs, suggest splits that do."""
    if ninth_pl.is_dir() and any(ninth_pl.rglob("*.png")):
        return
    alts: List[str] = []
    for sub in ("test", "val", "train", "all"):
        p = root / "predictions_ninthtry9_path_loss" / sub / "by_field" / "path_loss"
        if p.is_dir() and any(p.rglob("*.png")):
            alts.append(sub)
    if not alts:
        print(f"[warn] No path_loss PNGs under predictions_ninthtry9_path_loss/*/by_field/path_loss")
        return
    print(
        f"[warn] No path_loss PNGs for --split {split!r} (looked at {ninth_pl}). "
        f"Splits that have exports: {alts}. Example: --split {alts[0]}"
    )


def height_suffix_for_filename(height_m: Optional[float]) -> str:
    """Filename-safe tag, e.g. h120p5m for 120.5 m."""
    if height_m is None or not np.isfinite(height_m):
        return "h_unknown_m"
    # avoid dots in filenames for portability
    txt = f"{float(height_m):.4f}".rstrip("0").rstrip(".")
    return f"h{txt.replace('.', 'p')}m"


def build_panel_for_sample(
    sample_dir: Path,
    city: str,
    sample: str,
    ninth_pl: Path,
    spread_root: Path,
    cell_w: int,
    cell_h: int,
    label_bar_h: int,
    uav_height_m: Optional[float] = None,
    map_size_px: Optional[Tuple[int, int]] = None,
    scene_stats: Optional[Dict[str, float]] = None,
) -> Image.Image:
    """Build 2×4 grid with a readable title on each tile."""
    # Row 0: inputs / GT context
    t0 = load_rgb(sample_dir / "topology_map.png")
    t1 = load_rgb(sample_dir / "los_mask.png")
    t2 = load_rgb(sample_dir / "path_loss.png")
    t3 = load_rgb(sample_dir / "delay_spread.png")

    pred_pl_path = ninth_pl / city / f"{sample}_gt_pred.png"
    pred_delay_path = spread_root / "delay_spread" / city / f"{sample}_gt_pred.png"
    pred_ang_path = spread_root / "angular_spread" / city / f"{sample}_gt_pred.png"

    t4 = load_rgb(sample_dir / "angular_spread.png")

    p_pl = load_rgb(pred_pl_path)
    p_delay = load_rgb(pred_delay_path)
    p_ang = load_rgb(pred_ang_path)

    if p_pl is not None:
        p_pl = prediction_map_from_gt_pred_png(p_pl)
    if p_delay is not None:
        p_delay = prediction_map_from_gt_pred_png(p_delay)
    if p_ang is not None:
        p_ang = prediction_map_from_gt_pred_png(p_ang)

    row0_titles = [
        "Topology\n(input map)",
        "LoS\n(line-of-sight mask)",
        "Path loss\n(ground truth, dB)",
        "Delay spread\n(ground truth, ns)",
    ]
    row1_titles = [
        "Angular spread\n(ground truth, deg)",
        "Path loss\n(prediction, dB)",
        "Delay spread\n(prediction, ns)",
        "Angular spread\n(prediction, deg)",
    ]
    row0_ims = [t0, t1, t2, t3]
    row1_ims = [t4, p_pl, p_delay, p_ang]

    row_h = label_bar_h + cell_h
    header_h = 118  # several header lines (UAV height + map px + legend)
    grid_w = 4 * cell_w
    grid_h = header_h + 2 * row_h
    grid = Image.new("RGB", (grid_w, grid_h), (220, 222, 228))
    hdr = ImageDraw.Draw(grid)
    if uav_height_m is not None and bool(np.isfinite(float(uav_height_m))):
        h_txt = (
            f"UAV / antenna height: {float(uav_height_m):.4f} m "
            f"(from CKM_180326_antenna_height.csv antenna_height_m when available, else HDF5 uav_height)"
        )
    else:
        h_txt = (
            "UAV / antenna height: not found — pass matching --hdf5 and/or --scalar-csv "
            f"(sample {city}/{sample})"
        )
    ms_txt = (
        f"Map raster (topology PNG): {map_size_px[0]} x {map_size_px[1]} px"
        if map_size_px
        else "Map raster: (topology missing)"
    )
    panel_header = (
        "Row 1: topology, LoS, and ground-truth reference maps\n"
        "Row 2: angular spread (GT) and model predictions (path loss, delay spread, angular spread)\n"
        f"{h_txt}\n"
        + (
            "LoS/NLoS: "
            f"{scene_stats.get('los_pct', float('nan')):.2f}% / {scene_stats.get('nlos_pct', float('nan')):.2f}% | "
            "Buildings/non-buildings: "
            f"{scene_stats.get('building_pct', float('nan')):.2f}% / {scene_stats.get('non_building_pct', float('nan')):.2f}% | "
            "Mean building height: "
            f"{scene_stats.get('mean_building_height_m', float('nan')):.2f} m\n"
            if scene_stats is not None
            else "LoS/NLoS + building stats: unavailable\n"
        )
        f"{ms_txt}\n"
        "Filename tag 2x4 = this 2-by-4 tile grid; h…m suffix = same UAV height rounded for the file name."
    )
    hdr_font = _pick_font_for_bar(panel_header, grid_w, header_h - 8, max_size=12)
    _draw_multiline_centered(hdr, grid_w, header_h, panel_header, hdr_font, fill=(25, 28, 35))
    # separator under header
    hdr.line((0, header_h - 1, grid_w, header_h - 1), fill=(140, 145, 155), width=1)

    y0 = header_h
    for c in range(4):
        cell = cell_with_label_bar(row0_ims[c], row0_titles[c], cell_w, cell_h, label_bar_h)
        grid.paste(cell, (c * cell_w, y0))
    for c in range(4):
        cell = cell_with_label_bar(row1_ims[c], row1_titles[c], cell_w, cell_h, label_bar_h)
        grid.paste(cell, (c * cell_w, y0 + row_h))

    return grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Panel 2×4 (5 inputs context + 3 preds) → alltogether/")
    p.add_argument("--data-root", type=str, default="D:/Dataset_Imagenes", help="Root with raw_hdf5 and predictions_*")
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test", "all"),
        help="Must match the split used in export_dataset_and_predictions (path under predictions_* / by_field).",
    )
    p.add_argument(
        "--spread-label",
        type=str,
        default="auto",
        help="Folder predictions_<label>_delay_angular (e.g. secondtry2). "
        "Default 'auto' picks an existing folder for this --split (priority: secondtry2, thirdtry3, firsttry1).",
    )
    p.add_argument(
        "--verbose-paths",
        action="store_true",
        help="Print expected prediction paths for the first sample (debug Missing image).",
    )
    p.add_argument("--cell-w", type=int, default=320)
    p.add_argument("--cell-h", type=int, default=320)
    p.add_argument(
        "--label-bar-h",
        type=int,
        default=56,
        help="Height of the label bar above each tile",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--hdf5",
        type=str,
        default=str(DEFAULT_HDF5),
        help="HDF5 with uav_height per sample (for panel text and PNG filename suffix)",
    )
    p.add_argument(
        "--scalar-csv",
        type=str,
        default=str(DEFAULT_SCALAR_CSV),
        help=(
            "CKM / Slurm table: columns city,sample,antenna_height_m (e.g. Nagpur,sample_09802,183.71…). "
            "By default these values **override** HDF5 uav_height for the same (city,sample). "
            "Point to a missing path to skip CSV."
        ),
    )
    p.add_argument(
        "--prefer-hdf5-uav-height",
        action="store_true",
        help="If set, use HDF5 uav_height when present; CSV only fills (city,sample) missing from HDF5.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).resolve()
    by_sample = root / "raw_hdf5" / "by_sample"
    spread_label = resolve_spread_label(str(args.spread_label).strip(), root, args.split)
    ninth_pl = root / "predictions_ninthtry9_path_loss" / args.split / "by_field" / "path_loss"
    spread_base = root / f"predictions_{spread_label}_delay_angular" / args.split / "by_field"

    if not by_sample.is_dir():
        raise SystemExit(f"Missing dataset export: {by_sample}")

    warn_path_loss_split_mismatch(root, args.split, ninth_pl)
    if not spread_base.is_dir():
        print(f"[warn] Spread predictions folder missing: {spread_base}")
        alt = discover_spread_labels(root, args.split)
        if alt:
            print(f"[warn] For --split {args.split!r} these spread labels exist: {alt}. Try --spread-label {alt[0]}")

    hdf5_path = Path(args.hdf5)
    height_idx = build_antenna_height_index_m(hdf5_path)
    scene_stats_idx = build_scene_stats_index(hdf5_path)
    if not height_idx and hdf5_path.is_file():
        print(f"[warn] HDF5 found but no uav_height entries indexed: {hdf5_path}")
    elif not hdf5_path.is_file():
        print(f"[warn] HDF5 not found ({hdf5_path}); trying CSV only for UAV height")

    csv_path = Path(args.scalar_csv)
    try:
        csv_is_default = csv_path.resolve() == DEFAULT_SCALAR_CSV.resolve()
    except OSError:
        csv_is_default = False
    if csv_path.is_file():
        csv_idx = build_antenna_height_from_csv(csv_path)
        n_csv = len(csv_idx)
        if args.prefer_hdf5_uav_height:
            n_add = merge_csv_heights_into_index(height_idx, csv_path, fill_missing_only=True)
            if n_add:
                print(
                    f"[alltogether] UAV heights: CSV filled {n_add} keys missing in HDF5 "
                    f"({csv_path.name}); HDF5 kept when both had the sample."
                )
        elif n_csv:
            n_h5 = len(height_idx)
            height_idx.update(csv_idx)
            print(
                f"[alltogether] UAV heights: {n_h5} from HDF5 uav_height; "
                f"applied {n_csv} CSV rows from {csv_path.name} (**CSV wins** on same city/sample)"
            )
    elif str(args.scalar_csv).strip() and not csv_is_default:
        print(f"[warn] --scalar-csv not found ({csv_path})")

    out_root = root / "alltogether"
    out_root.mkdir(parents=True, exist_ok=True)

    samples = iter_samples(by_sample)
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]

    n_ok = 0
    for city, sample, sample_dir in samples:
        if args.verbose_paths and n_ok == 0:
            pp = ninth_pl / city / f"{sample}_gt_pred.png"
            pd = spread_base / "delay_spread" / city / f"{sample}_gt_pred.png"
            pa = spread_base / "angular_spread" / city / f"{sample}_gt_pred.png"
            print(f"[alltogether] First sample {city}/{sample} — try:")
            print(f"  path_loss:  {pp}  exists={pp.is_file()}")
            print(f"  delay_spread: {pd}  exists={pd.is_file()}")
            print(f"  angular_spread: {pa}  exists={pa.is_file()}")

        h_m = lookup_uav_height_m(height_idx, city, sample)
        map_wh: Optional[Tuple[int, int]] = None
        tp = sample_dir / "topology_map.png"
        if tp.is_file():
            try:
                with Image.open(tp) as im:
                    map_wh = im.size
            except OSError:
                map_wh = None

        panel = build_panel_for_sample(
            sample_dir,
            city,
            sample,
            ninth_pl,
            spread_base,
            args.cell_w,
            args.cell_h,
            args.label_bar_h,
            uav_height_m=h_m,
            map_size_px=map_wh,
            scene_stats=lookup_scene_stats(scene_stats_idx, city, sample),
        )
        dest_dir = out_root / city
        dest_dir.mkdir(parents=True, exist_ok=True)
        htag = height_suffix_for_filename(h_m)
        dest = dest_dir / f"{sample}_2x4_{htag}.png"
        panel.save(dest, quality=95)
        n_ok += 1

    print(f"[alltogether] Wrote {n_ok} panels under {out_root} (expected inputs under {by_sample})")


if __name__ == "__main__":
    main()
