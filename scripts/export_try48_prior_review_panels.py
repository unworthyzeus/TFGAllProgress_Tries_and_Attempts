#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PRACTICE_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Try 48 prior review panels for hard validation samples.")
    p.add_argument("--try-dir", default=str(PRACTICE_ROOT / "TFGFortyEighthTry48"))
    p.add_argument(
        "--config",
        default="experiments/fortyeighthtry48_pmnet_prior_gan/fortyeighthtry48_pmnet_prior_gan.yaml",
    )
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--scan-limit", type=int, default=120, help="Number of samples to scan before selecting the hardest ones.")
    p.add_argument("--top-k", type=int, default=6, help="Number of panels to export.")
    p.add_argument("--output-dir", default=str(PRACTICE_ROOT / "TFGFortyEighthTry48" / "prior_review"))
    return p.parse_args()


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _denormalize(values: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    return values.astype(np.float32, copy=False) * scale + offset


def _formula_channel_index(cfg: Dict[str, Any]) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):
        idx += 1
    if cfg["data"].get("distance_map_channel", False):
        idx += 1
    return idx


def _edge_band(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    edge = np.zeros_like(m, dtype=bool)
    edge[1:, :] |= m[1:, :] != m[:-1, :]
    edge[:-1, :] |= m[1:, :] != m[:-1, :]
    edge[:, 1:] |= m[:, 1:] != m[:, :-1]
    edge[:, :-1] |= m[:, 1:] != m[:, :-1]
    # thicken a little for a stable local boundary band
    band = edge.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(edge)
            ys = slice(max(0, dy), edge.shape[0] + min(0, dy))
            xs = slice(max(0, dx), edge.shape[1] + min(0, dx))
            yd = slice(max(0, -dy), edge.shape[0] + min(0, -dy))
            xd = slice(max(0, -dx), edge.shape[1] + min(0, -dx))
            shifted[yd, xd] = edge[ys, xs]
            band |= shifted
    return band


def _gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    gy = np.zeros_like(arr, dtype=np.float32)
    gx = np.zeros_like(arr, dtype=np.float32)
    gy[1:-1, :] = 0.5 * (arr[2:, :] - arr[:-2, :])
    gx[:, 1:-1] = 0.5 * (arr[:, 2:] - arr[:, :-2])
    return np.sqrt(gx * gx + gy * gy)


def _radial_ring_std(arr: np.ndarray, los_mask: np.ndarray) -> float:
    h, w = arr.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = rr.astype(np.int32)
    max_bin = int(bins.max())
    means: List[float] = []
    for b in range(max_bin + 1):
        mask = (bins == b) & los_mask
        if np.count_nonzero(mask) < 8:
            continue
        means.append(float(np.mean(arr[mask])))
    if len(means) < 11:
        return float("nan")
    vec = np.asarray(means, dtype=np.float32)
    kernel = np.ones(9, dtype=np.float32) / 9.0
    smooth = np.convolve(vec, kernel, mode="same")
    residual = vec - smooth
    return float(np.std(residual))


def _compute_metrics(prior_db: np.ndarray, gt_db: np.ndarray, los_mask: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    err = prior_db - gt_db
    los_valid = valid_mask & los_mask
    nlos_valid = valid_mask & (~los_mask)
    band = _edge_band(los_mask) & valid_mask
    gt_grad = _gradient_magnitude(gt_db)
    prior_grad = _gradient_magnitude(prior_db)

    def rmse(mask: np.ndarray) -> float:
        vals = err[mask]
        if vals.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(vals * vals)))

    def mean(mask: np.ndarray, arr: np.ndarray) -> float:
        vals = arr[mask]
        if vals.size == 0:
            return float("nan")
        return float(np.mean(vals))

    return {
        "overall_rmse_db": rmse(valid_mask),
        "los_rmse_db": rmse(los_valid),
        "nlos_rmse_db": rmse(nlos_valid),
        "mean_prior_nlos_db": mean(nlos_valid, prior_db),
        "mean_gt_nlos_db": mean(nlos_valid, gt_db),
        "nlos_bias_db": mean(nlos_valid, err),
        "gt_ring_std_db": _radial_ring_std(gt_db, los_valid),
        "prior_ring_std_db": _radial_ring_std(prior_db, los_valid),
        "gt_boundary_grad_db": mean(band, gt_grad),
        "prior_boundary_grad_db": mean(band, prior_grad),
        "valid_pixel_count": int(np.count_nonzero(valid_mask)),
        "los_ratio_valid": float(np.mean(los_mask[valid_mask])) if np.any(valid_mask) else float("nan"),
    }


def _norm_gray(arr: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    lo = float(np.min(a) if lo is None else lo)
    hi = float(np.max(a) if hi is None else hi)
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.uint8)
    out = ((a - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    return out


def _signed_error_rgb(arr: np.ndarray, clip_db: float = 30.0) -> np.ndarray:
    a = np.clip(arr.astype(np.float32, copy=False), -clip_db, clip_db) / clip_db
    rgb = np.zeros(a.shape + (3,), dtype=np.uint8)
    pos = np.clip(a, 0.0, 1.0)
    neg = np.clip(-a, 0.0, 1.0)
    rgb[..., 0] = (255.0 * pos + 255.0 * (1.0 - pos - neg)).clip(0, 255).astype(np.uint8)
    rgb[..., 1] = (255.0 * (1.0 - pos - neg)).clip(0, 255).astype(np.uint8)
    rgb[..., 2] = (255.0 * neg + 255.0 * (1.0 - pos - neg)).clip(0, 255).astype(np.uint8)
    return rgb


def _as_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2)


def _resize_rgb(arr: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(arr, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def _add_label(image: Image.Image, title: str) -> Image.Image:
    label_height = 30
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, label_height), fill=(240, 240, 240))
    draw.text((8, 9), title, fill=(0, 0, 0), font=font)
    return canvas


def _make_panel(
    topology: np.ndarray,
    los_mask: np.ndarray,
    gt_db: np.ndarray,
    prior_db: np.ndarray,
    valid_mask: np.ndarray,
    preview_size: int = 300,
) -> Image.Image:
    err = prior_db - gt_db
    masked_err = np.where(valid_mask, err, 0.0)
    shared_lo = float(min(np.min(gt_db[valid_mask]), np.min(prior_db[valid_mask]))) if np.any(valid_mask) else 0.0
    shared_hi = float(max(np.max(gt_db[valid_mask]), np.max(prior_db[valid_mask]))) if np.any(valid_mask) else 180.0
    tiles = [
        _add_label(_resize_rgb(_as_rgb(_norm_gray(topology)), preview_size), "topology"),
        _add_label(_resize_rgb(_as_rgb(_norm_gray(los_mask.astype(np.float32), 0.0, 1.0)), preview_size), "los_mask"),
        _add_label(_resize_rgb(_as_rgb(_norm_gray(gt_db, shared_lo, shared_hi)), preview_size), "gt_path_loss"),
        _add_label(_resize_rgb(_as_rgb(_norm_gray(prior_db, shared_lo, shared_hi)), preview_size), "prior_path_loss"),
        _add_label(_resize_rgb(_signed_error_rgb(masked_err, clip_db=30.0), preview_size), "prior_minus_gt"),
        _add_label(_resize_rgb(_as_rgb(_norm_gray(np.abs(masked_err), 0.0, 30.0)), preview_size), "abs_error"),
    ]

    cols = 3
    rows = 2
    tile_w = tiles[0].width
    tile_h = tiles[0].height
    panel = Image.new("RGB", (cols * tile_w, rows * tile_h), (225, 225, 225))
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        panel.paste(tile, (c * tile_w, r * tile_h))
    return panel


def main() -> None:
    args = parse_args()
    try_dir = _resolve(PRACTICE_ROOT, args.try_dir).resolve()
    config_path = _resolve(try_dir, args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(try_dir))
    config_utils = __import__("config_utils")
    data_utils = __import__("data_utils")

    cfg = config_utils.load_config(str(config_path))
    config_utils.anchor_data_paths_to_config_file(cfg, str(config_path))
    cfg["augmentation"] = dict(cfg.get("augmentation", {}))
    cfg["augmentation"]["enable"] = False
    splits = data_utils.build_dataset_splits_from_config(cfg)
    dataset = splits[args.split]
    target_meta = dict(cfg["target_metadata"]["path_loss"])
    formula_idx = _formula_channel_index(cfg)
    los_idx = 1 if cfg["data"].get("los_input_column") else None

    rows: List[Dict[str, Any]] = []
    scan_limit = min(int(args.scan_limit), len(dataset))
    for idx in range(scan_limit):
        city, sample = dataset.sample_refs[idx]
        x, y, m = dataset[idx][:3]
        topology = x[0].detach().cpu().numpy()
        los_mask = (x[los_idx].detach().cpu().numpy() > 0.5) if los_idx is not None else np.ones_like(topology, dtype=bool)
        prior_db = _denormalize(x[formula_idx].detach().cpu().numpy(), target_meta)
        gt_db = _denormalize(y[0].detach().cpu().numpy(), target_meta)
        dataset_valid_mask = m[0].detach().cpu().numpy() > 0.0
        topology_non_ground = topology > 0.0
        overlap_valid_non_ground = dataset_valid_mask & topology_non_ground
        # Be explicit in the review panels: never score building pixels even if the
        # dataset mask definition changes in the future.
        valid_mask = dataset_valid_mask & (~topology_non_ground)
        metrics = _compute_metrics(prior_db, gt_db, los_mask, valid_mask)
        rows.append(
            {
                "city": city,
                "sample": sample,
                "index": idx,
                "metrics": metrics,
                "topology": topology,
                "los_mask": los_mask,
                "prior_db": prior_db,
                "gt_db": gt_db,
                "valid_mask": valid_mask,
                "dataset_valid_pixels": int(np.count_nonzero(dataset_valid_mask)),
                "topology_non_ground_pixels": int(np.count_nonzero(topology_non_ground)),
                "valid_non_ground_overlap_pixels": int(np.count_nonzero(overlap_valid_non_ground)),
            }
        )

    hardest = sorted(
        rows,
        key=lambda row: (
            np.nan_to_num(row["metrics"]["nlos_rmse_db"], nan=-1.0),
            np.nan_to_num(row["metrics"]["overall_rmse_db"], nan=-1.0),
        ),
        reverse=True,
    )[: max(int(args.top_k), 1)]

    summary_rows: List[Dict[str, Any]] = []
    for rank, row in enumerate(hardest, start=1):
        safe_name = f"{rank:02d}_{row['city'].replace(' ', '_')}_{row['sample']}"
        panel_path = output_dir / f"{safe_name}_panel.png"
        panel = _make_panel(
            topology=row["topology"],
            los_mask=row["los_mask"],
            gt_db=row["gt_db"],
            prior_db=row["prior_db"],
            valid_mask=row["valid_mask"],
        )
        panel.save(panel_path)
        summary_rows.append(
            {
                "rank": rank,
                "city": row["city"],
                "sample": row["sample"],
                "index": row["index"],
                "panel_path": str(panel_path),
                "dataset_valid_pixels": row["dataset_valid_pixels"],
                "topology_non_ground_pixels": row["topology_non_ground_pixels"],
                "valid_non_ground_overlap_pixels": row["valid_non_ground_overlap_pixels"],
                **row["metrics"],
            }
        )

    payload = {
        "try_dir": str(try_dir),
        "config": str(config_path),
        "split": args.split,
        "scan_limit": scan_limit,
        "top_k": len(summary_rows),
        "formula": cfg["data"]["path_loss_formula_input"]["formula"],
        "hardest_samples": summary_rows,
    }
    (output_dir / "prior_review_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Try 48 Prior Visual Review",
        "",
        f"- Formula: `{payload['formula']}`",
        f"- Split: `{payload['split']}`",
        f"- Samples scanned: `{payload['scan_limit']}`",
        f"- Panels exported: `{payload['top_k']}`",
        "",
        "## Hardest Samples",
        "",
    ]
    for row in summary_rows:
        md_lines.append(
            f"- `{row['city']}/{row['sample']}`: overall `{row['overall_rmse_db']:.4f} dB`, "
            f"LoS `{row['los_rmse_db']:.4f} dB`, NLoS `{row['nlos_rmse_db']:.4f} dB`, "
            f"NLoS bias `{row['nlos_bias_db']:.4f} dB`, "
            f"valid/non-ground overlap `{row['valid_non_ground_overlap_pixels']}` px, "
            f"ring std GT/prior `{row['gt_ring_std_db']:.4f}/{row['prior_ring_std_db']:.4f}`, "
            f"boundary grad GT/prior `{row['gt_boundary_grad_db']:.4f}/{row['prior_boundary_grad_db']:.4f}`, "
            f"panel `{row['panel_path']}`"
        )
    (output_dir / "prior_review_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
