#!/usr/bin/env python3
"""
Plot validation metrics vs epoch from cluster validate_metrics_epoch_*_cgan.json.

- Runs with only path_loss → one subplot.
- Runs with delay_spread, angular_spread, path_loss, … → one subplot per output (rmse_physical).

  python scripts/plot_cluster_validate_metrics.py --tries TFGNinthTry9,TFGTenthTry10
  python scripts/plot_cluster_validate_metrics.py --root cluster_outputs --out D:/plots_val
  python scripts/plot_cluster_validate_metrics.py --metric mae_physical
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent
DEFAULT_CLUSTER_ROOT = PRACTICE_ROOT / "cluster_outputs"
DEFAULT_OUT = PRACTICE_ROOT / "cluster_plots" / "validate_metrics"
EPOCH_JSON = re.compile(r"^validate_metrics_epoch_(\d+)_cgan\.json$", re.IGNORECASE)

# Subplot order; any other keys with rmse_physical are appended sorted
METRIC_ORDER: Tuple[str, ...] = ("path_loss", "delay_spread", "angular_spread")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot rmse_physical (or mae_physical) vs epoch per cluster run")
    p.add_argument("--root", type=str, default=str(DEFAULT_CLUSTER_ROOT), help="cluster_outputs (or similar)")
    p.add_argument(
        "--tries",
        type=str,
        default="TFGNinthTry9,TFGTenthTry10",
        help="Comma-separated try folders under --root",
    )
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output directory for PNGs")
    p.add_argument(
        "--metric",
        type=str,
        default="rmse_physical",
        choices=("rmse_physical", "mae_physical", "rmse", "mae"),
        help="Y-axis quantity inside each output block (default: rmse_physical)",
    )
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--show", action="store_true", help="Show interactive windows (also saves PNG)")
    return p.parse_args()


def _has_metric_block(v: Dict[str, Any], metric_name: str) -> bool:
    return metric_name in v and isinstance(v.get(metric_name), (int, float))


def extract_metric_blocks(data: Dict[str, Any], metric_name: str) -> Dict[str, Dict[str, Any]]:
    blocks: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        if not k.startswith("_"):
            if _has_metric_block(v, metric_name):
                blocks[k] = v
            continue
        if k == "_checkpoint":
            continue
        for subk, subv in v.items():
            if isinstance(subv, dict) and _has_metric_block(subv, metric_name):
                blocks[subk] = subv
    return blocks


def order_metric_names(names: Sequence[str]) -> List[str]:
    names_set = set(names)
    ordered: List[str] = []
    for m in METRIC_ORDER:
        if m in names_set:
            ordered.append(m)
    rest = sorted(names_set - set(ordered))
    ordered.extend(rest)
    return ordered


def discover_plottable_metrics(
    samples: List[Dict[str, Any]],
    y_key: str,
) -> List[str]:
    """Union of output names that have y_key in at least one epoch JSON."""
    found: set[str] = set()
    for data in samples:
        for name in extract_metric_blocks(data, y_key).keys():
            found.add(name)
    return order_metric_names(found)


def load_epoch_points(run_dir: Path) -> List[Tuple[int, Dict[str, Any]]]:
    pts: List[Tuple[int, Dict[str, Any]]] = []
    for jp in sorted(run_dir.glob("validate_metrics_epoch_*_cgan.json")):
        m = EPOCH_JSON.match(jp.name)
        if not m:
            continue
        ep = int(m.group(1))
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ck = data.get("_checkpoint") or {}
        if isinstance(ck.get("epoch"), int):
            ep = int(ck["epoch"])
        pts.append((ep, data))
    pts.sort(key=lambda x: x[0])
    return pts


def safe_filename_part(s: str, max_len: int = 120) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    return s[:max_len] if len(s) > max_len else s


def plot_one_run(
    try_name: str,
    run_name: str,
    points: List[Tuple[int, Dict[str, Any]]],
    metric_y: str,
    out_path: Path,
    dpi: int,
    do_show: bool,
) -> bool:
    if not points:
        return False

    metrics = discover_plottable_metrics([p[1] for p in points], metric_y)
    if not metrics:
        print(f"[skip] no {metric_y} in {try_name}/{run_name}", file=sys.stderr)
        return False

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    epochs = [p[0] for p in points]
    n = len(metrics)
    fig_h = max(3.2, 2.8 * n)
    fig, axes = plt.subplots(n, 1, figsize=(9, fig_h), squeeze=False, sharex=True)
    axes_flat = axes.ravel()

    for ax, mname in zip(axes_flat, metrics):
        ys: List[float] = []
        es: List[int] = []
        unit = ""
        for ep, data in points:
            block = extract_metric_blocks(data, metric_y).get(mname)
            if not isinstance(block, dict):
                ys.append(float("nan"))
                es.append(ep)
                continue
            val = block.get(metric_y)
            if not isinstance(val, (int, float)):
                ys.append(float("nan"))
            else:
                ys.append(float(val))
            es.append(ep)
            if not unit and isinstance(block.get("unit_physical"), str):
                unit = str(block["unit_physical"])
            elif not unit and isinstance(block.get("unit"), str):
                unit = str(block["unit"])

        ax.plot(es, ys, "-o", markersize=3, linewidth=1.2)
        ax.set_ylabel(f"{metric_y}\n({unit})" if unit else metric_y)
        ax.set_title(mname.replace("__", " / ").replace("_", " "))
        ax.grid(True, alpha=0.3)
        finite = [y for y in ys if y == y]  # not nan
        if finite:
            ax.axhline(min(finite), color="green", linestyle="--", alpha=0.35, linewidth=0.8)

    axes_flat[-1].set_xlabel("epoch")
    title = f"{try_name}\n{run_name}"
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    tries = [t.strip() for t in args.tries.split(",") if t.strip()]
    n_ok = 0
    for t in tries:
        try_dir = root / t
        if not try_dir.is_dir():
            print(f"[warn] missing try: {try_dir}", file=sys.stderr)
            continue
        for run_dir in sorted(try_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            pts = load_epoch_points(run_dir)
            if not pts:
                continue
            fname = safe_filename_part(run_dir.name) + f"_{args.metric}.png"
            sub_try = t if t not in (".", "") else "_flat_root"
            dest = out_root / sub_try / fname
            if plot_one_run(t, run_dir.name, pts, args.metric, dest, args.dpi, args.show):
                print(dest)
                n_ok += 1

    print(f"[done] wrote {n_ok} figures under {out_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
