#!/usr/bin/env python3
"""
Scan cluster_outputs for NinthTry9 / TenthTry10 (or custom tries) and print
path_loss RMSE in physical dB for each validate_metrics_epoch_*_cgan.json.

Example:
  python scripts/summarize_path_loss_rmse_by_epoch.py
  python scripts/summarize_path_loss_rmse_by_epoch.py --tries TFGNinthTry9
  python scripts/summarize_path_loss_rmse_by_epoch.py --root D:/exports/cluster_outputs --format tsv
  python scripts/summarize_path_loss_rmse_by_epoch.py --summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_ROOT = SCRIPT_DIR.parent
DEFAULT_CLUSTER_ROOT = PRACTICE_ROOT / "cluster_outputs"
EPOCH_JSON = re.compile(r"^validate_metrics_epoch_(\d+)_cgan\.json$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="path_loss rmse_physical (dB) per run and epoch from cluster JSONs")
    p.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_CLUSTER_ROOT),
        help=f"Folder containing TFG*Try* (default: {DEFAULT_CLUSTER_ROOT})",
    )
    p.add_argument(
        "--tries",
        type=str,
        default="TFGNinthTry9,TFGTenthTry10",
        help="Comma-separated try directory names under --root",
    )
    p.add_argument(
        "--format",
        choices=("csv", "tsv", "table"),
        default="csv",
        help="Output format (default csv). Ignored if --summary.",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Per model_run: best/worst RMSE, epoch of best, mean, std, mean of last 5 epochs.",
    )
    p.add_argument(
        "--last-k",
        type=int,
        default=5,
        help="With --summary, trailing-epoch mean uses this many epochs (default 5).",
    )
    return p.parse_args()


def load_rmse_physical(path: Path) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """Returns (rmse_physical_db, epoch, eval_split)."""
    try:
        data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, None
    pl = data.get("path_loss") or {}
    rmse = pl.get("rmse_physical")
    if rmse is None:
        return None, None, None
    try:
        rmse_f = float(rmse)
    except (TypeError, ValueError):
        return None, None, None
    ck = data.get("_checkpoint") or {}
    ep = ck.get("epoch")
    if isinstance(ep, float) and ep == int(ep):
        ep = int(ep)
    elif not isinstance(ep, int):
        ep = None
    ev = data.get("_evaluation") or {}
    split = ev.get("split")
    split_s = str(split) if split is not None else None
    return rmse_f, ep, split_s


def iter_epoch_rows(cluster_root: Path, try_name: str) -> List[Dict[str, Any]]:
    try_dir = cluster_root / try_name
    rows: List[Dict[str, Any]] = []
    if not try_dir.is_dir():
        return rows
    for run_dir in sorted(try_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for jp in sorted(run_dir.glob("validate_metrics_epoch_*_cgan.json")):
            m = EPOCH_JSON.match(jp.name)
            if not m:
                continue
            epoch_from_name = int(m.group(1))
            rmse, epoch_ck, split = load_rmse_physical(jp)
            epoch = epoch_ck if epoch_ck is not None else epoch_from_name
            rows.append(
                {
                    "try": try_name,
                    "model_run": run_dir.name,
                    "epoch": epoch,
                    "path_loss_rmse_db": rmse,
                    "eval_split": split,
                    "json": str(jp.relative_to(cluster_root)),
                }
            )
    return rows


def print_summary_table(rows: List[Dict[str, Any]], last_k: int) -> None:
    """One line per (try, model_run) with aggregate RMSE stats."""
    by_run: DefaultDict[Tuple[str, str], List[Tuple[int, float]]] = defaultdict(list)
    for r in rows:
        rmse = r.get("path_loss_rmse_db")
        ep = r.get("epoch")
        if rmse is None or ep is None:
            continue
        try:
            rf = float(rmse)
            ei = int(ep)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rf):
            continue
        key = (str(r["try"]), str(r["model_run"]))
        by_run[key].append((ei, rf))

    lines: List[Dict[str, Any]] = []
    for (try_name, run_name), pairs in sorted(by_run.items()):
        pairs.sort(key=lambda x: x[0])
        values = [p[1] for p in pairs]
        epochs = [p[0] for p in pairs]
        best_i = min(range(len(values)), key=lambda i: values[i])
        worst_i = max(range(len(values)), key=lambda i: values[i])
        mean_v = sum(values) / len(values)
        var = sum((v - mean_v) ** 2 for v in values) / max(len(values) - 1, 1)
        std_v = math.sqrt(var) if len(values) > 1 else 0.0
        tail = values[-last_k:] if len(values) >= last_k else values
        tail_mean = sum(tail) / len(tail) if tail else float("nan")

        lines.append(
            {
                "try": try_name,
                "model_run": run_name,
                "n_epochs": len(values),
                "best_rmse_db": values[best_i],
                "best_epoch": epochs[best_i],
                "worst_rmse_db": values[worst_i],
                "worst_epoch": epochs[worst_i],
                "mean_rmse_db": mean_v,
                "std_rmse_db": std_v,
                f"mean_last_{last_k}_epochs_db": tail_mean,
            }
        )

    cols = [
        "try",
        "model_run",
        "n_epochs",
        "best_rmse_db",
        "best_epoch",
        "worst_rmse_db",
        "worst_epoch",
        "mean_rmse_db",
        "std_rmse_db",
        f"mean_last_{last_k}_epochs_db",
    ]
    widths = [len(c) for c in cols]
    for row in lines:
        for i, c in enumerate(cols):
            widths[i] = max(widths[i], len(str(row.get(c, ""))))

    header = "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    print(header)
    print("  ".join("-" * w for w in widths))
    for row in lines:
        print("  ".join(str(row.get(c, "")).ljust(widths[i]) for i, c in enumerate(cols)))


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"cluster root not found: {root}", file=sys.stderr)
        sys.exit(1)

    tries = [t.strip() for t in args.tries.split(",") if t.strip()]
    all_rows: List[Dict[str, Any]] = []
    for t in tries:
        all_rows.extend(iter_epoch_rows(root, t))

    # Stable sort: try, model_run, epoch
    all_rows.sort(key=lambda r: (r["try"], r["model_run"], r["epoch"] or -1))

    if args.summary:
        print_summary_table(all_rows, max(1, int(args.last_k)))
        return

    if args.format == "table":
        cols = ["try", "model_run", "epoch", "path_loss_rmse_db", "eval_split"]
        widths = [max(len(c), max((len(str(r.get(c, ""))) for r in all_rows), default=0)) for c in cols]
        header = "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
        print(header)
        print("  ".join("-" * w for w in widths))
        for r in all_rows:
            line = "  ".join(str(r.get(c, "")).ljust(widths[i]) for i, c in enumerate(cols))
            print(line)
        return

    delim = "\t" if args.format == "tsv" else ","
    cols = ["try", "model_run", "epoch", "path_loss_rmse_db", "eval_split", "json"]
    w = csv.DictWriter(sys.stdout, fieldnames=cols, delimiter=delim, lineterminator="\n")
    w.writeheader()
    for r in all_rows:
        w.writerow({k: r.get(k, "") for k in cols})


if __name__ == "__main__":
    main()
