"""Plot Try 77 training curves from outputs or cluster_outputs history files.

Usage:
    python scripts/plot_history.py outputs/try77_expert_open_sparse_lowrise_delay_spread
    python scripts/plot_history.py cluster_outputs/TFGSeventySeventhTry77/try77_expert_open_sparse_lowrise_delay_spread
    python scripts/plot_history.py outputs/try77_expert_*
    python scripts/plot_history.py --all outputs/ cluster_outputs/TFGSeventySeventhTry77

Writes:
    <run_dir>/history_curves.png
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_PANELS = [
    ("RMSE (native units)", [("train.rmse", "train"), ("val.rmse", "val")]),
    ("MAE (native units)", [("train.mae", "train"), ("val.mae", "val")]),
    (
        "Distribution losses",
        [
            ("train.map_nll", "train map_nll"),
            ("val.map_nll", "val map_nll"),
            ("train.dist_kl", "train dist_kl"),
            ("val.dist_kl", "val dist_kl"),
        ],
    ),
    (
        "Auxiliary losses",
        [
            ("train.moment_match", "train moment_match"),
            ("val.moment_match", "val moment_match"),
            ("train.outlier_budget", "train outlier_budget"),
            ("val.outlier_budget", "val outlier_budget"),
        ],
    ),
    (
        "Total + score",
        [
            ("train.total", "train total"),
            ("val.total", "val total"),
            ("score", "score"),
        ],
    ),
    ("LR + epoch time", [("lr", "lr"), ("elapsed_s", "epoch s")]),
]


def _lookup(row: Dict, dotted: str):
    node = row
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def plot_history(path: Path) -> Path:
    with path.open("r", encoding="utf-8") as handle:
        history: List[Dict] = json.load(handle)
    if not history:
        raise ValueError(f"Empty history at {path}")

    epochs = [int(row.get("epoch", i)) for i, row in enumerate(history)]
    fig, axes = plt.subplots(len(METRIC_PANELS), 1, figsize=(10, 3.2 * len(METRIC_PANELS)), sharex=True)
    if len(METRIC_PANELS) == 1:
        axes = [axes]

    for ax, (title, series) in zip(axes, METRIC_PANELS):
        plotted = 0
        for key, label in series:
            values = [_lookup(row, key) for row in history]
            if all(v is None for v in values):
                continue
            xs = [epoch for epoch, value in zip(epochs, values) if value is not None]
            ys = [float(value) for value in values if value is not None]
            ax.plot(xs, ys, marker=".", linewidth=1.2, label=label)
            plotted += 1
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("epoch")

    out_path = path.parent / "history_curves.png"
    fig.suptitle(path.parent.name, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def discover(root: Path) -> List[Path]:
    return sorted(Path(p) for p in glob.glob(str(root / "**" / "history.json"), recursive=True))


def unique_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def resolve_user_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Run dirs containing history.json, or the JSON files directly.")
    parser.add_argument("--all", action="store_true", help="Recursively discover history.json files under each path.")
    args = parser.parse_args()

    candidates: List[Path] = []
    try_root = Path(__file__).resolve().parents[1]
    project_root = try_root.parent
    default_paths = [
        try_root / "outputs",
        project_root / "cluster_outputs" / "TFGSeventySeventhTry77",
    ]

    for raw in (args.paths or default_paths):
        path = raw if isinstance(raw, Path) else resolve_user_path(raw, try_root)
        if path.is_file() and path.name == "history.json":
            candidates.append(path)
        elif path.is_dir():
            if args.all:
                candidates.extend(discover(path))
            else:
                history_path = path / "history.json"
                if history_path.is_file():
                    candidates.append(history_path)
                else:
                    candidates.extend(discover(path))
        else:
            pattern = str(raw) if isinstance(raw, Path) else str(resolve_user_path(raw, try_root))
            for hit in glob.glob(pattern):
                hit_path = Path(hit)
                if hit_path.is_file() and hit_path.name == "history.json":
                    candidates.append(hit_path)
                elif hit_path.is_dir():
                    history_path = hit_path / "history.json"
                    if history_path.is_file():
                        candidates.append(history_path)

    candidates = unique_paths(candidates)
    if not candidates:
        raise SystemExit("No history.json files found.")

    for history_path in candidates:
        try:
            out_path = plot_history(history_path)
            print(f"[ok] {history_path} -> {out_path}")
        except Exception as exc:
            print(f"[err] {history_path}: {exc}")


if __name__ == "__main__":
    main()
