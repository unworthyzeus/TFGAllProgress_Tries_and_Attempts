from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "cluster_outputs" / "TFGSixtyFifthTry65"


def _nested(payload: dict[str, Any], path: str, default: Any = float("nan")) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or(value: Any, default: int) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _epoch_from_path(path: Path) -> int:
    match = re.search(r"validate_metrics_epoch_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Could not parse epoch from {path.name}")
    return int(match.group(1))


def _output_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    dirs = {path.parent for path in root_dir.glob("*/validate_metrics_epoch_*.json")}
    return sorted(dirs, key=lambda path: str(path).lower())


def _expert_key(path: Path) -> str:
    name = path.name
    prefix = "sixtyfifthtry65_expert_"
    return name[len(prefix):] if name.startswith(prefix) else name


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("validate_metrics_epoch_*.json"), key=_epoch_from_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime = payload.get("runtime", {})
        if not isinstance(runtime, dict):
            runtime = payload.get("_train", {})
        if not isinstance(runtime, dict):
            runtime = {}
        checkpoint = payload.get("checkpoint", {})
        if not isinstance(checkpoint, dict):
            checkpoint = payload.get("_checkpoint", {})
        if not isinstance(checkpoint, dict):
            checkpoint = {}
        experiment = payload.get("experiment", {})
        if not isinstance(experiment, dict):
            experiment = {}
        rows.append(
            {
                "epoch": _int_or(checkpoint.get("epoch"), _epoch_from_path(path)),
                "val_rmse": _finite(_nested(payload, "metrics.path_loss.rmse_physical")),
                "train_rmse": _finite(_nested(payload, "metrics.train_path_loss.rmse_physical")),
                "los_rmse": _finite(_nested(payload, "focus.regimes.path_loss__los__LoS.rmse_physical")),
                "nlos_rmse": _finite(_nested(payload, "focus.regimes.path_loss__los__NLoS.rmse_physical")),
                "generator_loss": _finite(runtime.get("generator_loss")),
                "lr": _finite(runtime.get("learning_rate")),
                "best_epoch": _int_or(checkpoint.get("best_epoch"), _epoch_from_path(path)),
                "best_score": _finite(checkpoint.get("best_score")),
                "topology_class": experiment.get("topology_class"),
            }
        )
    return rows


def plot_output_dir(output_dir: Path, save_path: Path | None = None) -> dict[str, Any]:
    rows = load_rows(output_dir)
    if not rows:
        raise ValueError(f"No validate_metrics_epoch_*.json files found in {output_dir}")

    epochs = [r["epoch"] for r in rows]
    val_rmse = [r["val_rmse"] for r in rows]
    train_rmse = [r["train_rmse"] for r in rows]
    los_rmse = [r["los_rmse"] for r in rows]
    nlos_rmse = [r["nlos_rmse"] for r in rows]
    gen_loss = [r["generator_loss"] for r in rows]
    lr = [r["lr"] for r in rows]

    best_idx = min(range(len(rows)), key=lambda i: rows[i]["val_rmse"])
    best_epoch = rows[best_idx]["epoch"]
    topo = rows[0]["topology_class"] or _expert_key(output_dir)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False, constrained_layout=True)

    axes[0].plot(epochs, val_rmse, label="val RMSE", color="#0b7285", linewidth=2.1)
    axes[0].plot(epochs, train_rmse, label="train RMSE", color="#e8590c", linewidth=1.7)
    axes[0].axvline(best_epoch, color="#2b8a3e", linestyle=":", linewidth=1.4, label=f"best epoch {best_epoch}")
    axes[0].set_title(f"Try 65 grokking metrics - {topo}")
    axes[0].set_ylabel("RMSE (dB)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(epochs, los_rmse, label="LoS RMSE", color="#2f9e44", linewidth=1.8)
    axes[1].plot(epochs, nlos_rmse, label="NLoS RMSE", color="#c2255c", linewidth=1.8)
    axes[1].set_ylabel("dB")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(epochs, gen_loss, label="generator loss", color="#5f3dc4", linewidth=1.8)
    ax2 = axes[2].twinx()
    ax2.plot(epochs, lr, label="learning rate", color="#1c7ed6", linestyle="--", linewidth=1.4)
    axes[2].set_ylabel("Loss")
    ax2.set_ylabel("LR")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.25)
    lines_a, labels_a = axes[2].get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    axes[2].legend(lines_a + lines_b, labels_a + labels_b, loc="best")

    resolved_save_path = save_path or output_dir / "metrics_plot_try65.png"
    resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved_save_path, dpi=180)
    plt.close(fig)
    return {
        "output_dir": str(output_dir),
        "saved": str(resolved_save_path),
        "epochs": len(rows),
        "best_epoch": best_epoch,
        "best_val_rmse": rows[best_idx]["val_rmse"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Try 65 single-stage grokking metrics.")
    parser.add_argument("--output-dir", default="", help="Optional explicit output dir.")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT), help="Root containing downloaded Try65 output folders.")
    parser.add_argument("--expert-id", default="", help="Optional expert id.")
    parser.add_argument("--save-path", default="", help="PNG path for single dir mode or output dir for scan mode.")
    args = parser.parse_args()

    if args.output_dir:
        summary = plot_output_dir(Path(args.output_dir), Path(args.save_path) if args.save_path else None)
        print(json.dumps(summary, indent=2))
        return

    root_dir = Path(args.root_dir)
    dirs = _output_dirs(root_dir)
    if args.expert_id:
        dirs = [path for path in dirs if _expert_key(path) == args.expert_id]
    if not dirs:
        raise SystemExit(f"No Try65 output directories with validate metrics found under {root_dir}")

    save_root = Path(args.save_path) if args.save_path else None
    summaries: list[dict[str, Any]] = []
    for output_dir in dirs:
        save_path = None
        if save_root is not None:
            save_path = save_root / f"{_expert_key(output_dir)}_metrics_plot_try65.png"
        summaries.append(plot_output_dir(output_dir, save_path))

    print(json.dumps({"root_dir": str(root_dir), "plotted": summaries}, indent=2))


if __name__ == "__main__":
    main()
