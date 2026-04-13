from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "cluster_outputs" / "TFGFiftySeventhTry57"


def _epoch_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


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


def find_output_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    dirs = {path.parent for path in root_dir.glob("*/validate_metrics_epoch_*.json")}
    return sorted(dirs, key=lambda path: str(path).lower())


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("validate_metrics_epoch_*.json"), key=_epoch_from_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        train_payload = payload.get("runtime")
        if not isinstance(train_payload, dict):
            train_payload = payload.get("_train", {})
        if not isinstance(train_payload, dict):
            train_payload = {}

        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, dict):
            checkpoint = payload.get("_checkpoint", {})
        if not isinstance(checkpoint, dict):
            checkpoint = {}

        experiment = payload.get("experiment", {})
        if not isinstance(experiment, dict):
            experiment = {}

        row = {
            "epoch": _int_or(checkpoint.get("epoch"), _epoch_from_path(path)),
            "val_rmse": _finite(_nested(payload, "metrics.path_loss.rmse_physical")),
            "prior_rmse": _finite(_nested(payload, "metrics.prior_path_loss.rmse_physical")),
            "train_rmse": _finite(_nested(payload, "metrics.train_path_loss.rmse_physical")),
            "gain_db": _finite(_nested(payload, "metrics.improvement_vs_prior.rmse_gain_db")),
            "generator_loss": _finite(train_payload.get("generator_loss")),
            "train_no_data_loss": _finite(train_payload.get("no_data_loss")),
            "val_no_data_bce": _finite(_nested(payload, "metrics.no_data.bce")),
            "val_no_data_rmse": _finite(_nested(payload, "metrics.no_data.rmse")),
            "val_no_data_accuracy": _finite(_nested(payload, "metrics.no_data.accuracy")),
            "val_no_data_iou": _finite(_nested(payload, "metrics.no_data.iou")),
            "final_loss": _finite(_nested(train_payload, "loss_components.final_loss")),
            "residual_loss": _finite(_nested(train_payload, "loss_components.residual_loss")),
            "multiscale_loss": _finite(_nested(train_payload, "loss_components.multiscale_loss")),
            "gate_loss": _finite(_nested(train_payload, "loss_components.gate_loss")),
            "gan_loss": _finite(_nested(train_payload, "loss_components.gan_loss")),
            "term_final": _finite(_nested(train_payload, "loss_components.term_final")),
            "term_residual": _finite(_nested(train_payload, "loss_components.term_residual")),
            "term_multiscale": _finite(_nested(train_payload, "loss_components.term_multiscale")),
            "lr": _finite(train_payload.get("learning_rate")),
            "best_epoch": _int_or(checkpoint.get("best_epoch"), _epoch_from_path(path)),
            "best_score": _finite(checkpoint.get("best_score")),
            "topology_class": experiment.get("topology_class"),
        }
        rows.append(row)
    return rows


def plot_output_dir(output_dir: Path, save_path: Path | None = None) -> dict[str, Any]:
    rows = load_rows(output_dir)
    if not rows:
        raise ValueError(f"No validate_metrics_epoch_*.json files found in {output_dir}")

    epochs = [r["epoch"] for r in rows]
    val_rmse = [r["val_rmse"] for r in rows]
    train_rmse = [r["train_rmse"] for r in rows]
    prior_rmse = [r["prior_rmse"] for r in rows]
    gain_db = [r["gain_db"] for r in rows]
    gen_loss = [r["generator_loss"] for r in rows]
    train_no_data_loss = [r["train_no_data_loss"] for r in rows]
    val_no_data_bce = [r["val_no_data_bce"] for r in rows]
    val_no_data_acc = [r["val_no_data_accuracy"] for r in rows]
    val_no_data_iou = [r["val_no_data_iou"] for r in rows]
    final_loss = [r["final_loss"] for r in rows]
    residual_loss = [r["residual_loss"] for r in rows]
    multiscale_loss = [r["multiscale_loss"] for r in rows]
    gate_loss = [r["gate_loss"] for r in rows]
    gan_loss = [r["gan_loss"] for r in rows]
    term_final = [r["term_final"] for r in rows]
    term_residual = [r["term_residual"] for r in rows]
    term_multiscale = [r["term_multiscale"] for r in rows]
    lr = [r["lr"] for r in rows]

    best_idx = min(range(len(rows)), key=lambda i: rows[i]["val_rmse"])
    best_epoch = rows[best_idx]["epoch"]
    topo = rows[0]["topology_class"] or output_dir.name

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True, constrained_layout=True)

    axes[0].plot(epochs, val_rmse, label="val RMSE", color="#0b7285", linewidth=2.0)
    axes[0].plot(epochs, train_rmse, label="train RMSE", color="#e8590c", linewidth=1.8)
    axes[0].plot(epochs, prior_rmse, label="prior RMSE", color="#868e96", linestyle="--", linewidth=1.5)
    axes[0].axvline(best_epoch, color="#2b8a3e", linestyle=":", linewidth=1.5, label=f"best epoch {best_epoch}")
    axes[0].set_ylabel("RMSE (dB)")
    axes[0].set_title(f"Try 57 metrics - {topo}")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, gain_db, label="RMSE gain vs prior", color="#2f9e44", linewidth=2.0)
    axes[1].axhline(0.0, color="#868e96", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Gain (dB)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, gen_loss, label="generator loss", color="#5f3dc4", linewidth=2.0)
    axes[2].plot(epochs, train_no_data_loss, label="train no-data BCE", color="#c2255c", linewidth=1.5)
    axes[2].plot(epochs, val_no_data_bce, label="val no-data BCE", color="#495057", linewidth=1.6)
    ax2 = axes[2].twinx()
    ax2.plot(epochs, lr, label="learning rate", color="#1c7ed6", linestyle="--", linewidth=1.5)
    axes[2].set_ylabel("Loss")
    ax2.set_ylabel("LR")
    axes[2].grid(alpha=0.25)

    lines_a, labels_a = axes[2].get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    axes[2].legend(lines_a + lines_b, labels_a + labels_b, loc="best")

    axes[3].plot(epochs, final_loss, label="final loss", color="#1971c2", linewidth=1.8)
    axes[3].plot(epochs, residual_loss, label="residual loss", color="#e8590c", linewidth=1.8)
    axes[3].plot(epochs, multiscale_loss, label="multiscale loss", color="#5f3dc4", linewidth=1.6)
    axes[3].plot(epochs, gate_loss, label="gate loss", color="#2b8a3e", linewidth=1.2)
    axes[3].plot(epochs, gan_loss, label="gan loss", color="#c2255c", linewidth=1.2)
    ax3 = axes[3].twinx()
    ax3.plot(epochs, term_final, label="term final", color="#74c0fc", linestyle="--", linewidth=1.2)
    ax3.plot(epochs, term_residual, label="term residual", color="#ffa94d", linestyle="--", linewidth=1.2)
    ax3.plot(epochs, term_multiscale, label="term multiscale", color="#b197fc", linestyle="--", linewidth=1.2)
    axes[3].set_ylabel("Raw loss")
    ax3.set_ylabel("Weighted term")
    axes[3].grid(alpha=0.25)
    lines_a, labels_a = axes[3].get_legend_handles_labels()
    lines_b, labels_b = ax3.get_legend_handles_labels()
    axes[3].legend(lines_a + lines_b, labels_a + labels_b, loc="best", ncol=2)

    axes[4].plot(epochs, val_no_data_acc, label="val no-data accuracy", color="#2f9e44", linewidth=2.0)
    axes[4].plot(epochs, val_no_data_iou, label="val no-data IoU", color="#0b7285", linewidth=1.8)
    axes[4].set_ylabel("Mask quality")
    axes[4].set_xlabel("Epoch")
    axes[4].grid(alpha=0.25)
    axes[4].legend(loc="best")

    resolved_save_path = save_path or output_dir / "metrics_plot.png"
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
    parser = argparse.ArgumentParser(description="Plot Try 57 train/val RMSE trends from validate_metrics_epoch_*.json files.")
    parser.add_argument("--output-dir", default="", help="Optional single expert output dir. If omitted, scans --root-dir.")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT), help="Root containing downloaded Try57 expert output folders.")
    parser.add_argument("--save-path", default="", help="PNG path for single-dir mode, or directory for scan mode.")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
        summary = plot_output_dir(output_dir, Path(args.save_path) if args.save_path else None)
        print(json.dumps(summary, indent=2))
        return

    root_dir = Path(args.root_dir)
    output_dirs = find_output_dirs(root_dir)
    if not output_dirs:
        raise SystemExit(f"No validate_metrics_epoch_*.json files found under {root_dir}")

    save_root = Path(args.save_path) if args.save_path else None
    summaries: list[dict[str, Any]] = []
    for output_dir in output_dirs:
        save_path = None
        if save_root is not None:
            save_path = save_root / f"{output_dir.name}_metrics_plot.png"
        summaries.append(plot_output_dir(output_dir, save_path))

    print(json.dumps({"root_dir": str(root_dir), "plotted": summaries}, indent=2))


if __name__ == "__main__":
    main()
