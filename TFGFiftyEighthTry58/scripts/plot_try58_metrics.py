from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "cluster_outputs" / "TFGFiftyEighthTry58"


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


def _has_finite(values: list[float]) -> bool:
    return any(isinstance(v, float) and math.isfinite(v) for v in values)


def find_output_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    dirs = {path.parent for path in root_dir.glob("*/validate_metrics_epoch_*.json")}
    return sorted(dirs, key=lambda path: str(path).lower())


def _load_history_by_epoch(output_dir: Path) -> dict[int, dict[str, Any]]:
    history_path = output_dir / "history.json"
    if not history_path.exists():
        return {}
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, list):
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for row in payload:
        if isinstance(row, dict) and "epoch" in row:
            rows[_int_or(row.get("epoch"), -1)] = row
    return rows


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history_by_epoch = _load_history_by_epoch(output_dir)
    for path in sorted(output_dir.glob("validate_metrics_epoch_*.json"), key=_epoch_from_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = payload.get("_checkpoint", {})
        if not isinstance(checkpoint, dict):
            checkpoint = {}
        epoch = _int_or(checkpoint.get("epoch"), _epoch_from_path(path))
        history_row = history_by_epoch.get(epoch, {})
        if not isinstance(history_row, dict):
            history_row = {}
        train_payload = payload.get("_train", {})
        if not isinstance(train_payload, dict):
            train_payload = {}
        experiment = payload.get("experiment", {})
        if not isinstance(experiment, dict):
            experiment = {}

        row = {
            "epoch": epoch,
            "main_target": str(experiment.get("focused_target", "delay_spread")),
            "delay_rmse_physical": _finite(_nested(payload, "metrics.delay_spread.rmse_physical", _nested(payload, "delay_spread.rmse_physical"))),
            "angular_rmse_physical": _finite(_nested(payload, "metrics.angular_spread.rmse_physical", _nested(payload, "angular_spread.rmse_physical"))),
            "train_delay_rmse_physical": _finite(
                _nested(payload, "metrics.train_delay_spread.rmse_physical", _nested(payload, "train_delay_spread.rmse_physical"))
            ),
            "train_angular_rmse_physical": _finite(
                _nested(payload, "metrics.train_angular_spread.rmse_physical", _nested(payload, "train_angular_spread.rmse_physical"))
            ),
            "delay_loss": _finite(_nested(payload, "_loss.target_losses.delay_spread", _nested(payload, "metrics.delay_spread.mse", _nested(payload, "delay_spread.mse")))),
            "angular_loss": _finite(_nested(payload, "_loss.target_losses.angular_spread", _nested(payload, "metrics.angular_spread.mse", _nested(payload, "angular_spread.mse")))),
            "delay_rmse": _finite(_nested(payload, "metrics.delay_spread.rmse", _nested(payload, "delay_spread.rmse"))),
            "angular_rmse": _finite(_nested(payload, "metrics.angular_spread.rmse", _nested(payload, "angular_spread.rmse"))),
            "no_data_loss": _finite(_nested(payload, "metrics.no_data.bce", _nested(payload, "_loss.target_losses.no_data", _nested(payload, "no_data.bce", _nested(payload, "no_data.mse"))))),
            "no_data_rmse": _finite(_nested(payload, "metrics.no_data.rmse", _nested(payload, "no_data.rmse"))),
            "no_data_accuracy": _finite(_nested(payload, "metrics.no_data.accuracy", _nested(payload, "no_data.accuracy"))),
            "no_data_iou": _finite(_nested(payload, "metrics.no_data.iou", _nested(payload, "no_data.iou"))),
            "generator_loss": _finite(train_payload.get("generator_loss", history_row.get("generator_loss"))),
            "val_recon_loss": _finite(
                train_payload.get("val_recon_loss", _nested(payload, "_loss.val_recon_loss", history_row.get("val_recon_loss")))
            ),
            "recon_loss": _finite(_nested(train_payload, "loss_components.recon_loss", history_row.get("loss_components", {}).get("recon_loss"))),
            "multiscale_loss": _finite(_nested(train_payload, "loss_components.multiscale_loss", history_row.get("loss_components", {}).get("multiscale_loss"))),
            "gradient_loss": _finite(_nested(train_payload, "loss_components.gradient_loss", history_row.get("loss_components", {}).get("gradient_loss"))),
            "confidence_loss": _finite(_nested(train_payload, "loss_components.confidence_loss", history_row.get("loss_components", {}).get("confidence_loss"))),
            "term_recon": _finite(_nested(train_payload, "loss_components.term_recon", history_row.get("loss_components", {}).get("term_recon"))),
            "term_confidence": _finite(_nested(train_payload, "loss_components.term_confidence", history_row.get("loss_components", {}).get("term_confidence"))),
            "train_main_target_loss": _finite(_nested(train_payload, "loss_components.target_losses.delay_spread", history_row.get("loss_components", {}).get("target_losses", {}).get("delay_spread"))),
            "train_no_data_target_loss": _finite(_nested(train_payload, "loss_components.target_losses.no_data", history_row.get("loss_components", {}).get("target_losses", {}).get("no_data"))),
            "val_main_target_loss": _finite(_nested(payload, "_loss.target_losses.delay_spread")),
            "val_no_data_target_loss": _finite(_nested(payload, "_loss.target_losses.no_data")),
            "selection_value": _finite(
                train_payload.get("selection_metric_value", _nested(payload, "_selection.value", history_row.get("selection_metric_value")))
            ),
            "lr": _finite(train_payload.get("learning_rate")),
        }
        rows.append(row)
    return rows


def _best_idx(rows: list[dict[str, Any]]) -> int:
    def score(row: dict[str, Any]) -> float:
        value = row.get("selection_value")
        if isinstance(value, float) and math.isfinite(value):
            return value
        delay = row.get("delay_rmse_physical", float("nan"))
        angular = row.get("angular_rmse_physical", float("nan"))
        parts = [v for v in (delay, angular) if isinstance(v, float) and math.isfinite(v)]
        return sum(parts) / max(len(parts), 1) if parts else float("inf")

    return min(range(len(rows)), key=lambda i: score(rows[i]))


def plot_output_dir(output_dir: Path, save_path: Path | None = None) -> dict[str, Any]:
    rows = load_rows(output_dir)
    if not rows:
        raise ValueError(f"No validate_metrics_epoch_*.json files found in {output_dir}")

    epochs = [r["epoch"] for r in rows]
    delay_rmse_phys = [r["delay_rmse_physical"] for r in rows]
    angular_rmse_phys = [r["angular_rmse_physical"] for r in rows]
    train_delay_rmse_phys = [r["train_delay_rmse_physical"] for r in rows]
    train_angular_rmse_phys = [r["train_angular_rmse_physical"] for r in rows]
    gen_loss = [r["generator_loss"] for r in rows]
    val_recon = [r["val_recon_loss"] for r in rows]
    no_data_acc = [r["no_data_accuracy"] for r in rows]
    no_data_iou = [r["no_data_iou"] for r in rows]
    recon_loss = [r["recon_loss"] for r in rows]
    multiscale_loss = [r["multiscale_loss"] for r in rows]
    gradient_loss = [r["gradient_loss"] for r in rows]
    confidence_loss = [r["confidence_loss"] for r in rows]
    term_recon = [r["term_recon"] for r in rows]
    term_confidence = [r["term_confidence"] for r in rows]
    train_main_target_loss = [r["train_main_target_loss"] for r in rows]
    train_no_data_target_loss = [r["train_no_data_target_loss"] for r in rows]
    val_main_target_loss = [r["val_main_target_loss"] for r in rows]
    val_no_data_target_loss = [r["val_no_data_target_loss"] for r in rows]
    lr = [r["lr"] for r in rows]

    best_i = _best_idx(rows)
    best_epoch = rows[best_i]["epoch"]
    topo = output_dir.name.replace("fiftyeighthtry58_", "")

    has_delay = _has_finite(delay_rmse_phys) or _has_finite(train_delay_rmse_phys)
    has_angular = _has_finite(angular_rmse_phys) or _has_finite(train_angular_rmse_phys)
    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True, constrained_layout=True)

    if has_delay:
        axes[0].plot(epochs, delay_rmse_phys, label="val delay spread RMSE", color="#0b7285", linewidth=2.0)
        axes[0].plot(epochs, train_delay_rmse_phys, label="train delay spread RMSE", color="#0b7285", linestyle="--", linewidth=1.8)
    if has_angular:
        axes[0].plot(epochs, angular_rmse_phys, label="val angular spread RMSE", color="#e8590c", linewidth=2.0)
        axes[0].plot(epochs, train_angular_rmse_phys, label="train angular spread RMSE", color="#e8590c", linestyle="--", linewidth=1.8)
    axes[0].axvline(best_epoch, color="#2b8a3e", linestyle=":", linewidth=1.5, label=f"best epoch {best_epoch}")
    axes[0].set_ylabel("Physical RMSE")
    axes[0].set_title(f"Try 58 metrics - {topo}")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, train_main_target_loss, label="train delay target loss", color="#1971c2", linewidth=2.0)
    axes[1].plot(epochs, val_main_target_loss, label="val delay target loss", color="#74c0fc", linewidth=1.8)
    axes[1].plot(epochs, train_no_data_target_loss, label="train no-data BCE", color="#c2255c", linewidth=1.6)
    axes[1].plot(epochs, val_no_data_target_loss, label="val no-data BCE", color="#495057", linewidth=1.4)
    axes[1].set_ylabel("Target losses")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, gen_loss, label="train generator loss", color="#5f3dc4", linewidth=2.0)
    axes[2].plot(epochs, val_recon, label="val recon loss", color="#c2255c", linewidth=1.8)
    ax2 = axes[2].twinx()
    ax2.plot(epochs, lr, label="learning rate", color="#1c7ed6", linestyle="--", linewidth=1.5)
    axes[2].set_ylabel("Loss")
    ax2.set_ylabel("LR")
    axes[2].grid(alpha=0.25)
    lines_a, labels_a = axes[2].get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    axes[2].legend(lines_a + lines_b, labels_a + labels_b, loc="best")

    axes[3].plot(epochs, recon_loss, label="recon loss", color="#1971c2", linewidth=1.8)
    axes[3].plot(epochs, multiscale_loss, label="multiscale loss", color="#5f3dc4", linewidth=1.5)
    axes[3].plot(epochs, gradient_loss, label="gradient loss", color="#e8590c", linewidth=1.7)
    axes[3].plot(epochs, confidence_loss, label="confidence loss", color="#495057", linewidth=1.3)
    ax3 = axes[3].twinx()
    ax3.plot(epochs, term_recon, label="term recon", color="#74c0fc", linestyle="--", linewidth=1.2)
    ax3.plot(epochs, term_confidence, label="term confidence", color="#ff8787", linestyle="--", linewidth=1.2)
    axes[3].set_ylabel("Raw loss")
    ax3.set_ylabel("Weighted term")
    axes[3].grid(alpha=0.25)
    lines_a, labels_a = axes[3].get_legend_handles_labels()
    lines_b, labels_b = ax3.get_legend_handles_labels()
    axes[3].legend(lines_a + lines_b, labels_a + labels_b, loc="best", ncol=2)

    axes[4].plot(epochs, no_data_acc, label="val no-data accuracy", color="#2f9e44", linewidth=2.0)
    axes[4].plot(epochs, no_data_iou, label="val no-data IoU", color="#0b7285", linewidth=1.8)
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
        "best_delay_rmse_physical": rows[best_i]["delay_rmse_physical"] if has_delay else float("nan"),
        "best_angular_rmse_physical": rows[best_i]["angular_rmse_physical"] if has_angular else float("nan"),
        "best_train_delay_rmse_physical": rows[best_i]["train_delay_rmse_physical"] if has_delay else float("nan"),
        "best_train_angular_rmse_physical": rows[best_i]["train_angular_rmse_physical"] if has_angular else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Try 58 train/val delay and angular spread RMSE trends.")
    parser.add_argument("--output-dir", default="", help="Optional single expert output dir. If omitted, scans --root-dir.")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT), help="Root containing downloaded Try58 expert output folders.")
    parser.add_argument("--save-path", default="", help="PNG path for single-dir mode, or directory for scan mode.")
    args = parser.parse_args()

    if args.output_dir:
        summary = plot_output_dir(Path(args.output_dir), Path(args.save_path) if args.save_path else None)
        print(json.dumps(summary, indent=2))
        return

    root_dir = Path(args.root_dir)
    output_dirs = find_output_dirs(root_dir)
    if not output_dirs:
        raise SystemExit(f"No validate_metrics_epoch_*.json files found under {root_dir}")

    save_root = Path(args.save_path) if args.save_path else None
    summaries: list[dict[str, Any]] = []
    for output_dir in output_dirs:
        save_path = save_root / f"{output_dir.name}_metrics_plot.png" if save_root is not None else None
        summaries.append(plot_output_dir(output_dir, save_path))
    print(json.dumps({"root_dir": str(root_dir), "plotted": summaries}, indent=2))


if __name__ == "__main__":
    main()
