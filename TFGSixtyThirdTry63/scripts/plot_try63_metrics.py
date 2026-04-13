from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "cluster_outputs" / "TFGSixtyThirdTry63"


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


def _epoch_from_stage1(path: Path) -> int:
    match = re.search(r"validate_metrics_epoch_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Could not parse epoch from {path.name}")
    return int(match.group(1))


def _epoch_from_stage2(path: Path) -> int:
    match = re.search(r"validate_metrics_epoch_(\d+)_tail_refiner\.json$", path.name)
    if not match:
        raise ValueError(f"Could not parse epoch from {path.name}")
    return int(match.group(1))


def _stage1_output_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    dirs = {path.parent for path in root_dir.glob("*/validate_metrics_epoch_*.json") if not path.name.endswith("_tail_refiner.json")}
    return sorted(dirs, key=lambda path: str(path).lower())


def _stage2_output_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    dirs = {path.parent for path in root_dir.glob("*/validate_metrics_epoch_*_tail_refiner.json")}
    return sorted(dirs, key=lambda path: str(path).lower())


def _expert_key_from_stage1_dir(path: Path) -> str:
    name = path.name
    prefix = "sixtythirdtry63_expert_"
    return name[len(prefix):] if name.startswith(prefix) else name


def _expert_key_from_stage2_dir(path: Path) -> str:
    name = path.name
    prefix = "sixtythirdtry63_tail_refiner_"
    return name[len(prefix):] if name.startswith(prefix) else name


def load_stage1_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("validate_metrics_epoch_*.json"), key=_epoch_from_stage1):
        if path.name.endswith("_tail_refiner.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict):
            runtime = payload.get("_train", {})
        if not isinstance(runtime, dict):
            runtime = {}
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, dict):
            checkpoint = payload.get("_checkpoint", {})
        if not isinstance(checkpoint, dict):
            checkpoint = {}
        experiment = payload.get("experiment", {})
        if not isinstance(experiment, dict):
            experiment = {}
        rows.append(
            {
                "epoch": _int_or(checkpoint.get("epoch"), _epoch_from_stage1(path)),
                "val_rmse_128": _finite(_nested(payload, "metrics.path_loss_128.rmse_physical", _nested(payload, "metrics.path_loss.rmse_physical"))),
                "val_rmse_513": _finite(_nested(payload, "metrics.path_loss_513.rmse_physical")),
                "train_rmse": _finite(_nested(payload, "metrics.train_path_loss.rmse_physical")),
                "los_rmse": _finite(_nested(payload, "focus.regimes.path_loss__los__LoS.rmse_physical")),
                "nlos_rmse": _finite(_nested(payload, "focus.regimes.path_loss__los__NLoS.rmse_physical")),
                "generator_loss": _finite(runtime.get("generator_loss")),
                "lr": _finite(runtime.get("learning_rate")),
                "best_epoch": _int_or(checkpoint.get("best_epoch"), _epoch_from_stage1(path)),
                "best_score": _finite(checkpoint.get("best_score")),
                "topology_class": experiment.get("topology_class"),
            }
        )
    return rows


def load_stage2_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("validate_metrics_epoch_*_tail_refiner.json"), key=_epoch_from_stage2):
        payload = json.loads(path.read_text(encoding="utf-8"))
        train_payload = payload.get("_train", {})
        if not isinstance(train_payload, dict):
            train_payload = {}
        checkpoint = payload.get("_checkpoint", {})
        if not isinstance(checkpoint, dict):
            checkpoint = {}
        rows.append(
            {
                "epoch": _int_or(checkpoint.get("epoch"), _epoch_from_stage2(path)),
                "val_rmse": _finite(_nested(payload, "path_loss.rmse_physical")),
                "stage1_overall_rmse": _finite(_nested(payload, "path_loss__stage1__overall.rmse_physical")),
                "los_rmse": _finite(_nested(payload, "path_loss__los__LoS.rmse_physical")),
                "nlos_rmse": _finite(_nested(payload, "path_loss__los__NLoS.rmse_physical")),
                "tail_refiner_loss": _finite(train_payload.get("tail_refiner_loss")),
                "best_epoch": _int_or(checkpoint.get("best_epoch"), _epoch_from_stage2(path)),
                "best_score": _finite(checkpoint.get("best_score")),
                "los_fraction": _finite(_nested(payload, "_support.los_fraction")),
                "nlos_fraction": _finite(_nested(payload, "_support.nlos_fraction")),
            }
        )
    return rows


def _paired_dirs(root_dir: Path) -> list[tuple[str, Path, Path | None]]:
    stage1_dirs = { _expert_key_from_stage1_dir(p): p for p in _stage1_output_dirs(root_dir) }
    stage2_dirs = { _expert_key_from_stage2_dir(p): p for p in _stage2_output_dirs(root_dir) }
    keys = sorted(set(stage1_dirs) | set(stage2_dirs))
    return [(key, stage1_dirs.get(key), stage2_dirs.get(key)) for key in keys if stage1_dirs.get(key) is not None]


def plot_expert_pair(
    expert_id: str,
    stage1_dir: Path,
    stage2_dir: Path | None,
    save_path: Path | None = None,
) -> dict[str, Any]:
    stage1_rows = load_stage1_rows(stage1_dir)
    if not stage1_rows:
        raise ValueError(f"No stage1 validate_metrics_epoch_*.json files found in {stage1_dir}")
    stage2_rows = load_stage2_rows(stage2_dir) if stage2_dir and stage2_dir.exists() else []

    s1_epochs = [r["epoch"] for r in stage1_rows]
    s1_val_rmse_128 = [r["val_rmse_128"] for r in stage1_rows]
    s1_val_rmse_513 = [r["val_rmse_513"] for r in stage1_rows]
    s1_train_rmse = [r["train_rmse"] for r in stage1_rows]
    s1_los_rmse = [r["los_rmse"] for r in stage1_rows]
    s1_nlos_rmse = [r["nlos_rmse"] for r in stage1_rows]
    s1_gen_loss = [r["generator_loss"] for r in stage1_rows]
    s1_lr = [r["lr"] for r in stage1_rows]
    s1_best_idx = min(range(len(stage1_rows)), key=lambda i: stage1_rows[i]["val_rmse_513"] if math.isfinite(stage1_rows[i]["val_rmse_513"]) else stage1_rows[i]["val_rmse_128"])
    s1_best_epoch = stage1_rows[s1_best_idx]["epoch"]

    s2_epochs = [r["epoch"] for r in stage2_rows]
    s2_val_rmse = [r["val_rmse"] for r in stage2_rows]
    s2_stage1_rmse = [r["stage1_overall_rmse"] for r in stage2_rows]
    s2_los_rmse = [r["los_rmse"] for r in stage2_rows]
    s2_nlos_rmse = [r["nlos_rmse"] for r in stage2_rows]
    s2_loss = [r["tail_refiner_loss"] for r in stage2_rows]
    s2_gain = [
        (base - final) if math.isfinite(base) and math.isfinite(final) else float("nan")
        for base, final in zip(s2_stage1_rmse, s2_val_rmse)
    ]
    s2_best_epoch = None
    s2_best_val = float("nan")
    if stage2_rows:
        s2_best_idx = min(range(len(stage2_rows)), key=lambda i: stage2_rows[i]["val_rmse"])
        s2_best_epoch = stage2_rows[s2_best_idx]["epoch"]
        s2_best_val = stage2_rows[s2_best_idx]["val_rmse"]

    topo = stage1_rows[0]["topology_class"] or expert_id
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=False, constrained_layout=True)

    axes[0].plot(s1_epochs, s1_val_rmse_128, label="stage1 val RMSE 128", color="#0b7285", linewidth=2.0)
    if any(math.isfinite(v) for v in s1_val_rmse_513):
        axes[0].plot(s1_epochs, s1_val_rmse_513, label="stage1 val RMSE 513", color="#1098ad", linestyle="--", linewidth=1.8)
    axes[0].plot(s1_epochs, s1_train_rmse, label="stage1 train RMSE", color="#e8590c", linewidth=1.7)
    axes[0].axvline(s1_best_epoch, color="#2b8a3e", linestyle=":", linewidth=1.4, label=f"stage1 best {s1_best_epoch}")
    if stage2_rows:
        axes[0].plot(s2_epochs, s2_val_rmse, label="stage2 val RMSE", color="#5f3dc4", linewidth=2.0)
        axes[0].plot(s2_epochs, s2_stage1_rmse, label="stage2 teacher RMSE", color="#868e96", linestyle="--", linewidth=1.5)
        if s2_best_epoch is not None:
            axes[0].axvline(s2_best_epoch, color="#c2255c", linestyle=":", linewidth=1.4, label=f"stage2 best {s2_best_epoch}")
    axes[0].set_title(f"Try 63 two-stage metrics - {topo}")
    axes[0].set_ylabel("RMSE (dB)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(s1_epochs, s1_los_rmse, label="stage1 LoS RMSE", color="#2f9e44", linewidth=1.7)
    axes[1].plot(s1_epochs, s1_nlos_rmse, label="stage1 NLoS RMSE", color="#c2255c", linewidth=1.7)
    if stage2_rows:
        axes[1].plot(s2_epochs, s2_los_rmse, label="stage2 LoS RMSE", color="#74c0fc", linewidth=1.7)
        axes[1].plot(s2_epochs, s2_nlos_rmse, label="stage2 NLoS RMSE", color="#ff87b6", linewidth=1.7)
    axes[1].set_ylabel("dB")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", ncol=2)

    axes[2].plot(s1_epochs, s1_gen_loss, label="stage1 generator loss", color="#5f3dc4", linewidth=1.8)
    ax2 = axes[2].twinx()
    ax2.plot(s1_epochs, s1_lr, label="stage1 LR", color="#1c7ed6", linestyle="--", linewidth=1.4)
    if stage2_rows:
        axes[2].plot(s2_epochs, s2_loss, label="stage2 tail-refiner loss", color="#495057", linewidth=1.8)
    axes[2].set_ylabel("Loss")
    ax2.set_ylabel("LR")
    axes[2].grid(alpha=0.25)
    lines_a, labels_a = axes[2].get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    axes[2].legend(lines_a + lines_b, labels_a + labels_b, loc="best")

    if stage2_rows:
        axes[3].plot(s2_epochs, s2_gain, label="stage2 gain vs teacher", color="#2b8a3e", linewidth=2.0)
        axes[3].axhline(0.0, color="#adb5bd", linestyle="--", linewidth=1.2)
        axes[3].set_ylabel("Gain (dB)")
    else:
        axes[3].text(0.5, 0.5, "No stage2 metrics found yet", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_ylabel("Gain (dB)")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(alpha=0.25)
    if stage2_rows:
        axes[3].legend(loc="best")

    resolved_save_path = save_path or stage1_dir / "metrics_plot_try63_twostage.png"
    resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved_save_path, dpi=180)
    plt.close(fig)
    return {
        "expert_id": expert_id,
        "stage1_dir": str(stage1_dir),
        "stage2_dir": str(stage2_dir) if stage2_dir else "",
        "saved": str(resolved_save_path),
        "stage1_epochs": len(stage1_rows),
        "stage1_best_epoch": s1_best_epoch,
        "stage1_best_val_rmse_128": stage1_rows[s1_best_idx]["val_rmse_128"],
        "stage1_best_val_rmse_513": stage1_rows[s1_best_idx]["val_rmse_513"],
        "stage2_epochs": len(stage2_rows),
        "stage2_best_epoch": s2_best_epoch,
        "stage2_best_val_rmse": s2_best_val,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Try 63 two-stage metrics by pairing stage1 and tail-refiner outputs.")
    parser.add_argument("--expert-id", default="", help="Optional expert id, e.g. open_sparse_lowrise.")
    parser.add_argument("--stage1-dir", default="", help="Optional explicit stage1 dir.")
    parser.add_argument("--stage2-dir", default="", help="Optional explicit stage2 dir.")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT), help="Root containing downloaded Try63 output folders.")
    parser.add_argument("--save-path", default="", help="PNG path for single expert mode, or output directory for scan mode.")
    args = parser.parse_args()

    if args.stage1_dir:
        stage1_dir = Path(args.stage1_dir)
        stage2_dir = Path(args.stage2_dir) if args.stage2_dir else None
        expert_id = args.expert_id or _expert_key_from_stage1_dir(stage1_dir)
        summary = plot_expert_pair(expert_id, stage1_dir, stage2_dir, Path(args.save_path) if args.save_path else None)
        print(json.dumps(summary, indent=2))
        return

    root_dir = Path(args.root_dir)
    pairs = _paired_dirs(root_dir)
    if args.expert_id:
        pairs = [item for item in pairs if item[0] == args.expert_id]
    if not pairs:
        raise SystemExit(f"No Try62 stage1/stage2 output pairs found under {root_dir}")

    save_root = Path(args.save_path) if args.save_path else None
    summaries: list[dict[str, Any]] = []
    for expert_id, stage1_dir, stage2_dir in pairs:
        save_path = None
        if save_root is not None:
            save_path = save_root / f"{expert_id}_metrics_plot_try62_twostage.png"
        summaries.append(plot_expert_pair(expert_id, stage1_dir, stage2_dir, save_path))

    print(json.dumps({"root_dir": str(root_dir), "plotted": summaries}, indent=2))


if __name__ == "__main__":
    main()
