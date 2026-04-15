"""Plot Try 71 training metrics from validate_metrics_epoch_*.json files.

Reads the Try 71 JSON schema (single-stage 513x513, heteroscedastic NLL objective,
out_channels=2: mean + log_var, same RMSE validation as Try 68) and generates
multi-panel PNG plots per expert.

Try 71 key differences vs Try 68:
  generator_objective: heteroscedastic  — Kendall & Gal NLL loss
  out_channels: 2 (channel 1 = log_var, surfaced via no_data_logits during train)
  main_map_regression_loss = heteroscedastic NLL value (not MSE/Huber)
  nlos_focus_loss still enabled (weight=0.2)
  Validation RMSE is computed on the mean prediction only (channel 0)

JSON schema (same as Try 68 — uncertainty output not exposed in validation JSON):
  metrics.path_loss          .rmse_physical / .mae_physical
  metrics.path_loss_513      .rmse_physical / .mae_physical
  metrics.train_path_loss    .rmse_physical / .mae_physical
  metrics.prior_path_loss    .rmse_physical / .mae_physical
  metrics.improvement_vs_prior .rmse_gain_db / .mae_gain_db
  focus.regimes.path_loss__los__LoS     .rmse_physical
  focus.regimes.path_loss__los__NLoS    .rmse_physical
  focus.topology_class
  runtime.generator_loss / .learning_rate / .train_seconds / .val_seconds
  runtime.loss_components.{main_map_regression_loss (NLL), multiscale_loss, nlos_focus_loss}
  checkpoint.epoch / .best_epoch / .best_score
  selection_proxy.composite_nlos_weighted_rmse / .nlos_rmse_physical / .alpha
  model_info.val_uses_ema / .ema_decay / .note
  support.sample_count / .los_fraction / .nlos_fraction

Usage:
  python scripts/plot_try68_metrics.py
  python scripts/plot_try68_metrics.py --root D:/cluster_outputs/TFGSeventyFirstTry71
  python scripts/plot_try68_metrics.py --expert-id open_sparse_lowrise
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "cluster_outputs" / "TFGSeventyFirstTry71"
EXPERT_PREFIX = "seventyfirst_try71_expert_"


def _nested(payload: dict[str, Any], path: str, default: Any = float("nan")) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _epoch_from_path(path: Path) -> int:
    match = re.search(r"validate_metrics_epoch_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Cannot parse epoch from {path.name}")
    return int(match.group(1))


def _expert_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    dirs = {
        p.parent
        for p in root.glob("*/validate_metrics_epoch_*.json")
        if not p.name.endswith("_tail_refiner.json")
    }
    return sorted(dirs, key=lambda d: str(d).lower())


def _expert_key(path: Path) -> str:
    name = path.name
    if name.startswith(EXPERT_PREFIX):
        return name[len(EXPERT_PREFIX):]
    for prefix in ("seventyfirst_try71_expert_", "sixtysixth_expert_", "try71_expert_", "seventyfirst_try71_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    json_files = sorted(
        [f for f in output_dir.glob("validate_metrics_epoch_*.json") if not f.name.endswith("_tail_refiner.json")],
        key=_epoch_from_path,
    )
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ckpt = data.get("checkpoint", data.get("_checkpoint", {}))
        runtime = data.get("runtime", data.get("_train", {}))
        if not isinstance(runtime, dict):
            runtime = {}
        loss_components = runtime.get("loss_components", {})
        if not isinstance(loss_components, dict):
            loss_components = {}

        rows.append({
            "epoch": int(ckpt.get("epoch", _epoch_from_path(path))),
            "best_epoch": int(ckpt.get("best_epoch", 0)),
            "best_score": _f(ckpt.get("best_score")),
            # --- val metrics ---
            "val_rmse": _f(_nested(data, "metrics.path_loss.rmse_physical")),
            "val_mae": _f(_nested(data, "metrics.path_loss.mae_physical")),
            "val_rmse_513": _f(_nested(data, "metrics.path_loss_513.rmse_physical")),
            "val_mae_513": _f(_nested(data, "metrics.path_loss_513.mae_physical")),
            # --- train metrics ---
            "train_rmse": _f(_nested(data, "metrics.train_path_loss.rmse_physical")),
            "train_mae": _f(_nested(data, "metrics.train_path_loss.mae_physical")),
            # --- prior ---
            "prior_rmse": _f(_nested(data, "metrics.prior_path_loss.rmse_physical")),
            "prior_mae": _f(_nested(data, "metrics.prior_path_loss.mae_physical")),
            "rmse_gain_vs_prior": _f(_nested(data, "metrics.improvement_vs_prior.rmse_gain_db")),
            # --- LoS / NLoS ---
            "los_rmse": _f(_nested(data, "focus.regimes.path_loss__los__LoS.rmse_physical")),
            "nlos_rmse": _f(_nested(data, "focus.regimes.path_loss__los__NLoS.rmse_physical")),
            "los_mae": _f(_nested(data, "focus.regimes.path_loss__los__LoS.mae_physical")),
            "nlos_mae": _f(_nested(data, "focus.regimes.path_loss__los__NLoS.mae_physical")),
            # --- selection proxy ---
            "composite_nlos_weighted": _f(_nested(data, "selection_proxy.composite_nlos_weighted_rmse")),
            # --- loss ---
            "generator_loss": _f(runtime.get("generator_loss")),
            "lr": _f(runtime.get("learning_rate")),
            "train_seconds": _f(runtime.get("train_seconds")),
            "val_seconds": _f(runtime.get("val_seconds")),
            "huber_loss": _f(
                loss_components["main_map_regression_loss"]
                if "main_map_regression_loss" in loss_components
                else loss_components.get("final_loss", loss_components.get("generator_loss_total"))
            ),
            "multiscale_loss": _f(
                loss_components["term_multiscale"]
                if "term_multiscale" in loss_components
                else loss_components.get("multiscale_loss")
            ),
            "nlos_focus_loss": _f(loss_components.get("nlos_focus_loss")),
            # --- support ---
            "los_fraction": _f(_nested(data, "support.los_fraction")),
            "nlos_fraction": _f(_nested(data, "support.nlos_fraction")),
            # --- EMA info ---
            "val_uses_ema": bool(_nested(data, "model_info.val_uses_ema", False)),
            # --- topology ---
            "topology_class": _nested(data, "focus.topology_class", ""),
        })
    return rows


def _min_finite(values: list[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    return min(finite) if finite else None


def _plot_best_marker(ax, epochs, values, color="#2b8a3e"):
    finite_pairs = [(e, v) for e, v in zip(epochs, values) if math.isfinite(v)]
    if finite_pairs:
        best_e, best_v = min(finite_pairs, key=lambda p: p[1])
        ax.annotate(
            f"{best_v:.2f}",
            (best_e, best_v),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=7.5,
            color=color,
            fontweight="bold",
        )


def plot_expert(expert_id: str, output_dir: Path, save_path: Path | None = None) -> dict[str, Any]:
    rows = load_rows(output_dir)
    if not rows:
        raise ValueError(f"No validate_metrics_epoch_*.json found in {output_dir}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("pip install matplotlib")

    epochs = [r["epoch"] for r in rows]
    topo = rows[0].get("topology_class") or expert_id
    uses_ema = any(r["val_uses_ema"] for r in rows)

    fig, axes = plt.subplots(5, 1, figsize=(12, 18), constrained_layout=True)

    # ---- Panel 0: Overall RMSE (val vs train vs prior) ----
    ax = axes[0]
    val_rmse = [r["val_rmse"] for r in rows]
    train_rmse = [r["train_rmse"] for r in rows]
    prior_rmse = [r["prior_rmse"] for r in rows]
    val_rmse_513 = [r["val_rmse_513"] for r in rows]

    ax.plot(epochs, val_rmse, label="val RMSE", color="#0b7285", linewidth=2.2)
    if any(math.isfinite(v) for v in val_rmse_513):
        ax.plot(epochs, val_rmse_513, label="val RMSE 513", color="#1098ad", linestyle="--", linewidth=1.8)
    ema_suffix = " (online weights)" if uses_ema else ""
    ax.plot(epochs, train_rmse, label=f"train RMSE{ema_suffix}", color="#e8590c", linewidth=1.5, alpha=0.8)
    if any(math.isfinite(v) for v in prior_rmse):
        ax.plot(epochs, prior_rmse, label="prior RMSE", color="#868e96", linestyle=":", linewidth=1.3)
    best_idx = min(range(len(rows)), key=lambda i: rows[i]["val_rmse"] if math.isfinite(rows[i]["val_rmse"]) else 1e9)
    ax.axvline(rows[best_idx]["epoch"], color="#2b8a3e", linestyle=":", linewidth=1.2, label=f"best @ ep {rows[best_idx]['epoch']}")
    _plot_best_marker(ax, epochs, val_rmse)
    ax.set_ylabel("RMSE (dB)")
    ax.set_title(f"Try 71 — {topo} — Overall RMSE" + (" [EMA for val]" if uses_ema else ""))
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    # ---- Panel 1: LoS vs NLoS RMSE ----
    ax = axes[1]
    los_rmse = [r["los_rmse"] for r in rows]
    nlos_rmse = [r["nlos_rmse"] for r in rows]
    ax.plot(epochs, los_rmse, label="LoS RMSE", color="#2f9e44", linewidth=2.0)
    ax.plot(epochs, nlos_rmse, label="NLoS RMSE", color="#c2255c", linewidth=2.0)
    nlos_gap = [
        (n - l) if math.isfinite(n) and math.isfinite(l) else float("nan")
        for n, l in zip(nlos_rmse, los_rmse)
    ]
    ax2 = ax.twinx()
    ax2.plot(epochs, nlos_gap, label="NLoS−LoS gap", color="#ff922b", linewidth=1.2, alpha=0.6)
    ax2.set_ylabel("Gap (dB)", fontsize=8)
    _plot_best_marker(ax, epochs, nlos_rmse, color="#c2255c")
    _plot_best_marker(ax, epochs, los_rmse, color="#2f9e44")
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("LoS vs NLoS RMSE")
    ax.grid(alpha=0.25)
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc="best", fontsize=8)

    # ---- Panel 2: RMSE gain vs prior + selection proxy ----
    ax = axes[2]
    rmse_gain = [r["rmse_gain_vs_prior"] for r in rows]
    composite = [r["composite_nlos_weighted"] for r in rows]
    if any(math.isfinite(v) for v in rmse_gain):
        ax.plot(epochs, rmse_gain, label="RMSE gain vs prior (dB)", color="#5f3dc4", linewidth=1.8)
        ax.axhline(0, color="#adb5bd", linestyle="--", linewidth=0.8)
    if any(math.isfinite(v) for v in composite):
        ax_c = ax.twinx()
        ax_c.plot(epochs, composite, label="composite NLoS-weighted", color="#1c7ed6", linewidth=1.4, alpha=0.7)
        ax_c.set_ylabel("Composite score", fontsize=8)
        lines_c, labels_c = ax_c.get_legend_handles_labels()
    else:
        lines_c, labels_c = [], []
    ax.set_ylabel("RMSE gain (dB)")
    ax.set_title("Improvement vs Prior + Selection Proxy")
    ax.grid(alpha=0.25)
    lines_a, labels_a = ax.get_legend_handles_labels()
    ax.legend(lines_a + lines_c, labels_a + labels_c, loc="best", fontsize=8)

    # ---- Panel 3: Loss components ----
    ax = axes[3]
    gen_loss = [r["generator_loss"] for r in rows]
    huber = [r["huber_loss"] for r in rows]
    ms_loss = [r["multiscale_loss"] for r in rows]
    nlos_fl = [r["nlos_focus_loss"] for r in rows]
    ax.plot(epochs, gen_loss, label="total generator loss", color="#5f3dc4", linewidth=1.8)
    if any(math.isfinite(v) for v in huber):
        ax.plot(epochs, huber, label="main loss (heteroscedastic NLL)", color="#e8590c", linewidth=1.3, alpha=0.8)
    if any(math.isfinite(v) for v in ms_loss):
        ax.plot(epochs, ms_loss, label="multiscale loss", color="#1098ad", linewidth=1.2, alpha=0.7)
    if any(math.isfinite(v) for v in nlos_fl):
        ax.plot(epochs, nlos_fl, label="NLoS focus loss", color="#c2255c", linewidth=1.2, alpha=0.7)
    ax_lr = ax.twinx()
    lr = [r["lr"] for r in rows]
    ax_lr.plot(epochs, lr, label="LR", color="#1c7ed6", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_lr.set_ylabel("LR", fontsize=8)
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components + Learning Rate (Try 71: heteroscedastic NLL, log_var clamped [−10, 10])")
    ax.grid(alpha=0.25)
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax_lr.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc="best", fontsize=8, ncol=2)

    # ---- Panel 4: Timing ----
    ax = axes[4]
    t_train = [r["train_seconds"] for r in rows]
    t_val = [r["val_seconds"] for r in rows]
    t_total = [
        (t + v) if math.isfinite(t) and math.isfinite(v) else float("nan")
        for t, v in zip(t_train, t_val)
    ]
    ax.plot(epochs, t_train, label="train (s)", color="#e8590c", linewidth=1.5)
    ax.plot(epochs, t_val, label="val (s)", color="#1098ad", linewidth=1.5)
    ax.plot(epochs, t_total, label="total (s)", color="#495057", linewidth=1.8)
    avg_total = _min_finite(t_total)
    if avg_total is not None:
        mean_t = sum(v for v in t_total if math.isfinite(v)) / max(sum(1 for v in t_total if math.isfinite(v)), 1)
        ax.axhline(mean_t, color="#868e96", linestyle="--", linewidth=0.8, label=f"mean {mean_t:.0f}s")
    ax.set_ylabel("Seconds")
    ax.set_xlabel("Epoch")
    ax.set_title("Epoch Timing")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    resolved = save_path or output_dir / "metrics_plot_try71.png"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved, dpi=180, bbox_inches="tight")
    plt.close(fig)

    best_row = rows[best_idx]
    return {
        "expert_id": expert_id,
        "topology_class": topo,
        "output_dir": str(output_dir),
        "saved": str(resolved),
        "epochs": len(rows),
        "best_epoch": best_row["epoch"],
        "best_val_rmse": best_row["val_rmse"],
        "best_los_rmse": best_row["los_rmse"],
        "best_nlos_rmse": best_row["nlos_rmse"],
        "best_train_rmse": best_row["train_rmse"],
        "val_uses_ema": uses_ema,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Try 71 single-stage metrics per expert.")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT))
    parser.add_argument("--expert-id", default="")
    parser.add_argument("--output-dir", default="", help="Explicit output dir for a single expert.")
    parser.add_argument("--save-path", default="")
    args = parser.parse_args()

    if args.output_dir:
        eid = args.expert_id or _expert_key(Path(args.output_dir))
        summary = plot_expert(eid, Path(args.output_dir), Path(args.save_path) if args.save_path else None)
        print(json.dumps(summary, indent=2))
        return

    root = Path(args.root_dir)
    dirs = _expert_dirs(root)
    if args.expert_id:
        dirs = [d for d in dirs if _expert_key(d) == args.expert_id]
    if not dirs:
        raise SystemExit(f"No expert output dirs found under {root}")

    summaries: list[dict[str, Any]] = []
    for d in dirs:
        eid = _expert_key(d)
        save = Path(args.save_path) / f"{eid}_metrics_try71.png" if args.save_path else None
        try:
            summaries.append(plot_expert(eid, d, save))
        except ValueError as exc:
            print(f"[skip] {eid}: {exc}")

    print(json.dumps({"root_dir": str(root), "plotted": summaries}, indent=2))


if __name__ == "__main__":
    main()
