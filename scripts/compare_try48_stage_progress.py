#!/usr/bin/env python3
"""
Compare Try 48 Stage 1 and Stage 2 validation evolution against the prior.

Outputs:
- try48_stage_comparison_overview.png
- try48_stage_regime_deltas.png
- try48_stage_losses_and_proxies.png
- try48_stage_comparison_summary.json
- try48_stage_comparison_table.csv

Usage example:
  python scripts/compare_try48_stage_progress.py
  python scripts/compare_try48_stage_progress.py --show
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

DEFAULT_STAGE1_DIR = ROOT / "cluster_outputs" / "TFGFortyEighthTry48" / "fortyeighthtry48_pmnet_prior_stage1_t48_stage1_sep_base_4gpu"
DEFAULT_STAGE2_DIR = ROOT / "cluster_outputs" / "TFGFortyEighthTry48" / "fortyeighthtry48_pmnet_prior_stage2_t48_stage2_sep_4gpu"
DEFAULT_OUT_DIR = ROOT / "cluster_plots" / "try48_stage_compare"

EPOCH_RE = re.compile(r"^validate_metrics_epoch_(\d+)_cgan\.json$", re.IGNORECASE)


@dataclass
class EpochMetrics:
    stage: str
    epoch: int
    model_rmse: Optional[float]
    model_mae: Optional[float]
    model_mse: Optional[float]
    prior_rmse: Optional[float]
    prior_mae: Optional[float]
    prior_mse: Optional[float]
    los_rmse: Optional[float]
    los_mae: Optional[float]
    prior_los_rmse: Optional[float]
    prior_los_mae: Optional[float]
    nlos_rmse: Optional[float]
    nlos_mae: Optional[float]
    prior_nlos_rmse: Optional[float]
    prior_nlos_mae: Optional[float]
    gen_loss: Optional[float]
    disc_loss: Optional[float]

    @property
    def rmse_delta_vs_prior(self) -> Optional[float]:
        if self.model_rmse is None or self.prior_rmse is None:
            return None
        return self.model_rmse - self.prior_rmse

    @property
    def mae_delta_vs_prior(self) -> Optional[float]:
        if self.model_mae is None or self.prior_mae is None:
            return None
        return self.model_mae - self.prior_mae

    @property
    def los_rmse_delta_vs_prior(self) -> Optional[float]:
        if self.los_rmse is None or self.prior_los_rmse is None:
            return None
        return self.los_rmse - self.prior_los_rmse

    @property
    def nlos_rmse_delta_vs_prior(self) -> Optional[float]:
        if self.nlos_rmse is None or self.prior_nlos_rmse is None:
            return None
        return self.nlos_rmse - self.prior_nlos_rmse

    @property
    def nlos_los_gap_rmse(self) -> Optional[float]:
        if self.nlos_rmse is None or self.los_rmse is None:
            return None
        return self.nlos_rmse - self.los_rmse



def _fget(d: Dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None



def load_epoch_metrics(run_dir: Path, stage_name: str) -> List[EpochMetrics]:
    if not run_dir.exists():
        return []
    rows: List[EpochMetrics] = []
    for path in sorted(run_dir.glob("validate_metrics_epoch_*_cgan.json")):
        m = EPOCH_RE.match(path.name)
        if not m:
            continue
        epoch = int(m.group(1))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        ck_epoch = _fget(payload, "_checkpoint", "epoch")
        if ck_epoch is not None:
            epoch = int(ck_epoch)

        regs = payload.get("_regimes", {})
        train = payload.get("_train", {})
        rows.append(
            EpochMetrics(
                stage=stage_name,
                epoch=epoch,
                model_rmse=_fget(payload, "path_loss", "rmse_physical"),
                model_mae=_fget(payload, "path_loss", "mae_physical"),
                model_mse=_fget(payload, "path_loss", "mse_physical"),
                prior_rmse=_fget(payload, "path_loss__prior__overall", "rmse_physical"),
                prior_mae=_fget(payload, "path_loss__prior__overall", "mae_physical"),
                prior_mse=_fget(payload, "path_loss__prior__overall", "mse_physical"),
                los_rmse=_fget(regs, "path_loss__los__LoS", "rmse_physical"),
                los_mae=_fget(regs, "path_loss__los__LoS", "mae_physical"),
                prior_los_rmse=_fget(regs, "path_loss__prior__los__LoS", "rmse_physical"),
                prior_los_mae=_fget(regs, "path_loss__prior__los__LoS", "mae_physical"),
                nlos_rmse=_fget(regs, "path_loss__los__NLoS", "rmse_physical"),
                nlos_mae=_fget(regs, "path_loss__los__NLoS", "mae_physical"),
                prior_nlos_rmse=_fget(regs, "path_loss__prior__los__NLoS", "rmse_physical"),
                prior_nlos_mae=_fget(regs, "path_loss__prior__los__NLoS", "mae_physical"),
                gen_loss=_fget(train, "generator_loss"),
                disc_loss=_fget(train, "discriminator_loss"),
            )
        )
    rows.sort(key=lambda r: r.epoch)
    return rows



def _best_by_rmse(rows: List[EpochMetrics]) -> Optional[EpochMetrics]:
    valid = [r for r in rows if r.model_rmse is not None]
    if not valid:
        return None
    return min(valid, key=lambda r: float(r.model_rmse))



def write_csv(rows: List[EpochMetrics], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "stage",
                "epoch",
                "model_rmse",
                "prior_rmse",
                "rmse_delta_vs_prior",
                "model_mae",
                "prior_mae",
                "mae_delta_vs_prior",
                "los_rmse_delta_vs_prior",
                "nlos_rmse_delta_vs_prior",
                "nlos_los_gap_rmse",
                "generator_loss",
                "discriminator_loss",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.stage,
                    r.epoch,
                    r.model_rmse,
                    r.prior_rmse,
                    r.rmse_delta_vs_prior,
                    r.model_mae,
                    r.prior_mae,
                    r.mae_delta_vs_prior,
                    r.los_rmse_delta_vs_prior,
                    r.nlos_rmse_delta_vs_prior,
                    r.nlos_los_gap_rmse,
                    r.gen_loss,
                    r.disc_loss,
                ]
            )



def write_summary_json(stage1_rows: List[EpochMetrics], stage2_rows: List[EpochMetrics], out_json: Path) -> None:
    def row_to_dict(r: Optional[EpochMetrics]) -> Optional[Dict[str, Any]]:
        if r is None:
            return None
        return {
            "stage": r.stage,
            "epoch": r.epoch,
            "model_rmse": r.model_rmse,
            "prior_rmse": r.prior_rmse,
            "rmse_delta_vs_prior": r.rmse_delta_vs_prior,
            "model_mae": r.model_mae,
            "prior_mae": r.prior_mae,
            "mae_delta_vs_prior": r.mae_delta_vs_prior,
            "los_rmse_delta_vs_prior": r.los_rmse_delta_vs_prior,
            "nlos_rmse_delta_vs_prior": r.nlos_rmse_delta_vs_prior,
            "nlos_los_gap_rmse": r.nlos_los_gap_rmse,
            "generator_loss": r.gen_loss,
            "discriminator_loss": r.disc_loss,
        }

    stage1_best = _best_by_rmse(stage1_rows)
    stage2_best = _best_by_rmse(stage2_rows)
    stage1_latest = stage1_rows[-1] if stage1_rows else None
    stage2_latest = stage2_rows[-1] if stage2_rows else None

    payload = {
        "stage1": {
            "epochs_found": len(stage1_rows),
            "latest": row_to_dict(stage1_latest),
            "best_by_rmse": row_to_dict(stage1_best),
        },
        "stage2": {
            "epochs_found": len(stage2_rows),
            "latest": row_to_dict(stage2_latest),
            "best_by_rmse": row_to_dict(stage2_best),
        },
        "notes": {
            "stage2_model_is_combined": True,
            "meaning": "Stage 2 metrics correspond to (Stage1 base + Stage2 refiner) compared against the same prior baseline.",
            "adversarial_activity_proxy": "discriminator_loss (higher than zero indicates active adversarial training).",
            "hard_case_proxy": "nlos_los_gap_rmse (larger means harder NLoS relative to LoS).",
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def _plot_series(ax: Any, rows: List[EpochMetrics], y_getter: Any, label: str, color: str) -> None:
    xs = [r.epoch for r in rows]
    ys = [y_getter(r) for r in rows]
    xs2: List[int] = []
    ys2: List[float] = []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        xs2.append(x)
        ys2.append(float(y))
    if xs2:
        ax.plot(xs2, ys2, marker="o", markersize=3.2, linewidth=1.8, label=label, color=color)



def make_plots(stage1_rows: List[EpochMetrics], stage2_rows: List[EpochMetrics], out_dir: Path, show: bool, dpi: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install it with pip install matplotlib") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    c_stage1 = "#1f77b4"
    c_stage2 = "#d62728"
    c_prior = "#555555"

    # Figure 1: overall path-loss trend
    fig1, axs1 = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    _plot_series(axs1[0], stage1_rows, lambda r: r.model_rmse, "Stage1 model RMSE", c_stage1)
    _plot_series(axs1[0], stage2_rows, lambda r: r.model_rmse, "Stage2 combined RMSE", c_stage2)
    _plot_series(axs1[0], stage1_rows, lambda r: r.prior_rmse, "Prior RMSE", c_prior)
    axs1[0].set_title("Try48 Overall RMSE vs Epoch")
    axs1[0].set_ylabel("RMSE (dB)")
    axs1[0].legend(loc="best")

    _plot_series(axs1[1], stage1_rows, lambda r: r.model_mae, "Stage1 model MAE", c_stage1)
    _plot_series(axs1[1], stage2_rows, lambda r: r.model_mae, "Stage2 combined MAE", c_stage2)
    _plot_series(axs1[1], stage1_rows, lambda r: r.prior_mae, "Prior MAE", c_prior)
    axs1[1].set_title("Try48 Overall MAE vs Epoch")
    axs1[1].set_ylabel("MAE (dB)")
    axs1[1].set_xlabel("Epoch")
    axs1[1].legend(loc="best")

    fig1.tight_layout()
    fig1.savefig(out_dir / "try48_stage_comparison_overview.png", dpi=dpi, bbox_inches="tight")

    # Figure 2: regime deltas vs prior
    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    axs2[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    _plot_series(axs2[0], stage1_rows, lambda r: r.los_rmse_delta_vs_prior, "Stage1 LoS RMSE delta", c_stage1)
    _plot_series(axs2[0], stage1_rows, lambda r: r.nlos_rmse_delta_vs_prior, "Stage1 NLoS RMSE delta", "#17becf")
    _plot_series(axs2[0], stage2_rows, lambda r: r.los_rmse_delta_vs_prior, "Stage2 LoS RMSE delta", c_stage2)
    _plot_series(axs2[0], stage2_rows, lambda r: r.nlos_rmse_delta_vs_prior, "Stage2 NLoS RMSE delta", "#ff9896")
    axs2[0].set_title("LoS/NLoS RMSE Delta vs Prior (negative is better)")
    axs2[0].set_ylabel("Delta RMSE (dB)")
    axs2[0].legend(loc="best", ncol=2)

    axs2[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    _plot_series(axs2[1], stage1_rows, lambda r: r.rmse_delta_vs_prior, "Stage1 overall RMSE delta", c_stage1)
    _plot_series(axs2[1], stage2_rows, lambda r: r.rmse_delta_vs_prior, "Stage2 overall RMSE delta", c_stage2)
    _plot_series(axs2[1], stage1_rows, lambda r: r.mae_delta_vs_prior, "Stage1 overall MAE delta", "#2ca02c")
    _plot_series(axs2[1], stage2_rows, lambda r: r.mae_delta_vs_prior, "Stage2 overall MAE delta", "#98df8a")
    axs2[1].set_title("Overall Delta vs Prior")
    axs2[1].set_ylabel("Delta (dB)")
    axs2[1].set_xlabel("Epoch")
    axs2[1].legend(loc="best", ncol=2)

    fig2.tight_layout()
    fig2.savefig(out_dir / "try48_stage_regime_deltas.png", dpi=dpi, bbox_inches="tight")

    # Figure 3: losses + proxy metrics
    fig3, axs3 = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    _plot_series(axs3[0], stage1_rows, lambda r: r.gen_loss, "Stage1 generator loss", c_stage1)
    _plot_series(axs3[0], stage2_rows, lambda r: r.gen_loss, "Stage2 generator loss", c_stage2)
    _plot_series(axs3[0], stage1_rows, lambda r: r.disc_loss, "Stage1 discriminator loss", "#9467bd")
    _plot_series(axs3[0], stage2_rows, lambda r: r.disc_loss, "Stage2 discriminator loss", "#c5b0d5")
    axs3[0].set_title("Training Losses")
    axs3[0].set_ylabel("Loss")
    axs3[0].legend(loc="best", ncol=2)

    _plot_series(axs3[1], stage1_rows, lambda r: r.nlos_los_gap_rmse, "Stage1 NLoS-LoS RMSE gap", c_stage1)
    _plot_series(axs3[1], stage2_rows, lambda r: r.nlos_los_gap_rmse, "Stage2 NLoS-LoS RMSE gap", c_stage2)
    axs3[1].set_title("Hard-Case Proxy: NLoS-LoS RMSE Gap")
    axs3[1].set_ylabel("Gap (dB)")
    axs3[1].set_xlabel("Epoch")
    axs3[1].legend(loc="best")

    fig3.tight_layout()
    fig3.savefig(out_dir / "try48_stage_losses_and_proxies.png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create cool multi-plot comparison for Try48 prior vs Stage1 vs Stage2")
    p.add_argument("--stage1-dir", type=str, default=str(DEFAULT_STAGE1_DIR))
    p.add_argument("--stage2-dir", type=str, default=str(DEFAULT_STAGE2_DIR))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--show", action="store_true")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    stage1_dir = Path(args.stage1_dir).resolve()
    stage2_dir = Path(args.stage2_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    stage1_rows = load_epoch_metrics(stage1_dir, "stage1")
    stage2_rows = load_epoch_metrics(stage2_dir, "stage2")

    if not stage1_rows and not stage2_rows:
        raise SystemExit("No epoch metrics found for either stage. Check --stage1-dir/--stage2-dir.")

    all_rows = stage1_rows + stage2_rows
    all_rows.sort(key=lambda r: (r.stage, r.epoch))

    write_csv(all_rows, out_dir / "try48_stage_comparison_table.csv")
    write_summary_json(stage1_rows, stage2_rows, out_dir / "try48_stage_comparison_summary.json")
    make_plots(stage1_rows, stage2_rows, out_dir, show=bool(args.show), dpi=int(args.dpi))

    print(f"[ok] stage1_epochs={len(stage1_rows)} stage2_epochs={len(stage2_rows)}")
    print(f"[ok] wrote artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
