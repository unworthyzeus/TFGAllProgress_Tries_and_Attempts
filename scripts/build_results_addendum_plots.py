"""Build thesis addendum plots from current DirectML test summaries."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
THESIS_OUT = ROOT.parent / "FINAL_THESIS" / "TFG" / "img" / "thesis_figures" / "results_addendum"
TRY76_METRICS = ROOT / "TFGSeventySixthTry76" / "tmp_try76_directml_test_metrics.json"
TRY77_METRICS = ROOT / "TFGSeventySeventhTry77" / "tmp_try77_directml_test_metrics.json"
TRY76_HIST = ROOT / "TFGSeventySixthTry76" / "docs" / "histogram_study.json"
TRY76_CLASSES = ROOT / "TFGSeventySixthTry76" / "docs" / "distribution_classes.json"

ORDER = [
    "open_sparse_lowrise",
    "open_sparse_vertical",
    "mixed_compact_lowrise",
    "mixed_compact_midrise",
    "dense_block_midrise",
    "dense_block_highrise",
]
LABELS = [
    "open\nlow",
    "open\nvertical",
    "mixed\nlow",
    "mixed\nmid",
    "dense\nmid",
    "dense\nhigh",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, stem: str) -> None:
    THESIS_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(THESIS_OUT / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(THESIS_OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_try76() -> None:
    metrics = load_json(TRY76_METRICS)
    hist = load_json(TRY76_HIST)["by_class"]
    classes = load_json(TRY76_CLASSES)["by_class"]
    by = {(row["topology_class"], row["region_mode"]): row for row in metrics["per_expert"]}

    los = np.array([by[(cls, "los_only")]["pixel_weighted_rmse_db"] for cls in ORDER])
    nlos = np.array([by[(cls, "nlos_only")]["pixel_weighted_rmse_db"] for cls in ORDER])
    los_sigma = np.array([hist[cls]["path_loss|target_los"]["std"] for cls in ORDER])
    nlos_sigma = np.array([hist[cls]["path_loss|target_nlos"]["std"] for cls in ORDER])

    x = np.arange(len(ORDER))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    ax.bar(x - width / 2, los, width, label="LoS RMSE", color="#d9822b")
    ax.bar(x + width / 2, nlos, width, label="NLoS RMSE", color="#2f8f5b")
    ax.plot(x - width / 2, los_sigma, "D", color="#7a3f10", label="LoS target sigma", markersize=5)
    ax.plot(x + width / 2, nlos_sigma, "D", color="#155b37", label="NLoS target sigma", markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("dB")
    ax.set_title("Test RMSE and target spread")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)

    ax = axes[1]
    ax.bar(x - width / 2, los / los_sigma, width, label="LoS", color="#d9822b")
    ax.bar(x + width / 2, nlos / nlos_sigma, width, label="NLoS", color="#2f8f5b")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("RMSE / target sigma")
    ax.set_title("Normalised unexplained spread")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Try 76 DirectML test evaluation by expert", y=1.03)
    save(fig, "try76_los_vs_nlos")

    xs = np.linspace(72, 128, 700)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.8), sharex=True, sharey=True)
    for ax, cls, label in zip(axes.flat, ORDER, LABELS):
        for key, color, regime in [
            ("path_loss|target_los", "#d9822b", "LoS"),
            ("path_loss|target_nlos", "#2f8f5b", "NLoS"),
        ]:
            entry = classes[cls][key]
            pdf = np.zeros_like(xs)
            for pi, mu, sigma in entry["gmm5_components"]:
                sigma = float(sigma)
                mu = float(mu)
                component_pdf = (
                    float(pi)
                    * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
                    / (sigma * np.sqrt(2 * np.pi))
                )
                pdf += component_pdf
                ax.plot(xs, component_pdf, color=color, linewidth=0.8, linestyle=":", alpha=0.38)
            ax.plot(xs, pdf, color=color, linewidth=1.9, label=f"{regime} target")
            ax.axvline(float(entry["moments"]["mean"]), color=color, linestyle="--", alpha=0.45, linewidth=0.9)
        ax.set_title(label.replace("\n", " "), fontsize=10)
        ax.grid(alpha=0.22)
    for ax in axes[:, 0]:
        ax.set_ylabel("GMM density")
    for ax in axes[-1, :]:
        ax.set_xlabel("Path loss (dB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Try 76 target GMM-5 fits per topology expert", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, "try76_los_vs_nlos_distribution")


def plot_try77() -> None:
    metrics = load_json(TRY77_METRICS)
    by = {(row["topology_class"], row["metric"]): row for row in metrics["per_expert"]}
    delay_sample = np.array([by[(cls, "delay_spread")]["sample_mean_rmse"] for cls in ORDER])
    delay_pixel = np.array([by[(cls, "delay_spread")]["pixel_weighted_rmse"] for cls in ORDER])
    angular_sample = np.array([by[(cls, "angular_spread")]["sample_mean_rmse"] for cls in ORDER])
    angular_pixel = np.array([by[(cls, "angular_spread")]["pixel_weighted_rmse"] for cls in ORDER])

    x = np.arange(len(ORDER))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    ax = axes[0]
    ax.bar(x - width / 2, delay_sample, width, label="sample mean", color="#4c78a8")
    ax.bar(x + width / 2, delay_pixel, width, label="pixel weighted", color="#9ecae9")
    ax.axhline(50, linestyle="--", linewidth=1, color="#555", label="50 ns target")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("RMSE (ns)")
    ax.set_title("Delay spread")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - width / 2, angular_sample, width, label="sample mean", color="#7b5ea7")
    ax.bar(x + width / 2, angular_pixel, width, label="pixel weighted", color="#c5b0d5")
    ax.axhline(20, linestyle="--", linewidth=1, color="#555", label="20 degree target")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("RMSE (degrees)")
    ax.set_title("Angular spread")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Try 77 DirectML test evaluation by expert", y=1.03)
    save(fig, "try77_per_expert")


def plot_try78_by_topology() -> None:
    try76 = load_json(TRY76_METRICS)
    by76 = {row["topology_class"]: row for row in try76["per_expert"] if row["region_mode"] == "nlos_only"}
    try76_nlos = np.array([by76[cls]["pixel_weighted_rmse_db"] for cls in ORDER])

    try78_los = np.array([1.39, 1.43, 1.81, 1.91, 2.19, 2.44])
    try78_nlos = np.array([2.92, 3.91, 2.91, 3.84, 3.20, 4.09])
    try78_all = np.array([1.43, 1.55, 1.90, 2.21, 2.40, 2.97])

    x = np.arange(len(ORDER))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    ax.bar(x - width, try78_los, width, label="LoS", color="#d9822b")
    ax.bar(x, try78_nlos, width, label="NLoS", color="#2f8f5b")
    ax.bar(x + width, try78_all, width, label="overall", color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("Try 78 hybrid prior")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - width / 2, try76_nlos, width, label="Try 76 DL GMM", color="#2f8f5b")
    ax.bar(x + width / 2, try78_nlos, width, label="Try 78 analytic prior", color="#8f6bb1")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("NLoS RMSE (dB)")
    ax.set_title("NLoS comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Path-loss test/evaluation holdout by topology", y=1.03)
    save(fig, "try78_by_topology")


def main() -> None:
    plot_try76()
    plot_try77()
    plot_try78_by_topology()


if __name__ == "__main__":
    main()
