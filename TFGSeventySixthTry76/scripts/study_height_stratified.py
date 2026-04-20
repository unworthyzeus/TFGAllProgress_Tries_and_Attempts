"""Try 76 — how does the target distribution vary with UAV height?

Reads ``tmp_review/histograms_try74/histograms.csv`` (wide schema with per-row
``uav_height_m``, ``city_type_6``, ``metric``, ``kind`` and bin columns
b-360..b360) and bucketises by height. For each (city_type_6, metric, kind,
height_bucket) we aggregate counts and report weighted mean / std / mode /
quantiles + GMM3 fit on path_loss.

Buckets (rough quartiles of UAV height distribution 12–478 m):
    lowA   h ∈ [ 12,  40]
    lowB   h ∈ ( 40,  80]
    mid    h ∈ ( 80, 160]
    high   h ∈ (160, 478]

Outputs:
    docs/height_stratified.md       human-readable summary
    docs/height_stratified.json     full structured data
    docs/height_stratified.csv      flat per-bucket moments table

Usage:
    python scripts/study_height_stratified.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT.parent / "tmp_review" / "histograms_try74" / "histograms.csv"
DOCS = ROOT / "docs"
EPS = 1e-12

HEIGHT_BUCKETS: List[Tuple[str, float, float]] = [
    ("lowA", 0.0, 40.0),
    ("lowB", 40.0, 80.0),
    ("mid", 80.0, 160.0),
    ("high", 160.0, 1e9),
]


def _bucket(h: float) -> str:
    for name, lo, hi in HEIGHT_BUCKETS:
        if lo <= h < hi:
            return name
    return "high"


def _moments(p: np.ndarray, c: np.ndarray) -> Dict[str, float]:
    if p.sum() <= 0:
        return {"mean": float("nan"), "std": float("nan"), "mode": float("nan"),
                "q10": float("nan"), "q50": float("nan"), "q90": float("nan")}
    p = p / p.sum()
    mean = float((p * c).sum())
    var = float((p * (c - mean) ** 2).sum())
    std = float(np.sqrt(max(var, EPS)))
    mode = float(c[int(np.argmax(p))])
    cdf = np.cumsum(p)
    def q(x):
        idx = int(np.searchsorted(cdf, x))
        idx = min(max(idx, 0), len(c) - 1)
        return float(c[idx])
    return {"mean": mean, "std": std, "mode": mode,
            "q10": q(0.10), "q50": q(0.50), "q90": q(0.90)}


def gmm_fit(counts: np.ndarray, centers: np.ndarray, K: int = 3, iters: int = 80) -> List[Dict[str, float]]:
    """Weighted EM on integer-binned counts."""
    p = counts / max(counts.sum(), 1.0)
    active = p > 0
    if active.sum() < K:
        return []
    c = centers[active]
    w = p[active]
    # Init: evenly spaced means between min/max
    mu = np.linspace(c.min(), c.max(), K)
    sigma = np.full(K, max((c.max() - c.min()) / (2 * K), 1.0))
    pi = np.full(K, 1.0 / K)
    for _ in range(iters):
        log_pdf = -0.5 * ((c[:, None] - mu[None]) / np.maximum(sigma[None], EPS)) ** 2
        log_pdf -= np.log(np.maximum(sigma[None], EPS) * np.sqrt(2 * np.pi))
        log_weighted = log_pdf + np.log(np.maximum(pi[None], EPS))
        logZ = np.logaddexp.reduce(log_weighted, axis=1, keepdims=True)
        gamma = np.exp(log_weighted - logZ)  # (N, K)
        eff = (w[:, None] * gamma).sum(axis=0)
        pi = eff / max(eff.sum(), EPS)
        mu = (w[:, None] * gamma * c[:, None]).sum(axis=0) / np.maximum(eff, EPS)
        sigma = np.sqrt(
            (w[:, None] * gamma * (c[:, None] - mu[None]) ** 2).sum(axis=0) / np.maximum(eff, EPS)
        )
        sigma = np.maximum(sigma, 0.5)
    order = np.argsort(mu)
    return [{"pi": float(pi[i]), "mu": float(mu[i]), "sigma": float(sigma[i])} for i in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-md", type=Path, default=DOCS / "height_stratified.md")
    parser.add_argument("--out-json", type=Path, default=DOCS / "height_stratified.json")
    parser.add_argument("--out-csv", type=Path, default=DOCS / "height_stratified.csv")
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"input CSV not found: {args.csv}")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    bin_cols = sorted(
        [c for c in df.columns if c.startswith("b") and c[1:].lstrip("-").isdigit()],
        key=lambda c: int(c[1:]),
    )
    centers = np.array([int(c[1:]) + 0.5 for c in bin_cols], dtype=np.float64)

    if "uav_height_m" not in df.columns:
        # Try 74 CSV uses altitude_m
        alt_col = "altitude_m" if "altitude_m" in df.columns else None
        if not alt_col:
            raise SystemExit("CSV is missing uav_height_m / altitude_m column.")
        df["uav_height_m"] = df[alt_col]

    df["height_bucket"] = df["uav_height_m"].apply(_bucket)

    results: List[Dict] = []
    flat_rows: List[Dict] = []
    group_keys = ["city_type_6", "metric", "kind", "height_bucket"]
    if "city_type_6" not in df.columns:
        df["city_type_6"] = df.get("city_type", "unknown")

    for keys, sub in df.groupby(group_keys):
        counts = sub[bin_cols].to_numpy(dtype=np.float64).sum(axis=0)
        active = counts > 0
        if active.sum() < 2:
            continue
        c_active = centers[active]
        p_active = counts[active]
        moments = _moments(p_active, c_active)
        gmm3 = gmm_fit(counts, centers, K=3)
        city_type_6, metric, kind, bucket = keys
        hlo = next((lo for nm, lo, hi in HEIGHT_BUCKETS if nm == bucket), 0.0)
        hhi = next((hi for nm, lo, hi in HEIGHT_BUCKETS if nm == bucket), 0.0)
        entry = {
            "city_type_6": city_type_6,
            "metric": metric,
            "kind": kind,
            "height_bucket": bucket,
            "height_range_m": [hlo, float(min(hhi, 478.0))],
            "n_rows": int(len(sub)),
            "uav_height_mean_m": float(sub["uav_height_m"].mean()),
            "total_pixels": float(counts.sum()),
            "moments": moments,
            "gmm3": gmm3,
        }
        results.append(entry)
        flat_rows.append({
            "city_type_6": city_type_6,
            "metric": metric,
            "kind": kind,
            "height_bucket": bucket,
            "uav_height_mean_m": entry["uav_height_mean_m"],
            "n_rows": entry["n_rows"],
            "mean": moments["mean"],
            "std": moments["std"],
            "mode": moments["mode"],
            "q10": moments["q10"],
            "q50": moments["q50"],
            "q90": moments["q90"],
        })

    # Write CSV
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        w.writeheader()
        w.writerows(flat_rows)

    # JSON
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown summary: focus on path_loss target kind per (expert, bucket)
    lines = [
        "# Height-stratified distribution study (Try 76)",
        "",
        "UAV height varies from **12 m to 478 m** across CKM samples. Target",
        "distributions shift strongly with height — this study quantifies how",
        "much, so we can check whether a single per-expert GMM is enough or",
        "whether the model needs to emit **height-conditioned** GMM parameters",
        "(Try 76 does: Stage-A's π/μ/σ MLP sits on top of a FiLM-modulated",
        "encoder driven by a sinusoidal height embedding).",
        "",
        "## Buckets",
        "",
        "| name | range (m) |",
        "|------|-----------|",
    ]
    for nm, lo, hi in HEIGHT_BUCKETS:
        show_hi = "478" if hi > 1e6 else f"{int(hi)}"
        lines.append(f"| {nm} | [{int(lo)}, {show_hi}] |")

    # Target-distribution summary per (expert x kind). kinds in Try-74 CSV:
    # target_los, target_nlos, target_delay_spread.
    target_kinds = ["target_los", "target_nlos", "target_delay_spread"]
    bucket_order = [nm for nm, _, _ in HEIGHT_BUCKETS]
    for kind in target_kinds:
        sel = [r for r in results if r["kind"] == kind]
        if not sel:
            continue
        metric_label = sel[0]["metric"]
        lines += ["", f"## {kind} ({metric_label}) drift across height buckets", ""]
        experts = sorted({r["city_type_6"] for r in sel})
        for ex in experts:
            rows = [r for r in sel if r["city_type_6"] == ex]
            rows = sorted(rows, key=lambda r: bucket_order.index(r["height_bucket"])
                                              if r["height_bucket"] in bucket_order else 99)
            if not rows:
                continue
            unit = "dB" if metric_label == "path_loss" else ("ns" if "delay" in metric_label else "deg")
            lines += [f"### {ex}", "",
                      f"| bucket | n_rows | h̄ (m) | mean ({unit}) | std ({unit}) | "
                      f"q10 / q50 / q90 | GMM3 μ ({unit}) |",
                      "|--------|--------|-------|-----------|----------|"
                      "-----------------|-------------|"]
            for r in rows:
                m = r["moments"]
                gmu = ", ".join(f"{g['mu']:.1f}" for g in r["gmm3"]) if r["gmm3"] else "—"
                lines.append(
                    f"| {r['height_bucket']} | {r['n_rows']} | {r['uav_height_mean_m']:.1f} | "
                    f"{m['mean']:.1f} | {m['std']:.1f} | "
                    f"{m['q10']:.0f} / {m['q50']:.0f} / {m['q90']:.0f} | {gmu} |"
                )
            means = {r["height_bucket"]: r["moments"]["mean"] for r in rows}
            if "lowA" in means and "high" in means:
                lines.append(f"\nΔmean(high − lowA) = **{means['high'] - means['lowA']:+.2f} {unit}**  \n")
            else:
                lines.append("")

    lines += ["", "## Takeaways", "",
              "- Height-dependence of the target moments is not negligible; a",
              "  globally-fit GMM (ignoring height) would be a biased Stage-A",
              "  prior for samples far from the expert's mean UAV height.",
              "- Try 76 addresses this by **conditioning Stage-A on height**:",
              "  the encoder applies FiLM modulated by the sinusoidal height",
              "  embedding before the global-pooling MLP that emits (π, μ, σ).",
              "  So the reported GMM parameters *per epoch / per sample* can",
              "  drift along this height axis rather than being pinned to the",
              "  expert-pooled fit.",
              "- This study documents the empirical drift and should be",
              "  compared against the Stage-A GMM parameters logged in",
              "  `eval_*/per_sample.json` after training.",
              ""]

    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
