"""Try 76 — histogram study.

Aggregate the per-sample dB histograms produced by the Try 74 review script
(``tmp_review/histograms_try74/histograms.csv``) and summarise the *target*
distribution shapes per (topology class, LoS/NLoS region, metric).

The goal is purely descriptive: we want to know what the target pixel-value
distributions look like so the Try 76 network can output a parametric
description of them (mixture of Gaussians over dB) rather than trying to
regress every pixel independently.

Outputs (under the same tmp_review folder for repeatability):

    docs/histogram_study.json        full numeric summary
    docs/histogram_study.md          human-readable table

Usage:
    python scripts/study_histograms.py \
        --csv ../tmp_review/histograms_try74/histograms.csv

No dependencies beyond numpy + pandas.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


BIN_LO_DEFAULT = -360
BIN_HI_DEFAULT = 361  # exclusive upper for bin b<HI-1>
METRICS = ("path_loss", "delay_spread", "angular_spread")
GROUPING_KEYS = ("city_type_6", "metric", "kind")


@dataclass
class DistSummary:
    total_pixels: int
    mean: float
    std: float
    skew: float
    kurt: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    mode: float
    nonzero_lo: int
    nonzero_hi: int
    gmm2: List[Tuple[float, float, float]]  # [(pi, mu, sigma), ...]


def load_csv(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path, low_memory=False)
    bin_cols = [c for c in df.columns if c.startswith("b") and c[1:].lstrip("-").isdigit()]
    bin_cols.sort(key=lambda c: int(c[1:]))
    bin_centers = np.array([int(c[1:]) + 0.5 for c in bin_cols], dtype=np.float64)
    return df, bin_centers, bin_cols


def aggregate_counts(group: pd.DataFrame, bin_cols: List[str]) -> np.ndarray:
    mat = group[bin_cols].to_numpy(dtype=np.float64, copy=False)
    return mat.sum(axis=0)


def weighted_moments(counts: np.ndarray, centers: np.ndarray) -> Tuple[float, float, float, float]:
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    p = counts / total
    mu = float((p * centers).sum())
    var = float((p * (centers - mu) ** 2).sum())
    std = float(np.sqrt(max(var, 1e-12)))
    skew = float((p * ((centers - mu) / max(std, 1e-6)) ** 3).sum())
    kurt = float((p * ((centers - mu) / max(std, 1e-6)) ** 4).sum() - 3.0)
    return mu, std, skew, kurt


def weighted_quantiles(counts: np.ndarray, centers: np.ndarray, qs: Iterable[float]) -> List[float]:
    total = counts.sum()
    if total <= 0:
        return [0.0 for _ in qs]
    cdf = np.cumsum(counts) / total
    out = []
    for q in qs:
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = max(0, min(idx, centers.size - 1))
        out.append(float(centers[idx]))
    return out


def nonzero_range(counts: np.ndarray, centers: np.ndarray) -> Tuple[int, int]:
    nz = np.flatnonzero(counts > 0)
    if nz.size == 0:
        return 0, 0
    return int(round(centers[nz[0]] - 0.5)), int(round(centers[nz[-1]] - 0.5))


def fit_gmm_1d(counts: np.ndarray, centers: np.ndarray, K: int = 2, iters: int = 60) -> List[Tuple[float, float, float]]:
    """Weighted EM on binned 1D data. Returns K components sorted by mu."""
    counts = np.asarray(counts, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    total = counts.sum()
    if total <= 0 or K < 1:
        return [(1.0 / max(K, 1), 0.0, 1.0)] * max(K, 1)

    w = counts / total
    mu, std, *_ = weighted_moments(counts, centers)
    rng = np.random.default_rng(42)
    if K == 1:
        return [(1.0, float(mu), float(max(std, 1.0)))]

    qs = weighted_quantiles(counts, centers, np.linspace(0.1, 0.9, K))
    mus = np.array(qs, dtype=np.float64)
    sigmas = np.full(K, max(std, 5.0), dtype=np.float64)
    pis = np.full(K, 1.0 / K, dtype=np.float64)

    N = centers.size
    for _ in range(iters):
        # E-step
        diff = centers[None, :] - mus[:, None]
        log_pdf = -0.5 * (diff / sigmas[:, None]) ** 2 - np.log(sigmas[:, None] * np.sqrt(2 * np.pi))
        log_weighted = np.log(pis[:, None] + 1e-12) + log_pdf
        m = log_weighted.max(axis=0, keepdims=True)
        log_norm = m + np.log(np.exp(log_weighted - m).sum(axis=0, keepdims=True) + 1e-30)
        log_resp = log_weighted - log_norm
        resp = np.exp(log_resp)  # (K, N)

        # Weight each bin by count
        r_w = resp * w[None, :]
        nk = r_w.sum(axis=1) + 1e-12
        mus = (r_w * centers[None, :]).sum(axis=1) / nk
        variances = (r_w * (centers[None, :] - mus[:, None]) ** 2).sum(axis=1) / nk
        sigmas = np.sqrt(np.maximum(variances, 1.0))  # floor 1 dB
        pis = nk / nk.sum()

    order = np.argsort(mus)
    return [(float(pis[i]), float(mus[i]), float(sigmas[i])) for i in order]


def summarise(counts: np.ndarray, centers: np.ndarray) -> DistSummary:
    mu, std, skew, kurt = weighted_moments(counts, centers)
    p05, p25, p50, p75, p95 = weighted_quantiles(counts, centers, (0.05, 0.25, 0.50, 0.75, 0.95))
    mode_idx = int(np.argmax(counts))
    lo, hi = nonzero_range(counts, centers)
    gmm2 = fit_gmm_1d(counts, centers, K=2)
    return DistSummary(
        total_pixels=int(counts.sum()),
        mean=round(mu, 3),
        std=round(std, 3),
        skew=round(skew, 3),
        kurt=round(kurt, 3),
        p05=p05,
        p25=p25,
        p50=p50,
        p75=p75,
        p95=p95,
        mode=float(centers[mode_idx]),
        nonzero_lo=lo,
        nonzero_hi=hi,
        gmm2=[(round(p, 3), round(m, 2), round(s, 2)) for p, m, s in gmm2],
    )


def build_report(df: pd.DataFrame, centers: np.ndarray, bin_cols: List[str]) -> Dict[str, Dict[str, DistSummary]]:
    keep_kinds = {
        "target_los", "target_nlos", "pred_los", "pred_nlos",
        "target_delay_spread", "pred_delay_spread",
        "target_angular_spread", "pred_angular_spread",
    }
    df = df[df["kind"].isin(keep_kinds)].copy()

    report: Dict[str, Dict[str, DistSummary]] = {}
    for (class6, metric, kind), grp in df.groupby(list(GROUPING_KEYS), dropna=False):
        counts = aggregate_counts(grp, bin_cols)
        key_class = str(class6)
        key_leaf = f"{metric}|{kind}"
        report.setdefault(key_class, {})[key_leaf] = summarise(counts, centers)
    return report


def overall_by_kind(df: pd.DataFrame, centers: np.ndarray, bin_cols: List[str]) -> Dict[str, DistSummary]:
    """Collapsed-over-classes summary, per (metric, kind)."""
    out: Dict[str, DistSummary] = {}
    for (metric, kind), grp in df.groupby(["metric", "kind"], dropna=False):
        counts = aggregate_counts(grp, bin_cols)
        out[f"{metric}|{kind}"] = summarise(counts, centers)
    return out


def per_city_summary(df: pd.DataFrame, centers: np.ndarray, bin_cols: List[str]) -> Dict[str, Dict[str, DistSummary]]:
    out: Dict[str, Dict[str, DistSummary]] = {}
    for (city, metric, kind), grp in df.groupby(["city", "metric", "kind"], dropna=False):
        counts = aggregate_counts(grp, bin_cols)
        out.setdefault(str(city), {})[f"{metric}|{kind}"] = summarise(counts, centers)
    return out


def format_summary_row(label: str, s: DistSummary) -> str:
    gmm_str = " ; ".join(
        f"(pi={p:.2f}, mu={m:.1f}, sigma={sd:.1f})" for (p, m, sd) in s.gmm2
    )
    return (
        f"| {label} | {s.total_pixels:,} | {s.mean:.1f} | {s.std:.1f} | "
        f"{s.skew:+.2f} | {s.kurt:+.2f} | {s.p05:.0f}/{s.p50:.0f}/{s.p95:.0f} | "
        f"{s.mode:.0f} | {s.nonzero_lo}..{s.nonzero_hi} | {gmm_str} |"
    )


def write_markdown(path: Path, by_class: Dict, overall: Dict, per_city: Dict) -> None:
    lines: List[str] = []
    lines.append("# Try 76 — histogram study (auto-generated)\n")
    lines.append("Source: `tmp_review/histograms_try74/histograms.csv` — aggregated targets and predictions from Try 74 / Try 75 experts.\n")
    lines.append("\nBin width = 1 dB (or 1 ns / 1 deg). Summaries use weighted moments over aggregated bin counts. `gmm2` = K=2 Gaussian mixture fit in dB.\n")

    header = "| group | N | mean | std | skew | kurt | p05/p50/p95 | mode | nonzero range | gmm2 |"
    sep = "|---|---:|---:|---:|---:|---:|---|---:|---|---|"

    lines.append("\n## Overall (all cities collapsed)\n")
    lines.append(header)
    lines.append(sep)
    for leaf, s in sorted(overall.items()):
        lines.append(format_summary_row(leaf, s))

    lines.append("\n## By topology class (city_type_6) x kind\n")
    for klass in sorted(by_class.keys()):
        lines.append(f"\n### {klass}\n")
        lines.append(header)
        lines.append(sep)
        for leaf, s in sorted(by_class[klass].items()):
            lines.append(format_summary_row(leaf, s))

    lines.append("\n## Per-city quick stats (path_loss targets only)\n")
    lines.append("| city | target_los mean/std/p50 | target_nlos mean/std/p50 |")
    lines.append("|---|---|---|")
    for city in sorted(per_city.keys()):
        leafs = per_city[city]
        los = leafs.get("path_loss|target_los")
        nlos = leafs.get("path_loss|target_nlos")
        def fmt(s):
            if s is None:
                return "-"
            return f"{s.mean:.1f}/{s.std:.1f}/{s.p50:.0f}"
        lines.append(f"| {city} | {fmt(los)} | {fmt(nlos)} |")

    path.write_text("\n".join(lines), encoding="utf-8")


def summary_to_dict(s: DistSummary) -> Dict:
    return {
        "total_pixels": s.total_pixels,
        "mean": s.mean,
        "std": s.std,
        "skew": s.skew,
        "kurt": s.kurt,
        "p05": s.p05,
        "p25": s.p25,
        "p50": s.p50,
        "p75": s.p75,
        "p95": s.p95,
        "mode": s.mode,
        "nonzero_lo": s.nonzero_lo,
        "nonzero_hi": s.nonzero_hi,
        "gmm2": [{"pi": p, "mu": m, "sigma": s_} for p, m, s_ in s.gmm2],
    }


def write_json(path: Path, by_class: Dict, overall: Dict, per_city: Dict) -> None:
    payload = {
        "overall": {k: summary_to_dict(v) for k, v in overall.items()},
        "by_class": {
            klass: {leaf: summary_to_dict(v) for leaf, v in leafs.items()}
            for klass, leafs in by_class.items()
        },
        "per_city": {
            city: {leaf: summary_to_dict(v) for leaf, v in leafs.items()}
            for city, leafs in per_city.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_json or (out_dir / "histogram_study.json")
    out_md = args.out_md or (out_dir / "histogram_study.md")

    df, centers, bin_cols = load_csv(args.csv)
    by_class = build_report(df, centers, bin_cols)
    overall = overall_by_kind(df, centers, bin_cols)
    per_city = per_city_summary(df, centers, bin_cols)

    write_json(out_json, by_class, overall, per_city)
    write_markdown(out_md, by_class, overall, per_city)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
