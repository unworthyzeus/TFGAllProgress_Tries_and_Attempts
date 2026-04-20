"""Try 76 — distribution-family identification per (topology class x kind).

Complements ``study_histograms.py``. For every aggregated histogram we fit a
handful of parametric families on the 1-dB (or 1-ns / 1-deg) bins and rank
them by KL divergence against the empirical distribution. The best-fitting
family is then reported — this is what we use to decide *which* parametric
output the Stage-A head of Try 76 should emit per expert.

Families considered (all with closed-form MLE or moment-based fits, no SciPy
dependency):

    - gaussian           single Gaussian (baseline)
    - laplace            L1, heavier tails than Gaussian
    - skew_normal        Azzalini, moment-based (approximate)
    - lognormal          for non-negative data with right skew (shift if needed)
    - gamma              shape/scale via method of moments
    - weibull            method of moments
    - gmm2 / gmm3        Gaussian mixture on the bin centres (reuse EM)
    - exp_tail           degenerate-at-0 spike + exponential tail (for spreads)

Output:
    docs/distribution_classes.md
    docs/distribution_classes.json

Usage:
    python scripts/classify_distributions.py --csv ../tmp_review/histograms_try74/histograms.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


GROUPING_KEYS = ("city_type_6", "metric", "kind")
EPS = 1e-12


# ---------------------------------------------------------------------------
# I/O (mirror study_histograms.py)
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = pd.read_csv(path, low_memory=False)
    bin_cols = [c for c in df.columns if c.startswith("b") and c[1:].lstrip("-").isdigit()]
    bin_cols.sort(key=lambda c: int(c[1:]))
    bin_centers = np.array([int(c[1:]) + 0.5 for c in bin_cols], dtype=np.float64)
    return df, bin_centers, bin_cols


def aggregate_counts(group: pd.DataFrame, bin_cols: List[str]) -> np.ndarray:
    return group[bin_cols].to_numpy(dtype=np.float64, copy=False).sum(axis=0)


# ---------------------------------------------------------------------------
# Weighted moments (shared)
# ---------------------------------------------------------------------------

def _moments(p: np.ndarray, centers: np.ndarray) -> Tuple[float, float, float, float]:
    mu = float((p * centers).sum())
    var = float((p * (centers - mu) ** 2).sum())
    std = float(np.sqrt(max(var, EPS)))
    skew = float((p * ((centers - mu) / max(std, 1e-6)) ** 3).sum())
    kurt = float((p * ((centers - mu) / max(std, 1e-6)) ** 4).sum())
    return mu, std, skew, kurt


def _empirical_pmf(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts) + 1.0 / counts.size
    return counts / total


# ---------------------------------------------------------------------------
# Parametric PMFs on bin centres (binwidth = 1 unit)
# ---------------------------------------------------------------------------

def _norm_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-3)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _pmf_from_pdf(pdf: np.ndarray) -> np.ndarray:
    s = pdf.sum()
    if s <= 0:
        return np.full_like(pdf, 1.0 / pdf.size)
    return pdf / s


def pmf_gaussian(centers: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return _pmf_from_pdf(_norm_pdf(centers, mu, sigma))


def pmf_laplace(centers: np.ndarray, mu: float, b: float) -> np.ndarray:
    b = max(b, 1e-3)
    pdf = (1.0 / (2 * b)) * np.exp(-np.abs(centers - mu) / b)
    return _pmf_from_pdf(pdf)


def pmf_skew_normal(centers: np.ndarray, mu: float, sigma: float, skew: float) -> np.ndarray:
    """Azzalini SN via delta = sign(skew) * sqrt(|skew|^(2/3) / (c + |skew|^(2/3))),
    with c = (2/(pi-2))^(2/3). Matches moments approximately for |skew| <= 0.95.
    """
    c = (2.0 / (np.pi - 2.0)) ** (2.0 / 3.0)
    s_abs = min(abs(skew), 0.99)
    delta = np.sign(skew) * np.sqrt(s_abs ** (2.0 / 3.0) / (c + s_abs ** (2.0 / 3.0)))
    alpha = delta / max(np.sqrt(1.0 - delta ** 2), 1e-6)
    # Convert (mean, std, skew) -> (xi, omega, alpha) Azzalini
    omega = max(sigma / np.sqrt(1.0 - 2.0 * (delta ** 2) / np.pi), 1e-3)
    xi = mu - omega * delta * np.sqrt(2.0 / np.pi)
    z = (centers - xi) / omega
    pdf = 2.0 * _norm_pdf(z, 0.0, 1.0) * 0.5 * (1.0 + np.math.erf(alpha * z / np.sqrt(2)) if False else _sn_cdf(alpha * z))
    return _pmf_from_pdf(pdf)


def _sn_cdf(x: np.ndarray) -> np.ndarray:
    # Vectorised Phi via erf
    from math import erf
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2)))


def pmf_lognormal(centers: np.ndarray, counts_mean: float, counts_std: float, shift: float) -> np.ndarray:
    """Fit log-normal on (x - shift). shift = min(support) - 0.5 so domain is > 0."""
    x = centers - shift
    pos = x > 0
    pdf = np.zeros_like(centers)
    if not np.any(pos):
        return _pmf_from_pdf(np.ones_like(centers))
    # Moment-match on positive support
    # mu_log = ln(mean^2 / sqrt(mean^2 + var))
    # sigma_log^2 = ln(1 + var / mean^2)
    m = max(counts_mean - shift, 1e-3)
    v = max(counts_std ** 2, 1e-3)
    mu_log = np.log(m * m / np.sqrt(m * m + v))
    sigma_log = np.sqrt(max(np.log(1.0 + v / (m * m)), 1e-4))
    xx = x[pos]
    pdf_pos = (1.0 / (xx * sigma_log * np.sqrt(2 * np.pi))) * np.exp(-(np.log(xx) - mu_log) ** 2 / (2 * sigma_log ** 2))
    pdf[pos] = pdf_pos
    return _pmf_from_pdf(pdf)


def pmf_gamma(centers: np.ndarray, counts_mean: float, counts_std: float, shift: float) -> np.ndarray:
    x = centers - shift
    pos = x > 0
    pdf = np.zeros_like(centers)
    if not np.any(pos):
        return _pmf_from_pdf(np.ones_like(centers))
    m = max(counts_mean - shift, 1e-3)
    v = max(counts_std ** 2, 1e-3)
    shape = m * m / v
    scale = v / m
    from math import lgamma
    xx = x[pos]
    log_pdf = (shape - 1) * np.log(xx) - xx / scale - shape * np.log(scale) - lgamma(shape)
    pdf_pos = np.exp(np.clip(log_pdf, -50, 50))
    pdf[pos] = pdf_pos
    return _pmf_from_pdf(pdf)


def pmf_weibull(centers: np.ndarray, counts_mean: float, counts_std: float, shift: float) -> np.ndarray:
    x = centers - shift
    pos = x > 0
    pdf = np.zeros_like(centers)
    if not np.any(pos):
        return _pmf_from_pdf(np.ones_like(centers))
    m = max(counts_mean - shift, 1e-3)
    v = max(counts_std ** 2, 1e-3)
    cv = np.sqrt(v) / m  # coefficient of variation
    # Approximate shape k from CV using Johnson's inversion
    k = (cv ** -1.086)
    k = float(np.clip(k, 0.3, 20.0))
    from math import gamma as Gamma
    lam = m / Gamma(1.0 + 1.0 / k)
    xx = x[pos]
    pdf_pos = (k / lam) * (xx / lam) ** (k - 1.0) * np.exp(-((xx / lam) ** k))
    pdf[pos] = pdf_pos
    return _pmf_from_pdf(pdf)


def pmf_gmm(centers: np.ndarray, counts: np.ndarray, K: int, iters: int = 80) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
    p = _empirical_pmf(counts)
    mu0, std0, _, _ = _moments(p, centers)
    if K == 1:
        return pmf_gaussian(centers, mu0, std0), [(1.0, mu0, std0)]
    qs = np.quantile_weighted(p, centers, np.linspace(0.1, 0.9, K)) if hasattr(np, "quantile_weighted") else None
    if qs is None:
        cdf = np.cumsum(p)
        qs = [float(centers[int(np.searchsorted(cdf, q))]) for q in np.linspace(0.1, 0.9, K)]
    mus = np.array(qs, dtype=np.float64)
    sigmas = np.full(K, max(std0, 3.0), dtype=np.float64)
    pis = np.full(K, 1.0 / K, dtype=np.float64)
    for _ in range(iters):
        diff = centers[None, :] - mus[:, None]
        log_pdf = -0.5 * (diff / sigmas[:, None]) ** 2 - np.log(sigmas[:, None] * np.sqrt(2 * np.pi))
        log_weighted = np.log(pis[:, None] + EPS) + log_pdf
        m = log_weighted.max(axis=0, keepdims=True)
        log_norm = m + np.log(np.exp(log_weighted - m).sum(axis=0, keepdims=True) + EPS)
        resp = np.exp(log_weighted - log_norm)
        rw = resp * p[None, :]
        nk = rw.sum(axis=1) + EPS
        mus = (rw * centers[None, :]).sum(axis=1) / nk
        variances = (rw * (centers[None, :] - mus[:, None]) ** 2).sum(axis=1) / nk
        sigmas = np.sqrt(np.maximum(variances, 1.0))
        pis = nk / nk.sum()
    # PMF
    pdfs = np.zeros((K, centers.size))
    for k in range(K):
        pdfs[k] = _norm_pdf(centers, mus[k], sigmas[k])
    mixture = (pis[:, None] * pdfs).sum(axis=0)
    pmf = _pmf_from_pdf(mixture)
    comps = [(float(pis[k]), float(mus[k]), float(sigmas[k])) for k in np.argsort(mus)]
    return pmf, comps


def pmf_spike_plus_exp(centers: np.ndarray, counts: np.ndarray, spike_bins: int = 2) -> np.ndarray:
    """Model used for delay_spread / angular_spread targets: a delta-at-zero
    plus a shifted exponential tail. Returns the fitted PMF on the bin grid.
    """
    p = _empirical_pmf(counts)
    # Find the "zero" region: bins with centre < spike_bins*binwidth after first
    # nonzero bin. Binwidth = 1 for our CSVs.
    nz = np.flatnonzero(counts > 0)
    if nz.size == 0:
        return np.ones_like(centers) / centers.size
    start = nz[0]
    spike_idx = np.arange(start, min(start + spike_bins, centers.size))
    spike_mass = float(p[spike_idx].sum())
    tail_idx = np.arange(spike_idx[-1] + 1, centers.size)
    tail_pmf = np.zeros_like(centers)
    if tail_idx.size > 0 and p[tail_idx].sum() > 0:
        tail_centers = centers[tail_idx] - centers[spike_idx[-1]]
        tail_p = p[tail_idx] / p[tail_idx].sum()
        mean_tail = float((tail_p * tail_centers).sum())
        if mean_tail > 0:
            lam = 1.0 / max(mean_tail, 1e-3)
            pdf_tail = lam * np.exp(-lam * tail_centers)
            tail_pmf[tail_idx] = _pmf_from_pdf(pdf_tail) * (1.0 - spike_mass)
    spike_pmf = np.zeros_like(centers)
    spike_pmf[spike_idx] = 1.0 / spike_idx.size
    return spike_mass * spike_pmf + tail_pmf


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def kl(empirical: np.ndarray, model: np.ndarray) -> float:
    e = np.clip(empirical, EPS, None)
    m = np.clip(model, EPS, None)
    return float((e * (np.log(e) - np.log(m))).sum())


def tvd(empirical: np.ndarray, model: np.ndarray) -> float:
    return float(0.5 * np.abs(empirical - model).sum())


def _centroid_support(counts: np.ndarray, centers: np.ndarray) -> Tuple[int, int]:
    nz = np.flatnonzero(counts > 0)
    if nz.size == 0:
        return 0, 0
    return int(round(centers[nz[0]] - 0.5)), int(round(centers[nz[-1]] - 0.5))


def classify_one(counts: np.ndarray, centers: np.ndarray) -> Dict:
    p = _empirical_pmf(counts)
    mu, std, skew, kurt = _moments(p, centers)
    lo, hi = _centroid_support(counts, centers)
    shift = float(lo - 0.5)  # so support starts at ~ bin centre lo + 0.5 > 0

    models: Dict[str, np.ndarray] = {}
    models["gaussian"] = pmf_gaussian(centers, mu, std)
    models["laplace"] = pmf_laplace(centers, mu, std / np.sqrt(2))
    try:
        models["skew_normal"] = pmf_skew_normal(centers, mu, std, skew)
    except Exception:
        pass
    if hi > lo:
        try:
            models["lognormal"] = pmf_lognormal(centers, mu, std, shift)
        except Exception:
            pass
        try:
            models["gamma"] = pmf_gamma(centers, mu, std, shift)
        except Exception:
            pass
        try:
            models["weibull"] = pmf_weibull(centers, mu, std, shift)
        except Exception:
            pass
    models["spike_plus_exp"] = pmf_spike_plus_exp(centers, counts)
    gmm_components: Dict[int, list] = {}
    for K in (2, 3, 4, 5):
        try:
            models[f"gmm{K}"], comps = pmf_gmm(centers, counts, K=K)
            gmm_components[K] = comps
        except Exception:
            gmm_components[K] = []

    ranked = sorted(models.items(), key=lambda kv: kl(p, kv[1]))
    best_name, best_pmf = ranked[0]
    out = {
        "moments": {"mean": round(mu, 3), "std": round(std, 3), "skew": round(skew, 3), "kurt": round(kurt - 3.0, 3)},
        "support": {"lo": lo, "hi": hi},
        "best_family": best_name,
        "kl_ranking": [(name, round(kl(p, pmf), 4), round(tvd(p, pmf), 4)) for name, pmf in ranked],
    }
    for K, comps in gmm_components.items():
        if comps:
            out[f"gmm{K}_components"] = [(round(p_, 3), round(m_, 2), round(s_, 2)) for p_, m_, s_ in comps]
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_by_class(df: pd.DataFrame, centers: np.ndarray, bin_cols: List[str]) -> Dict:
    keep = {
        "target_los", "target_nlos", "pred_los", "pred_nlos",
        "target_delay_spread", "pred_delay_spread",
        "target_angular_spread", "pred_angular_spread",
    }
    df = df[df["kind"].isin(keep)].copy()
    out: Dict[str, Dict[str, Dict]] = {}
    for (class6, metric, kind), grp in df.groupby(list(GROUPING_KEYS), dropna=False):
        counts = aggregate_counts(grp, bin_cols)
        out.setdefault(str(class6), {})[f"{metric}|{kind}"] = classify_one(counts, centers)
    return out


def build_overall(df: pd.DataFrame, centers: np.ndarray, bin_cols: List[str]) -> Dict:
    keep = {
        "target_los", "target_nlos", "pred_los", "pred_nlos",
        "target_delay_spread", "pred_delay_spread",
        "target_angular_spread", "pred_angular_spread",
    }
    df = df[df["kind"].isin(keep)].copy()
    out: Dict[str, Dict] = {}
    for (metric, kind), grp in df.groupby(["metric", "kind"], dropna=False):
        counts = aggregate_counts(grp, bin_cols)
        out[f"{metric}|{kind}"] = classify_one(counts, centers)
    return out


def format_ranking(r: Dict) -> str:
    top3 = r["kl_ranking"][:3]
    return " > ".join(f"{name}(KL={k:.3f})" for name, k, _ in top3)


def write_md(path: Path, overall: Dict, by_class: Dict) -> None:
    lines: List[str] = []
    lines.append("# Try 76 — distribution-class identification\n")
    lines.append("Auto-generated from `tmp_review/histograms_try74/histograms.csv`.\n")
    lines.append("For each aggregated histogram we fit a handful of parametric families on the 1-dB/ns/deg bins and rank them by KL(empirical || model). The smaller the KL, the better the fit.\n")
    lines.append("Families: `gaussian`, `laplace`, `skew_normal`, `lognormal`, `gamma`, `weibull`, `spike_plus_exp` (for heavy-tailed sparse metrics), `gmm2`, `gmm3`, `gmm4`, `gmm5`.\n")

    lines.append("\n## Overall (all cities collapsed)\n")
    lines.append("| group | best family | mean | std | skew | excess kurt | top-3 by KL |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for leaf in sorted(overall.keys()):
        r = overall[leaf]
        m = r["moments"]
        lines.append(
            f"| {leaf} | **{r['best_family']}** | {m['mean']:.1f} | {m['std']:.1f} | "
            f"{m['skew']:+.2f} | {m['kurt']:+.2f} | {format_ranking(r)} |"
        )

    lines.append("\n## By topology class (city_type_6) x kind\n")
    for klass in sorted(by_class.keys()):
        lines.append(f"\n### {klass}\n")
        lines.append("| group | best family | mean | std | skew | excess kurt | top-3 by KL |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for leaf in sorted(by_class[klass].keys()):
            r = by_class[klass][leaf]
            m = r["moments"]
            lines.append(
                f"| {leaf} | **{r['best_family']}** | {m['mean']:.1f} | {m['std']:.1f} | "
                f"{m['skew']:+.2f} | {m['kurt']:+.2f} | {format_ranking(r)} |"
            )

    lines.append("\n## Takeaways\n")
    lines.append(_summary_takeaways(overall, by_class))
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_takeaways(overall: Dict, by_class: Dict) -> str:
    def best_for(group: str) -> str:
        r = overall.get(group)
        return r["best_family"] if r else "-"

    lines = []
    lines.append(f"- `path_loss | target_los` → **{best_for('path_loss|target_los')}** — the only metric where a single near-Gaussian is competitive.")
    lines.append(f"- `path_loss | target_nlos` → **{best_for('path_loss|target_nlos')}** — a narrow peak around 107 dB with a secondary low-σ mode; a 2-component Gaussian mixture wins, a single Gaussian is too wide.")
    lines.append(f"- `delay_spread | target_delay_spread` → **{best_for('delay_spread|target_delay_spread')}** — not Gaussian; a spike-at-≈3 ns plus a long exponential tail is the right parametric form.")
    lines.append(f"- `angular_spread | target_angular_spread` → **{best_for('angular_spread|target_angular_spread')}** — same shape as delay spread: spike-at-0 + heavy tail; Gaussian is a bad choice.")
    lines.append("\nImplication for Try 76: the Stage-A distribution head must be **family-aware per expert**. Path-loss experts output a 3-component Gaussian mixture, but delay/angular experts (not trained in Try 76 but noted for Try 77+) should output `(π_spike, λ_tail)` for a degenerate+exponential model, not a GMM.")
    return "\n".join(lines)


def write_json(path: Path, overall: Dict, by_class: Dict) -> None:
    path.write_text(json.dumps({"overall": overall, "by_class": by_class}, indent=2), encoding="utf-8")


def write_ranking_csv(path: Path, overall: Dict, by_class: Dict) -> None:
    """Flat CSV: one row per (scope, city_type_6, metric, kind, family) ranking entry."""
    rows: List[List] = []
    header = ["scope", "city_type_6", "metric", "kind", "family", "kl", "tvd", "best", "mean", "std", "skew", "excess_kurt"]
    rows.append(header)

    def emit(scope: str, klass: str, leaf_key: str, r: Dict) -> None:
        metric, kind = leaf_key.split("|", 1)
        m = r["moments"]
        best = r["best_family"]
        for name, k, t in r["kl_ranking"]:
            rows.append([
                scope, klass, metric, kind, name,
                f"{k:.6f}", f"{t:.6f}",
                1 if name == best else 0,
                m["mean"], m["std"], m["skew"], m["kurt"],
            ])

    for leaf, r in overall.items():
        emit("overall", "(all)", leaf, r)
    for klass, leafs in by_class.items():
        for leaf, r in leafs.items():
            emit("by_class", klass, leaf, r)

    with open(path, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def write_raw_histograms(path: Path, df: pd.DataFrame, centers: np.ndarray, bin_cols: List[str]) -> None:
    """Per (city_type_6, metric, kind) aggregated raw counts — long-form CSV.

    Columns: city_type_6, metric, kind, bin_lo, bin_hi, count
    """
    import csv

    keep = {
        "target_los", "target_nlos", "pred_los", "pred_nlos",
        "target_delay_spread", "pred_delay_spread",
        "target_angular_spread", "pred_angular_spread",
    }
    df = df[df["kind"].isin(keep)].copy()

    rows_out = []
    for (klass, metric, kind), grp in df.groupby(["city_type_6", "metric", "kind"], dropna=False):
        counts = aggregate_counts(grp, bin_cols)
        nz = np.flatnonzero(counts > 0)
        if nz.size == 0:
            continue
        start = int(nz[0])
        end = int(nz[-1]) + 1  # inclusive
        for i in range(start, end):
            c = int(counts[i])
            if c == 0:
                continue
            bin_lo = int(round(centers[i] - 0.5))
            rows_out.append([str(klass), str(metric), str(kind), bin_lo, bin_lo + 1, c])

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city_type_6", "metric", "kind", "bin_lo", "bin_hi", "count"])
        w.writerows(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "distribution_classes.md"
    out_json = out_dir / "distribution_classes.json"
    out_ranking_csv = out_dir / "distribution_classes_ranking.csv"
    out_raw_csv = out_dir / "histograms_raw_by_class.csv"

    df, centers, bin_cols = load_csv(args.csv)
    overall = build_overall(df, centers, bin_cols)
    by_class = build_by_class(df, centers, bin_cols)
    write_md(out_md, overall, by_class)
    write_json(out_json, overall, by_class)
    write_ranking_csv(out_ranking_csv, overall, by_class)
    write_raw_histograms(out_raw_csv, df, centers, bin_cols)
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_ranking_csv}")
    print(f"Wrote {out_raw_csv}")


if __name__ == "__main__":
    main()
