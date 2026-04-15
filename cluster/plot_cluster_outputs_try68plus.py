#!/usr/bin/env python3
"""Scan ``cluster_outputs`` for tries >= 68, plot each expert (like per-try ``scripts/plot_try*_metrics.py``).

Discovers ``cluster_outputs/TFG*Try{N}/**/validate_metrics_epoch_*.json``, resolves the matching
``TFG*Try{N}`` folder under the repo, loads ``plot_expert`` from that try's ``scripts/`` (same
search order as manual runs: ``plot_try{N}_metrics.py`` then ``plot_try68_metrics.py`` fallback),
and runs plotting in parallel processes.

Also writes ``cluster_outputs/_plots_try68plus/summary_best_val_rmse.csv`` and a small overview PNG.

Examples (from ``TFGpractice``)::

  python cluster/plot_cluster_outputs_try68plus.py
  python cluster/plot_cluster_outputs_try68plus.py --min-try 69 --workers 8
  python cluster/plot_cluster_outputs_try68plus.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLUSTER_OUT = REPO_ROOT / "cluster_outputs"
SUMMARY_DIRNAME = "_plots_try68plus"


def try_number_from_try_folder(name: str) -> int | None:
    m = re.search(r"Try(\d+)\s*$", name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def find_plot_script(try_folder: Path, try_num: int) -> Path | None:
    scripts = try_folder / "scripts"
    if not scripts.is_dir():
        return None
    candidates = [
        scripts / f"plot_try{try_num}_metrics.py",
        scripts / "plot_try68_metrics.py",
        scripts / "plot_try69_metrics.py",
    ]
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p
    return None


def expert_id_from_output_dir(d: Path) -> str:
    name = d.name
    for pat in (r"^try\d+_expert_(.+)$", r"^try\d+_(.+)$"):
        m = re.match(pat, name)
        if m:
            return m.group(1)
    return name


def discover_expert_dirs(cluster_try_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in cluster_try_dir.rglob("validate_metrics_epoch_*.json"):
        if p.name.endswith("_tail_refiner.json"):
            continue
        out.append(p.parent)
    # unique, stable order
    uniq = sorted({d.resolve() for d in out}, key=lambda x: str(x).lower())
    return [Path(x) for x in uniq]


def load_plot_expert(plot_script: Path) -> Callable[..., Any]:
    mod_name = "plot_try_dyn_" + hashlib.sha1(str(plot_script).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(mod_name, plot_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {plot_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "plot_expert", None)
    if not callable(fn):
        raise RuntimeError(f"No plot_expert() in {plot_script}")
    return fn


@dataclass(frozen=True)
class PlotTask:
    try_num: int
    try_name: str
    expert_dir: str
    plot_script: str
    save_path: str


def _run_one_task(task: PlotTask) -> dict[str, Any]:
    plot_fn = load_plot_expert(Path(task.plot_script))
    expert_dir = Path(task.expert_dir)
    save = Path(task.save_path)
    save.parent.mkdir(parents=True, exist_ok=True)
    eid = expert_id_from_output_dir(expert_dir)
    summary = plot_fn(eid, expert_dir, save)
    if isinstance(summary, dict):
        summary = dict(summary)
        summary.setdefault("try_num", task.try_num)
        summary.setdefault("try_name", task.try_name)
        return summary
    return {"try_num": task.try_num, "try_name": task.try_name, "expert_dir": str(expert_dir), "saved": str(save)}


def _best_val_rmse_from_dir(expert_dir: Path) -> tuple[int, float] | None:
    """Return (epoch, rmse) from latest epoch file without loading all."""
    files = sorted(
        expert_dir.glob("validate_metrics_epoch_*.json"),
        key=lambda p: int(re.search(r"_(\d+)\.json$", p.name).group(1)) if re.search(r"_(\d+)\.json$", p.name) else 0,
    )
    if not files:
        return None
    best_ep, best_v = None, float("inf")
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        m = data.get("metrics") or {}
        pl = m.get("path_loss") or {}
        v = float(pl.get("rmse_physical", float("nan")))
        if not math.isfinite(v):
            continue
        ck = data.get("checkpoint") or {}
        ep = int(ck.get("epoch", 0))
        if v < best_v:
            best_v, best_ep = v, ep
    if best_ep is None:
        return None
    return best_ep, best_v


def write_summary_csv_and_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_best_val_rmse.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["try_num", "try_name", "expert_id", "best_epoch", "best_val_rmse", "plot_png"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = [f"T{r['try_num']}\n{r['expert_id'][:18]}" for r in rows]
    vals = [float(r["best_val_rmse"]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.45), 5))
    x = range(len(rows))
    ax.bar(x, vals, color="#1098ad", edgecolor="#0b7285")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
    ax.set_ylabel("Best val RMSE (dB)")
    ax.set_title("cluster_outputs — best validation RMSE per run (try >= 68)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_best_val_rmse.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel plots from cluster_outputs JSON (tries >= 68).")
    ap.add_argument("--cluster-outputs", type=Path, default=DEFAULT_CLUSTER_OUT)
    ap.add_argument("--min-try", type=int, default=68)
    ap.add_argument("--workers", type=int, default=0, help="0 = CPU count.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    root = args.cluster_outputs.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    tasks: list[PlotTask] = []
    summary_scan: list[dict[str, Any]] = []

    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        n = try_number_from_try_folder(child.name)
        if n is None or n < args.min_try:
            continue
        repo_try = REPO_ROOT / child.name
        plot_script = find_plot_script(repo_try, n)
        if plot_script is None:
            print(f"[skip try {n}] no plot script under {repo_try / 'scripts'}", file=sys.stderr)
            continue

        out_base = root / SUMMARY_DIRNAME / child.name
        for expert_dir in discover_expert_dirs(child):
            eid = expert_id_from_output_dir(expert_dir)
            save = out_base / f"{eid}_metrics_try{n}.png"
            tasks.append(
                PlotTask(
                    try_num=n,
                    try_name=child.name,
                    expert_dir=str(expert_dir),
                    plot_script=str(plot_script),
                    save_path=str(save),
                )
            )
            b = _best_val_rmse_from_dir(expert_dir)
            if b:
                ep, rmse = b
                try:
                    plot_rel = str(save.relative_to(root))
                except ValueError:
                    plot_rel = str(save)
                summary_scan.append(
                    {
                        "try_num": n,
                        "try_name": child.name,
                        "expert_id": eid,
                        "best_epoch": ep,
                        "best_val_rmse": round(rmse, 4),
                        "plot_png": plot_rel,
                    }
                )

    print(f"Found {len(tasks)} expert plot job(s) (tries >= {args.min_try}).")
    if args.dry_run:
        for t in tasks:
            print(f"  Try{t.try_num} {expert_id_from_output_dir(Path(t.expert_dir))} -> {t.save_path}")
        return

    workers = args.workers if args.workers > 0 else max(1, (mp.cpu_count() or 4))
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one_task, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append(f"{t.try_name} {t.expert_dir}: {e}")

    for r in sorted(results, key=lambda x: (x.get("try_num", 0), str(x.get("output_dir", "")))):
        line = {
            "try": r.get("try_num"),
            "expert": r.get("expert_id"),
            "epochs": r.get("epochs"),
            "best_val_rmse": r.get("best_val_rmse"),
            "saved": r.get("saved"),
        }
        print(json.dumps(line, ensure_ascii=False))

    for err in errors:
        print(f"[error] {err}", file=sys.stderr)

    if not args.no_summary and summary_scan:
        summary_scan.sort(key=lambda r: (r["try_num"], r["expert_id"]))
        write_summary_csv_and_plot(summary_scan, root / SUMMARY_DIRNAME)
        print(f"\nWrote {root / SUMMARY_DIRNAME / 'summary_best_val_rmse.csv'}")


if __name__ == "__main__":
    # Windows + multiprocessing
    mp.freeze_support()
    main()
