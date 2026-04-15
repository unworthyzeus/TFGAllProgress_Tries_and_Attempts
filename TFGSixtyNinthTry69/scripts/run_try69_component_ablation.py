#!/usr/bin/env python3
"""Short component ablations for Try 69 (same expert YAML, same seed).

Runs several training jobs with ``--epochs-override`` and patched YAMLs, then reads
``validate_metrics_latest.json`` from each run's output directory.

Example (from ``TFGSixtyNinthTry69/`` on a dev machine with 1 GPU):

  python scripts/run_try69_component_ablation.py \\
    --base-config experiments/sixtyninth_try69_experts/try69_expert_open_sparse_lowrise.yaml \\
    --epochs-override 4 \\
    --output-json outputs/try69_ablation_table.json

Requires: same dataset path as in the base config (HDF5 reachable from cwd).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _deep_set(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur: Any = d
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _ablation_presets() -> List[Tuple[str, Dict[Tuple[str, ...], Any]]]:
    """Name -> list of (nested key path, value) overrides."""
    return [
        ("baseline", {}),
        (
            "no_corridor",
            {("corridor_weighting", "enabled"): False},
        ),
        (
            "no_dual_head",
            {
                ("dual_los_nlos_head", "enabled"): False,
                ("model", "out_channels"): 1,
            },
        ),
        (
            "no_pde",
            {("pde_residual_loss", "enabled"): False},
        ),
        (
            "no_multiscale",
            {("multiscale_path_loss", "enabled"): False},
        ),
    ]


def _merge_overrides(cfg: Dict[str, Any], overrides: Dict[Tuple[str, ...], Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for path, val in overrides.items():
        _deep_set(out, path, val)
    return out


def _read_val_rmse(out_dir: Path) -> float:
    p = out_dir / "validate_metrics_latest.json"
    if not p.exists():
        return float("nan")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pl = data.get("path_loss") or {}
    return float(pl.get("rmse_physical", float("nan")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Try 69 short component ablations")
    parser.add_argument("--base-config", type=Path, required=True, help="Expert YAML (Try 69)")
    parser.add_argument("--epochs-override", type=int, default=4, help="Short train for each variant")
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=ROOT / "experiments" / "sixtyninth_try69_ablation_staging",
        help="Where to write temporary YAMLs",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Write summary table JSON here")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    base_path = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    with base_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    staging = args.staging_dir
    staging.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    train_py = ROOT / "train_partitioned_pathloss_expert.py"

    for name, overrides in _ablation_presets():
        cfg = _merge_overrides(base_cfg, overrides)
        # Deterministic short run: unique output dir, frequent val JSON, no early stop noise.
        cfg.setdefault("training", {})
        cfg["training"]["epochs"] = int(args.epochs_override)
        cfg["training"]["save_every"] = 1
        es = dict(cfg["training"].get("early_stopping") or {})
        es["enabled"] = False
        cfg["training"]["early_stopping"] = es
        out_rel = f"outputs/try69_ablation_{name}"
        cfg.setdefault("runtime", {})["output_dir"] = out_rel

        tmp_yaml = staging / f"try69_ablation_{name}.yaml"
        with tmp_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        cmd = [
            sys.executable,
            str(train_py),
            "--config",
            str(tmp_yaml.resolve()),
            "--epochs-override",
            str(int(args.epochs_override)),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        print(json.dumps({"ablation": name, "cmd": cmd}, indent=2))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
        rmse = _read_val_rmse(ROOT / out_rel)
        rows.append({"name": name, "val_rmse_physical_db": rmse, "output_dir": out_rel, "overrides": {str(k): v for k, v in overrides.items()}})

    summary = {"base_config": str(base_path), "epochs": int(args.epochs_override), "rows": rows}
    print(json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
