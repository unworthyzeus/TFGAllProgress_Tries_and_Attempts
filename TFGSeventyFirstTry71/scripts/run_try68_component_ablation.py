#!/usr/bin/env python3
"""Short component ablations for Try 71 (same expert YAML, same seed).

Patches YAML variants, runs training with ``--epochs-override``, then reads
``validate_metrics_latest.json`` from each run's output directory.

**Eval-only (checkpoint, sin entrenar):** ``--eval-only --checkpoint ruta/best_model.pt``
recorre solo presets que tienen sentido en inferencia (``baseline``, ``ema``,
``test_split``) llamando a ``evaluate.py``. Los presets tipo ``no_multiscale``
cambian solo la pérdida de entrenamiento: con un peso fijo no son una ablation
válida en evaluate.

Try 71 trainer does **not** read ``corridor_weighting`` (YAML may still list it
from the Try 66 template; toggling it here is a no-op until wired in code).

Example (from ``TFGSeventyFirstTry71/``):

  python scripts/run_try68_component_ablation.py \\
    --base-config experiments/seventyfirst_try71_experts/try71_expert_open_sparse_lowrise.yaml \\
    --epochs-override 4 \\
    --output-json outputs/try68_ablation_table.json

Windows + DirectML (``pip install torch-directml``), workers en 0, solo ``baseline``:

  python scripts/run_try68_component_ablation.py \\
    --base-config experiments/seventyfirst_try71_experts/try71_expert_open_sparse_lowrise.yaml \\
    --epochs-override 1 --runtime-device directml --local-dataloader \\
    --presets baseline \\
    --hdf5-path C:/ruta/a/CKM_Dataset_270326.h5

Solo validación con checkpoint ya entrenado (DirectML):

  python scripts/run_try68_component_ablation.py \\
    --eval-only --checkpoint outputs/try71_expert_open_sparse_lowrise/best_model.pt \\
    --base-config experiments/seventyfirst_try71_experts/try71_expert_open_sparse_lowrise.yaml \\
    --runtime-device directml --local-dataloader \\
    --presets baseline ema \\
    --hdf5-path C:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _deep_set(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur: Any = d
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _ablation_presets() -> List[Tuple[str, Dict[Tuple[str, ...], Any]]]:
    """Name -> nested-key overrides. Only keys honored by Try 71 train/data."""
    return [
        ("baseline", {}),
        (
            "no_multiscale",
            {("multiscale_path_loss", "enabled"): False},
        ),
        (
            "no_cutmix",
            {("training", "cutmix_prob"): 0.0},
        ),
        (
            "uniform_nlos_loss_weights",
            {("training", "nlos_reweight_factor"): 1.0},
        ),
        (
            "no_tta",
            {("test_time_augmentation", "enabled"): False},
        ),
        (
            "rmse_main_instead_of_huber",
            {("loss", "loss_type"): "rmse"},
        ),
    ]


def _merge_overrides(cfg: Dict[str, Any], overrides: Dict[Tuple[str, ...], Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for path, val in overrides.items():
        _deep_set(out, path, val)
    return out


def _eval_ablation_presets() -> List[Tuple[str, Dict[Tuple[str, ...], Any], List[str]]]:
    """Eval-only: name, YAML overrides, extra CLI tokens for ``evaluate.py``."""
    return [
        ("baseline", {}, []),
        ("ema", {}, ["--use-ema"]),
        ("test_split", {}, ["--split", "test"]),
    ]


def _eval_presets_filtered(names: Optional[List[str]]) -> List[Tuple[str, Dict[Tuple[str, ...], Any], List[str]]]:
    all_presets = _eval_ablation_presets()
    if not names:
        return all_presets
    allowed = {n for n, _, _ in all_presets}
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise SystemExit(
            f"Unknown --presets for --eval-only: {unknown!r}. Valid: {sorted(allowed)}. "
            "Training-loss presets (no_multiscale, no_cutmix, …) require training, not evaluate.py."
        )
    pick = set(names)
    return [(n, o, e) for n, o, e in all_presets if n in pick]


def _ablation_presets_filtered(names: Optional[List[str]]) -> List[Tuple[str, Dict[Tuple[str, ...], Any]]]:
    all_presets = _ablation_presets()
    if not names:
        return all_presets
    allowed = {n for n, _ in all_presets}
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise SystemExit(f"Unknown --presets {unknown!r}. Choose from: {sorted(allowed)}")
    pick = set(names)
    return [(n, o) for n, o in all_presets if n in pick]


def _apply_local_runtime_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "runtime_device", None):
        cfg.setdefault("runtime", {})["device"] = str(args.runtime_device)
    if getattr(args, "local_dataloader", False):
        data = cfg.setdefault("data", {})
        data["num_workers"] = 0
        data["val_num_workers"] = 0
        data["persistent_workers"] = False
        data["val_persistent_workers"] = False
    if getattr(args, "hdf5_path", None):
        p = Path(args.hdf5_path).expanduser()
        cfg.setdefault("data", {})["hdf5_path"] = str(p.resolve())
    if getattr(args, "training_batch_size", None) is not None:
        cfg.setdefault("training", {})["batch_size"] = int(args.training_batch_size)
    if getattr(args, "val_batch_size", None) is not None:
        cfg.setdefault("data", {})["val_batch_size"] = int(args.val_batch_size)


def _read_rmse_from_evaluate_stdout(raw_stdout: str) -> float:
    for line in reversed(raw_stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            data = json.loads(line)
            pl = data.get("path_loss") or {}
            return float(pl.get("rmse_physical", float("nan")))
    return float("nan")


def _read_val_rmse(out_dir: Path) -> float:
    p = out_dir / "validate_metrics_latest.json"
    if not p.exists():
        return float("nan")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pl = data.get("path_loss") or {}
    return float(pl.get("rmse_physical", float("nan")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Try 71 short component ablations")
    parser.add_argument("--base-config", type=Path, required=True, help="Expert YAML (Try 71)")
    parser.add_argument("--epochs-override", type=int, default=4, help="Short train for each variant")
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=ROOT / "experiments" / "seventyfirst_try71_ablation_staging",
        help="Where to write temporary YAMLs",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Write summary table JSON here")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument(
        "--runtime-device",
        type=str,
        default=None,
        help="Override runtime.device (e.g. directml, cuda, cpu, auto).",
    )
    parser.add_argument(
        "--local-dataloader",
        action="store_true",
        help="Set data num_workers/val_num_workers to 0 and disable persistent_workers (Windows-friendly).",
    )
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=None,
        help="Override data.hdf5_path (absolute recommended if not under TFGpractice/Datasets/).",
    )
    parser.add_argument(
        "--training-batch-size",
        type=int,
        default=None,
        help="Override training.batch_size (e.g. 1 for tight VRAM on DirectML).",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Override data.val_batch_size.",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help="Subset of ablation names (default: all). E.g. --presets baseline no_multiscale",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Do not train: run evaluate.py per eval preset (baseline / ema / test_split) on --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Generator checkpoint (.pt); required with --eval-only.",
    )
    args = parser.parse_args()

    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit("--eval-only requires --checkpoint")
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)

    base_path = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    with base_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    staging = args.staging_dir
    staging.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    train_py = ROOT / "train_partitioned_pathloss_expert.py"
    evaluate_py = ROOT / "evaluate.py"
    chk = str(Path(args.checkpoint).resolve()) if args.checkpoint else ""

    if args.eval_only:
        for name, overrides, eval_extra in _eval_presets_filtered(args.presets):
            cfg = _merge_overrides(base_cfg, overrides)
            _apply_local_runtime_overrides(cfg, args)
            tmp_yaml = staging / f"try68_eval_ablation_{name}.yaml"
            with tmp_yaml.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

            cmd = [
                sys.executable,
                str(evaluate_py),
                "--config",
                str(tmp_yaml.resolve()),
                "--checkpoint",
                chk,
            ]
            if args.runtime_device:
                cmd += ["--runtime-device", str(args.runtime_device)]
            cmd += list(eval_extra)
            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env["TQDM_DISABLE"] = "1"
            print(json.dumps({"ablation": name, "mode": "eval", "cmd": cmd}, indent=2))
            rmse = float("nan")
            if not args.dry_run:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                try:
                    rmse = _read_rmse_from_evaluate_stdout(proc.stdout)
                except json.JSONDecodeError:
                    print(proc.stdout)
                    print(proc.stderr, file=sys.stderr)
                    raise
            rows.append(
                {
                    "name": name,
                    "mode": "eval",
                    "rmse_physical_db": rmse,
                    "checkpoint": chk,
                    "evaluate_extra": list(eval_extra),
                    "overrides": {str(k): v for k, v in overrides.items()},
                }
            )
        summary: Dict[str, Any] = {
            "mode": "eval_only",
            "base_config": str(base_path),
            "checkpoint": chk,
            "rows": rows,
        }
    else:
        for name, overrides in _ablation_presets_filtered(args.presets):
            cfg = _merge_overrides(base_cfg, overrides)
            _apply_local_runtime_overrides(cfg, args)
            cfg.setdefault("training", {})
            cfg["training"]["epochs"] = int(args.epochs_override)
            cfg["training"]["save_every"] = 1
            es = dict(cfg["training"].get("early_stopping") or {})
            es["enabled"] = False
            cfg["training"]["early_stopping"] = es
            out_rel = f"outputs/try68_ablation_{name}"
            cfg.setdefault("runtime", {})["output_dir"] = out_rel

            tmp_yaml = staging / f"try68_ablation_{name}.yaml"
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
            rows.append(
                {
                    "name": name,
                    "mode": "train",
                    "val_rmse_physical_db": rmse,
                    "output_dir": out_rel,
                    "overrides": {str(k): v for k, v in overrides.items()},
                }
            )

        summary = {"mode": "train", "base_config": str(base_path), "epochs": int(args.epochs_override), "rows": rows}
    print(json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
