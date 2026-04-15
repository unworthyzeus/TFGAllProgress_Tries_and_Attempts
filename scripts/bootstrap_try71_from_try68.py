#!/usr/bin/env python3
"""Copy TFGSixtyEighthTry68 -> TFGSeventyFirstTry71 with try68/try71 string renames.

Only the open_sparse_lowrise expert config is retained; the other five expert YAMLs
and the cluster multi-expert submit scripts are removed so nothing accidentally submits
the wrong expert.

Run from repo root:
  python TFGpractice/scripts/bootstrap_try71_from_try68.py

Post-run patches needed in TFGSeventyFirstTry71/:
  1. model_pmhhnet.py          -- out_channels=2 split: mean residual (ch0) + log_var (ch1)
  2. train_partitioned_pathloss_expert.py
                               -- heteroscedastic NLL loss + RMSE-vs-coverage evaluation
  3. experiments/seventyfirst_try71_experts/try71_expert_open_sparse_lowrise.yaml
                               -- model.out_channels: 2, resume/init checkpoint path
  4. cluster/run_seventyfirst_try71_*.slurm
                               -- TRY71_INIT_CHECKPOINT env var pointing to Try 68 best_model.pt
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TEXT_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".slurm",
    ".md",
    ".tex",
    ".json",
    ".toml",
    ".txt",
    ".sh",
    ".cfg",
}

CONTENT_REPLACEMENTS: list[tuple[str, str]] = [
    ("TFGSixtyEighthTry68", "TFGSeventyFirstTry71"),
    ("sixtyeighth_try68", "seventyfirst_try71"),
    ("try68_expert_registry", "try71_expert_registry"),
    ("try68_topology", "try71_topology"),
    ("try68_expert", "try71_expert"),
    ("logs_train_try68", "logs_train_try71"),
    ("logs_cleanup_try68", "logs_cleanup_try71"),
    ("submit_try68", "submit_try71"),
    ("run_sixtyeighth_try68", "run_seventyfirst_try71"),
    ("Try 68", "Try 71"),
    ("Try68", "Try71"),
    ("try68 chain", "try71 chain"),
    ("_fetch_try68", "_fetch_try71"),
    ("_relaunch_try68", "_relaunch_try71"),
    ("launch_try68_resume_from_try66", "launch_try71_resume_from_try68"),
    # Master port: try68 uses 29968, try71 uses 29971 (avoids collision with 67/68/69/70)
    ("MASTER_PORT:-29968", "MASTER_PORT:-29971"),
    ("MASTER_PORT=${MASTER_PORT:-29968}", "MASTER_PORT=${MASTER_PORT:-29971}"),
    # Submitter base-master-port: try68 uses 30268, try71 uses 30271
    ("30268", "30271"),
    # Output dirs
    ("outputs/try68_expert", "outputs/try71_expert"),
    # Remote cluster path suffix
    ("TFGSixtyEighthTry68", "TFGSeventyFirstTry71"),
]


def _rename_path(path: Path) -> Path:
    name = path.name
    new_name = name
    for old, new in (
        ("sixtyeighth_try68_experts", "seventyfirst_try71_experts"),
        ("sixtyeighth_try68_classifier", "seventyfirst_try71_classifier"),
        ("sixtyeighth_try68_ablation_staging", "seventyfirst_try71_ablation_staging"),
        ("try68_expert_registry", "try71_expert_registry"),
        ("try68_topology_classifier", "try71_topology_classifier"),
        ("try68_expert_", "try71_expert_"),
        ("run_sixtyeighth_try68_", "run_seventyfirst_try71_"),
        ("submit_try68_", "submit_try71_"),
        ("launch_try68_resume_from_try66", "launch_try71_resume_from_try68"),
        ("_fetch_try68_", "_fetch_try71_"),
        ("_fetch_try68.", "_fetch_try71."),
        ("_relaunch_try68", "_relaunch_try71"),
        ("TRY68_DESIGN.md", "TRY71_DESIGN.md"),
        ("TRY68_CHANGES.md", "TRY71_CHANGES.md"),
    ):
        if old in new_name:
            new_name = new_name.replace(old, new)
    if new_name != name:
        dest = path.with_name(new_name)
        path.rename(dest)
        return dest
    return path


def _transform_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    out = raw
    for a, b in CONTENT_REPLACEMENTS:
        out = out.replace(a, b)
    if out != raw:
        path.write_text(out, encoding="utf-8", newline="\n")


# Expert YAML names to DELETE (keep only open_sparse_lowrise)
_REMOVE_EXPERT_SUFFIXES = [
    "dense_block_highrise",
    "dense_block_midrise",
    "mixed_compact_lowrise",
    "mixed_compact_midrise",
    "open_sparse_vertical",
]

# Cluster submit scripts that cover all 6 experts — delete after copy
_REMOVE_CLUSTER_SCRIPTS = [
    "submit_try71_experts_4gpu_sequential.py",
    "submit_try71_experts_2gpu_sequential.py",
    "submit_try71_experts_1gpu_sequential.py",
    "submit_try71_chain_keep_running_cleanup.py",
    "submit_try71_stage1_stage2_4gpu.py",
]


def main() -> int:
    practice = Path(__file__).resolve().parents[1]
    src = practice / "TFGSixtyEighthTry68"
    dst = practice / "TFGSeventyFirstTry71"

    if not src.is_dir():
        print(f"Missing source: {src}", file=sys.stderr)
        return 1
    if dst.exists():
        print(f"Removing existing {dst}")
        shutil.rmtree(dst)

    print(f"Copying {src.name} -> {dst.name}")
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "outputs"),
    )

    # Rename subdirectories (deepest first to avoid path invalidation)
    all_dirs = sorted(
        [p for p in dst.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for d in all_dirs:
        new_name = d.name
        for old, new in (
            ("sixtyeighth_try68_experts", "seventyfirst_try71_experts"),
            ("sixtyeighth_try68_classifier", "seventyfirst_try71_classifier"),
            ("sixtyeighth_try68_ablation_staging", "seventyfirst_try71_ablation_staging"),
        ):
            if old in new_name:
                new_name = new_name.replace(old, new)
        if new_name != d.name:
            d.rename(d.with_name(new_name))

    # Rename files
    for f in list(dst.rglob("*")):
        if f.is_file():
            _rename_path(f)

    # Replace text content
    for f in dst.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() in TEXT_SUFFIXES or f.name.endswith(".slurm"):
            _transform_file(f)

    # Remove non-open_sparse_lowrise expert YAMLs
    experts_dir = dst / "experiments" / "seventyfirst_try71_experts"
    if experts_dir.is_dir():
        for suffix in _REMOVE_EXPERT_SUFFIXES:
            yaml_path = experts_dir / f"try71_expert_{suffix}.yaml"
            if yaml_path.exists():
                yaml_path.unlink()
                print(f"  Removed {yaml_path.name}")

    # Remove multi-expert cluster submit scripts (only open_sparse_lowrise will be submitted)
    cluster_dir = dst / "cluster"
    if cluster_dir.is_dir():
        for script_name in _REMOVE_CLUSTER_SCRIPTS:
            s = cluster_dir / script_name
            if s.exists():
                s.unlink()
                print(f"  Removed {script_name}")

    # Create empty outputs dir so the trainer can write checkpoints
    (dst / "outputs").mkdir(exist_ok=True)

    print(
        "\nDone. Manual patches still needed:\n"
        "  1. model_pmhhnet.py: out_channels=2 (mean + log_var channels)\n"
        "  2. train_partitioned_pathloss_expert.py: NLL loss + RMSE-vs-coverage eval\n"
        "  3. experiments/seventyfirst_try71_experts/try71_expert_open_sparse_lowrise.yaml:\n"
        "       model.out_channels: 2\n"
        "  4. cluster/launch_try71_resume_from_try68.py: set Try 68 ckpt path\n"
        "  5. cluster/run_seventyfirst_try71_*.slurm: TRY71_INIT_CHECKPOINT env var\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
