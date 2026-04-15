#!/usr/bin/env python3
"""Copy TFGSixtySeventhTry67 -> TFGSixtyNinthTry69 with try67/try69 string renames (cluster ports bumped).

Run from repo root:
  python TFGpractice/scripts/bootstrap_try69_from_try67.py

Post-run: patch Try 69 `train_partitioned_pathloss_expert.py` (corridor loss) and
`scripts/generate_try69_configs.py` (SOA defaults), then regenerate expert YAMLs.
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
    ("TFGSixtySeventhTry67", "TFGSixtyNinthTry69"),
    ("sixtyseventh_try67", "sixtyninth_try69"),
    ("try67_expert_registry", "try69_expert_registry"),
    ("try67_topology", "try69_topology"),
    ("try67_expert", "try69_expert"),
    ("logs_train_try67", "logs_train_try69"),
    ("logs_cleanup_try67", "logs_cleanup_try69"),
    ("submit_try67", "submit_try69"),
    ("run_sixtyseventh_try67", "run_sixtyninth_try69"),
    ("Try 67", "Try 69"),
    ("Try67", "Try69"),
    ("try67 chain", "try69 chain"),
    ("generate_try67_configs", "generate_try69_configs"),
    ("plot_try67_metrics", "plot_try69_metrics"),
    ("_fetch_try67", "_fetch_try69"),
    ("MASTER_PORT:-29966", "MASTER_PORT:-29976"),
    ("MASTER_PORT=${MASTER_PORT:-29966}", "MASTER_PORT=${MASTER_PORT:-29976}"),
    ("30266", "30286"),
]


def _rename_path(path: Path) -> Path:
    name = path.name
    new_name = name
    for old, new in (
        ("sixtyseventh_try67_experts", "sixtyninth_try69_experts"),
        ("sixtyseventh_try67_classifier", "sixtyninth_try69_classifier"),
        ("try67_expert_registry", "try69_expert_registry"),
        ("try67_topology_classifier", "try69_topology_classifier"),
        ("try66_topology_classifier", "try69_topology_classifier"),
        ("try67_expert_", "try69_expert_"),
        ("run_sixtyseventh_try67_", "run_sixtyninth_try69_"),
        ("submit_try67_", "submit_try69_"),
        ("generate_try67_configs", "generate_try69_configs"),
        ("plot_try67_metrics", "plot_try69_metrics"),
        ("_fetch_try67_", "_fetch_try69_"),
        ("_relaunch_try67", "_relaunch_try69"),
        ("TRY67_DESIGN.md", "TRY69_DESIGN.md"),
        ("TRY67_CHANGES.md", "TRY69_CHANGES.md"),
        ("TRY67_IMPLEMENTATION.md", "TRY69_IMPLEMENTATION.md"),
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


def main() -> int:
    practice = Path(__file__).resolve().parents[1]
    src = practice / "TFGSixtySeventhTry67"
    dst = practice / "TFGSixtyNinthTry69"
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
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )

    all_dirs = sorted([p for p in dst.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True)
    for d in all_dirs:
        new_name = d.name
        for old, new in (
            ("sixtyseventh_try67_experts", "sixtyninth_try69_experts"),
            ("sixtyseventh_try67_classifier", "sixtyninth_try69_classifier"),
        ):
            if old in new_name:
                new_name = new_name.replace(old, new)
        if new_name != d.name:
            d.rename(d.with_name(new_name))

    for f in list(dst.rglob("*")):
        if f.is_file():
            _rename_path(f)

    for f in dst.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() in TEXT_SUFFIXES or f.name.endswith(".slurm"):
            _transform_file(f)

    print("Done. Apply Try 69 SOA patches (corridor loss, generate_try69_configs) and regenerate YAMLs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
