"""Build per-expert path-loss clamps from the histogram study.

Reads ``docs/histogram_study.json`` (produced by ``study_histograms.py``) and
emits ``docs/expert_clamps.json`` with ``(clamp_lo, clamp_hi)`` for each of the
12 experts, applying a small margin on both sides.

The Try 76 Stage-A head uses these clamps to keep Gaussian-mixture means inside
the physical range. They are read at config-load time via
``src/config_try76.load_clamp_table``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


MARGIN_DB = 3.0

REGION_LABEL = {
    "los_only": "target_los",
    "nlos_only": "target_nlos",
}


def build(study_json: Path, margin: float = MARGIN_DB) -> Dict[str, Dict[str, float]]:
    with open(study_json, "r", encoding="utf-8") as f:
        study = json.load(f)
    by_class = study.get("by_class", {})
    overall = study.get("overall", {})

    clamps: Dict[str, Dict[str, float]] = {}
    for klass, leafs in by_class.items():
        for region_mode, target_kind in REGION_LABEL.items():
            key = f"path_loss|{target_kind}"
            leaf = leafs.get(key)
            if leaf is None:
                leaf = overall.get(key)
            if leaf is None:
                continue
            lo = float(leaf.get("nonzero_lo", 0)) - margin
            hi = float(leaf.get("nonzero_hi", 180)) + margin
            clamps[f"{klass}_{region_mode.replace('_only', '')}"] = {
                "clamp_lo": max(float(int(round(lo))), 0.0),
                "clamp_hi": min(float(int(round(hi))), 180.0),
            }
    return clamps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-json", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "histogram_study.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "expert_clamps.json")
    parser.add_argument("--margin", type=float, default=MARGIN_DB)
    args = parser.parse_args()

    clamps = build(args.study_json, margin=args.margin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clamps, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(clamps)} experts)")


if __name__ == "__main__":
    main()
