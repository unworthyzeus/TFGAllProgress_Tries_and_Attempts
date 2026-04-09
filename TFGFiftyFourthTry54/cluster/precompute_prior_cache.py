#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


TRY_DIR = Path(__file__).resolve().parent.parent
if str(TRY_DIR) not in sys.path:
    sys.path.insert(0, str(TRY_DIR))

import config_utils
import data_utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute formula prior cache files for Try 48.")
    p.add_argument("--config", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--limit", type=int, default=0, help="Optional max samples per split; 0 means all.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_utils.load_config(args.config)
    config_utils.anchor_data_paths_to_config_file(cfg, args.config)
    cfg["augmentation"] = dict(cfg.get("augmentation", {}))
    cfg["augmentation"]["enable"] = False
    splits = data_utils.build_dataset_splits_from_config(cfg)

    started = time.time()
    for split_name in args.splits:
        if split_name not in splits:
            continue
        dataset = splits[split_name]
        total = len(dataset)
        limit = total if int(args.limit) <= 0 else min(int(args.limit), total)
        print(f"[prior-cache:{split_name}] {limit}/{total} samples", flush=True)
        for idx in range(limit):
            _ = dataset[idx]
            if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == limit:
                elapsed = time.time() - started
                print(
                    f"[prior-cache:{split_name}] {idx + 1}/{limit} elapsed={elapsed/60.0:.1f} min",
                    flush=True,
                )


if __name__ == "__main__":
    main()
