#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PRACTICE_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute Try 48 prior cache files by walking dataset samples.")
    p.add_argument("--try-dir", default=str(PRACTICE_ROOT / "TFGFortyEighthTry48"))
    p.add_argument(
        "--config",
        default="experiments/fortyeighthtry48_pmnet_prior_gan/fortyeighthtry48_pmnet_prior_gan.yaml",
    )
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=0, help="Optional max samples per split; 0 means all.")
    return p.parse_args()


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def main() -> None:
    args = parse_args()
    try_dir = _resolve(PRACTICE_ROOT, args.try_dir).resolve()
    config_path = _resolve(try_dir, args.config).resolve()

    sys.path.insert(0, str(try_dir))
    import config_utils
    import data_utils

    cfg = config_utils.load_config(str(config_path))
    config_utils.anchor_data_paths_to_config_file(cfg, str(config_path))
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
