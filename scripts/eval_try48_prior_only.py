#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from torch.utils.data import DataLoader


PRACTICE_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the formula prior channel only for Try 48.")
    p.add_argument("--try-dir", default=str(PRACTICE_ROOT / "TFGFortyEighthTry48"))
    p.add_argument(
        "--config",
        default="experiments/fortyeighthtry48_pmnet_prior_gan/fortyeighthtry48_pmnet_prior_gan.yaml",
    )
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _formula_channel_index(cfg: dict) -> int:
    idx = 1
    if cfg["data"].get("los_input_column"):
        idx += 1
    if bool(cfg["data"].get("distance_map_channel", False)):
        idx += 1
    return idx


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
    dataset = data_utils.build_dataset_splits_from_config(cfg)[args.split]

    loader_kwargs = {}
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        persistent_workers=int(args.num_workers) > 0,
        **loader_kwargs,
    )

    scale = float(cfg["target_metadata"]["path_loss"]["scale"])
    count = sq = abs_ = 0.0
    count_l = sq_l = abs_l = 0.0
    count_n = sq_n = abs_n = 0.0

    total = len(dataset) if int(args.limit) <= 0 else min(int(args.limit), len(dataset))
    for i, (x, y, m) in enumerate(loader, start=1):
        if i > total:
            break
        ch = _formula_channel_index(cfg)
        prior = x[:, ch : ch + 1] * scale
        tgt = y[:, 0:1] * scale
        valid = m[:, 0:1] > 0
        los = x[:, 1:2] > 0.5

        diff = (prior - tgt)[valid]
        if diff.numel():
            count += diff.numel()
            sq += float((diff * diff).sum())
            abs_ += float(diff.abs().sum())

        diff_l = (prior - tgt)[valid & los]
        if diff_l.numel():
            count_l += diff_l.numel()
            sq_l += float((diff_l * diff_l).sum())
            abs_l += float(diff_l.abs().sum())

        diff_n = (prior - tgt)[valid & (~los)]
        if diff_n.numel():
            count_n += diff_n.numel()
            sq_n += float((diff_n * diff_n).sum())
            abs_n += float(diff_n.abs().sum())

        if i == 1 or i % 250 == 0 or i == total:
            print(
                json.dumps(
                    {
                        "i": i,
                        "total": total,
                        "overall_rmse": round(math.sqrt(sq / max(count, 1.0)), 4),
                        "los_rmse": round(math.sqrt(sq_l / max(count_l, 1.0)), 4) if count_l else None,
                        "nlos_rmse": round(math.sqrt(sq_n / max(count_n, 1.0)), 4) if count_n else None,
                    }
                ),
                flush=True,
            )

    result = {
        "split": args.split,
        "samples": total,
        "overall_rmse": math.sqrt(sq / count),
        "overall_mae": abs_ / count,
        "los_rmse": math.sqrt(sq_l / count_l),
        "los_mae": abs_l / count_l,
        "nlos_rmse": math.sqrt(sq_n / count_n),
        "nlos_mae": abs_n / count_n,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
