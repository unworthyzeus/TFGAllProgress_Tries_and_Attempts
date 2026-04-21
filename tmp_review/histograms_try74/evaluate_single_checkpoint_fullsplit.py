from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from compute_histograms import run_checkpoint


def _empty_bucket() -> dict[str, float | int]:
    return {"sse": 0.0, "sae": 0.0, "count": 0}


def _update_bucket(bucket: dict[str, float | int], diff: np.ndarray) -> None:
    if diff.size == 0:
        return
    diff64 = diff.astype(np.float64, copy=False)
    bucket["sse"] = float(bucket["sse"]) + float(np.square(diff64).sum())
    bucket["sae"] = float(bucket["sae"]) + float(np.abs(diff64).sum())
    bucket["count"] = int(bucket["count"]) + int(diff.size)


def _finalize_bucket(bucket: dict[str, float | int]) -> dict[str, float | int]:
    count = int(bucket["count"])
    if count <= 0:
        return {"rmse_db": float("nan"), "mae_db": float("nan"), "count": 0}
    sse = float(bucket["sse"])
    sae = float(bucket["sae"])
    return {
        "rmse_db": float(math.sqrt(sse / count)),
        "mae_db": float(sae / count),
        "count": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one local path-loss checkpoint on the full split without expert routing."
    )
    parser.add_argument("--try-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="directml", choices=["auto", "cuda", "directml", "dml", "cpu"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    buckets = {
        "ground": _empty_bucket(),
        "los": _empty_bucket(),
        "nlos": _empty_bucket(),
    }
    per_city_type: dict[str, dict[str, dict[str, float | int]]] = {}
    sample_count = 0

    for (
        _key,
        pred_np,
        tgt_np,
        los_bool,
        vm_np,
        city,
        sample_ref,
        ant_h,
        city_type_3,
        city_type_6,
        exclude_non_ground_targets,
        _delay_gt,
        _angular_gt,
    ) in run_checkpoint(
        try_root=args.try_root.resolve(),
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        split=args.split,
        max_samples=args.max_samples,
        device_str=args.device,
        eval_batch_size=args.batch_size,
        eval_num_workers=args.num_workers,
        progress_every=args.progress_every,
        all_samples=True,
        force_full_region_mask=True,
        exclude_building_pixels=True,
        sample_meta_cache={},
    ):
        del city, sample_ref, ant_h, exclude_non_ground_targets
        sample_count += 1
        diff = pred_np - tgt_np

        ground_sel = vm_np
        los_sel = vm_np & los_bool
        nlos_sel = vm_np & (~los_bool)

        _update_bucket(buckets["ground"], diff[ground_sel])
        _update_bucket(buckets["los"], diff[los_sel])
        _update_bucket(buckets["nlos"], diff[nlos_sel])

        city_bucket = per_city_type.setdefault(
            city_type_3 or city_type_6 or "unknown",
            {"ground": _empty_bucket(), "los": _empty_bucket(), "nlos": _empty_bucket()},
        )
        _update_bucket(city_bucket["ground"], diff[ground_sel])
        _update_bucket(city_bucket["los"], diff[los_sel])
        _update_bucket(city_bucket["nlos"], diff[nlos_sel])

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "try_root": str(args.try_root.resolve()),
        "device": args.device,
        "split": args.split,
        "sample_count": sample_count,
        "metrics": {name: _finalize_bucket(bucket) for name, bucket in buckets.items()},
        "metrics_by_city_type_3": {
            name: {metric: _finalize_bucket(bucket) for metric, bucket in metric_buckets.items()}
            for name, metric_buckets in sorted(per_city_type.items())
        },
    }

    print(json.dumps(summary, indent=2))
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
