"""Run the full Try80 quality metric comparison on validation and test.

This wrapper keeps validation and test outputs in one dated output root and
passes the local Windows paths explicitly, so the cluster paths stored in the
resolved training config do not leak into local reruns.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TFGPRACTICE = ROOT.parent
DEFAULT_EXPERIMENT = TFGPRACTICE / "cluster_outputs" / "TFGEightiethTry80" / "try80_joint_huge_pathloss_finetune"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["val", "test"],
        default=["val", "test"],
        help="Splits to evaluate sequentially.",
    )
    parser.add_argument("--device", default="directml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT / "resolved_config.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_EXPERIMENT / "best_model.pt")
    parser.add_argument("--hdf5-path", type=Path, default=TFGPRACTICE / "Datasets" / "CKM_Dataset_270326.h5")
    parser.add_argument("--try78-los-calibration-json", type=Path, default=ROOT / "calibrations" / "try78_los_two_ray_calibration.json")
    parser.add_argument("--try78-nlos-calibration-json", type=Path, default=ROOT / "calibrations" / "try78_nlos_regime_calibration.json")
    parser.add_argument("--try79-calibration-json", type=Path, default=ROOT / "calibrations" / "try79_calibration.json")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "all_quality_metrics_val_test_dml_b1_after_range_update",
    )
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits": args.splits,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "hdf5_path": str(args.hdf5_path.resolve()),
        "try78_los_calibration_json": str(args.try78_los_calibration_json.resolve()),
        "try78_nlos_calibration_json": str(args.try78_nlos_calibration_json.resolve()),
        "try79_calibration_json": str(args.try79_calibration_json.resolve()),
    }
    (args.out_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    compare_script = ROOT / "scripts" / "compare_prior_try80_all_quality_metrics.py"
    for split in args.splits:
        out_dir = args.out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(compare_script),
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--out-dir",
            str(out_dir),
            "--split",
            split,
            "--hdf5-path",
            str(args.hdf5_path),
            "--try78-los-calibration-json",
            str(args.try78_los_calibration_json),
            "--try78-nlos-calibration-json",
            str(args.try78_nlos_calibration_json),
            "--try79-calibration-json",
            str(args.try79_calibration_json),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--progress-every",
            str(args.progress_every),
        ]
        if args.mixed_precision:
            cmd.append("--mixed-precision")
        print(f"[{split}] running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)

    done = json.loads((args.out_root / "run_metadata.json").read_text(encoding="utf-8"))
    done["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.out_root / "run_metadata.json").write_text(json.dumps(done, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
