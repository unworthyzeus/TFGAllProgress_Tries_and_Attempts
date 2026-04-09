#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import paramiko
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "TFGFortyNinthTry49"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGFortyNinthTry49"
HOST = "sert.ac.upc.edu"
USER = "gmoreno"

STAGE1_INITIAL_CFG = (
    "experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/"
    "fortyninthtry49_pmnet_prior_stage1_widen112_initial_mae_dominant.yaml"
)
STAGE1_RESUME_CFG = (
    "experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/"
    "fortyninthtry49_pmnet_prior_stage1_widen112_resume_mae_dominant.yaml"
)
STAGE1_OUTPUT_SUFFIX = "t49_stage1_w112_4gpu"
STAGE1_SOURCE_CKPT = (
    "outputs/fortyninthtry49_pmnet_prior_stage1_t49_stage1_w112_4gpu/"
    "reduced_try49_epoch30_128_to_112.pt"
)
STAGE1_REDUCED_CKPT = (
    "outputs/fortyninthtry49_pmnet_prior_stage1_mae_dominant_t49_stage1_w112_4gpu/"
    "reduced_try49_epoch30_128_to_112.pt"
)


def run_local(command: list[str]) -> None:
    print("LOCAL>", " ".join(command))
    subprocess.run(command, check=True, cwd=str(ROOT))


def parse_job_id(text: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", text)
    if not match:
        raise RuntimeError(f"Could not parse job id from: {text!r}")
    return match.group(1)


def remote_sbatch(client: paramiko.SSHClient, command: str) -> str:
    full_cmd = f"cd {REMOTE_DIR} && {command}"
    print("REMOTE>", full_cmd)
    _, stdout, stderr = client.exec_command(full_cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err:
        raise RuntimeError(err)
    print(out)
    return parse_job_id(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Try49 and submit shrink->stage2->stage1->stage2->stage1 chain.")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--exclude-node", default="")
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password:
        raise SystemExit(f"Set environment variable {args.password_env}")

    if not args.skip_upload:
        run_local(
            [
                sys.executable,
                str(ROOT / "cluster" / "upload_and_submit_experiments.py"),
                "--local-dir",
                str(LOCAL_DIR),
                "--upload-only",
                "--no-clean-outputs",
                "--skip-datasets",
            ]
        )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=30)
    try:
        sbatch_prefix = "sbatch"
        if args.exclude_node:
            sbatch_prefix += f" --exclude={args.exclude_node}"
        shrink = remote_sbatch(
            client,
            f"{sbatch_prefix} cluster/run_fortyninthtry49_tail_refiner_stage2_shrink_84_1gpu.slurm",
        )
        stage2_a = remote_sbatch(
            client,
            f"{sbatch_prefix} --dependency=afterany:{shrink} cluster/run_fortyninthtry49_tail_refiner_stage2_4gpu.slurm",
        )
        stage1_a = remote_sbatch(
            client,
            f"{sbatch_prefix} "
            f"--dependency=afterany:{stage2_a} "
            f"--export=ALL,CONFIG_PATH={STAGE1_INITIAL_CFG},OUTPUT_SUFFIX={STAGE1_OUTPUT_SUFFIX},"
            f"SOURCE_CKPT={STAGE1_SOURCE_CKPT},"
            f"REDUCED_BASE_CKPT={STAGE1_REDUCED_CKPT} "
            "cluster/run_fortyninthtry49_pmnet_prior_stage1_4gpu.slurm",
        )
        stage2_b = remote_sbatch(
            client,
            f"{sbatch_prefix} --dependency=afterany:{stage1_a} cluster/run_fortyninthtry49_tail_refiner_stage2_4gpu.slurm",
        )
        stage1_b = remote_sbatch(
            client,
            f"{sbatch_prefix} "
            f"--dependency=afterany:{stage2_b} "
            f"--export=ALL,CONFIG_PATH={STAGE1_RESUME_CFG},OUTPUT_SUFFIX={STAGE1_OUTPUT_SUFFIX} "
            "cluster/run_fortyninthtry49_pmnet_prior_stage1_resume_4gpu.slurm",
        )
        print(
            {
                "shrink": shrink,
                "stage2_a": stage2_a,
                "stage1_a": stage1_a,
                "stage2_b": stage2_b,
                "stage1_b": stage1_b,
            }
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
