#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import paramiko
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "TFGFiftyFirstTry51"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGFiftyFirstTry51"
HOST = "sert.ac.upc.edu"
USER = "gmoreno"
TARGET_NODE = "sert-2001"


def run_local(command: list[str]) -> None:
    print("LOCAL>", " ".join(command))
    subprocess.run(command, check=True, cwd=str(ROOT))


def parse_job_id(text: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", text)
    if not match:
        raise RuntimeError(f"Could not parse job id from: {text!r}")
    return match.group(1)


def remote_exec(client: paramiko.SSHClient, command: str, *, check: bool = True) -> tuple[str, str]:
    print("REMOTE>", command)
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err:
        print(err)
    if check and err:
        raise RuntimeError(err)
    return out, err


def remote_sbatch(client: paramiko.SSHClient, command: str) -> str:
    out, err = remote_exec(client, f"cd {REMOTE_DIR} && {command}")
    if err:
        raise RuntimeError(err)
    return parse_job_id(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload Try51, cancel existing jobs, and submit cleanup->stage1->cleanup->stage2 chain on sert-2001."
    )
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all-user-jobs", action="store_true", default=True)
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
                "--skip-datasets",
            ]
        )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=30)
    try:
        if args.cancel_all_user_jobs:
            remote_exec(client, f"scancel -u {args.user}", check=False)
            time.sleep(2.0)

        remote_exec(client, f"squeue -u {args.user}", check=False)

        sbatch_prefix = f"sbatch --nodelist={args.node}"

        cleanup0 = remote_sbatch(
            client,
            f"{sbatch_prefix} cluster/run_fiftyfirsttry51_cleanup_sert2001_1gpu.slurm",
        )
        stage1_a = remote_sbatch(
            client,
            f"{sbatch_prefix} --dependency=afterany:{cleanup0} cluster/run_fiftyfirsttry51_pmnet_prior_stage1_4gpu.slurm",
        )
        cleanup1 = remote_sbatch(
            client,
            f"{sbatch_prefix} --dependency=afterany:{stage1_a} cluster/run_fiftyfirsttry51_cleanup_sert2001_1gpu.slurm",
        )
        stage2_a = remote_sbatch(
            client,
            f"{sbatch_prefix} --dependency=afterany:{cleanup1} cluster/run_fiftyfirsttry51_tail_refiner_stage2_4gpu.slurm",
        )
        cleanup2 = remote_sbatch(
            client,
            f"{sbatch_prefix} --dependency=afterany:{stage2_a} cluster/run_fiftyfirsttry51_cleanup_sert2001_1gpu.slurm",
        )
        stage1_b = remote_sbatch(
            client,
            f"{sbatch_prefix} --dependency=afterany:{cleanup2} cluster/run_fiftyfirsttry51_pmnet_prior_stage1_resume_4gpu.slurm",
        )
        cleanup3 = remote_sbatch(
            client,
            f"{sbatch_prefix} --dependency=afterany:{stage1_b} cluster/run_fiftyfirsttry51_cleanup_sert2001_1gpu.slurm",
        )
        stage2_b = remote_sbatch(
            client,
            f"{sbatch_prefix} --dependency=afterany:{cleanup3} cluster/run_fiftyfirsttry51_tail_refiner_stage2_4gpu.slurm",
        )
        cleanup4 = remote_sbatch(
            client,
            f"{sbatch_prefix} --dependency=afterany:{stage2_b} cluster/run_fiftyfirsttry51_cleanup_sert2001_1gpu.slurm",
        )

        print(
            {
                "cleanup0": cleanup0,
                "stage1_a": stage1_a,
                "cleanup1": cleanup1,
                "stage2_a": stage2_a,
                "cleanup2": cleanup2,
                "stage1_b": stage1_b,
                "cleanup3": cleanup3,
                "stage2_b": stage2_b,
                "cleanup4": cleanup4,
            }
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
