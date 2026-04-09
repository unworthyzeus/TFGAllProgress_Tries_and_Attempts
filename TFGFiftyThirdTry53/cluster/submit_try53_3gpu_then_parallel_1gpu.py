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
LOCAL_DIR = ROOT / "TFGFiftyThirdTry53"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGFiftyThirdTry53"
HOST = "sert.ac.upc.edu"
USER = "gmoreno"


def run_local(command: list[str]) -> None:
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
        description="Upload Try53 and submit: stage1(3gpu,1h) -> stage2(3gpu,2h) -> stage3(3gpu,2h), then stage1/stage2/stage3 long 1gpu jobs in parallel."
    )
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--cancel-all-user-jobs", action="store_true", default=False)
    parser.add_argument("--wipe-try53-outputs", action="store_true", default=False)
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password and not args.ssh_key:
        raise SystemExit(f"Set environment variable {args.password_env} or pass --ssh-key")

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
    connect_kwargs = {
        "hostname": args.host,
        "username": args.user,
        "timeout": 30,
        "allow_agent": True,
        "look_for_keys": True,
    }
    if password:
        connect_kwargs["password"] = password
    if args.ssh_key:
        connect_kwargs["key_filename"] = args.ssh_key
    client.connect(**connect_kwargs)
    try:
        if args.cancel_all_user_jobs:
            remote_exec(client, f"scancel -u {args.user}", check=False)
        if args.wipe_try53_outputs:
            remote_exec(client, f"rm -rf {REMOTE_DIR}/outputs", check=False)
            remote_exec(client, f"rm -f {REMOTE_DIR}/logs_train_fiftythirdtry53_*", check=False)
            remote_exec(client, f"rm -f {REMOTE_DIR}/logs_cleanup_fiftythirdtry53_*", check=False)
            remote_exec(client, f"mkdir -p {REMOTE_DIR}/outputs", check=False)

        cleanup = remote_sbatch(client, "sbatch cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
        stage1_short = remote_sbatch(
            client,
            f"sbatch --dependency=afterany:{cleanup} cluster/run_fiftythirdtry53_stage1_bootstrap_3gpu_1h.slurm",
        )
        stage2_short = remote_sbatch(
            client,
            f"sbatch --dependency=afterany:{stage1_short} cluster/run_fiftythirdtry53_stage2_3gpu_2h.slurm",
        )
        stage3_short = remote_sbatch(
            client,
            f"sbatch --dependency=afterany:{stage2_short} cluster/run_fiftythirdtry53_stage3_3gpu_2h.slurm",
        )
        stage1_long = remote_sbatch(
            client,
            f"sbatch --dependency=afterany:{stage3_short} cluster/run_fiftythirdtry53_stage1_feedback_1gpu_2d_smallgpu.slurm",
        )
        stage2_long = remote_sbatch(
            client,
            f"sbatch --dependency=afterany:{stage3_short} cluster/run_fiftythirdtry53_stage2_1gpu_2d_smallgpu.slurm",
        )
        stage3_long = remote_sbatch(
            client,
            f"sbatch --dependency=afterany:{stage3_short} cluster/run_fiftythirdtry53_stage3_1gpu_2d_smallgpu.slurm",
        )
        print(
            {
                "cleanup": cleanup,
                "stage1_short": stage1_short,
                "stage2_short": stage2_short,
                "stage3_short": stage3_short,
                "stage1_long_parallel": stage1_long,
                "stage2_long_parallel": stage2_long,
                "stage3_long_parallel": stage3_long,
            }
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
