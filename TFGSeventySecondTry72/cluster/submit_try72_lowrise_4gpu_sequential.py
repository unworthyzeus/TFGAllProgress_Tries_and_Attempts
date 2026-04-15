#!/usr/bin/env python3
"""Upload Try 72, optionally cancel all user Slurm jobs, submit chained 4-GPU jobs for open_sparse_lowrise.

Each job uses ``cluster/run_seventysecond_try72_4gpu.slurm`` (4× RTX2080, 4:00:00 walltime). Jobs are
chained with ``--dependency=afterany:<prev>`` so only one runs at a time.

By default every job exports ``RESUME_CHECKPOINT`` to the expert ``best_model.pt`` under
``outputs/try72_expert_open_sparse_lowrise/`` (path relative to the repo on the cluster). Each
segment reloads that path at startup, so after job *k* updates the checkpoint, job *k+1* picks up
the latest weights.
"""
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
LOCAL_DIR = ROOT / "TFGSeventySecondTry72"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGSeventySecondTry72"
HOST = "sert.ac.upc.edu"
USER = "gmoreno"
TARGET_NODE = "sert-2001"
DEFAULT_CONFIG = "experiments/seventysecond_try72_experts/try72_expert_open_sparse_lowrise.yaml"
DEFAULT_RESUME_CHECKPOINT = "outputs/try72_expert_open_sparse_lowrise/best_model.pt"


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
    parser = argparse.ArgumentParser(description="Submit N sequential Try72 open_sparse_lowrise 4-GPU jobs.")
    parser.add_argument("--count", type=int, default=4, help="Number of chained jobs (default: 4).")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all-user-jobs", action="store_true", help="Run scancel -u USER before submitting.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="CONFIG_PATH relative to repo root on cluster.")
    parser.add_argument("--base-master-port", type=int, default=30166, help="MASTER_PORT for first job; +i per job.")
    parser.add_argument(
        "--resume-checkpoint",
        default=DEFAULT_RESUME_CHECKPOINT,
        help=(
            "Path passed as RESUME_CHECKPOINT to Slurm (relative to repo on cluster). "
            "Ignored if --no-resume is set."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Omit RESUME_CHECKPOINT (trainer resolves best_model.pt from output_dir).",
    )
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password and not args.ssh_key:
        raise SystemExit(f"Set environment variable {args.password_env} or pass --ssh-key")

    if not args.skip_upload:
        run_local([
            sys.executable,
            str(ROOT / "cluster" / "upload_and_submit_experiments.py"),
            "--local-dir",
            str(LOCAL_DIR),
            "--upload-only",
            "--skip-datasets",
        ])

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict = {
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

        remote_exec(client, f"squeue -u {args.user}", check=False)

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        current_dep = ""
        submitted: list[tuple[str, str]] = []

        for i in range(args.count):
            job_name = f"t72-osl-4gpu-{i + 1}"
            exports = [
                f"CONFIG_PATH={args.config}",
                "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
                f"MASTER_PORT={args.base_master_port + i}",
            ]
            if not args.no_resume and str(args.resume_checkpoint).strip():
                exports.append(f"RESUME_CHECKPOINT={args.resume_checkpoint}")
            cmd = (
                f"{sbatch_prefix} {current_dep}-J {job_name} "
                f"--export=ALL,{','.join(exports)} "
                "cluster/run_seventysecond_try72_4gpu.slurm"
            )
            job_id = remote_sbatch(client, cmd)
            submitted.append((job_name, job_id))
            current_dep = f"--dependency=afterany:{job_id} "

        print("\n=== Submitted jobs (sequential) ===")
        for name, jid in submitted:
            print(f"  {name}: job {jid}")
        print(f"\nTotal: {len(submitted)} jobs.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
