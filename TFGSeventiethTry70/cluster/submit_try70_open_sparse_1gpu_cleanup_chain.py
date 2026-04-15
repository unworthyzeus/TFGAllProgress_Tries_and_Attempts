#!/usr/bin/env python3
"""Upload Try 70, optionally scancel all user jobs, submit chained 1-GPU train jobs with 1-GPU cleanup between.

Uses ``cluster/run_seventieth_try70_1gpu.slurm`` + ``cluster/run_seventieth_try70_cleanup_sert2001_1gpu.slurm``.

Example (from ``TFGpractice``)::

  python TFGSeventiethTry70/cluster/submit_try70_open_sparse_1gpu_cleanup_chain.py --cancel-all-user-jobs

After a successful run, the script prints ``LAST_TRY70_CLEANUP_JOB_ID=…``. Use that **numeric** id with
Try68's ``--dependency-afterany`` if you want Try68 to start only after the whole Try70 chain finishes.
Omit ``--dependency-afterany`` on Try68/Try69 for **independent** chains (same node may still serialize
under load).
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
LOCAL_DIR = ROOT / "TFGSeventiethTry70"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGSeventiethTry70"
HOST = "sert.ac.upc.edu"
USER = "gmoreno"
TARGET_NODE = "sert-2001"
DEFAULT_CONFIG = "experiments/seventieth_try70_experts/try70_expert_open_sparse_lowrise.yaml"
DEFAULT_TRY70_INIT = (
    "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68/outputs/try68_expert_open_sparse_lowrise/best_model.pt"
)
DEFAULT_RESUME_CHECKPOINT = "outputs/try70_expert_open_sparse_lowrise/best_model.pt"
TRAIN_SLURM = "cluster/run_seventieth_try70_1gpu.slurm"
CLEANUP_SLURM = "cluster/run_seventieth_try70_cleanup_sert2001_1gpu.slurm"


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
    parser = argparse.ArgumentParser(description="Try70 open_sparse: N×1-GPU train + cleanup between segments.")
    parser.add_argument("--count", type=int, default=4, help="Number of train segments (default: 4).")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all-user-jobs", action="store_true")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--base-master-port", type=int, default=30290)
    parser.add_argument("--try70-init-checkpoint", default=DEFAULT_TRY70_INIT)
    parser.add_argument("--resume-checkpoint", default=DEFAULT_RESUME_CHECKPOINT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-try70-init", action="store_true")
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
            job_name = f"t70-osl-1gpu-{i + 1}"
            exports = [
                f"CONFIG_PATH={args.config}",
                "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
                f"MASTER_PORT={args.base_master_port + i}",
            ]
            if i == 0 and not args.no_try70_init and str(args.try70_init_checkpoint).strip():
                exports.append(f"TRY70_INIT_CHECKPOINT={args.try70_init_checkpoint}")
            if not args.no_resume and str(args.resume_checkpoint).strip():
                exports.append(f"RESUME_CHECKPOINT={args.resume_checkpoint}")
            train_cmd = (
                f"{sbatch_prefix} {current_dep}-J {job_name} "
                f"--export=ALL,{','.join(exports)} "
                f"{TRAIN_SLURM}"
            )
            train_id = remote_sbatch(client, train_cmd)
            submitted.append((job_name, train_id))

            cleanup_name = f"t70-cleanup-{i + 1}"
            cleanup_cmd = (
                f"{sbatch_prefix} --dependency=afterany:{train_id} "
                f"-J {cleanup_name} {CLEANUP_SLURM}"
            )
            cleanup_id = remote_sbatch(client, cleanup_cmd)
            submitted.append((cleanup_name, cleanup_id))
            current_dep = f"--dependency=afterany:{cleanup_id} "

        print("\n=== Submitted (train 1gpu -> cleanup 1gpu) ===")
        for name, jid in submitted:
            tag = "  [cleanup]" if "cleanup" in name else "  [train]  "
            print(f"{tag} {name}: job {jid}")
        print(f"\nTotal: {len(submitted)} jobs ({args.count} train + {args.count} cleanup).")
        last_name, last_id = submitted[-1]
        if "cleanup" in last_name:
            print(
                f"\nLAST_TRY70_CLEANUP_JOB_ID={last_id}\n"
                f"(Pass this number to Try68: "
                f"python TFGSixtyEighthTry68/cluster/submit_try68_experts_1gpu_sequential.py "
                f"--skip-upload --dependency-afterany {last_id})"
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
