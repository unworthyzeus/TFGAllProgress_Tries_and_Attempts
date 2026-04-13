#!/usr/bin/env python3
"""Upload Try 66 and submit 6 single-stage expert jobs on 4 GPUs.

Single-stage pipeline: 513x513 direct, no Stage 2 refiner.
Each expert is submitted sequentially (afterany dependency chain).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import paramiko
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "TFGSixtySixthTry66"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtySixthTry66"
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


def load_registry(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    experts = list(data.get("experts", [])) if isinstance(data, dict) else []
    if not experts:
        raise RuntimeError(f"No experts found in registry: {path}")
    return experts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload Try 66 and submit 6 single-stage expert jobs on 4 GPUs."
    )
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all-user-jobs", action="store_true")
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password and not args.ssh_key:
        raise SystemExit(f"Set environment variable {args.password_env} or pass --ssh-key")

    if not args.skip_upload:
        run_local([
            sys.executable,
            str(ROOT / "cluster" / "upload_and_submit_experiments.py"),
            "--local-dir", str(LOCAL_DIR),
            "--upload-only",
            "--skip-datasets",
        ])

    registry = load_registry(
        LOCAL_DIR / "experiments" / "sixtysixth_try66_experts" / "try66_expert_registry.yaml"
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

        remote_exec(client, f"squeue -u {args.user}", check=False)

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        current_dep = ""
        base_port = 30066
        submitted: list[dict[str, str]] = []

        for idx, expert in enumerate(registry):
            tc = str(expert["topology_class"])
            config = str(expert["config"])
            job_name = f"t66-{tc.replace('_', '-')[:32]}"
            exports = [
                f"CONFIG_PATH={config}",
                "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
                f"MASTER_PORT={base_port + idx}",
            ]
            cmd = (
                f"{sbatch_prefix} {current_dep}-J {job_name} "
                f"--export=ALL,{','.join(exports)} "
                "cluster/run_sixtysixth_try66_4gpu.slurm"
            )
            job_id = remote_sbatch(client, cmd)
            submitted.append({
                "job_id": job_id,
                "expert_id": str(expert["expert_id"]),
                "config": config,
            })
            current_dep = f"--dependency=afterany:{job_id} "

        print("\n=== Submitted jobs ===")
        for s in submitted:
            print(f"  {s['expert_id']}: job {s['job_id']}")
        print(f"\nTotal: {len(submitted)} jobs submitted.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
