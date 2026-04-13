#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

try:
    import paramiko
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "TFGFiftySeventhTry57"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGFiftySeventhTry57"
HOST = "sert-entry-3"
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


def load_registry() -> list[dict[str, Any]]:
    registry_path = (
        LOCAL_DIR
        / "experiments"
        / "fiftyseventhtry57_partitioned_stage1"
        / "fiftyseventhtry57_expert_registry.yaml"
    )
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    experts = list(data.get("experts", [])) if isinstance(data, dict) else []
    if not experts:
        raise RuntimeError(f"No experts found in registry: {registry_path}")
    return experts


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Try57 and submit the 6 partitioned experts as 2-GPU jobs with cleanup between jobs.")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all-user-jobs", action="store_true", default=True)
    parser.add_argument("--wipe-try57-outputs", action="store_true", default=False)
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

    experts = load_registry()

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
            time.sleep(2.0)

        if args.wipe_try57_outputs:
            remote_exec(
                client,
                f"find {REMOTE_DIR}/outputs -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + 2>/dev/null || true; "
                f"rm -f {REMOTE_DIR}/logs_train_fiftyseventhtry57_* {REMOTE_DIR}/logs_cleanup_fiftyseventhtry57_*",
                check=False,
            )

        remote_exec(client, f"squeue -u {args.user}", check=False)

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        initial_cleanup = remote_sbatch(
            client,
            f"{sbatch_prefix} cluster/run_fiftyseventhtry57_cleanup_sert2001_1gpu.slurm",
        )
        current_dependency = initial_cleanup

        submitted: list[dict[str, str]] = []
        base_port = 29881
        for idx, expert in enumerate(experts):
            topology_class = str(expert["topology_class"])
            config = str(expert["config"])
            job_name = f"ckm-t55-{topology_class.replace('_', '-')[:40]}"
            master_port = str(base_port + idx)
            extra_exports = [
                f"CONFIG_PATH={config}",
                f"MASTER_PORT={master_port}",
                "PRECOMPUTE_PRIOR_CACHE=1",
            ]

            cmd = (
                f"{sbatch_prefix} --dependency=afterany:{current_dependency} -J {job_name} "
                f"--export=ALL,{','.join(extra_exports)} "
                "cluster/run_fiftyseventhtry57_partitioned_expert_2gpu.slurm"
            )
            job_id = remote_sbatch(client, cmd)
            cleanup_after = remote_sbatch(
                client,
                f"{sbatch_prefix} --dependency=afterany:{job_id} cluster/run_fiftyseventhtry57_cleanup_sert2001_1gpu.slurm",
            )
            submitted.append(
                {
                    "topology_class": topology_class,
                    "job_id": job_id,
                    "cleanup_after": cleanup_after,
                    "config": config,
                    "master_port": master_port,
                }
            )
            current_dependency = cleanup_after

        print({"initial_cleanup": initial_cleanup, "submitted": submitted})
    finally:
        client.close()


if __name__ == "__main__":
    main()
