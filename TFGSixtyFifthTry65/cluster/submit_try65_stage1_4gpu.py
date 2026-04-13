#!/usr/bin/env python3
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
LOCAL_DIR = ROOT / "TFGSixtyFifthTry65"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyFifthTry65"
HOST = "sert-entry-4"
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
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    experts = list(data.get("experts", [])) if isinstance(data, dict) else []
    if not experts:
        raise RuntimeError(f"No experts found in registry: {path}")
    return experts


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Try65 and submit single-stage grokking expert chain on 4 GPUs.")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all-user-jobs", action="store_true")
    parser.add_argument("--wipe-try65-outputs", action="store_true")
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

    stage1_registry = load_registry(
        LOCAL_DIR / "experiments" / "sixtyfifthtry65_grokking_stage1" / "sixtyfifthtry65_expert_registry.yaml"
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

        if args.wipe_try65_outputs:
            remote_exec(
                client,
                f"find {REMOTE_DIR}/outputs -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + 2>/dev/null || true; "
                f"rm -f {REMOTE_DIR}/logs_train_sixtyfifthtry65_*",
                check=False,
            )

        remote_exec(client, f"squeue -u {args.user}", check=False)
        remote_exec(
            client,
            (
                f"ssh -T {args.node} "
                "\""
                "pkill -9 -u gmoreno -f 'torchrun.*TFGSixtyFifthTry65' || true; "
                "pkill -9 -u gmoreno -f 'train_partitioned_pathloss_expert.py' || true; "
                "sleep 3; "
                "ps -fu gmoreno | egrep 'torchrun|train_partitioned_pathloss_expert' || true; "
                "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true"
                "\""
            ),
            check=False,
        )

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        current_dependency_arg = ""
        base_port = 30161
        submitted: list[dict[str, str]] = []

        for idx, expert in enumerate(stage1_registry):
            topology_class = str(expert["topology_class"])
            stage1_config = str(expert["config"])
            job_name = f"ckm-t65-{topology_class.replace('_', '-')[:32]}"
            exports = [
                f"CONFIG_PATH={stage1_config}",
                "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
                f"MASTER_PORT={base_port + idx}",
            ]
            cmd = (
                f"{sbatch_prefix} {current_dependency_arg}-J {job_name} "
                f"--export=ALL,{','.join(exports)} "
                "cluster/run_sixtyfifthtry65_4gpu.slurm"
            )
            job_id = remote_sbatch(client, cmd)
            submitted.append({"job_id": job_id, "stage": "stage1", "expert_id": str(expert["expert_id"]), "config": stage1_config})
            current_dependency_arg = f"--dependency=afterany:{job_id} "

        print({"submitted": submitted})
    finally:
        client.close()


if __name__ == "__main__":
    main()
