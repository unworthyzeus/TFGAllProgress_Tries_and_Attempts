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
LOCAL_DIR = ROOT / "TFGSixtyFourthTry64"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyFourthTry64"
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
    parser = argparse.ArgumentParser(description="Upload Try64 and submit coarse-to-fine 128->513 stage1->stage2 expert chain on 4 GPUs.")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all-user-jobs", action="store_true")
    parser.add_argument("--wipe-try64-outputs", action="store_true")
    parser.add_argument("--stage1-only", action="store_true")
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
        LOCAL_DIR / "experiments" / "sixtyfourthtry64_partitioned_stage1" / "sixtyfourthtry64_expert_registry.yaml"
    )
    stage2_registry = load_registry(
        LOCAL_DIR / "experiments" / "sixtyfourthtry64_tail_refiner_stage2" / "sixtyfourthtry64_tail_refiner_registry.yaml"
    )
    stage2_by_id = {str(item["expert_id"]): item for item in stage2_registry}

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

        if args.wipe_try64_outputs:
            remote_exec(
                client,
                f"find {REMOTE_DIR}/outputs -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + 2>/dev/null || true; "
                f"rm -f {REMOTE_DIR}/logs_train_sixtyfourthtry64_*",
                check=False,
            )

        remote_exec(client, f"squeue -u {args.user}", check=False)
        remote_exec(
            client,
            (
                f"ssh -T {args.node} "
                "\""
                "pkill -9 -u gmoreno -f 'torchrun.*TFGSixtyFourthTry64' || true; "
                "pkill -9 -u gmoreno -f 'train_partitioned_pathloss_expert.py' || true; "
                "pkill -9 -u gmoreno -f 'train_pmnet_tail_refiner.py' || true; "
                "sleep 3; "
                "ps -fu gmoreno | egrep 'torchrun|train_partitioned_pathloss_expert|train_pmnet_tail_refiner' || true; "
                "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true"
                "\""
            ),
            check=False,
        )

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        current_dependency_arg = ""
        base_port = 29961
        submitted: list[dict[str, str]] = []
        stage1_job_ids: dict[str, str] = {}
        stage2_exports_by_id: dict[str, tuple[str, str, str]] = {}

        for idx, expert in enumerate(stage1_registry):
            expert_id = str(expert["expert_id"])
            topology_class = str(expert["topology_class"])
            stage1_config = str(expert["config"])
            stage2_config = str(stage2_by_id[expert_id]["config"])

            stage1_job_name = f"ckm-t64-s1-{topology_class.replace('_', '-')[:32]}"
            stage1_exports = [
                f"CONFIG_PATH={stage1_config}",
                "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
                f"MASTER_PORT={base_port + (idx * 2)}",
            ]
            stage1_cmd = (
                f"{sbatch_prefix} {current_dependency_arg}-J {stage1_job_name} "
                f"--export=ALL,{','.join(stage1_exports)} "
                "cluster/run_sixtyfourthtry64_4gpu.slurm"
            )
            stage1_job_id = remote_sbatch(client, stage1_cmd)
            stage1_job_ids[expert_id] = stage1_job_id
            submitted.append({"job_id": stage1_job_id, "stage": "stage1", "expert_id": expert_id, "config": stage1_config})
            stage2_exports_by_id[expert_id] = (
                topology_class,
                stage2_config,
                str(base_port + (idx * 2) + 1),
            )

            if args.stage1_only:
                current_dependency_arg = f"--dependency=afterany:{stage1_job_id} "
                continue

            current_dependency_arg = f"--dependency=afterany:{stage1_job_id} "

        if not args.stage1_only:
            last_stage1_dependency = current_dependency_arg
            for expert in stage1_registry:
                expert_id = str(expert["expert_id"])
                topology_class, stage2_config, master_port = stage2_exports_by_id[expert_id]
                stage2_job_name = f"ckm-t64-s2-{topology_class.replace('_', '-')[:32]}"
                stage2_exports = [
                    f"CONFIG_PATH={stage2_config}",
                    "TRAIN_SCRIPT=train_pmnet_tail_refiner.py",
                    f"MASTER_PORT={master_port}",
                ]
                stage2_cmd = (
                    f"{sbatch_prefix} {last_stage1_dependency}-J {stage2_job_name} "
                    f"--export=ALL,{','.join(stage2_exports)} "
                    "cluster/run_sixtyfourthtry64_4gpu.slurm"
                )
                stage2_job_id = remote_sbatch(client, stage2_cmd)
                submitted.append({"job_id": stage2_job_id, "stage": "stage2", "expert_id": expert_id, "config": stage2_config})
                last_stage1_dependency = f"--dependency=afterany:{stage2_job_id} "

        print({"submitted": submitted})
    finally:
        client.close()


if __name__ == "__main__":
    main()
