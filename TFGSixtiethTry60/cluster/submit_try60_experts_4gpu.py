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
LOCAL_DIR = ROOT / "TFGSixtiethTry60"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtiethTry60"
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


def load_registry() -> list[dict[str, Any]]:
    registry_path = (
        LOCAL_DIR
        / "experiments"
        / "sixtiethtry60_partitioned_stage1"
        / "sixtiethtry60_expert_registry.yaml"
    )
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    experts = list(data.get("experts", [])) if isinstance(data, dict) else []
    if not experts:
        raise RuntimeError(f"No experts found in registry: {registry_path}")
    for expert in experts:
        config_path = LOCAL_DIR / str(expert["config"])
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        expert["base_channels"] = int(cfg.get("model", {}).get("base_channels", 0))
    experts.sort(key=lambda item: (int(item.get("base_channels", 0)), str(item.get("topology_class", ""))), reverse=True)
    return experts


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Try60 and submit the 6 topology experts as 4-GPU jobs with cleanup between jobs.")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all-user-jobs", action="store_true")
    parser.add_argument("--wipe-try60-outputs", action="store_true")
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

        if args.wipe_try60_outputs:
            remote_exec(
                client,
                f"find {REMOTE_DIR}/outputs -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + 2>/dev/null || true; "
                f"rm -f {REMOTE_DIR}/logs_train_sixtiethtry60_* {REMOTE_DIR}/logs_cleanup_sixtiethtry60_*",
                check=False,
            )

        remote_exec(client, f"squeue -u {args.user}", check=False)

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        cleanup_script = "cluster/run_sixtiethtry60_cleanup_sert2001_1gpu.slurm"
        initial_cleanup = remote_sbatch(client, f"{sbatch_prefix} {cleanup_script}")
        submitted: list[dict[str, str]] = []
        current_dependency = initial_cleanup
        base_port = 29941
        for idx, expert in enumerate(experts):
            topology_class = str(expert["topology_class"])
            config = str(expert["config"])
            job_name = f"ckm-t60-{topology_class.replace('_', '-')[:40]}"
            master_port = str(base_port + idx)
            extra_exports = [
                f"CONFIG_PATH={config}",
                f"MASTER_PORT={master_port}",
            ]
            cmd = (
                f"{sbatch_prefix} --dependency=afterany:{current_dependency} -J {job_name} "
                f"--export=ALL,{','.join(extra_exports)} "
                "cluster/run_sixtiethtry60_partitioned_expert_4gpu.slurm"
            )
            job_id = remote_sbatch(client, cmd)
            cleanup_after = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{job_id} {cleanup_script}")
            submitted.append(
                {
                    "topology_class": topology_class,
                    "job_id": job_id,
                    "config": config,
                    "master_port": master_port,
                    "cleanup_after": cleanup_after,
                }
            )
            current_dependency = cleanup_after

        print({"initial_cleanup": initial_cleanup, "submitted": submitted})
    finally:
        client.close()


if __name__ == "__main__":
    main()
