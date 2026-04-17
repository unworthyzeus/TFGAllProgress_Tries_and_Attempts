#!/usr/bin/env python3
"""Submit Try75 experts as parallel 2-GPU Slurm jobs (no sequential dependency chain).

Each train job uses cluster/run_seventyfifth_try75_2gpu.slurm.
Experts are resolved from experiments/seventyfifth_try75_experts/try75_expert_registry.yaml.

Default behavior (no positional args): submit all experts in registry order.
Pass explicit expert_ids to submit a subset.
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
LOCAL_DIR = ROOT / "TFGSeventyFifthTry75"
REGISTRY = LOCAL_DIR / "experiments" / "seventyfifth_try75_experts" / "try75_expert_registry.yaml"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGSeventyFifthTry75"
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


def load_registry() -> list[dict[str, Any]]:
    with REGISTRY.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    experts = list(data.get("experts", [])) if isinstance(data, dict) else []
    if not experts:
        raise RuntimeError(f"No experts in {REGISTRY}")
    return experts


def resolve_expert(expert_id: str, experts: list[dict[str, Any]]) -> dict[str, Any]:
    for row in experts:
        if str(row.get("expert_id")) == expert_id or str(row.get("topology_class")) == expert_id:
            return row
    ids = [str(r.get("expert_id")) for r in experts]
    raise SystemExit(f"Unknown expert_id {expert_id!r}. Known: {ids}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit parallel Try75 2-GPU jobs (one Slurm job per expert, same time)."
    )
    parser.add_argument(
        "expert_ids",
        nargs="*",
        help="expert_id values from try75_expert_registry.yaml. Defaults to all experts in registry order.",
    )
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all", action="store_true", help="scancel all user jobs before submitting.")
    parser.add_argument("--cancel-job-ids", default="", help="Comma-separated Slurm job IDs to scancel before submitting.")
    parser.add_argument("--base-master-port", type=int, default=30486)
    parser.add_argument("--no-resume", action="store_true", help="Skip RESUME_CHECKPOINT (start fresh from epoch 0).")
    parser.add_argument(
        "--final-cleanup",
        action="store_true",
        help="Submit one cleanup job after all parallel train jobs finish.",
    )
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password and not args.ssh_key:
        raise SystemExit(f"Set environment variable {args.password_env} or pass --ssh-key")

    all_experts = load_registry()
    rows = [resolve_expert(eid, all_experts) for eid in args.expert_ids] if args.expert_ids else all_experts

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
    connect_kwargs: dict[str, Any] = {
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
        if args.cancel_all:
            remote_exec(client, f"scancel -u {args.user}", check=False)
        cancel_raw = str(args.cancel_job_ids).strip()
        if cancel_raw:
            for jid in cancel_raw.split(","):
                jid = jid.strip()
                if jid:
                    remote_exec(client, f"scancel {jid}", check=False)
        remote_exec(client, f"squeue -u {args.user}", check=False)

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        submitted: list[tuple[str, str]] = []
        train_job_ids: list[str] = []

        for i, row in enumerate(rows):
            expert_id = str(row["expert_id"])
            config = str(row["config"])
            job_name = f"t75-{expert_id.replace('_', '-')[:28]}"
            exports = [
                f"CONFIG_PATH={config}",
                "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
                f"MASTER_PORT={args.base_master_port + i * 2}",
            ]
            if not args.no_resume:
                checkpoint = str(row.get("checkpoint", "")).strip()
                if checkpoint:
                    exports.append(f"RESUME_CHECKPOINT={checkpoint}")

            train_cmd = (
                f"{sbatch_prefix} -J {job_name} "
                f"--export=ALL,{','.join(exports)} "
                "cluster/run_seventyfifth_try75_2gpu.slurm"
            )
            train_job_id = remote_sbatch(client, train_cmd)
            submitted.append((job_name, train_job_id))
            train_job_ids.append(train_job_id)

        if args.final_cleanup and train_job_ids:
            deps = ":".join(train_job_ids)
            cleanup_cmd = (
                f"{sbatch_prefix} --dependency=afterany:{deps} "
                "-J t75-cleanup-final "
                "cluster/run_seventyfifth_try75_cleanup_sert2001_1gpu.slurm"
            )
            cleanup_job_id = remote_sbatch(client, cleanup_cmd)
            submitted.append(("t75-cleanup-final", cleanup_job_id))

        print("\n=== Submitted parallel jobs ===")
        for name, jid in submitted:
            tag = "  [cleanup]" if "cleanup" in name else "  [train]  "
            print(f"{tag} {name}: job {jid}")

        print(f"\nTotal train jobs: {len(train_job_ids)}")
        if args.final_cleanup:
            print("Final cleanup: enabled")
        else:
            print("Final cleanup: disabled")
    finally:
        client.close()


if __name__ == "__main__":
    main()
