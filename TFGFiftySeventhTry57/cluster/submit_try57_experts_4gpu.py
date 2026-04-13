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
REMOTE_TRY49 = "/scratch/nas/3/gmoreno/TFGpractice/TFGFortyNinthTry49"
REMOTE_TRY52 = "/scratch/nas/3/gmoreno/TFGpractice/TFGFiftySecondTry52"
REMOTE_TRY54 = "/scratch/nas/3/gmoreno/TFGpractice/TFGFiftyFourthTry54"
HOST = "sert-entry-3"
USER = "gmoreno"
TARGET_NODE = "sert-2001"
SHARED_FORMULA_CACHE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/Datasets/try57_shared_formula_cache"
SHARED_FORMULA_PRECOMPUTED_HDF5 = f"{SHARED_FORMULA_CACHE_DIR}/try57_shared_formula_cache.h5"


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
    for expert in experts:
        config_path = LOCAL_DIR / str(expert["config"])
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        expert["base_channels"] = int(cfg.get("model", {}).get("base_channels", 0))
    experts.sort(key=lambda item: int(item.get("base_channels", 0)), reverse=True)
    return experts


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Try57 and submit the 6 topology experts as 4-GPU jobs with cleanup between jobs.")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all-user-jobs", action="store_true")
    parser.add_argument("--wipe-try57-outputs", action="store_true")
    parser.add_argument("--delete-try52", action="store_true")
    parser.add_argument("--delete-try49", action="store_true")
    parser.add_argument("--delete-try54", action="store_true")
    parser.add_argument("--shared-prior-cache-dir", default=SHARED_FORMULA_CACHE_DIR)
    parser.add_argument("--shared-prior-precomputed-hdf5", default=SHARED_FORMULA_PRECOMPUTED_HDF5)
    parser.add_argument("--submit-shared-prior-precompute", action="store_true")
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
            time.sleep(3.0)

        if args.delete_try52:
            remote_exec(client, f"rm -rf {REMOTE_TRY52}", check=False)
        if args.delete_try49:
            remote_exec(client, f"rm -rf {REMOTE_TRY49}", check=False)
        if args.delete_try54:
            remote_exec(client, f"rm -rf {REMOTE_TRY54}", check=False)
        if args.wipe_try57_outputs:
            remote_exec(
                client,
                f"find {REMOTE_DIR}/outputs -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + 2>/dev/null || true; "
                f"rm -f {REMOTE_DIR}/logs_train_fiftyseventhtry57_* {REMOTE_DIR}/logs_cleanup_fiftyseventhtry57_*",
                check=False,
            )
        remote_exec(client, f"squeue -u {args.user}", check=False)

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        submitted: list[dict[str, str]] = []
        dependency_arg = ""
        if args.submit_shared_prior_precompute:
            precompute_cmd = (
                f"{sbatch_prefix} -J ckm-t55-prior-cache "
                f"--export=ALL,SHARED_FORMULA_CACHE_DIR={args.shared_prior_cache_dir},"
                f"SHARED_FORMULA_PRECOMPUTED_HDF5={args.shared_prior_precomputed_hdf5},PRECOMPUTE_SPLITS='train val test' "
                "cluster/run_fiftyseventhtry57_shared_prior_precompute_1gpu.slurm"
            )
            precompute_job_id = remote_sbatch(client, precompute_cmd)
            submitted.append(
                {
                    "kind": "prior_precompute",
                    "job_id": precompute_job_id,
                    "shared_cache_dir": args.shared_prior_cache_dir,
                }
            )
            dependency_arg = f"--dependency=afterok:{precompute_job_id} "

        initial_cleanup = remote_sbatch(
            client,
            f"{sbatch_prefix} {dependency_arg}cluster/run_fiftyseventhtry57_cleanup_sert2001_1gpu.slurm",
        )
        current_dependency_arg = f"--dependency=afterany:{initial_cleanup} "
        base_port = 29861
        for idx, expert in enumerate(experts):
            topology_class = str(expert["topology_class"])
            config = str(expert["config"])
            job_name = f"ckm-t55-{topology_class.replace('_', '-')[:40]}"
            master_port = str(base_port + idx)
            extra_exports = [
                f"CONFIG_PATH={config}",
                f"MASTER_PORT={master_port}",
                "PRECOMPUTE_PRIOR_CACHE=0",
                f"FORMULA_CACHE_DIR={args.shared_prior_cache_dir}",
                f"FORMULA_PRECOMPUTED_HDF5={args.shared_prior_precomputed_hdf5}",
                "CLEANUP_FORMULA_CACHE=0",
            ]
            cmd = (
                f"{sbatch_prefix} {current_dependency_arg}-J {job_name} "
                f"--export=ALL,{','.join(extra_exports)} "
                "cluster/run_fiftyseventhtry57_partitioned_expert_4gpu.slurm"
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
            current_dependency_arg = f"--dependency=afterany:{cleanup_after} "

        print({"initial_cleanup": initial_cleanup, "submitted": submitted})
    finally:
        client.close()


if __name__ == "__main__":
    main()
