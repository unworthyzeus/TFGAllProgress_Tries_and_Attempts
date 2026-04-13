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


ROOT = Path(__file__).resolve().parents[1]
HOST = "sert-entry-3"
USER = "gmoreno"
TARGET_NODE = "sert-2001"
PASSWORD_ENV = "SSH_PASSWORD"
TRY57_SHARED_CACHE = "/scratch/nas/3/gmoreno/TFGpractice/Datasets/try57_shared_formula_cache/try57_shared_formula_cache.h5"
TRY55_SHARED_CACHE = "/scratch/nas/3/gmoreno/TFGpractice/Datasets/try55_shared_formula_cache/try55_shared_formula_cache.h5"

TRY_SPECS = {
    "try57": {
        "local_dir": ROOT / "TFGFiftySeventhTry57",
        "remote_dir": "/scratch/nas/3/gmoreno/TFGpractice/TFGFiftySeventhTry57",
        "registry_path": ROOT / "TFGFiftySeventhTry57" / "experiments" / "fiftyseventhtry57_partitioned_stage1" / "fiftyseventhtry57_expert_registry.yaml",
        "run_slurm": "cluster/run_fiftyseventhtry57_partitioned_expert_2gpu.slurm",
        "cleanup_slurm": "cluster/run_fiftyseventhtry57_cleanup_sert2001_1gpu.slurm",
        "job_prefix": "ckm-t57",
        "base_port": 31100,
        "extra_exports": {
            "PRECOMPUTE_PRIOR_CACHE": "0",
            "FORMULA_CACHE_DIR": "/scratch/nas/3/gmoreno/TFGpractice/Datasets/try57_shared_formula_cache",
            "FORMULA_PRECOMPUTED_HDF5": "/scratch/nas/3/gmoreno/TFGpractice/Datasets/try57_shared_formula_cache/try57_shared_formula_cache.h5",
            "CLEANUP_FORMULA_CACHE": "0",
        },
        "log_glob": "logs_train_fiftyseventhtry57_* logs_cleanup_fiftyseventhtry57_*",
    },
    "try58": {
        "local_dir": ROOT / "TFGFiftyEighthTry58",
        "remote_dir": "/scratch/nas/3/gmoreno/TFGpractice/TFGFiftyEighthTry58",
        "registry_path": ROOT / "TFGFiftyEighthTry58" / "experiments" / "fiftyeighthtry58_topology_experts" / "fiftyeighthtry58_expert_registry.yaml",
        "run_slurm": "cluster/run_try58_topology_expert_1gpu.slurm",
        "cleanup_slurm": "cluster/run_fiftyeighthtry58_cleanup_sert2001_1gpu.slurm",
        "job_prefix": "ckm-t58",
        "base_port": 31200,
        "extra_exports": {},
        "log_glob": "logs_try58_expert_* logs_cleanup_fiftyeighthtry58_*",
    },
    "try59": {
        "local_dir": ROOT / "TFGFiftyNinthTry59",
        "remote_dir": "/scratch/nas/3/gmoreno/TFGpractice/TFGFiftyNinthTry59",
        "registry_path": ROOT / "TFGFiftyNinthTry59" / "experiments" / "fiftyninthtry59_topology_experts" / "fiftyninthtry59_expert_registry.yaml",
        "run_slurm": "cluster/run_try59_topology_expert_1gpu.slurm",
        "cleanup_slurm": "cluster/run_fiftyninthtry59_cleanup_sert2001_1gpu.slurm",
        "job_prefix": "ckm-t59",
        "base_port": 31300,
        "extra_exports": {},
        "log_glob": "logs_try59_expert_* logs_cleanup_fiftyninthtry59_*",
    },
}

TOPOLOGY_ORDER = [
    "open_sparse_lowrise",
    "open_sparse_vertical",
    "mixed_compact_lowrise",
    "mixed_compact_midrise",
    "dense_block_midrise",
    "dense_block_highrise",
]


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


def remote_sbatch(client: paramiko.SSHClient, remote_dir: str, command: str) -> str:
    out, err = remote_exec(client, f"cd {remote_dir} && {command}")
    if err:
        raise RuntimeError(err)
    return parse_job_id(out)


def load_registry_map(registry_path: Path) -> dict[str, dict[str, Any]]:
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    experts = list(data.get("experts", [])) if isinstance(data, dict) else []
    mapping: dict[str, dict[str, Any]] = {}
    for expert in experts:
        topology_class = str(expert["topology_class"])
        mapping[topology_class] = dict(expert)
    missing = [name for name in TOPOLOGY_ORDER if name not in mapping]
    if missing:
        raise RuntimeError(f"Registry {registry_path} is missing topology classes: {missing}")
    return mapping


def build_job_waves(registries: dict[str, dict[str, dict[str, Any]]]) -> list[list[dict[str, str]]]:
    waves: list[list[dict[str, str]]] = []
    for topology_class in TOPOLOGY_ORDER:
        wave: list[dict[str, str]] = []
        for try_name in ("try57", "try58", "try59"):
            expert = registries[try_name][topology_class]
            wave.append(
                {
                    "try_name": try_name,
                    "topology_class": topology_class,
                    "config": str(expert["config"]),
                }
            )
        waves.append(wave)
    return waves


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload and submit Try57/58/59 expert jobs interleaved by topology on 1 GPU.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default=PASSWORD_ENV)
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--cancel-all-user-jobs", action="store_true")
    parser.add_argument("--wipe-outputs", action="store_true")
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password and not args.ssh_key:
        raise SystemExit(f"Set environment variable {args.password_env} or pass --ssh-key")

    if not args.skip_upload:
        for spec in TRY_SPECS.values():
            run_local(
                [
                    sys.executable,
                    str(ROOT / "cluster" / "upload_and_submit_experiments.py"),
                    "--local-dir",
                    str(spec["local_dir"]),
                    "--upload-only",
                    "--skip-datasets",
                ]
            )

    registries = {try_name: load_registry_map(Path(spec["registry_path"])) for try_name, spec in TRY_SPECS.items()}
    waves = build_job_waves(registries)

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
        if args.cancel_all_user_jobs:
            remote_exec(client, f"scancel -u {args.user}", check=False)
            time.sleep(3.0)

        if args.wipe_outputs:
            for spec in TRY_SPECS.values():
                remote_exec(
                    client,
                    f"find {spec['remote_dir']}/outputs -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + 2>/dev/null || true; "
                    f"rm -f {spec['remote_dir']}/{spec['log_glob']}",
                    check=False,
                )

        remote_exec(client, f"squeue -u {args.user}", check=False)

        remote_exec(
            client,
            f"mkdir -p /scratch/nas/3/gmoreno/TFGpractice/Datasets/try57_shared_formula_cache && "
            f"if [ ! -f {TRY57_SHARED_CACHE} ] && [ -f {TRY55_SHARED_CACHE} ]; then cp {TRY55_SHARED_CACHE} {TRY57_SHARED_CACHE}; fi",
            check=False,
        )

        initial_spec = TRY_SPECS["try57"]
        dependency = remote_sbatch(
            client,
            str(initial_spec["remote_dir"]),
            f"sbatch --nodelist={args.node} {initial_spec['cleanup_slurm']}",
        )

        submitted: list[dict[str, str]] = []
        port_offset = 0

        for wave in waves:
            wave_job_ids: list[str] = []
            for item in wave:
                spec = TRY_SPECS[item["try_name"]]
                job_name = f"{spec['job_prefix']}-{item['topology_class'].replace('_', '-')[:32]}"
                exports = {
                    "CONFIG_PATH": item["config"],
                    "MASTER_PORT": str(int(spec["base_port"]) + port_offset),
                    "OUTPUT_SUFFIX": "",
                }
                exports.update({str(k): str(v) for k, v in dict(spec["extra_exports"]).items()})
                export_clause = ",".join(f"{key}={value}" for key, value in exports.items())
                job_id = remote_sbatch(
                    client,
                    str(spec["remote_dir"]),
                    f"sbatch --nodelist={args.node} --dependency=afterany:{dependency} -J {job_name} "
                    f"--export=ALL,{export_clause} {spec['run_slurm']}",
                )
                wave_job_ids.append(job_id)
                submitted.append(
                    {
                        "try_name": item["try_name"],
                        "topology_class": item["topology_class"],
                        "job_id": job_id,
                        "config": item["config"],
                        "depends_on": dependency,
                    }
                )
                port_offset += 1

            cleanup_spec = TRY_SPECS["try57"]
            dependency = remote_sbatch(
                client,
                str(cleanup_spec["remote_dir"]),
                f"sbatch --nodelist={args.node} --dependency=afterany:{':'.join(wave_job_ids)} {cleanup_spec['cleanup_slurm']}",
            )

        print({"initial_cleanup": submitted[0]["depends_on"] if submitted else dependency, "submitted": submitted, "final_cleanup": dependency})
        remote_exec(client, f"squeue -u {args.user}", check=False)
    finally:
        client.close()


if __name__ == "__main__":
    main()
