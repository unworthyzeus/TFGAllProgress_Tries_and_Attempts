#!/usr/bin/env python3
"""One-shot: scancel user jobs, optional rm Try66/67 on cluster, upload Try70+68+69, submit chained Slurm:

  Try70 (N× train 1gpu → cleanup 1gpu) → Try68 (6 experts, 1gpu+cleanup) → Try69 (6 experts, 1gpu+cleanup).

Run from ``TFGpractice``::

  export SSH_PASSWORD=...
  python cluster/submit_chain_try70_try68_try69_1gpu_cleanup.py

Uses the same remote layout as the per-try submitters.
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

ROOT = Path(__file__).resolve().parent.parent
HOST = "sert.ac.upc.edu"
USER = "gmoreno"
NODE = "sert-2001"
SCRATCH = "/scratch/nas/3/gmoreno/TFGpractice"
R70 = f"{SCRATCH}/TFGSeventiethTry70"
R68 = f"{SCRATCH}/TFGSixtyEighthTry68"
R69 = f"{SCRATCH}/TFGSixtyNinthTry69"
L70 = ROOT / "TFGSeventiethTry70"
L68 = ROOT / "TFGSixtyEighthTry68"
L69 = ROOT / "TFGSixtyNinthTry69"
REG68 = L68 / "experiments" / "sixtyeighth_try68_experts" / "try68_expert_registry.yaml"
REG69 = L69 / "experiments" / "sixtyninth_try69_experts" / "try69_expert_registry.yaml"

DEFAULT_TRY70_INIT = f"{R68}/outputs/try68_expert_open_sparse_lowrise/best_model.pt"
TRY70_CONFIG = "experiments/seventieth_try70_experts/try70_expert_open_sparse_lowrise.yaml"
TRY70_RESUME = "outputs/try70_expert_open_sparse_lowrise/best_model.pt"


def run_local(cmd: list[str]) -> None:
    print("LOCAL>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def parse_job_id(text: str) -> str:
    m = re.search(r"Submitted batch job (\d+)", text)
    if not m:
        raise RuntimeError(f"Could not parse job id from: {text!r}")
    return m.group(1)


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


def sbatch(client: paramiko.SSHClient, remote_dir: str, cmd: str) -> str:
    out, err = remote_exec(client, f"cd {remote_dir} && {cmd}")
    if err:
        raise RuntimeError(err)
    return parse_job_id(out)


def load_yaml_reg(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rows = list(data.get("experts", [])) if isinstance(data, dict) else []
    if not rows:
        raise RuntimeError(f"No experts in {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Chain Try70 → Try68 → Try69 (all 1-GPU train + cleanups).")
    ap.add_argument("--password-env", default="SSH_PASSWORD")
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--try70-segments", type=int, default=4)
    ap.add_argument("--try70-base-port", type=int, default=30290)
    ap.add_argument("--try68-base-port", type=int, default=30310)
    ap.add_argument("--try69-base-port", type=int, default=30340)
    ap.add_argument("--no-scancel", action="store_true")
    ap.add_argument(
        "--rm-try66-try67",
        action="store_true",
        help=f"rm -rf {SCRATCH}/TFGSixtySixthTry66 and TFGSixtySeventhTry67 on the cluster.",
    )
    ap.add_argument("--no-try70-init", action="store_true")
    ap.add_argument("--no-resume-try70", action="store_true")
    ap.add_argument("--no-resume-try68", action="store_true")
    ap.add_argument("--no-resume-try69", action="store_true")
    args = ap.parse_args()

    pw = os.environ.get(args.password_env, "")
    if not pw and not args.ssh_key:
        raise SystemExit(f"Set {args.password_env} or pass --ssh-key")

    if not args.skip_upload:
        for local in (L70, L68, L69):
            run_local(
                [
                    sys.executable,
                    str(ROOT / "cluster" / "upload_and_submit_experiments.py"),
                    "--local-dir",
                    str(local),
                    "--upload-only",
                    "--skip-datasets",
                ]
            )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw: dict = {"hostname": HOST, "username": USER, "timeout": 30, "allow_agent": True, "look_for_keys": True}
    if pw:
        kw["password"] = pw
    if args.ssh_key:
        kw["key_filename"] = args.ssh_key
    client.connect(**kw)

    submitted: list[tuple[str, str]] = []
    try:
        if not args.no_scancel:
            remote_exec(client, f"scancel -u {USER}", check=False)
        if args.rm_try66_try67:
            remote_exec(client, f"rm -rf {SCRATCH}/TFGSixtySixthTry66 {SCRATCH}/TFGSixtySeventhTry67", check=False)
        remote_exec(client, f"squeue -u {USER}", check=False)

        sb = f"sbatch --nodelist={NODE}"
        dep = ""

        # --- Try 70 ---
        for i in range(args.try70_segments):
            name = f"t70-osl-1gpu-{i + 1}"
            ex = [
                f"CONFIG_PATH={TRY70_CONFIG}",
                "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
                f"MASTER_PORT={args.try70_base_port + i}",
            ]
            if i == 0 and not args.no_try70_init:
                ex.append(f"TRY70_INIT_CHECKPOINT={DEFAULT_TRY70_INIT}")
            if not args.no_resume_try70:
                ex.append(f"RESUME_CHECKPOINT={TRY70_RESUME}")
            tid = sbatch(
                client,
                R70,
                f"{sb} {dep}-J {name} --export=ALL,{','.join(ex)} cluster/run_seventieth_try70_1gpu.slurm",
            )
            submitted.append((name, tid))
            cname = f"t70-cleanup-{i + 1}"
            cid = sbatch(
                client,
                R70,
                f"{sb} --dependency=afterany:{tid} -J {cname} cluster/run_seventieth_try70_cleanup_sert2001_1gpu.slurm",
            )
            submitted.append((cname, cid))
            dep = f"--dependency=afterany:{cid} "

        # --- Try 68 ---
        rows68 = load_yaml_reg(REG68)
        for i, row in enumerate(rows68):
            eid = str(row["expert_id"])
            name = f"t68-1gpu-{eid.replace('_', '-')[:24]}"
            ex = [
                f"CONFIG_PATH={row['config']}",
                "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
                f"MASTER_PORT={args.try68_base_port + i * 2}",
            ]
            if not args.no_resume_try68:
                ck = str(row.get("checkpoint", "")).strip()
                if ck:
                    ex.append(f"RESUME_CHECKPOINT={ck}")
            tid = sbatch(
                client,
                R68,
                f"{sb} {dep}-J {name} --export=ALL,{','.join(ex)} cluster/run_sixtyeighth_try68_1gpu.slurm",
            )
            submitted.append((name, tid))
            cname = f"t68-cleanup-{eid.replace('_', '-')[:18]}"
            cid = sbatch(
                client,
                R68,
                f"{sb} --dependency=afterany:{tid} -J {cname} cluster/run_sixtyeighth_try68_cleanup_sert2001_1gpu.slurm",
            )
            submitted.append((cname, cid))
            dep = f"--dependency=afterany:{cid} "

        # --- Try 69 ---
        rows69 = load_yaml_reg(REG69)
        for i, row in enumerate(rows69):
            eid = str(row["expert_id"])
            name = f"t69-1gpu-{eid.replace('_', '-')[:24]}"
            ex = [
                f"CONFIG_PATH={row['config']}",
                "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
                f"MASTER_PORT={args.try69_base_port + i * 2}",
            ]
            if not args.no_resume_try69:
                ck = str(row.get("checkpoint", "")).strip()
                if ck:
                    ex.append(f"RESUME_CHECKPOINT={ck}")
            tid = sbatch(
                client,
                R69,
                f"{sb} {dep}-J {name} --export=ALL,{','.join(ex)} cluster/run_sixtyninth_try69_1gpu.slurm",
            )
            submitted.append((name, tid))
            cname = f"t69-cleanup-{eid.replace('_', '-')[:18]}"
            cid = sbatch(
                client,
                R69,
                f"{sb} --dependency=afterany:{tid} -J {cname} cluster/run_sixtyninth_try69_cleanup_sert2001_1gpu.slurm",
            )
            submitted.append((cname, cid))
            dep = f"--dependency=afterany:{cid} "

        print("\n=== Full chain submitted ===")
        for n, j in submitted:
            tag = "[cl]" if "cleanup" in n else "[tr]"
            print(f"  {tag} {n}: {j}")
        print(f"\nTotal jobs: {len(submitted)}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
