#!/usr/bin/env python3
"""Upload Try69 + Try70, optionally scancel user jobs, submit Try70 1gpu+cleanup chain then Try69 1gpu+cleanup.

Try69 jobs follow ``try69_expert_registry.yaml`` (six topology experts by default).
Use after fixing OOM-related YAML/Slurm (Try69 batch 1 + checkpointing; Try70 lighter aux).

From ``TFGpractice``::

  export SSH_PASSWORD=...
  python cluster/relaunch_try69_try70_1gpu_cleanup.py --cancel-all-user-jobs
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

try:
    import paramiko
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko

ROOT = Path(__file__).resolve().parent.parent
R70 = "/scratch/nas/3/gmoreno/TFGpractice/TFGSeventiethTry70"
R69 = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyNinthTry69"
HOST = "sert.ac.upc.edu"
USER = "gmoreno"
NODE = "sert-2001"

TRY70_SLURM = "cluster/run_seventieth_try70_1gpu.slurm"
TRY70_CLEAN = "cluster/run_seventieth_try70_cleanup_sert2001_1gpu.slurm"
TRY69_SLURM = "cluster/run_sixtyninth_try69_1gpu.slurm"
TRY69_CLEAN = "cluster/run_sixtyninth_try69_cleanup_sert2001_1gpu.slurm"

DEFAULT_TRY70_INIT = f"/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68/outputs/try68_expert_open_sparse_lowrise/best_model.pt"
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


def remote_exec(c: paramiko.SSHClient, cmd: str, *, check: bool = True) -> tuple[str, str]:
    print("REMOTE>", cmd)
    _, o, e = c.exec_command(cmd)
    out = o.read().decode().strip()
    err = e.read().decode().strip()
    if out:
        print(out)
    if err:
        print(err)
    if check and err:
        raise RuntimeError(err)
    return out, err


def sbatch(c: paramiko.SSHClient, rdir: str, cmd: str) -> str:
    out, err = remote_exec(c, f"cd {rdir} && {cmd}")
    if err:
        raise RuntimeError(err)
    return parse_job_id(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Relaunch Try70 then Try69 (1-GPU train + cleanups).")
    ap.add_argument("--password-env", default="SSH_PASSWORD")
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--cancel-all-user-jobs", action="store_true")
    ap.add_argument("--try70-segments", type=int, default=4)
    ap.add_argument("--try70-base-port", type=int, default=30290)
    ap.add_argument("--try69-base-port", type=int, default=30340)
    ap.add_argument("--no-try70-init", action="store_true")
    ap.add_argument("--no-resume-try70", action="store_true")
    ap.add_argument("--no-resume-try69", action="store_true")
    args = ap.parse_args()

    pw = os.environ.get(args.password_env, "")
    if not pw and not args.ssh_key:
        raise SystemExit(f"Set {args.password_env} or pass --ssh-key")

    if not args.skip_upload:
        for local in (ROOT / "TFGSeventiethTry70", ROOT / "TFGSixtyNinthTry69"):
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

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw: dict = {"hostname": HOST, "username": USER, "timeout": 30, "allow_agent": True, "look_for_keys": True}
    if pw:
        kw["password"] = pw
    if args.ssh_key:
        kw["key_filename"] = args.ssh_key
    c.connect(**kw)

    sb = f"sbatch --nodelist={NODE}"
    submitted: list[tuple[str, str]] = []
    try:
        if args.cancel_all_user_jobs:
            remote_exec(c, f"scancel -u {USER}", check=False)
        remote_exec(c, f"squeue -u {USER}", check=False)

        dep = ""
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
            jid = sbatch(
                c,
                R70,
                f"{sb} {dep}-J {name} --export=ALL,{','.join(ex)} {TRY70_SLURM}",
            )
            submitted.append((name, jid))
            cname = f"t70-cleanup-{i + 1}"
            cid = sbatch(
                c,
                R70,
                f"{sb} --dependency=afterany:{jid} -J {cname} {TRY70_CLEAN}",
            )
            submitted.append((cname, cid))
            dep = f"--dependency=afterany:{cid} "

        reg = ROOT / "TFGSixtyNinthTry69" / "experiments" / "sixtyninth_try69_experts" / "try69_expert_registry.yaml"
        with reg.open("r", encoding="utf-8") as f:
            rows = list(yaml.safe_load(f).get("experts", []))

        for i, row in enumerate(rows):
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
            jid = sbatch(
                c,
                R69,
                f"{sb} {dep}-J {name} --export=ALL,{','.join(ex)} {TRY69_SLURM}",
            )
            submitted.append((name, jid))
            cname = f"t69-cleanup-{eid.replace('_', '-')[:18]}"
            cid = sbatch(
                c,
                R69,
                f"{sb} --dependency=afterany:{jid} -J {cname} {TRY69_CLEAN}",
            )
            submitted.append((cname, cid))
            dep = f"--dependency=afterany:{cid} "

        print("\n=== Relaunch submitted ===")
        for n, j in submitted:
            tag = "[cl]" if "cleanup" in n else "[tr]"
            print(f"  {tag} {n}: {j}")
        last = submitted[-1]
        if "cleanup" in last[0]:
            print(f"\nLAST_JOB_ID={last[1]} (final cleanup)")
    finally:
        c.close()


if __name__ == "__main__":
    main()
