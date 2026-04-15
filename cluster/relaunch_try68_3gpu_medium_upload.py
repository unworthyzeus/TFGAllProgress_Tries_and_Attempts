#!/usr/bin/env python3
"""Upload Try 68 only, optionally cancel all your Slurm jobs, submit the 3-GPU chain.

Uses ``cluster/run_sixtyeighth_try68_3gpu_medium_gpu.slurm`` (``medium_gpu``: SERT ``big_gpu``
often has MinGRES=4, so 3-GPU jobs get ``QOSMinGRES``). Same train→cleanup→… pattern as
the 1-GPU relauncher, but **only Try 68** (registry experts).

From ``TFGpractice``::

  set SSH_PASSWORD=...
  python cluster/relaunch_try68_3gpu_medium_upload.py --cancel-all-user-jobs

Copy ``experiments/dataloader_batch_presets/presets/3gpu_medium.yml`` into each expert YAML
before training (same recipe as 4-GPU).
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
R68 = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68"

REG68 = ROOT / "TFGSixtyEighthTry68" / "experiments" / "sixtyeighth_try68_experts" / "try68_expert_registry.yaml"

T68_TRAIN_3GPU = "cluster/run_sixtyeighth_try68_3gpu_medium_gpu.slurm"
T68_CLEAN = "cluster/run_sixtyeighth_try68_cleanup_sert2001_1gpu.slurm"


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


def remote_sbatch(c: paramiko.SSHClient, rdir: str, cmd: str) -> str:
    out, err = remote_exec(c, f"cd {rdir} && {cmd}")
    if err:
        raise RuntimeError(err)
    return parse_job_id(out)


def load_registry(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    experts = list(data.get("experts", [])) if isinstance(data, dict) else []
    if not experts:
        raise RuntimeError(f"No experts in {path}")
    return experts


def chain_try68_3gpu(
    c: paramiko.SSHClient,
    *,
    nodelist: str,
    base_port: int,
    resume: bool,
) -> list[tuple[str, str]]:
    rows = load_registry(REG68)
    sb = f"sbatch --nodelist={nodelist}"
    dep = ""
    out: list[tuple[str, str]] = []
    for i, row in enumerate(rows):
        eid = str(row["expert_id"])
        exports = [
            f"CONFIG_PATH={row['config']}",
            "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
            f"MASTER_PORT={base_port + i * 2}",
            f"NPROC_PER_NODE=3",
        ]
        if resume:
            ck = str(row.get("checkpoint", "")).strip()
            if ck:
                exports.append(f"RESUME_CHECKPOINT={ck}")
        name = f"t68-3gpu-{eid.replace('_', '-')[:22]}"
        tid = remote_sbatch(
            c,
            R68,
            f"{sb} {dep}-J {name} --export=ALL,{','.join(exports)} {T68_TRAIN_3GPU}",
        )
        out.append((name, tid))
        cname = f"t68-cleanup-{eid.replace('_', '-')[:18]}"
        cid = remote_sbatch(
            c,
            R68,
            f"{sb} --dependency=afterany:{tid} -J {cname} {T68_CLEAN}",
        )
        out.append((cname, cid))
        dep = f"--dependency=afterany:{cid} "
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Upload Try 68, optionally scancel all user jobs, submit 3-GPU medium_gpu expert chain."
    )
    ap.add_argument("--password-env", default="SSH_PASSWORD")
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--user", default=USER)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--cancel-all-user-jobs",
        action="store_true",
        help="Run scancel -u USER before submitting (stops all your Slurm jobs).",
    )
    ap.add_argument("--nodelist", default="sert-2001", help="sbatch --nodelist= (Slurm script also pins -w).")
    ap.add_argument("--no-resume", action="store_true", help="Do not pass RESUME_CHECKPOINT from registry.")
    ap.add_argument("--port68", type=int, default=31310, help="Base MASTER_PORT for first expert.")
    args = ap.parse_args()

    pw = os.environ.get(args.password_env, "")
    if not pw and not args.ssh_key:
        raise SystemExit(f"Set {args.password_env} or pass --ssh-key")
    resume = not args.no_resume

    if not args.skip_upload:
        run_local(
            [
                sys.executable,
                str(ROOT / "cluster" / "upload_and_submit_experiments.py"),
                "--local-dir",
                str(ROOT / "TFGSixtyEighthTry68"),
                "--upload-only",
                "--skip-datasets",
            ]
        )

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw: dict = {"hostname": args.host, "username": args.user, "timeout": 30, "allow_agent": True, "look_for_keys": True}
    if pw:
        kw["password"] = pw
    if args.ssh_key:
        kw["key_filename"] = args.ssh_key
    c.connect(**kw)

    try:
        if args.cancel_all_user_jobs:
            remote_exec(c, f"scancel -u {args.user}", check=False)
        remote_exec(c, f"squeue -u {args.user}", check=False)

        print("\n=== Try 68 — 3-GPU medium_gpu chain (registry) ===")
        submitted: list[tuple[str, str]] = []
        for name, jid in chain_try68_3gpu(
            c, nodelist=args.nodelist.strip(), base_port=args.port68, resume=resume
        ):
            submitted.append((name, jid))
            tag = "cl" if "cleanup" in name else "tr"
            print(f"  [{tag}] {name}: {jid}")
        print("\nSubmitted", len(submitted), "jobs (train+cleanup per expert).")
    finally:
        c.close()


if __name__ == "__main__":
    main()
