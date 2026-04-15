#!/usr/bin/env python3
"""Cancel (optional), upload Try 68 + Try 69, submit two independent chains: **68 @ 2 GPU**, **69 @ 1 GPU**.

No Slurm dependency between Try 68 and Try 69: both first train jobs can run together if the
node has ≥3 free GPUs (e.g. 2+1 on ``sert-2001``).

From ``TFGpractice``::

  set SSH_PASSWORD=...
  python cluster/relaunch_try68_2gpu_try69_1gpu_parallel_upload.py --cancel-all-user-jobs

Apply ``experiments/dataloader_batch_presets/presets/2gpu.yml`` to Try 68 expert YAMLs and
``1gpu.yml`` (or your usual preset) to Try 69 before expecting stable DataLoader memory.
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
R69 = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyNinthTry69"

REG68 = ROOT / "TFGSixtyEighthTry68" / "experiments" / "sixtyeighth_try68_experts" / "try68_expert_registry.yaml"
REG69 = ROOT / "TFGSixtyNinthTry69" / "experiments" / "sixtyninth_try69_experts" / "try69_expert_registry.yaml"

T68_TRAIN_2GPU = "cluster/run_sixtyeighth_try68_2gpu.slurm"
T68_CLEAN = "cluster/run_sixtyeighth_try68_cleanup_sert2001_1gpu.slurm"
T69_TRAIN_1GPU = "cluster/run_sixtyninth_try69_1gpu.slurm"
T69_CLEAN = "cluster/run_sixtyninth_try69_cleanup_sert2001_1gpu.slurm"


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


def chain_try68_2gpu(
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
            "NPROC_PER_NODE=2",
        ]
        if resume:
            ck = str(row.get("checkpoint", "")).strip()
            if ck:
                exports.append(f"RESUME_CHECKPOINT={ck}")
        name = f"t68-2gpu-{eid.replace('_', '-')[:22]}"
        tid = remote_sbatch(
            c,
            R68,
            f"{sb} {dep}-J {name} --export=ALL,{','.join(exports)} {T68_TRAIN_2GPU}",
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


def chain_try69_1gpu(
    c: paramiko.SSHClient,
    *,
    nodelist: str,
    base_port: int,
    resume: bool,
) -> list[tuple[str, str]]:
    rows = load_registry(REG69)
    sb = f"sbatch --nodelist={nodelist}"
    dep = ""
    out: list[tuple[str, str]] = []
    for i, row in enumerate(rows):
        eid = str(row["expert_id"])
        exports = [
            f"CONFIG_PATH={row['config']}",
            "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
            f"MASTER_PORT={base_port + i * 2}",
        ]
        if resume:
            ck = str(row.get("checkpoint", "")).strip()
            if ck:
                exports.append(f"RESUME_CHECKPOINT={ck}")
        name = f"t69-1gpu-{eid.replace('_', '-')[:24]}"
        tid = remote_sbatch(
            c,
            R69,
            f"{sb} {dep}-J {name} --export=ALL,{','.join(exports)} {T69_TRAIN_1GPU}",
        )
        out.append((name, tid))
        cname = f"t69-cleanup-{eid.replace('_', '-')[:18]}"
        cid = remote_sbatch(
            c,
            R69,
            f"{sb} --dependency=afterany:{tid} -J {cname} {T69_CLEAN}",
        )
        out.append((cname, cid))
        dep = f"--dependency=afterany:{cid} "
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Upload Try 68+69; optional scancel; submit Try 68 2-GPU chain + Try 69 1-GPU chain in parallel."
    )
    ap.add_argument("--password-env", default="SSH_PASSWORD")
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--user", default=USER)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--cancel-all-user-jobs",
        action="store_true",
        help="Run scancel -u USER before submitting.",
    )
    ap.add_argument("--nodelist-68", default="", metavar="NODE", help="Default: same as --nodelist.")
    ap.add_argument("--nodelist-69", default="", metavar="NODE", help="Default: same as --nodelist.")
    ap.add_argument("--nodelist", default="sert-2001", help="Default Slurm nodelist for both tries.")
    ap.add_argument("--no-resume", action="store_true", help="Disable RESUME_CHECKPOINT from registries.")
    ap.add_argument("--port68", type=int, default=30310)
    ap.add_argument("--port69", type=int, default=30410)
    args = ap.parse_args()

    pw = os.environ.get(args.password_env, "")
    if not pw and not args.ssh_key:
        raise SystemExit(f"Set {args.password_env} or pass --ssh-key")

    n68 = args.nodelist_68.strip() or args.nodelist
    n69 = args.nodelist_69.strip() or args.nodelist
    resume = not args.no_resume

    if not args.skip_upload:
        for local in (ROOT / "TFGSixtyEighthTry68", ROOT / "TFGSixtyNinthTry69"):
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

        print("\n=== Try 68 — 2-GPU chain (registry) ===")
        for name, jid in chain_try68_2gpu(c, nodelist=n68, base_port=args.port68, resume=resume):
            tag = "cl" if "cleanup" in name else "tr"
            print(f"  [{tag}] {name}: {jid}")

        print("\n=== Try 69 — 1-GPU chain (registry) ===")
        for name, jid in chain_try69_1gpu(c, nodelist=n69, base_port=args.port69, resume=resume):
            tag = "cl" if "cleanup" in name else "tr"
            print(f"  [{tag}] {name}: {jid}")

        print(
            "\nFirst train job of Try 68 (2 GPU) and Try 69 (1 GPU) have no cross-dependency; "
            "Slurm may run them together if ≥3 GPUs are free on the node(s)."
        )
    finally:
        c.close()


if __name__ == "__main__":
    main()
