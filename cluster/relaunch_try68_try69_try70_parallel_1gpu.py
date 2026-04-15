#!/usr/bin/env python3
"""Cancel (optional), upload Try 68/69/70, submit three **independent** 1-GPU chains in parallel.

Each try only chains **train → cleanup → …** inside itself. There is **no** Slurm dependency
between Try 68, Try 69, and Try 70, so the first train job of each try can run at the same
time if the cluster has enough GPUs on the requested node(s).

From ``TFGpractice``::

  set SSH_PASSWORD=...
  python cluster/relaunch_try68_try69_try70_parallel_1gpu.py --cancel-all-user-jobs

Use ``--nodelist-68``, ``--nodelist-69``, ``--nodelist-70`` to pin each try to a different
host if a single node has only one GPU (default: all use ``sert-2001``; Slurm will queue
what does not fit).
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
R70 = "/scratch/nas/3/gmoreno/TFGpractice/TFGSeventiethTry70"

REG68 = ROOT / "TFGSixtyEighthTry68" / "experiments" / "sixtyeighth_try68_experts" / "try68_expert_registry.yaml"
REG69 = ROOT / "TFGSixtyNinthTry69" / "experiments" / "sixtyninth_try69_experts" / "try69_expert_registry.yaml"

T68_TRAIN = "cluster/run_sixtyeighth_try68_1gpu.slurm"
T68_CLEAN = "cluster/run_sixtyeighth_try68_cleanup_sert2001_1gpu.slurm"
T69_TRAIN = "cluster/run_sixtyninth_try69_1gpu.slurm"
T69_CLEAN = "cluster/run_sixtyninth_try69_cleanup_sert2001_1gpu.slurm"
T70_TRAIN = "cluster/run_seventieth_try70_1gpu.slurm"
T70_CLEAN = "cluster/run_seventieth_try70_cleanup_sert2001_1gpu.slurm"

TRY70_CONFIG = "experiments/seventieth_try70_experts/try70_expert_open_sparse_lowrise.yaml"
TRY70_INIT = (
    "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68/outputs/"
    "try68_expert_open_sparse_lowrise/best_model.pt"
)
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


def chain_try68(
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
        ]
        if resume:
            ck = str(row.get("checkpoint", "")).strip()
            if ck:
                exports.append(f"RESUME_CHECKPOINT={ck}")
        name = f"t68-1gpu-{eid.replace('_', '-')[:24]}"
        tid = remote_sbatch(
            c,
            R68,
            f"{sb} {dep}-J {name} --export=ALL,{','.join(exports)} {T68_TRAIN}",
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


def chain_try69(
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
            f"{sb} {dep}-J {name} --export=ALL,{','.join(exports)} {T69_TRAIN}",
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


def chain_try70(
    c: paramiko.SSHClient,
    *,
    nodelist: str,
    segments: int,
    base_port: int,
    resume: bool,
    try70_init: str,
    no_try70_init: bool,
) -> list[tuple[str, str]]:
    sb = f"sbatch --nodelist={nodelist}"
    dep = ""
    out: list[tuple[str, str]] = []
    for i in range(segments):
        exports = [
            f"CONFIG_PATH={TRY70_CONFIG}",
            "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
            f"MASTER_PORT={base_port + i}",
        ]
        if i == 0 and not no_try70_init and str(try70_init).strip():
            exports.append(f"TRY70_INIT_CHECKPOINT={try70_init}")
        if resume and str(TRY70_RESUME).strip():
            exports.append(f"RESUME_CHECKPOINT={TRY70_RESUME}")
        name = f"t70-osl-1gpu-{i + 1}"
        tid = remote_sbatch(
            c,
            R70,
            f"{sb} {dep}-J {name} --export=ALL,{','.join(exports)} {T70_TRAIN}",
        )
        out.append((name, tid))
        cname = f"t70-cleanup-{i + 1}"
        cid = remote_sbatch(
            c,
            R70,
            f"{sb} --dependency=afterany:{tid} -J {cname} {T70_CLEAN}",
        )
        out.append((cname, cid))
        dep = f"--dependency=afterany:{cid} "
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Upload 68/69/70 and submit three parallel 1-GPU Slurm chains (no cross-try deps)."
    )
    ap.add_argument("--password-env", default="SSH_PASSWORD")
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--user", default=USER)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--cancel-all-user-jobs", action="store_true")
    ap.add_argument(
        "--nodelist",
        default="sert-2001",
        help="Default Slurm nodelist for all three tries (override per try below).",
    )
    ap.add_argument("--nodelist-68", default="", metavar="NODE")
    ap.add_argument("--nodelist-69", default="", metavar="NODE")
    ap.add_argument("--nodelist-70", default="", metavar="NODE")
    ap.add_argument("--try70-segments", type=int, default=4)
    ap.add_argument("--try70-init-checkpoint", default=TRY70_INIT)
    ap.add_argument("--no-try70-init", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="Disable RESUME_CHECKPOINT / TRY70 init for all.")
    ap.add_argument("--port68", type=int, default=30310)
    ap.add_argument("--port69", type=int, default=30410)
    ap.add_argument("--port70", type=int, default=30510)
    args = ap.parse_args()

    pw = os.environ.get(args.password_env, "")
    if not pw and not args.ssh_key:
        raise SystemExit(f"Set {args.password_env} or pass --ssh-key")

    n68 = args.nodelist_68.strip() or args.nodelist
    n69 = args.nodelist_69.strip() or args.nodelist
    n70 = args.nodelist_70.strip() or args.nodelist
    resume = not args.no_resume

    if not args.skip_upload:
        for local in (
            ROOT / "TFGSixtyEighthTry68",
            ROOT / "TFGSixtyNinthTry69",
            ROOT / "TFGSeventiethTry70",
        ):
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

        all_submitted: list[tuple[str, str, str]] = []

        print("\n=== Try 68 chain (isolated) ===")
        for name, jid in chain_try68(c, nodelist=n68, base_port=args.port68, resume=resume):
            all_submitted.append(("68", name, jid))

        print("\n=== Try 69 chain (isolated) ===")
        for name, jid in chain_try69(c, nodelist=n69, base_port=args.port69, resume=resume):
            all_submitted.append(("69", name, jid))

        print("\n=== Try 70 chain (isolated) ===")
        for name, jid in chain_try70(
            c,
            nodelist=n70,
            segments=args.try70_segments,
            base_port=args.port70,
            resume=resume,
            try70_init=args.try70_init_checkpoint,
            no_try70_init=args.no_try70_init,
        ):
            all_submitted.append(("70", name, jid))

        print("\n=== All submitted (no cross-try dependencies) ===")
        for t, n, j in all_submitted:
            tag = "cl" if "cleanup" in n else "tr"
            print(f"  [{t}] [{tag}] {n}: {j}")
        print(
            "\nFirst train job of each try has no dependency on the other tries; "
            "Slurm may run them together if GPUs are free on the chosen node(s)."
        )
    finally:
        c.close()


if __name__ == "__main__":
    main()
