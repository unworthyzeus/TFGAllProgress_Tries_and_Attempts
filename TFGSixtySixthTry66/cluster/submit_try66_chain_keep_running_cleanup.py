#!/usr/bin/env python3
"""Cancel all Slurm jobs for the user except one running job, then queue training with cleanup between runs.

Typical use: one ``t66-*`` job is already **R** on ``sert-2001``; cancel pending/held duplicates and PD
chains, then schedule (after the kept job ends) cleanup → train → cleanup → train → …

Default train queue (after the kept running job + cleanup): **one** ``open_sparse_lowrise`` segment
(resume from ``best_model.pt``), then each **other** registry expert once (no resume): vertical,
mixed compact low/mid, dense mid/high. Cleanup runs after the kept job and between every train job.
Override with ``--train-spec`` (see below).
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
LOCAL_DIR = ROOT / "TFGSixtySixthTry66"
REGISTRY = LOCAL_DIR / "experiments" / "sixtysixth_try66_experts" / "try66_expert_registry.yaml"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtySixthTry66"
HOST = "sert.ac.upc.edu"
USER = "gmoreno"
TARGET_NODE = "sert-2001"
CLEANUP_SLURM = "cluster/run_sixtysixth_try66_cleanup_sert2001_1gpu.slurm"
TRAIN_SLURM = "cluster/run_sixtysixth_try66_4gpu.slurm"


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


def squeue_job_ids(client: paramiko.SSHClient, user: str) -> list[str]:
    out, _ = remote_exec(client, f"squeue -u {user} -h -o %i", check=False)
    return [line.strip() for line in out.splitlines() if line.strip().isdigit()]


def squeue_first_running_id(client: paramiko.SSHClient, user: str, node: str) -> str | None:
    out, _ = remote_exec(
        client,
        f"squeue -u {user} -w {node} -t R -h -o %i",
        check=False,
    )
    for line in out.splitlines():
        s = line.strip()
        if s.isdigit():
            return s
    return None


def default_train_specs(experts: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    """(expert_id, resume_relpath or None) — one extra lowrise, then every non-lowrise expert."""
    low = resolve_expert("open_sparse_lowrise", experts)
    ck = str(low["checkpoint"]).replace("\\", "/")
    seg: list[tuple[str, str | None]] = [("open_sparse_lowrise", ck)]
    for eid in (
        "open_sparse_vertical",
        "mixed_compact_lowrise",
        "mixed_compact_midrise",
        "dense_block_midrise",
        "dense_block_highrise",
    ):
        seg.append((eid, None))
    return seg


def parse_train_specs(specs: list[str], experts: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for token in specs:
        if ":" in token:
            eid, flag = token.split(":", 1)
            eid = eid.strip()
            flag = flag.strip().lower()
            if flag in {"resume", "r"}:
                row = resolve_expert(eid, experts)
                out.append((eid, str(row["checkpoint"]).replace("\\", "/")))
            elif flag in {"fresh", "f", "no"}:
                out.append((eid, None))
            else:
                raise SystemExit(f"Bad train-spec suffix in {token!r}; use :resume or :fresh")
        else:
            out.append((token.strip(), None))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cancel all jobs except one R job; chain train + cleanup.")
    parser.add_argument(
        "--keep-job-id",
        default="",
        help="Slurm job ID to never scancel. If empty, auto-pick first R job on --node for this user.",
    )
    parser.add_argument(
        "--no-keep",
        action="store_true",
        help="Do not keep any job: scancel everything, then start chain with no leading dependency.",
    )
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--base-master-port", type=int, default=30366)
    parser.add_argument(
        "--no-cleanup-between",
        action="store_true",
        help="Omit cleanup Slurm jobs between training jobs.",
    )
    parser.add_argument(
        "--train-spec",
        nargs="*",
        default=[],
        help=(
            "Ordered segments: expert_id or expert_id:resume or expert_id:fresh. "
            "If omitted, uses default 4× lowrise (resume) + vertical + mixed_compact_lowrise + midrise."
        ),
    )
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password and not args.ssh_key:
        raise SystemExit(f"Set environment variable {args.password_env} or pass --ssh-key")

    experts = load_registry()
    if args.train_spec:
        train_specs = parse_train_specs(list(args.train_spec), experts)
    else:
        train_specs = default_train_specs(experts)

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
    connect_kwargs: dict = {
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
        keep: str | None = None
        if not args.no_keep:
            keep = str(args.keep_job_id).strip() or None
            if not keep:
                keep = squeue_first_running_id(client, args.user, args.node)
            if keep:
                print(f"\n=== Keeping running job id {keep} (will not scancel) ===")
            else:
                print("\n=== No running job on node to keep; will scancel all queued jobs ===")

        all_ids = squeue_job_ids(client, args.user)
        for jid in all_ids:
            if keep and jid == keep:
                continue
            remote_exec(client, f"scancel {jid}", check=False)

        remote_exec(client, f"squeue -u {args.user}", check=False)

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        dependency = ""
        if keep:
            dependency = f"--dependency=afterany:{keep} "

        cleanup_between = not args.no_cleanup_between
        submitted: list[tuple[str, str]] = []
        train_idx = 0

        if keep and cleanup_between:
            cjid = remote_sbatch(
                client,
                f"{sbatch_prefix} {dependency}-J t66-cleanup-after-keep "
                f"--export=ALL {CLEANUP_SLURM}",
            )
            submitted.append(("cleanup-after-keep", cjid))
            dependency = f"--dependency=afterany:{cjid} "

        for seg_i, (expert_id, resume_path) in enumerate(train_specs):
            row = resolve_expert(expert_id, experts)
            config = str(row["config"])
            job_name = f"t66-{expert_id.replace('_', '-')[:24]}-{train_idx}"
            exports = [
                f"CONFIG_PATH={config}",
                "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
                f"MASTER_PORT={args.base_master_port + train_idx}",
            ]
            if resume_path:
                exports.append(f"RESUME_CHECKPOINT={resume_path}")
            tjid = remote_sbatch(
                client,
                f"{sbatch_prefix} {dependency}-J {job_name} "
                f"--export=ALL,{','.join(exports)} {TRAIN_SLURM}",
            )
            submitted.append((job_name, tjid))
            train_idx += 1
            dependency = f"--dependency=afterany:{tjid} "

            if cleanup_between and seg_i < len(train_specs) - 1:
                cjid = remote_sbatch(
                    client,
                    f"{sbatch_prefix} {dependency}-J t66-cleanup-between "
                    f"--export=ALL {CLEANUP_SLURM}",
                )
                submitted.append(("cleanup-between", cjid))
                dependency = f"--dependency=afterany:{cjid} "

        print("\n=== Submitted (train + cleanups) ===")
        for name, jid in submitted:
            print(f"  {name}: {jid}")
        print(f"\nTotal submitted: {len(submitted)} Slurm jobs.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
