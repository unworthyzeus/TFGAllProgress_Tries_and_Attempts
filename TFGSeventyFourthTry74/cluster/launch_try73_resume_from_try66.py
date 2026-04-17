#!/usr/bin/env python3
"""Copy cluster training outputs from Try 66 into Try 73, then optionally submit Try 73 jobs.

Try 73 is the same recipe as Try 66 with the PMHHNet HF-at-stem fix (see VERSIONS.md). Checkpoints
are layout-compatible (same state dict keys); training resumes from ``best_model.pt`` paths in
``try73_expert_registry.yaml`` after this copy.

Prerequisites:
  * SSH access to the cluster (``SSH_PASSWORD`` or key).
  * Try 66 outputs already exist under ``.../TFGSixtySixthTry66/outputs/try66_expert_*``.

Examples:
  # Dry-run: print the remote shell script only
  python cluster/launch_try73_resume_from_try66.py --dry-run

  # Upload Try73 code (does not wipe remote outputs), copy Try66 checkpoints into Try73 outputs, then sbatch
  python cluster/launch_try73_resume_from_try66.py --submit

  # Copy only on the cluster (no code upload, no Slurm)
  python cluster/launch_try73_resume_from_try66.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import paramiko
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko

ROOT = Path(__file__).resolve().parents[2]
TRY68_DIR = ROOT / "TFGSeventyThirdTry73"

REMOTE_BASE = "/scratch/nas/3/gmoreno/TFGpractice"
REMOTE_TRY66 = f"{REMOTE_BASE}/TFGSixtySixthTry66/outputs"
REMOTE_TRY68 = f"{REMOTE_BASE}/TFGSeventyThirdTry73/outputs"

HOST = "sert.ac.upc.edu"
USER = "gmoreno"


def _remote_copy_script() -> str:
    # Same filesystem: cp -a preserves checkpoints, JSON logs, and tensorboard if present.
    return f"""set -euo pipefail
mkdir -p "{REMOTE_TRY68}"
shopt -s nullglob
for src in "{REMOTE_TRY66}"/try66_expert_*; do
  [ -d "$src" ] || continue
  base=$(basename "$src")
  suf=${{base#try66_expert_}}
  dst="{REMOTE_TRY68}/try73_expert_${{suf}}"
  echo "[sync] $src -> $dst"
  rm -rf "$dst"
  cp -a "$src" "$dst"
done
echo "[sync] done. Listing Try73 outputs:"
ls -la "{REMOTE_TRY68}" || true
"""


def _connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get(args.password_env, "")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict = {
        "hostname": args.host,
        "username": args.user,
        "timeout": 45,
        "allow_agent": True,
        "look_for_keys": True,
    }
    if password:
        connect_kwargs["password"] = password
    if args.ssh_key:
        connect_kwargs["key_filename"] = args.ssh_key
    client.connect(**connect_kwargs)
    return client


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the remote bash script and exit (no SSH).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Upload Try73 with --no-clean-outputs, sync checkpoints from Try66, then run "
            "submit_try73_experts_4gpu_sequential.py --skip-upload."
        ),
    )
    args = parser.parse_args()

    script = _remote_copy_script()
    if args.dry_run:
        print(script)
        return 0

    if not os.environ.get(args.password_env, "") and not args.ssh_key:
        print(f"Set {args.password_env} or pass --ssh-key", file=sys.stderr)
        return 2

    if args.submit:
        upload_cmd = [
            sys.executable,
            str(ROOT / "cluster" / "upload_and_submit_experiments.py"),
            "--local-dir",
            "TFGSeventyThirdTry73",
            "--upload-only",
            "--skip-datasets",
            "--no-clean-outputs",
        ]
        print("LOCAL>", " ".join(upload_cmd))
        subprocess.run(upload_cmd, check=True, cwd=str(ROOT))

    client = _connect(args)
    try:
        print("REMOTE> bash -s  (sync Try66 outputs -> Try73)")
        stdin, stdout, stderr = client.exec_command("bash -s", timeout=3600)
        stdin.write(script)
        stdin.channel.shutdown_write()
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        exit_status = stdout.channel.recv_exit_status()
        if out:
            print(out)
        if err:
            print(err, file=sys.stderr)
        if exit_status != 0:
            return exit_status
    finally:
        client.close()

    if args.submit:
        submit = TRY68_DIR / "cluster" / "submit_try73_experts_4gpu_sequential.py"
        if not submit.is_file():
            print(f"Missing {submit}", file=sys.stderr)
            return 1
        subprocess.run(
            [sys.executable, str(submit), "--skip-upload"],
            check=True,
            cwd=str(ROOT),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
