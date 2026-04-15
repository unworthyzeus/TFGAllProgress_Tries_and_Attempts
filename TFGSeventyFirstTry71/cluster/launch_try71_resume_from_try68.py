#!/usr/bin/env python3
"""Copy the open_sparse_lowrise Try 68 checkpoint into Try 71 on the cluster, then optionally submit.

Try 71 extends Try 68 with heteroscedastic uncertainty (out_channels=2). The backbone weights are
fully compatible; the new log_var output head initialises randomly. Training resumes via
``load_state_dict(strict=False)`` from the Try 68 ``best_model.pt``.

Only the ``open_sparse_lowrise`` expert is included in Try 71.

Prerequisites:
  * SSH access to the cluster (``SSH_PASSWORD`` env or --ssh-key).
  * Try 68 ``outputs/try68_expert_open_sparse_lowrise/`` exists on the cluster.

Examples:
  # Dry-run: print the remote shell script and exit
  python cluster/launch_try71_resume_from_try68.py --dry-run

  # Upload Try 71 code (--no-clean-outputs), copy Try 68 checkpoint, then sbatch
  python cluster/launch_try71_resume_from_try68.py --submit

  # Copy only on the cluster (no code upload, no Slurm)
  python cluster/launch_try71_resume_from_try68.py
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
TRY71_DIR = ROOT / "TFGSeventyFirstTry71"

REMOTE_BASE = "/scratch/nas/3/gmoreno/TFGpractice"
REMOTE_TRY68_OUTPUTS = f"{REMOTE_BASE}/TFGSixtyEighthTry68/outputs"
REMOTE_TRY71_OUTPUTS = f"{REMOTE_BASE}/TFGSeventyFirstTry71/outputs"

HOST = "sert.ac.upc.edu"
USER = "gmoreno"

EXPERT = "open_sparse_lowrise"


def _remote_copy_script() -> str:
    src = f"{REMOTE_TRY68_OUTPUTS}/try68_expert_{EXPERT}"
    dst = f"{REMOTE_TRY71_OUTPUTS}/try71_expert_{EXPERT}"
    return f"""set -euo pipefail
mkdir -p "{REMOTE_TRY71_OUTPUTS}"
if [ -d "{src}" ]; then
  echo "[sync] {src} -> {dst}"
  rm -rf "{dst}"
  cp -a "{src}" "{dst}"
  echo "[sync] done."
else
  echo "[warn] Source not found: {src}" >&2
  echo "[info] Try 71 will start from scratch (no resume)."
fi
echo "[info] Try 71 outputs:"
ls -la "{REMOTE_TRY71_OUTPUTS}" || true
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
            "Upload Try 71 with --no-clean-outputs, copy Try 68 checkpoint, then run "
            "submit_try71_lowrise_4gpu_sequential.py --skip-upload."
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
            "TFGSeventyFirstTry71",
            "--upload-only",
            "--skip-datasets",
            "--no-clean-outputs",
        ]
        print("LOCAL>", " ".join(upload_cmd))
        subprocess.run(upload_cmd, check=True, cwd=str(ROOT))

    client = _connect(args)
    try:
        print(f"REMOTE> bash -s  (sync Try68 open_sparse_lowrise -> Try71)")
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
        submit = TRY71_DIR / "cluster" / "submit_try71_lowrise_4gpu_sequential.py"
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
