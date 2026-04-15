#!/usr/bin/env python3
"""SSH to the cluster and print disk usage for scratch / home / Try 71 tree.

Usage (PowerShell):
  $env:SSH_PASSWORD = 'your_password'
  python cluster/check_cluster_disk.py

Or with an SSH key:
  python cluster/check_cluster_disk.py --ssh-key ~/.ssh/id_ed25519
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

HOST = "sert.ac.upc.edu"
USER = "gmoreno"
# Paths used by Try 71 jobs (see submit_try71_stage1_stage2_4gpu.py)
REMOTE_SCRATCH_USER = "/scratch/nas/3/gmoreno"
REMOTE_TFG_ROOT = f"{REMOTE_SCRATCH_USER}/TFGpractice"
REMOTE_TRY66 = f"{REMOTE_TFG_ROOT}/TFGSeventyFirstTry71"
REMOTE_DATASET = f"{REMOTE_TFG_ROOT}/Datasets"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check free disk space on the cluster via SSH.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password and not args.ssh_key:
        raise SystemExit(
            f"Set {args.password_env} or pass --ssh-key. Example:\n"
            f'  $env:{args.password_env} = "<password>"\n'
            f"  python {Path(__file__).name}"
        )

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

    # Single shell script: df on mount points, du on project dirs, optional quota.
    remote_script = rf"""
set -e
echo "========== df -h (relevant mounts) =========="
df -h {REMOTE_SCRATCH_USER} 2>/dev/null || df -h "$(dirname {REMOTE_SCRATCH_USER})" 2>/dev/null || true
df -h "$HOME" 2>/dev/null || true
echo ""
echo "========== Try 71 / TFG tree sizes =========="
for d in "{REMOTE_TRY66}" "{REMOTE_DATASET}" "{REMOTE_TFG_ROOT}"; do
  if [ -d "$d" ]; then
    du -sh "$d" 2>/dev/null || echo "(du failed: $d)"
  else
    echo "(missing: $d)"
  fi
done
echo ""
echo "========== Largest items under Try 71 outputs (top 12) =========="
if [ -d "{REMOTE_TRY66}/outputs" ]; then
  du -sh "{REMOTE_TRY66}/outputs"/* 2>/dev/null | sort -hr | head -12
else
  echo "(no outputs dir)"
fi
echo ""
echo "========== quota (if available) =========="
command -v quota >/dev/null 2>&1 && quota -s 2>/dev/null || echo "(quota not installed or no user quota)"
command -v lfs >/dev/null 2>&1 && lfs quota -h -u "$USER" "$(dirname {REMOTE_SCRATCH_USER})" 2>/dev/null || true
"""
    try:
        _, stdout, stderr = client.exec_command(remote_script)
        out = stdout.read().decode()
        err = stderr.read().decode()
        sys.stdout.write(out)
        if err.strip():
            sys.stderr.write(err)
    finally:
        client.close()


if __name__ == "__main__":
    main()
