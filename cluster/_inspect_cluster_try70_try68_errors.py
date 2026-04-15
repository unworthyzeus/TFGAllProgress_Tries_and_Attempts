#!/usr/bin/env python3
"""Tail recent Slurm .err for Try70 / Try68 1gpu jobs on sert (needs SSH agent or SSH_PASSWORD)."""
from __future__ import annotations

import os
import sys

try:
    import paramiko
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "paramiko"])
    import paramiko


def main() -> None:
    pw = os.environ.get("SSH_PASSWORD", "")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw: dict = {
        "hostname": "sert.ac.upc.edu",
        "username": "gmoreno",
        "timeout": 30,
        "allow_agent": True,
        "look_for_keys": True,
    }
    if pw:
        kw["password"] = pw
    c.connect(**kw)

    def run(cmd: str) -> str:
        _, stdout, stderr = c.exec_command(cmd, timeout=120)
        return (stdout.read() + stderr.read()).decode("utf-8", "replace")

    print(run("squeue -u gmoreno"))
    for base, label in (
        ("/scratch/nas/3/gmoreno/TFGpractice/TFGSeventiethTry70", "try70"),
        ("/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68", "try68"),
    ):
        print(f"\n========== {label} .err (newest 3 files, tail 100) ==========")
        print(
            run(
                rf"bash -c 'for f in $(ls -t {base}/logs_train_*1gpu*.err 2>/dev/null | head -3); do "
                rf"echo --- \"$f\" ---; tail -100 \"$f\"; done'"
            )
        )
        print(f"\n========== {label} .out (newest 1 file, tail 60) ==========")
        print(
            run(
                rf"bash -c 'f=$(ls -t {base}/logs_train_*1gpu*.out 2>/dev/null | head -1); "
                rf"echo \"$f\"; tail -60 \"$f\"'"
            )
        )
    c.close()


if __name__ == "__main__":
    main()
