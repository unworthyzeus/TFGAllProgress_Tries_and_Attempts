#!/usr/bin/env python3
"""SSH: list recent Try68 1gpu Slurm logs and tail .err (password: SSH_PASSWORD)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import paramiko
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "paramiko"])
    import paramiko

BASE = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68"


def main() -> None:
    pw = os.environ.get("SSH_PASSWORD", "")
    if not pw:
        print("Set SSH_PASSWORD", file=sys.stderr)
        sys.exit(2)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("sert.ac.upc.edu", username="gmoreno", password=pw, timeout=45)

    def run(cmd: str, t: int = 120) -> str:
        _, o, e = c.exec_command(cmd, timeout=t)
        return (o.read() + e.read()).decode("utf-8", "replace")

    out_path = os.environ.get("TRY68_LOG_DUMP", "cluster/_try68_cluster_log_dump.txt")
    parts: list[str] = []
    parts.append("=== squeue -u gmoreno ===\n")
    parts.append(run("squeue -u gmoreno"))
    parts.append("\n=== newest logs_train_try68_1gpu*.err (15 files, tail 40 each) ===\n")
    parts.append(
        run(
            f"bash -c 'i=0; for f in $(ls -t {BASE}/logs_train_try68_1gpu*.err 2>/dev/null); do "
            f"echo ========== \"$f\" ==========; tail -40 \"$f\"; i=$((i+1)); [ $i -ge 15 ] && break; done'"
        )
    )
    parts.append("\n=== newest logs_train_try68_1gpu*.out (5 files, tail 25 each) ===\n")
    parts.append(
        run(
            f"bash -c 'i=0; for f in $(ls -t {BASE}/logs_train_try68_1gpu*.out 2>/dev/null); do "
            f"echo ========== \"$f\" ==========; tail -25 \"$f\"; i=$((i+1)); [ $i -ge 5 ] && break; done'"
        )
    )
    text = "".join(parts)
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} ({len(text)} chars). First 4000 chars ASCII-safe:")
    safe = text.encode("ascii", errors="replace").decode("ascii")
    print(safe[:4000])
    c.close()


if __name__ == "__main__":
    main()
