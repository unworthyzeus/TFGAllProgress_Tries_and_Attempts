#!/usr/bin/env python3
"""SSH: squeue + sacct failures + tail recent .err for Try 68/69/70 training logs.

Try 68 includes **all** ``logs_train_try68_*.err`` (1-GPU, 2-GPU ``ckm-t68-2gpu``, etc.).
Try 69/70 still use the 1-GPU log glob names used by default Slurm scripts.

  set SSH_PASSWORD=...
  python cluster/_fetch_try68_69_70_slurm_err.py

Writes UTF-8 dump to ``cluster/_try68_69_70_cluster_err_dump.txt`` and prints an ASCII-safe prefix.
"""
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

ROOT = Path(__file__).resolve().parent.parent
REMOTE_BASE = "/scratch/nas/3/gmoreno/TFGpractice"
TRIES: list[tuple[str, str, str]] = [
    ("68", "TFGSixtyEighthTry68", "logs_train_try68_*.err"),
    ("69", "TFGSixtyNinthTry69", "logs_train_try69_1gpu*.err"),
    ("70", "TFGSeventiethTry70", "logs_train_try70_1gpu*.err"),
]


def main() -> None:
    pw = os.environ.get("SSH_PASSWORD", "")
    if not pw:
        print("Set SSH_PASSWORD", file=sys.stderr)
        sys.exit(2)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("sert.ac.upc.edu", username="gmoreno", password=pw, timeout=60)

    def run(cmd: str, t: int = 180) -> str:
        _, o, e = c.exec_command(cmd, timeout=t)
        return (o.read() + e.read()).decode("utf-8", "replace")

    parts: list[str] = []
    parts.append("=== squeue -u gmoreno ===\n")
    parts.append(run("squeue -u gmoreno"))
    parts.append(
        "\n=== sacct last 80 rows (non-success states last 3 days) ===\n"
        "Note: sacct date format is cluster-dependent; empty is OK.\n"
    )
    parts.append(
        run(
            "sacct -u gmoreno -S $(date -d '3 days ago' +%Y-%m-%d) --state=FAILED,OUT_OF_MEMORY,"
            "CANCELLED,TIMEOUT,NODE_FAIL,PREEMPTED --format=JobID,JobName%20,State,ExitCode,"
            "DerivedExitCode,MaxRSS,Elapsed,NodeList -n -P 2>/dev/null | tail -n 80"
        )
    )
    parts.append(
        "\n=== sacct last 60 rows (ALL states, today) — see COMPLETED vs FAILED ===\n"
    )
    parts.append(
        run(
            "sacct -u gmoreno -S today --format=JobID,JobName%24,State,ExitCode,"
            "DerivedExitCode,Elapsed,MaxRSS,NodeList -X -n -P 2>/dev/null | tail -n 60"
        )
    )
    for label, folder, glob_err in TRIES:
        base = f"{REMOTE_BASE}/{folder}"
        parts.append(f"\n=== Try {label}: newest {glob_err} (12 files, tail 50 each) ===\n")
        inner = (
            f"i=0; for f in $(ls -t {base}/{glob_err} 2>/dev/null); do "
            'echo "========== $f =========="; tail -50 "$f"; '
            "i=$((i+1)); [ $i -ge 12 ] && break; done"
        )
        parts.append(run(f"bash -lc {inner!r}"))

    text = "".join(parts)
    out_path = Path(os.environ.get("CLUSTER_ERR_DUMP", str(ROOT / "cluster" / "_try68_69_70_cluster_err_dump.txt")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} ({len(text)} chars). First 6000 chars (ASCII-safe):")
    safe = text.encode("ascii", errors="replace").decode("ascii")
    print(safe[:6000])
    c.close()


if __name__ == "__main__":
    main()
