"""One-off: fetch Try 72 cluster log snippets (timing). Run: python _fetch_try72_logs.py"""
from __future__ import annotations

import json
import os
import sys

try:
    import paramiko
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "paramiko"])
    import paramiko

HOST = "sert.ac.upc.edu"
USER = "gmoreno"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGSeventySecondTry72"


def main() -> None:
    password = os.environ.get("SSH_PASSWORD", "")
    if not password:
        print("Set SSH_PASSWORD", file=sys.stderr)
        sys.exit(2)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=HOST, username=USER, password=password, timeout=45)

    def run(cmd: str) -> str:
        _, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            print(err, file=sys.stderr)
        return out

    print("=== squeue (gmoreno) ===")
    print(run("squeue -u gmoreno 2>/dev/null | head -20"))

    print("\n=== Newest SLURM .out ===")
    out = run(f"ls -t {REMOTE_DIR}/logs_train_try72_*.out 2>/dev/null | head -8")
    paths = [p.strip() for p in out.splitlines() if p.strip()]
    if not paths:
        print("(no logs_train_try72_*.out yet)")
    for p in paths[:5]:
        print(" ", p)

    if paths:
        p0 = paths[0]
        print(f"\n=== tail {p0} (last 120 lines) ===")
        print(run(f"tail -n 120 '{p0}'"))
        print(f"\n=== wc -l {p0} ===")
        print(run(f"wc -l '{p0}'"))

    print("\n=== grep JSON lines with timing (last 25) ===")
    pat = "train_seconds|seconds_per_train_batch|epoch_seconds|train_batches_per_second"
    g = run(f"grep -hE '{pat}' {REMOTE_DIR}/logs_train_try72_*.out 2>/dev/null | tail -25")
    print(g if g.strip() else "(no grep hits)")

    print("\n=== validate_metrics_latest (first expert dir found) ===")
    find_out = run(
        f"find {REMOTE_DIR}/outputs -maxdepth 2 -name validate_metrics_latest.json "
        f"2>/dev/null | head -1"
    )
    vpath = find_out.strip().splitlines()[0] if find_out.strip() else ""
    if vpath:
        raw = run(f"cat '{vpath}'")
        try:
            data = json.loads(raw)
            rt = data.get("runtime", {})
            print("path:", vpath)
            for k in (
                "train_seconds",
                "val_seconds",
                "epoch_seconds",
                "train_batches_per_second",
                "seconds_per_train_batch",
                "seconds_per_val_sample",
            ):
                if k in rt:
                    print(f"  {k}: {rt[k]}")
        except json.JSONDecodeError:
            print(raw[:2000])
    else:
        print("(no validate_metrics_latest.json yet)")

    print("\n=== train_progress_latest (if any) ===")
    tp = run(
        f"find {REMOTE_DIR}/outputs -maxdepth 2 -name train_progress_latest.json "
        f"2>/dev/null | head -1"
    ).strip().splitlines()
    if tp:
        print(run(f"cat '{tp[0]}'")[:2500])
    else:
        print("(none)")

    client.close()


if __name__ == "__main__":
    main()
