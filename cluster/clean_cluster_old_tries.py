#!/usr/bin/env python3
"""Delete old TFG try directories from cluster to free quota.

Keeps: TFGFiftySeventhTry57, TFGFiftyEighthTry58, TFGFiftyNinthTry59,
       TFGSixtyEighthTry68, TFGSixtyNinthTry69, Datasets (and anything
       that is not a TFG*Try* folder).

Usage (from TFGpractice/):
    set SSH_PASSWORD=Slenderman,2004
    python cluster/clean_cluster_old_tries.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

try:
    import paramiko
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko

HOST = "sert.ac.upc.edu"
USER = "gmoreno"
REMOTE_BASE = "/scratch/nas/3/gmoreno/TFGpractice"

KEEP = {
    "TFGFiftySeventhTry57",
    "TFGFiftyEighthTry58",
    "TFGFiftyNinthTry59",
    "TFGSixtyEighthTry68",
    "TFGSixtyNinthTry69",
    "Datasets",
}


def remote_exec(c: paramiko.SSHClient, cmd: str) -> tuple[str, str]:
    print("REMOTE>", cmd)
    _, o, e = c.exec_command(cmd)
    out = o.read().decode().strip()
    err = e.read().decode().strip()
    if out:
        print(out)
    if err:
        print("  STDERR:", err)
    return out, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print what would be deleted, do nothing")
    ap.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = ap.parse_args()

    password = os.environ.get("SSH_PASSWORD", "")
    if not password:
        import getpass
        password = getpass.getpass(f"SSH password for {USER}@{HOST}: ")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=password, timeout=30)

    # List all entries in REMOTE_BASE
    out, _ = remote_exec(client, f"ls -1 {REMOTE_BASE}/")
    entries = [e.strip() for e in out.splitlines() if e.strip()]

    to_delete = []
    for entry in entries:
        if re.match(r"^TFG.*Try\d+$", entry) and entry not in KEEP:
            to_delete.append(entry)

    if not to_delete:
        print("\nNothing to delete.")
        client.close()
        return

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Will delete {len(to_delete)} directories:")
    total = 0
    for d in to_delete:
        out, _ = remote_exec(client, f"du -sh {REMOTE_BASE}/{d} 2>/dev/null | cut -f1")
        size = out.strip() or "?"
        print(f"  {d}  [{size}]")

    if args.dry_run:
        print("\nDry run — nothing deleted.")
        client.close()
        return

    if not args.yes:
        confirm = input(f"\nType YES to delete these {len(to_delete)} directories: ").strip()
        if confirm != "YES":
            print("Aborted.")
            client.close()
            return

    for d in to_delete:
        remote_exec(client, f"rm -rf {REMOTE_BASE}/{d}")
        print(f"  Deleted {d}")

    # Show updated quota
    print("\n--- Updated disk usage ---")
    remote_exec(client, "quota -s 2>/dev/null || echo '(quota command not available)'")
    remote_exec(client, f"du -sh {REMOTE_BASE}/ 2>/dev/null")

    client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
