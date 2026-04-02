#!/usr/bin/env python3
"""
Download experiment artifacts from the cluster into the local workspace.

Downloads (by default) from:
  /scratch/nas/3/gmoreno/TFGpractice/<TRY>/outputs/<RUN>/

Into local:
  c:\\TFG\\TFGpractice\\cluster_outputs\\<TRY>\\<RUN>\\

Default behavior:
  - Download EVERYTHING under outputs/ (mirrors the remote tree).

Use --filter json_and_best_cgan to pull metrics (*.json) + only best_cgan.pt (skip epoch_*_cgan.pt).

Auth:
  - Requires SSH_PASSWORD env var (so we never hardcode credentials).

Usage (PowerShell):
  $env:SSH_PASSWORD="***"
  python c:\\TFG\\TFGpractice\\cluster\\download_cluster_outputs.py --tries TFGFifthTry5,TFGSixthTry6
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import paramiko
except ImportError:
    os.system(f"{sys.executable} -m pip install paramiko -q")
    import paramiko


DEFAULT_REMOTE_ROOT = "/scratch/nas/3/gmoreno/TFGpractice"
DEFAULT_LOCAL_ROOT = r"c:\TFG\TFGpractice\cluster_outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download outputs/ artifacts from cluster.")
    p.add_argument("--host", default="sert.ac.upc.edu")
    p.add_argument("--user", default="gmoreno")
    p.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    p.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    p.add_argument(
        "--tries",
        default="",
        help="Comma-separated try folders (e.g. TFGFifthTry5,TFGSixthTry6). If empty, auto-detect TFG*Try* under remote root.",
    )
    p.add_argument(
        "--filter",
        default="all",
        choices=["all", "json_and_pt", "json_only", "json_and_best_cgan"],
        help=(
            "Download mode. 'all' = mirror outputs/ (default). "
            "'json_and_pt' = *.json and every *.pt (includes epoch_*_cgan.pt). "
            "'json_only' = *.json only. "
            "'json_and_best_cgan' = *.json + only best_cgan.pt (no epoch checkpoints)."
        ),
    )
    p.add_argument(
        "--include-prior-calibration",
        action="store_true",
        help="Also download files under <TRY>/prior_calibration into cluster_outputs/<TRY>/prior_calibration.",
    )
    return p.parse_args()


def sftp_isdir(sftp: "paramiko.SFTPClient", remote_path: str) -> bool:
    import stat

    try:
        st = sftp.stat(remote_path)
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(st.st_mode)


def sftp_listdir(sftp: "paramiko.SFTPClient", remote_path: str) -> List[str]:
    try:
        return sorted(sftp.listdir(remote_path))
    except FileNotFoundError:
        return []


def should_download(rel_path: str, mode: str) -> bool:
    if mode == "all":
        return True
    name = os.path.basename(rel_path)
    if mode == "json_and_pt":
        return name.endswith(".json") or name.endswith(".pt")
    if mode == "json_only":
        return name.endswith(".json")
    if mode == "json_and_best_cgan":
        if name.endswith(".json"):
            return True
        if name.endswith(".pt"):
            return (name == "best_cgan.pt" or name == "best_model.pt")
        return False
    return True


def mkdir_p(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_tree_filtered(
    sftp: "paramiko.SFTPClient",
    remote_base: str,
    local_base: Path,
    mode: str,
) -> Tuple[int, int]:
    """Returns (files_downloaded, bytes_downloaded)."""
    import stat

    files = 0
    total_bytes = 0

    stack: List[Tuple[str, str]] = [("", remote_base)]
    while stack:
        rel_dir, remote_dir = stack.pop()
        try:
            entries = sftp.listdir_attr(remote_dir)
        except FileNotFoundError:
            continue
        for ent in entries:
            remote_path = f"{remote_dir}/{ent.filename}"
            rel_path = f"{rel_dir}/{ent.filename}" if rel_dir else ent.filename
            if stat.S_ISDIR(ent.st_mode):
                stack.append((rel_path, remote_path))
                continue
            if not should_download(rel_path, mode=mode):
                continue

            local_path = local_base / rel_path
            mkdir_p(local_path.parent)

            # Best checkpoints are frequently overwritten with the same tensor
            # shapes, so size equality alone is not enough to decide whether
            # the local copy is up to date.
            if local_path.exists():
                local_stat = local_path.stat()
                same_size = local_stat.st_size == int(ent.st_size)
                local_mtime = int(local_stat.st_mtime)
                remote_mtime = int(ent.st_mtime)
                if same_size and local_mtime >= remote_mtime:
                    continue

            sftp.get(remote_path, str(local_path))
            try:
                os.utime(local_path, (int(ent.st_atime), int(ent.st_mtime)))
            except Exception:
                pass
            files += 1
            total_bytes += int(ent.st_size)
    return files, total_bytes


def autodetect_tries(sftp: "paramiko.SFTPClient", remote_root: str) -> List[str]:
    tries = []
    for name in sftp_listdir(sftp, remote_root):
        if fnmatch.fnmatch(name, "TFG*Try*") and sftp_isdir(sftp, f"{remote_root}/{name}"):
            tries.append(name)
    return sorted(tries)


def main() -> None:
    args = parse_args()
    password = os.environ.get("SSH_PASSWORD", "")
    if not password:
        raise SystemExit("Missing SSH_PASSWORD env var.")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=30)

    local_root = Path(args.local_root)
    mkdir_p(local_root)

    try:
        sftp = client.open_sftp()
        if args.tries.strip():
            tries = [t.strip() for t in args.tries.split(",") if t.strip()]
        else:
            tries = autodetect_tries(sftp, args.remote_root)

        if not tries:
            print("No try folders found.")
            return

        for t in tries:
            remote_outputs = f"{args.remote_root}/{t}/outputs"
            local_outputs = local_root / t
            total_files = 0
            total_bytes = 0

            if sftp_isdir(sftp, remote_outputs):
                files, nbytes = download_tree_filtered(
                    sftp,
                    remote_outputs,
                    local_outputs,
                    mode=str(args.filter),
                )
                total_files += files
                total_bytes += nbytes

            if args.include_prior_calibration:
                remote_prior = f"{args.remote_root}/{t}/prior_calibration"
                local_prior = local_outputs / "prior_calibration"
                if sftp_isdir(sftp, remote_prior):
                    files, nbytes = download_tree_filtered(
                        sftp,
                        remote_prior,
                        local_prior,
                        mode="all",
                    )
                    total_files += files
                    total_bytes += nbytes

            if total_files:
                print(f"{t}: downloaded {total_files} files ({total_bytes/1024/1024:.1f} MiB) into {local_outputs}")
            else:
                print(f"{t}: nothing new to download")
        sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()

