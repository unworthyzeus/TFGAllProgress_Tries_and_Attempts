#!/usr/bin/env python3
"""
Upload TFGSixthTry6 to the cluster (excluding outputs/caches) and ensure the new dataset
is present on the cluster. Then submit the 2-GPU Slurm job.

Required env:
  - SSH_PASSWORD: SSH password for gmoreno@sert.ac.upc.edu

Usage (PowerShell):
  $env:SSH_PASSWORD="***"
  python cluster/upload_and_submit_sixthtry6.py
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

try:
    import paramiko
except ImportError:
    os.system(f"{sys.executable} -m pip install paramiko -q")
    import paramiko

_CLUSTER_DIR = Path(__file__).resolve().parent
if str(_CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLUSTER_DIR))
import upload_dataset_helpers as udh

HOST = "sert.ac.upc.edu"
USER = "gmoreno"
REMOTE_BASE = "/scratch/nas/3/gmoreno/TFGpractice"

LOCAL_DIR = "TFGSixthTry6"
LOCAL_DATASET_PATH = r"c:\TFG\TFGpractice\Datasets\CKM_Dataset_180326.h5"
REMOTE_DATASET_DIR = f"{REMOTE_BASE}/Datasets"
REMOTE_DATASET_PATH = f"{REMOTE_DATASET_DIR}/CKM_Dataset_180326.h5"

EXCLUDE_DIRS = {"outputs", "__pycache__", ".git", ".venv"}
EXCLUDE_EXTS = {".pt", ".pyc", ".pyo", ".h5"}


def _mkdir_p(sftp, remote_path: str) -> None:
    udh.mkdir_p_with_quota_hint(sftp, remote_path)


def _upload_file_if_missing(sftp, local_path: str, remote_path: str) -> None:
    udh.upload_if_missing_file(sftp, _mkdir_p, Path(local_path), remote_path)


def _should_skip(name: str) -> bool:
    if name in EXCLUDE_DIRS:
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext in EXCLUDE_EXTS:
        return True
    return False


def _collect_files(local_root: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(local_root):
        dirs[:] = [d for d in dirs if not _should_skip(d)]
        rel_root = os.path.relpath(root, local_root)
        for f in files:
            if _should_skip(f):
                continue
            local_full = os.path.join(root, f)
            rel = os.path.join(rel_root, f).replace("\\", "/")
            out.append((local_full, rel))
    return out


def _clean_remote_trash(sftp, remote_dir: str) -> None:
    # Remove outputs and cached bytecode under the experiment folder only.
    def rm_r(path: str) -> None:
        try:
            entries = sftp.listdir_attr(path)
        except FileNotFoundError:
            return
        for ent in entries:
            full = f"{path}/{ent.filename}"
            if stat.S_ISDIR(ent.st_mode):
                if ent.filename in {"outputs", "__pycache__"}:
                    rm_r(full)
                    try:
                        sftp.rmdir(full)
                    except OSError:
                        pass
                else:
                    rm_r(full)
            else:
                ext = os.path.splitext(ent.filename)[1].lower()
                if ext in {".pyc", ".pyo"}:
                    try:
                        sftp.remove(full)
                    except OSError:
                        pass

    rm_r(remote_dir)


def main() -> None:
    password = "Slenderman,2004"
    # if not password:
    #     raise SystemExit("Missing SSH_PASSWORD env var.")

    local_root = Path(r"c:\TFG\TFGpractice") / LOCAL_DIR
    if not local_root.exists():
        local_root = Path.cwd() / LOCAL_DIR
    if not local_root.exists():
        raise SystemExit(f"Local folder not found: {LOCAL_DIR}")

    dataset_local = Path(LOCAL_DATASET_PATH)
    if not dataset_local.exists():
        raise SystemExit(f"Local dataset not found: {dataset_local}")

    files = _collect_files(local_root)
    print(f"Uploading {len(files)} files for {LOCAL_DIR} (excluding outputs, .h5, .pt)...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=password, timeout=30)
    udh.prepare_ssh_transport_for_large_uploads(client)

    try:
        sftp = client.open_sftp()

        # Ensure dataset exists on cluster once.
        _upload_file_if_missing(sftp, str(dataset_local), REMOTE_DATASET_PATH)

        remote_dir = f"{REMOTE_BASE}/{LOCAL_DIR}"
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            _mkdir_p(sftp, remote_dir)

        print("Cleaning remote trash (outputs/__pycache__/pyc) under experiment dir...")
        _clean_remote_trash(sftp, remote_dir)

        # Upload code/configs.
        for idx, (local_full, rel_path) in enumerate(files, start=1):
            remote_full = f"{remote_dir}/{rel_path}"
            remote_parent = os.path.dirname(remote_full).replace("\\", "/")
            _mkdir_p(sftp, remote_parent)
            sftp.put(local_full, remote_full)
            if idx % 25 == 0 or idx == len(files):
                print(f"  {idx}/{len(files)}")

        sftp.close()
        print("Upload done.")

        cmd = f"cd {remote_dir} && sbatch cluster/run_sixthtry6_2gpu.slurm"
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(out)
        if err:
            print("stderr:", err)
    finally:
        client.close()


if __name__ == "__main__":
    main()

