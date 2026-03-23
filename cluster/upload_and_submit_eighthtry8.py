#!/usr/bin/env python3
"""
Upload TFG EighthTry8 to the cluster (excluding outputs/code artifacts) and submit.

Dataset (shared):
  /scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_180326.h5

Usage (PowerShell):
  $env:SSH_PASSWORD="(your ssh password)"
  python cluster\\upload_and_submit_eighthtry8.py --gpus 2

  python cluster\\upload_and_submit_eighthtry8.py --gpus 4
"""

from __future__ import annotations

import argparse
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

LOCAL_DIR = "TFGEighthTry8"
LOCAL_DATASET_PATH = r"c:\TFG\TFGpractice\Datasets\CKM_Dataset_180326.h5"
REMOTE_DATASET_PATH = f"{REMOTE_BASE}/Datasets/CKM_Dataset_180326.h5"

EXCLUDE_DIRS = {"outputs", "__pycache__", ".git", ".venv", ".cursor"}
EXCLUDE_EXTS = {".h5", ".pt", ".pth", ".pyc", ".pyo"}


def mkdir_p(sftp, remote_path: str) -> None:
    udh.mkdir_p_with_quota_hint(sftp, remote_path)


def upload_if_missing(sftp, local_path: str, remote_path: str) -> None:
    udh.upload_if_missing_file(sftp, mkdir_p, Path(local_path), remote_path)


def should_skip(name: str) -> bool:
    if name in EXCLUDE_DIRS:
        return True
    ext = os.path.splitext(name)[1].lower()
    return ext in EXCLUDE_EXTS


def collect_files(local_root: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(local_root):
        dirs[:] = [d for d in dirs if not should_skip(d)]
        rel_root = os.path.relpath(root, local_root)
        for f in files:
            if should_skip(f):
                continue
            local_full = os.path.join(root, f)
            rel = os.path.join(rel_root, f).replace("\\", "/")
            out.append((local_full, rel))
    return out


def clean_remote_experiment_dir(sftp, remote_dir: str) -> None:
    # Remove outputs and cached artifacts under remote_dir only.
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
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, choices=[2, 4], default=2)
    args = p.parse_args()

    password = os.environ.get("SSH_PASSWORD", "")
    if not password:
        raise SystemExit("Missing SSH_PASSWORD env var.")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    local_root = project_root / LOCAL_DIR

    if not local_root.exists():
        raise SystemExit(f"Local folder not found: {local_root}")
    if not Path(LOCAL_DATASET_PATH).exists():
        raise SystemExit(f"Local dataset not found: {LOCAL_DATASET_PATH}")

    slurm_script = (
        "cluster/run_eighthtry8_2gpu.slurm" if args.gpus == 2 else "cluster/run_eighthtry8_4gpu_4h.slurm"
    )

    files = collect_files(local_root)
    print(f"Uploading {len(files)} files for {LOCAL_DIR}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=password, timeout=30)
    udh.prepare_ssh_transport_for_large_uploads(client)

    try:
        sftp = client.open_sftp()
        upload_if_missing(sftp, LOCAL_DATASET_PATH, REMOTE_DATASET_PATH)

        remote_dir = f"{REMOTE_BASE}/{LOCAL_DIR}"
        mkdir_p(sftp, remote_dir)

        print("Cleaning remote outputs...")
        clean_remote_experiment_dir(sftp, remote_dir)

        for i, (local_full, rel_path) in enumerate(files, start=1):
            remote_full = f"{remote_dir}/{rel_path}"
            mkdir_p(sftp, os.path.dirname(remote_full).replace("\\", "/"))
            sftp.put(local_full, remote_full)
            if i % 25 == 0 or i == len(files):
                print(f"  {i}/{len(files)}")

        sftp.close()

        cmd = f"cd {remote_dir} && sbatch {slurm_script}"
        print(f"Submitting: {cmd}")
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

