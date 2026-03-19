#!/usr/bin/env python3
"""
Upload TFGSeventhTry7 to the cluster (excluding outputs/code artifacts) and submit.

Dataset:
  /scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_180326.h5 (shared)
If missing, it will be uploaded once.

Usage (PowerShell):
  $env:SSH_PASSWORD="(your ssh password)"
  python cluster\\upload_and_submit_seventhtry7.py --gpus 2

  python cluster\\upload_and_submit_seventhtry7.py --gpus 4
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

HOST = "sert.ac.upc.edu"
USER = "gmoreno"
REMOTE_BASE = "/scratch/nas/3/gmoreno/TFGpractice"

LOCAL_DIR = "TFGSeventhTry7"
LOCAL_DATASET_PATH = r"c:\TFG\TFGpractice\Datasets\CKM_Dataset_180326.h5"
REMOTE_DATASET_PATH = f"{REMOTE_BASE}/Datasets/CKM_Dataset_180326.h5"

EXCLUDE_DIRS = {"outputs", "__pycache__", ".git", ".venv", ".cursor"}
EXCLUDE_EXTS = {".h5", ".pt", ".pth", ".pyc", ".pyo"}


def mkdir_p(sftp, remote_path: str) -> None:
    parts = [p for p in remote_path.strip("/").split("/") if p]
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_if_missing(sftp, local_path: str, remote_path: str) -> None:
    try:
        sftp.stat(remote_path)
        print(f"Dataset already present: {remote_path}")
        return
    except FileNotFoundError:
        pass

    mkdir_p(sftp, os.path.dirname(remote_path).replace("\\", "/"))
    print(f"Uploading dataset: {local_path} -> {remote_path} (one-time)...")
    sftp.put(local_path, remote_path)
    print("Dataset upload done.")


def should_skip(name: str) -> bool:
    if name in EXCLUDE_DIRS:
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext in EXCLUDE_EXTS:
        return True
    return False


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
    # Remove only outputs and cached artifacts under the experiment folder.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, choices=[2, 4], default=2)
    args = parser.parse_args()

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

    slurm_script = "cluster/run_seventhtry7_2gpu.slurm" if args.gpus == 2 else "cluster/run_seventhtry7_4gpu_4h.slurm"
    local_files = collect_files(local_root)
    print(f"Uploading {len(local_files)} files for {LOCAL_DIR} (excluding outputs and checkpoints)...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=password, timeout=30)

    try:
        sftp = client.open_sftp()
        upload_if_missing(sftp, LOCAL_DATASET_PATH, REMOTE_DATASET_PATH)

        remote_dir = f"{REMOTE_BASE}/{LOCAL_DIR}"
        mkdir_p(sftp, remote_dir)

        print("Cleaning remote experiment outputs...")
        clean_remote_experiment_dir(sftp, remote_dir)

        # Upload code/configs
        for i, (local_full, rel_path) in enumerate(local_files, start=1):
            remote_full = f"{remote_dir}/{rel_path}"
            remote_parent = os.path.dirname(remote_full).replace("\\", "/")
            mkdir_p(sftp, remote_parent)
            sftp.put(local_full, remote_full)
            if i % 25 == 0 or i == len(local_files):
                print(f"  {i}/{len(local_files)}")

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

