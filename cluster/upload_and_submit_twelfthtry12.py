#!/usr/bin/env python3
"""Upload TFGTwelfthTry12; --variant los|nlos --gpus 1|2. Set SSH_PASSWORD."""
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
LOCAL_DIR = "TFGTwelfthTry12"
LOCAL_H5 = Path(__file__).resolve().parent.parent / "Datasets" / "CKM_Dataset_180326.h5"
LOCAL_CSV = Path(__file__).resolve().parent.parent / "Datasets" / "CKM_180326_antenna_height.csv"
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


def upload_if_missing(sftp, local_path: Path, remote_path: str) -> None:
    try:
        sftp.stat(remote_path)
        return
    except FileNotFoundError:
        pass
    mkdir_p(sftp, os.path.dirname(remote_path).replace("\\", "/"))
    sftp.put(str(local_path), remote_path)


def upload_csv_always(sftp, local_path: Path, remote_path: str) -> None:
    if not local_path.is_file():
        return
    mkdir_p(sftp, os.path.dirname(remote_path).replace("\\", "/"))
    print(f"Upload antenna CSV -> {remote_path}")
    sftp.put(str(local_path), remote_path)


def should_skip(name: str) -> bool:
    if name in EXCLUDE_DIRS:
        return True
    return os.path.splitext(name)[1].lower() in EXCLUDE_EXTS


def collect_files(local_root: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(local_root):
        dirs[:] = [d for d in dirs if not should_skip(d)]
        for f in files:
            if should_skip(f):
                continue
            lf = Path(root) / f
            rel = os.path.relpath(lf, local_root).replace("\\", "/")
            out.append((str(lf), rel))
    return out


def clean_remote_outputs(sftp, remote_dir: str) -> None:
    def rm_r(path: str) -> None:
        try:
            entries = sftp.listdir_attr(path)
        except FileNotFoundError:
            return
        for ent in entries:
            full = f"{path}/{ent.filename}"
            if stat.S_ISDIR(ent.st_mode) and ent.filename in {"outputs", "__pycache__"}:
                rm_r(full)
                try:
                    sftp.rmdir(full)
                except OSError:
                    pass
            elif stat.S_ISDIR(ent.st_mode):
                rm_r(full)

    rm_r(remote_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["los", "nlos"], required=True)
    p.add_argument("--gpus", type=int, choices=[1, 2], default=2)
    args = p.parse_args()
    pw = os.environ.get("SSH_PASSWORD", "")
    if not pw:
        raise SystemExit("Set SSH_PASSWORD")

    root = Path(__file__).resolve().parent.parent
    local_try = root / LOCAL_DIR
    if not local_try.is_dir() or not LOCAL_H5.is_file():
        raise SystemExit("Missing try folder or HDF5")

    v = args.variant
    g = args.gpus
    slurm = f"cluster/run_twelfthtry12_{v}_{g}gpu.slurm"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=pw, timeout=30)
    try:
        sftp = client.open_sftp()
        upload_if_missing(sftp, LOCAL_H5, f"{REMOTE_BASE}/Datasets/CKM_Dataset_180326.h5")
        upload_csv_always(sftp, LOCAL_CSV, f"{REMOTE_BASE}/Datasets/CKM_180326_antenna_height.csv")
        rdir = f"{REMOTE_BASE}/{LOCAL_DIR}"
        mkdir_p(sftp, rdir)
        clean_remote_outputs(sftp, rdir)
        for lf, rel in collect_files(local_try):
            rf = f"{rdir}/{rel}"
            mkdir_p(sftp, os.path.dirname(rf).replace("\\", "/"))
            sftp.put(lf, rf)
        sftp.close()
        cmd = f"cd {rdir} && sbatch {slurm}"
        print(cmd)
        _, stdout, stderr = client.exec_command(cmd)
        print(stdout.read().decode().strip())
        e = stderr.read().decode().strip()
        if e:
            print("stderr:", e)
    finally:
        client.close()


if __name__ == "__main__":
    main()
