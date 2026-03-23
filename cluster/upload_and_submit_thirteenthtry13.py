#!/usr/bin/env python3
"""Upload TFGThirteenthTry13 (un solo modelo FiLM, dataset completo) al cluster.

YAML: experiments/thirteenthtry13_film/thirteenthtry13_film.yaml
Slurm: cluster/run_thirteenthtry13_film_1gpu.slurm

Set SSH_PASSWORD.

Por defecto: sube y hace sbatch una vez. --upload-only solo sincroniza.
"""
from __future__ import annotations

import argparse
import io
import os
import stat
import sys
from pathlib import Path

_CLUSTER_DIR = Path(__file__).resolve().parent
if str(_CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLUSTER_DIR))
import upload_dataset_helpers as udh

try:
    import paramiko
except ImportError:
    os.system(f"{sys.executable} -m pip install paramiko -q")
    import paramiko

HOST = "sert.ac.upc.edu"
USER = "gmoreno"
REMOTE_BASE = "/scratch/nas/3/gmoreno/TFGpractice"
LOCAL_DIR = "TFGThirteenthTry13"
EXCLUDE_DIRS = {"outputs", "__pycache__", ".git", ".venv", ".cursor"}
EXCLUDE_EXTS = {".h5", ".pt", ".pth", ".pyc", ".pyo"}


def mkdir_p(sftp, remote_path: str) -> None:
    udh.mkdir_p_with_quota_hint(sftp, remote_path)


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


def sftp_put_slurm_lf(sftp, local_path: str, remote_path: str, rel: str) -> None:
    if rel.replace("\\", "/").lower().endswith(".slurm"):
        with open(local_path, "rb") as f:
            blob = f.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        sftp.putfo(io.BytesIO(blob), remote_path)
    else:
        sftp.put(local_path, remote_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upload-only", action="store_true", help="Sync only; no sbatch.")
    p.add_argument(
        "--no-clean-outputs",
        action="store_true",
        help="Do not delete remote TFGThirteenthTry13/outputs before upload.",
    )
    args = p.parse_args()
    pw = os.environ.get("SSH_PASSWORD", "")
    if not pw:
        raise SystemExit("Set environment variable SSH_PASSWORD")

    slurm_jobs: list[str] = []
    if not args.upload_only:
        slurm_jobs = ["cluster/run_thirteenthtry13_film_1gpu.slurm"]

    root = Path(__file__).resolve().parent.parent
    local_try = root / LOCAL_DIR
    if not local_try.is_dir():
        raise SystemExit(f"Missing folder {local_try}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=pw, timeout=30)
    udh.prepare_ssh_transport_for_large_uploads(client)
    try:
        sftp = client.open_sftp()
        # FiLM 13: solo el HDF5 con alturas (mapas + escalares); no subir CKM_Dataset_180326.h5 aquí.
        ant_loc = udh.resolve_antenna_height_h5(root)
        if ant_loc:
            print(f"Dataset upload (FiLM): {ant_loc.name} -> .../Datasets/{udh.REMOTE_ANTENNA_H5_NAME}")
            udh.upload_if_missing_file(sftp, mkdir_p, ant_loc, udh.remote_antenna_h5_path(REMOTE_BASE))
        else:
            print(
                f"Warning: local antenna HDF5 not found (tried {udh.LOCAL_ANTENNA_H5_CANDIDATES} under Datasets/). "
                f"Put CKM_180326_antenna_height.h5 there, or ensure cluster already has {udh.REMOTE_ANTENNA_H5_NAME}."
            )
        rdir = f"{REMOTE_BASE}/{LOCAL_DIR}"
        mkdir_p(sftp, rdir)
        if not args.no_clean_outputs:
            print(f"Cleaning remote outputs/ under {LOCAL_DIR} (if any)...")
            clean_remote_outputs(sftp, rdir)
        files = collect_files(local_try)
        print(f"Syncing {len(files)} files under {LOCAL_DIR}/ ...")
        for i, (lf, rel) in enumerate(files, start=1):
            rf = f"{rdir}/{rel}"
            mkdir_p(sftp, os.path.dirname(rf).replace("\\", "/"))
            sftp_put_slurm_lf(sftp, lf, rf, rel)
            if i == 1 or i == len(files) or i % 40 == 0:
                print(f"  ... {i}/{len(files)} {rel}")
        sftp.close()
        for slurm in slurm_jobs:
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
