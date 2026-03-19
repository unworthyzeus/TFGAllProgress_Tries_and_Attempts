#!/usr/bin/env python3
"""
Upload TFGFifthTry5 to cluster (excluding dataset, outputs) and submit sbatch via Paramiko.
Dataset on cluster: /scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_old_and_small.h5 (shared).
Usage: python cluster/upload_and_submit.py
"""
import os
import stat
import sys

try:
    import paramiko
except ImportError:
    print("Installing paramiko...")
    os.system(f"{sys.executable} -m pip install paramiko -q")
    import paramiko

# Config
HOST = "sert.ac.upc.edu"
USER = "gmoreno"
PASSWORD = "Slenderman,2004"
REMOTE_BASE = "/scratch/nas/3/gmoreno/TFGpractice"
LOCAL_DIR = "TFGFifthTry5"
# Shared dataset on cluster - do NOT upload
HDF5_CLUSTER_PATH = "/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_old_and_small.h5"

# Exclude from upload: dataset, outputs, caches, checkpoints
EXCLUDE_NAMES = {
    "CKM_Dataset.h5",
    "outputs",
    "__pycache__",
    ".git",
    ".venv",
}
EXCLUDE_EXT = {".h5", ".pt", ".pyc", ".pyo"}


def should_skip(_path: str, name: str) -> bool:
    if name in EXCLUDE_NAMES:
        return True
    _, ext = os.path.splitext(name)
    if ext.lower() in EXCLUDE_EXT:
        return True
    return False


def _rm_r(sftp, path: str, dirs_to_remove: set, exts_to_remove: set) -> None:
    """Recursively remove dirs in dirs_to_remove and files with exts in exts_to_remove."""
    try:
        entries = sftp.listdir_attr(path)
    except FileNotFoundError:
        return
    for ent in entries:
        full = f"{path}/{ent.filename}"
        if stat.S_ISDIR(ent.st_mode):
            if ent.filename in dirs_to_remove:
                _rm_r(sftp, full, dirs_to_remove, exts_to_remove)
                try:
                    sftp.rmdir(full)
                except OSError:
                    pass
            else:
                _rm_r(sftp, full, dirs_to_remove, exts_to_remove)
        else:
            ext = os.path.splitext(ent.filename)[1].lower()
            if ext in exts_to_remove or ent.filename == "CKM_Dataset.h5":
                try:
                    sftp.remove(full)
                except OSError:
                    pass


def collect_files(local_root: str) -> list[tuple[str, str]]:
    """Returns [(local_path, remote_rel_path), ...]"""
    out = []
    for root, dirs, files in os.walk(local_root):
        dirs[:] = [d for d in dirs if not should_skip(root, d)]
        rel_root = os.path.relpath(root, local_root)
        for f in files:
            if should_skip(root, f):
                continue
            local = os.path.join(root, f)
            rel = os.path.join(rel_root, f).replace("\\", "/")
            out.append((local, rel))
    return out


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    local_path = os.path.join(project_root, LOCAL_DIR)
    if not os.path.isdir(local_path):
        local_path = os.path.join(os.getcwd(), LOCAL_DIR)
    if not os.path.isdir(local_path):
        print(f"ERROR: {LOCAL_DIR} not found")
        sys.exit(1)

    files = collect_files(local_path)
    print(f"Uploading {len(files)} files (excluding dataset, outputs, .h5, .pt)...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    try:
        sftp = client.open_sftp()
        remote_dir = f"{REMOTE_BASE}/{LOCAL_DIR}"

        # Remove previously uploaded trash (outputs, dataset, caches, checkpoints)
        print("Removing remote trash (outputs, CKM_Dataset.h5, __pycache__, *.pt, *.h5)...")
        _rm_r(sftp, remote_dir, {"outputs", "__pycache__"}, {".pt", ".h5", ".pyc", ".pyo"})

        # Ensure remote dir exists
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            sftp.mkdir(remote_dir)

        for i, (local_full, rel_path) in enumerate(files):
            remote_full = f"{remote_dir}/{rel_path}"
            remote_parent = os.path.dirname(remote_full)
            parts = rel_path.split("/")
            for j in range(1, len(parts)):
                sub = "/".join(parts[:j])
                rsub = f"{remote_dir}/{sub}"
                try:
                    sftp.stat(rsub)
                except FileNotFoundError:
                    sftp.mkdir(rsub)

            sftp.put(local_full, remote_full)
            if (i + 1) % 20 == 0 or i + 1 == len(files):
                print(f"  {i + 1}/{len(files)}")

        sftp.close()
        print("Upload done.")

        # Submit
        cmd = f"cd {remote_dir} && sbatch cluster/run_fifthtry5_2gpu.slurm"
        print(f"Running: {cmd}")
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode()
        err = stderr.read().decode()
        if out:
            print(out.strip())
        if err:
            print("stderr:", err.strip())
    finally:
        client.close()

    print("Done.")


if __name__ == "__main__":
    main()
