#!/usr/bin/env python3
"""Upload Try 53 materials to RunPod over direct TCP SSH/SFTP.

This replaces the older PTY/base64 uploader that talked to the RunPod SSH proxy.
The new pod exposes a direct TCP SSH endpoint, so we can use Paramiko SFTP
reliably for tarball upload + remote extraction.

Default remote layout:
  /workspace/TFGpractice/<repo folders>
  /workspace/TFGpractice/TFGFiftyFirstTry51/outputs (from cluster_outputs)

To preserve absolute paths used by configs, this script also creates:
  /scratch/nas/3/gmoreno/TFGpractice -> /workspace/TFGpractice
"""

from __future__ import annotations

import argparse
import os
import posixpath
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import paramiko
except ImportError:
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko


HOST = "38.65.239.14"
PORT = 35484
USER = "root"
REMOTE_BASE = "/workspace/TFGpractice"
REMOTE_SCRATCH_BASE = "/scratch/nas/3/gmoreno/TFGpractice"


@dataclass(frozen=True)
class UploadSpec:
    key: str
    local_rel: str
    remote_rel: str
    kind: str


UPLOAD_SPECS: dict[str, UploadSpec] = {
    "datasets": UploadSpec("datasets", "Datasets", "Datasets", "data"),
    "try51_stage1_bootstrap": UploadSpec(
        "try51_stage1_bootstrap",
        "cluster_outputs/TFGFiftyFirstTry51/fiftyfirsttry51_pmnet_prior_stage1_literature_t51_stage1_w112_4gpu",
        "TFGFiftyFirstTry51/outputs/fiftyfirsttry51_pmnet_prior_stage1_literature_t51_stage1_w112_4gpu",
        "outputs",
    ),
    "try53": UploadSpec("try53", "TFGFiftyThirdTry53", "TFGFiftyThirdTry53", "code"),
    "cluster": UploadSpec("cluster", "cluster", "cluster", "code"),
}

DEFAULT_ORDER = ("datasets", "try51_stage1_bootstrap", "try53")
COMMON_SKIP_DIRS = {"__pycache__", ".git", ".venv", ".cursor", ".ipynb_checkpoints"}
CODE_SKIP_EXTS = {".h5", ".pt", ".pth", ".pyc", ".pyo"}
OUTPUT_SKIP_EXTS = {".pyc", ".pyo"}


def should_skip_dir(name: str, kind: str) -> bool:
    if name in COMMON_SKIP_DIRS:
        return True
    if kind == "code" and name == "outputs":
        return True
    return False


def should_skip_file(name: str, kind: str) -> bool:
    suffix = Path(name).suffix.lower()
    if kind == "code":
        return suffix in CODE_SKIP_EXTS
    if kind == "outputs":
        return suffix in OUTPUT_SKIP_EXTS
    return suffix in {".pyc", ".pyo"}


def resolve_local_root(repo_root: Path, local_rel: str) -> Path:
    candidate = repo_root / local_rel
    if not candidate.exists():
        raise SystemExit(f"Missing local folder: {candidate}")
    return candidate


def count_files(local_root: Path, kind: str) -> int:
    total = 0
    for root, dirs, filenames in os.walk(local_root):
        dirs[:] = [directory for directory in dirs if not should_skip_dir(directory, kind)]
        for filename in filenames:
            if should_skip_file(filename, kind):
                continue
            total += 1
    return total


def build_tarball(local_root: Path, archive_prefix: str, kind: str, temp_dir: Path) -> Path:
    safe_prefix = archive_prefix.strip("/").replace("/", "_")
    tar_path = temp_dir / f"{safe_prefix}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        for root, dirs, filenames in os.walk(local_root):
            dirs[:] = [directory for directory in dirs if not should_skip_dir(directory, kind)]
            for filename in filenames:
                if should_skip_file(filename, kind):
                    continue
                local_path = Path(root) / filename
                rel_path = os.path.relpath(local_path, local_root).replace("\\", "/")
                archive.add(str(local_path), arcname=f"{archive_prefix}/{rel_path}")
    return tar_path


def parse_items(raw_items: str) -> Sequence[str]:
    items = [item.strip() for item in raw_items.split(",") if item.strip()]
    if not items:
        raise SystemExit("No upload items selected.")
    unknown = [item for item in items if item not in UPLOAD_SPECS]
    if unknown:
        known = ", ".join(sorted(UPLOAD_SPECS))
        raise SystemExit(f"Unknown upload items: {', '.join(unknown)}. Known items: {known}")
    return items


def mkdir_p_sftp(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    remote_dir = posixpath.normpath(remote_dir)
    parts = remote_dir.split("/")
    current = "/"
    for part in parts:
        if not part:
            continue
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def remote_exec(client: paramiko.SSHClient, command: str) -> None:
    print("REMOTE>", command)
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err:
        raise RuntimeError(err)


def load_pkey(ssh_key: str) -> paramiko.PKey:
    key_path = Path(ssh_key).expanduser()
    errors: list[str] = []
    key_classes = [
        getattr(paramiko, "Ed25519Key", None),
        getattr(paramiko, "RSAKey", None),
        getattr(paramiko, "ECDSAKey", None),
        getattr(paramiko, "DSSKey", None),
    ]
    for cls in [cls for cls in key_classes if cls is not None]:
        try:
            return cls.from_private_key_file(str(key_path))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cls.__name__}: {exc}")
    joined = "; ".join(errors)
    raise SystemExit(f"Could not load SSH key {key_path}: {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Try 53 materials to RunPod via SFTP.")
    parser.add_argument(
        "--ssh-key",
        default=str(Path(__file__).resolve().parent.parent / "runpod_ssh"),
        help="Path to the private SSH key used by RunPod.",
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--remote-base", default=REMOTE_BASE)
    parser.add_argument("--scratch-base", default=REMOTE_SCRATCH_BASE)
    parser.add_argument(
        "--items",
        default=",".join(DEFAULT_ORDER),
        help="Comma-separated upload item keys.",
    )
    parser.add_argument(
        "--no-link-scratch",
        action="store_true",
        help="Do not create the /scratch -> /workspace symlink expected by configs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    selected_items = parse_items(args.items)

    print("RunPod upload plan:")
    for key in selected_items:
        spec = UPLOAD_SPECS[key]
        local_root = resolve_local_root(repo_root, spec.local_rel)
        remote_root = f"{args.remote_base.rstrip('/')}/{spec.remote_rel}"
        file_count = count_files(local_root, spec.kind)
        print(f"  {key}: {spec.local_rel} -> {remote_root} ({file_count} files)")

    temp_dir = Path(tempfile.mkdtemp(prefix="runpod_try53_upload_"))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = load_pkey(args.ssh_key)
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        pkey=pkey,
        timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    sftp = client.open_sftp()

    try:
        remote_exec(client, f"mkdir -p {args.remote_base}")
        for key in selected_items:
            spec = UPLOAD_SPECS[key]
            local_root = resolve_local_root(repo_root, spec.local_rel)
            remote_root = f"{args.remote_base.rstrip('/')}/{spec.remote_rel}"
            remote_parent = posixpath.dirname(remote_root.rstrip("/"))
            mkdir_p_sftp(sftp, remote_parent)
            tar_path = build_tarball(local_root, spec.remote_rel, spec.kind, temp_dir)
            remote_tmp = f"/tmp/{Path(tar_path).name}"
            print(f"Uploading {key} -> {remote_root}")
            sftp.put(str(tar_path), remote_tmp)
            remote_exec(
                client,
                f"rm -rf {remote_root} && mkdir -p {remote_parent} && tar -xzf {remote_tmp} -C {remote_parent} && rm -f {remote_tmp}",
            )

        if not args.no_link_scratch:
            scratch_parent = posixpath.dirname(args.scratch_base.rstrip("/"))
            remote_exec(client, f"mkdir -p {scratch_parent} && ln -sfn {args.remote_base} {args.scratch_base}")
    finally:
        try:
            sftp.close()
        except Exception:
            pass
        client.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("RunPod upload complete.")


if __name__ == "__main__":
    main()
