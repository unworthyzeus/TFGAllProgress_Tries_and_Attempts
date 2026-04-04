from __future__ import annotations

import argparse
import os
import posixpath
import sys
from getpass import getpass
from pathlib import Path

import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload and submit Try49 prior-calibration job using Paramiko"
    )
    parser.add_argument("--host", default="sert.ac.upc.edu", help="SSH host")
    parser.add_argument("--username", default="gmoreno", help="SSH username")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument(
        "--password-env",
        default="SERT_PASSWORD",
        help="Environment variable that may contain SSH password",
    )
    parser.add_argument(
        "--remote-root",
        default="/scratch/nas/3/gmoreno/TFGpractice",
        help="Remote project root",
    )
    parser.add_argument(
        "--local-try49",
        default="TFGFortyNinthTry49",
        help="Local Try49 directory",
    )
    parser.add_argument(
        "--local-calibration-script",
        default="scripts/fit_formula_prior_obstruction_calibration.py",
        help="Local calibration script path",
    )
    parser.add_argument(
        "--remote-job-script",
        default="cluster/run_fortyninthtry49_prior_calibration_1gpu.slurm",
        help="Remote Try49 job script relative path",
    )
    parser.add_argument(
        "--cancel-job-id",
        default="",
        help="Optional old job id to cancel before submit",
    )
    parser.add_argument(
        "--no-upload-try49",
        action="store_true",
        help="Skip uploading full Try49 folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect and print operations without submitting",
    )
    return parser.parse_args()


def _password_from_env_or_prompt(env_name: str) -> str:
    value = os.environ.get(env_name, "")
    if value:
        return value
    return getpass("SSH password: ")


def _exec(ssh: paramiko.SSHClient, command: str, check: bool = True) -> tuple[int, str, str]:
    stdin, stdout, stderr = ssh.exec_command(command)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if check and code != 0:
        raise RuntimeError(f"Command failed ({code}): {command}\nSTDERR:\n{err}\nSTDOUT:\n{out}")
    return code, out, err


def _sftp_upload_dir(sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str) -> None:
    for root, dirs, files in os.walk(local_dir):
        rel = Path(root).relative_to(local_dir).as_posix()
        target_root = remote_dir if rel == "." else posixpath.join(remote_dir, rel)
        try:
            sftp.mkdir(target_root)
        except OSError:
            pass
        for d in dirs:
            d_remote = posixpath.join(target_root, d)
            try:
                sftp.mkdir(d_remote)
            except OSError:
                pass
        for file_name in files:
            lpath = Path(root) / file_name
            rpath = posixpath.join(target_root, file_name)
            sftp.put(str(lpath), rpath)


def main() -> int:
    args = parse_args()
    local_try49 = Path(args.local_try49).resolve()
    local_cal = Path(args.local_calibration_script).resolve()

    if not local_try49.exists():
        raise FileNotFoundError(f"Missing local Try49 folder: {local_try49}")
    if not local_cal.exists():
        raise FileNotFoundError(f"Missing local calibration script: {local_cal}")

    password = _password_from_env_or_prompt(args.password_env)

    remote_root = args.remote_root.rstrip("/")
    remote_try49 = f"{remote_root}/TFGFortyNinthTry49"
    remote_scripts = f"{remote_root}/scripts"
    remote_cal = f"{remote_scripts}/fit_formula_prior_obstruction_calibration.py"
    remote_job = f"{remote_try49}/{args.remote_job_script}"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"[INFO] Connecting to {args.username}@{args.host}:{args.port}")
        ssh.connect(
            hostname=args.host,
            port=args.port,
            username=args.username,
            password=password,
            look_for_keys=False,
            allow_agent=False,
            timeout=30,
        )
        _exec(ssh, f"mkdir -p {remote_try49} {remote_scripts}")

        sftp = ssh.open_sftp()
        try:
            if not args.no_upload_try49:
                print(f"[INFO] Uploading Try49 folder -> {remote_try49}")
                _sftp_upload_dir(sftp, local_try49, remote_try49)
            print(f"[INFO] Uploading calibration script -> {remote_cal}")
            sftp.put(str(local_cal), remote_cal)
        finally:
            sftp.close()

        # Ensure Slurm script line endings are Linux-compatible.
        _exec(ssh, f"sed -i 's/\\r$//' {remote_job}")

        if args.cancel_job_id:
            _exec(ssh, f"scancel {args.cancel_job_id} || true", check=False)

        if args.dry_run:
            print("[INFO] Dry run mode enabled, skipping sbatch")
            return 0

        code, out, _ = _exec(
            ssh,
            f"cd {remote_try49} && sbatch {args.remote_job_script}",
        )
        print(out.strip())

        _, queue_out, _ = _exec(ssh, f"squeue -u {args.username} | head -n 20", check=False)
        print("\n[INFO] Queue snapshot:\n" + queue_out.strip())
        return code
    finally:
        ssh.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user", file=sys.stderr)
        raise
