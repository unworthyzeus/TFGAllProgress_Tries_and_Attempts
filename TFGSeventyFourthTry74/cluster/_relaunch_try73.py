"""Cancel all jobs, wipe Try 73 logs (keep checkpoints), upload code, and relaunch all experts."""
from __future__ import annotations
import os, sys, re, time
from pathlib import Path

try:
    import paramiko
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "paramiko"])
    import paramiko

HOST = "sert.ac.upc.edu"
USER = "gmoreno"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGSeventyThirdTry73"
LOCAL_DIR = Path(__file__).resolve().parents[1]
TARGET_NODE = "sert-2001"


def run(client, cmd: str, timeout: int = 30, check: bool = False) -> str:
    print(f"  > {cmd}")
    _, out, err = client.exec_command(cmd, timeout=timeout)
    o = out.read().decode("utf-8", errors="replace").strip()
    e = err.read().decode("utf-8", errors="replace").strip()
    result = (o + "\n" + e).strip()
    if result:
        for line in result.split("\n")[:15]:
            print(f"    {line}")
    return o


def upload_file_unix(sftp, local_path: Path, remote_path: str) -> None:
    """Upload a file, converting line endings to Unix."""
    content = local_path.read_bytes()
    content = content.replace(b"\r\n", b"\n")
    with sftp.file(remote_path, "wb") as f:
        f.write(content)


def upload_dir(sftp, local: Path, remote: str, exclude: set[str] | None = None) -> int:
    exclude = exclude or set()
    count = 0
    for item in sorted(local.iterdir()):
        if item.name in exclude or item.name.startswith(".") or item.name == "__pycache__":
            continue
        remote_path = f"{remote}/{item.name}"
        if item.is_dir():
            try:
                sftp.stat(remote_path)
            except FileNotFoundError:
                sftp.mkdir(remote_path)
            count += upload_dir(sftp, item, remote_path, exclude)
        elif item.is_file() and item.stat().st_size < 5_000_000:
            upload_file_unix(sftp, item, remote_path)
            count += 1
    return count


def parse_job_id(text: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", text)
    if not match:
        raise RuntimeError(f"Could not parse job id from: {text!r}")
    return match.group(1)


def main() -> None:
    pw = os.environ.get("SSH_PASSWORD", "")
    if not pw:
        print("Set SSH_PASSWORD"); sys.exit(2)

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=pw, timeout=45)

    # 1. Cancel all user jobs
    print("\n=== Step 1: Cancel all jobs ===")
    run(c, f"scancel -u {USER}")
    time.sleep(3)
    run(c, f"squeue -u {USER}")

    # 2. Wipe logs only (keep checkpoints for resume)
    print("\n=== Step 2: Wipe old logs (keeping checkpoints) ===")
    run(
        c,
        f"rm -f {REMOTE_DIR}/logs_train_try73_*.out {REMOTE_DIR}/logs_train_try73_*.err "
        f"{REMOTE_DIR}/logs_train_try73_chain_*.out {REMOTE_DIR}/logs_train_try73_chain_*.err",
    )

    # 3. Upload code
    print("\n=== Step 3: Upload code ===")
    run(c, f"mkdir -p {REMOTE_DIR}/experiments/seventythird_try73_experts")
    run(c, f"mkdir -p {REMOTE_DIR}/experiments/seventythird_try73_classifier")
    run(c, f"mkdir -p {REMOTE_DIR}/cluster")
    run(c, f"mkdir -p {REMOTE_DIR}/scripts")
    run(c, f"mkdir -p {REMOTE_DIR}/prior_calibration")

    sftp = c.open_sftp()
    exclude = {"outputs", "prior_cache", "precomputed", "cluster_outputs", "__pycache__", ".git", "Datasets"}
    n = upload_dir(sftp, LOCAL_DIR, REMOTE_DIR, exclude)
    sftp.close()
    print(f"  Uploaded {n} files")

    # 4. Submit open_sparse_lowrise 3× chained (each 48h, 2 GPUs, auto-resume).
    print("\n=== Step 4: Submit open_sparse_lowrise x3 (48h each, 2 GPUs) ===")
    config = "experiments/seventythird_try73_experts/try73_expert_open_sparse_lowrise.yaml"
    job_name = "t73-open-sparse-lowrise"
    cleanup_script = "cluster/run_seventythird_try73_cleanup_sert2001_1gpu.slurm"
    submitted = []
    current_dep = ""

    for run_idx in range(3):
        exports = [
            f"CONFIG_PATH={config}",
            "TRAIN_SCRIPT=train_partitioned_pathloss_expert.py",
            "MASTER_PORT=30166",
        ]
        cmd = (
            f"cd {REMOTE_DIR} && "
            f"sbatch --nodelist={TARGET_NODE} {current_dep}"
            f"-J {job_name}-r{run_idx + 1} "
            f"--export=ALL,{','.join(exports)} "
            "cluster/run_seventythird_try73_3gpu.slurm"
        )
        out = run(c, cmd)
        job_id = parse_job_id(out)
        cleanup_out = run(
            c,
            f"cd {REMOTE_DIR} && sbatch --nodelist={TARGET_NODE} "
            f"--dependency=afterany:{job_id} {cleanup_script}",
        )
        cleanup_id = parse_job_id(cleanup_out)
        submitted.append({"run": run_idx + 1, "job_id": job_id, "cleanup_id": cleanup_id})
        current_dep = f"--dependency=afterany:{cleanup_id} "

    print(f"\n=== Submitted {len(submitted)} chained jobs ===")
    for s in submitted:
        print(f"  run {s['run']}: train={s['job_id']} -> cleanup={s['cleanup_id']}")

    # 5. Verify
    print("\n=== Final queue ===")
    run(c, f"squeue -u {USER}")

    c.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
