#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import paramiko
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko


ROOT = Path(__file__).resolve().parents[2]
HOST = "sert.ac.upc.edu"
USER = "gmoreno"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGFiftyThirdTry53"
TARGET_NODE = "sert-2001"


def parse_job_id(text: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", text)
    if not match:
        raise RuntimeError(f"Could not parse job id from: {text!r}")
    return match.group(1)


def remote_exec(client: paramiko.SSHClient, command: str, *, check: bool = True) -> tuple[str, str]:
    print("REMOTE>", command)
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err:
        print(err)
    if check and err:
        raise RuntimeError(err)
    return out, err


def remote_sbatch(client: paramiko.SSHClient, command: str) -> str:
    out, err = remote_exec(client, f"cd {REMOTE_DIR} && {command}")
    if err:
        raise RuntimeError(err)
    return parse_job_id(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace all pending Try53 jobs after a running stage3 with new 55m/1h25m 4-GPU scripts.")
    parser.add_argument("--after-job", required=True, help="Running stage3 job id to continue after")
    parser.add_argument("--chain-repeats", type=int, default=3)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password and not args.ssh_key:
        raise SystemExit(f"Set environment variable {args.password_env} or pass --ssh-key")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = {
        "hostname": args.host,
        "username": args.user,
        "timeout": 30,
        "allow_agent": True,
        "look_for_keys": True,
    }
    if password:
        connect_kwargs["password"] = password
    if args.ssh_key:
        connect_kwargs["key_filename"] = args.ssh_key
    client.connect(**connect_kwargs)
    try:
        remote_exec(
            client,
            f"""python - <<'PY'\nimport subprocess\nlines=subprocess.check_output(['squeue','-u','{args.user}','-h','-o','%i %t']).decode().splitlines()\nfor line in lines:\n    jid, st = line.split()\n    if jid != '{args.after_job}' and st == 'PD':\n        subprocess.run(['scancel', jid], check=False)\nPY""",
            check=False,
        )

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        prev = args.after_job
        remaining_repeats = max(args.chain_repeats, 1)

        for repeat_index in range(remaining_repeats):
            if repeat_index == 0:
                cleanup0 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{prev} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                stage1_feedback = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{cleanup0} cluster/run_fiftythirdtry53_stage1_feedback_4gpu.slurm")
                cleanup1 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{stage1_feedback} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                stage2_b = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{cleanup1} cluster/run_fiftythirdtry53_stage2_4gpu.slurm")
                cleanup2 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{stage2_b} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                stage3_b = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{cleanup2} cluster/run_fiftythirdtry53_stage3_4gpu.slurm")
                cleanup3 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{stage3_b} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                prev = cleanup3
            else:
                cleanup0 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{prev} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                stage1_bootstrap = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{cleanup0} cluster/run_fiftythirdtry53_stage1_bootstrap_4gpu.slurm")
                cleanup1 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{stage1_bootstrap} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                stage2_a = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{cleanup1} cluster/run_fiftythirdtry53_stage2_4gpu.slurm")
                cleanup2 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{stage2_a} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                stage3_a = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{cleanup2} cluster/run_fiftythirdtry53_stage3_4gpu.slurm")
                cleanup3 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{stage3_a} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                stage1_feedback = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{cleanup3} cluster/run_fiftythirdtry53_stage1_feedback_4gpu.slurm")
                cleanup4 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{stage1_feedback} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                stage2_b = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{cleanup4} cluster/run_fiftythirdtry53_stage2_4gpu.slurm")
                cleanup5 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{stage2_b} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                stage3_b = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{cleanup5} cluster/run_fiftythirdtry53_stage3_4gpu.slurm")
                cleanup6 = remote_sbatch(client, f"{sbatch_prefix} --dependency=afterany:{stage3_b} cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm")
                prev = cleanup6
    finally:
        client.close()


if __name__ == "__main__":
    main()
