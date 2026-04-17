"""
Push the 3 corrected YAMLs (base_channels 48→40, hf_channels 20→16) to the cluster.
Run from repo root or TFGSeventyThirdTry73/.
"""
import paramiko
import pathlib

SSH_HOST = "sert.ac.upc.edu"
SSH_USER = "gmoreno"
SSH_PASS = "Slenderman,2004"

CLUSTER_EXPERIMENTS = (
    "/scratch/nas/3/gmoreno/TFGpractice/TFGSeventyThirdTry73/experiments/seventythird_try73_experts"
)

HERE = pathlib.Path(__file__).parent.parent  # TFGSeventyThirdTry73/
LOCAL_YAMLS_DIR = HERE / "experiments" / "seventythird_try73_experts"

EXPERTS_TO_FIX = [
    "try73_expert_mixed_compact_midrise.yaml",
    "try73_expert_dense_block_midrise.yaml",
    "try73_expert_dense_block_highrise.yaml",
]

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {SSH_HOST}...")
    client.connect(SSH_HOST, username=SSH_USER, password=SSH_PASS, timeout=30)
    sftp = client.open_sftp()

    for fname in EXPERTS_TO_FIX:
        local_path = LOCAL_YAMLS_DIR / fname
        remote_path = f"{CLUSTER_EXPERIMENTS}/{fname}"
        print(f"  Uploading {fname} ...")
        sftp.put(str(local_path), remote_path)
        print(f"    -> {remote_path}  OK")

    sftp.close()

    # Quick verify: print base_channels from each uploaded file
    print("\nVerifying on cluster:")
    for fname in EXPERTS_TO_FIX:
        remote_path = f"{CLUSTER_EXPERIMENTS}/{fname}"
        _, stdout, _ = client.exec_command(
            f"grep -E 'base_channels|hf_channels' {remote_path}"
        )
        lines = stdout.read().decode().strip()
        print(f"  {fname}:\n    {lines}")

    client.close()
    print("\nDone. Re-submit the 3 jobs so they pick up the new architecture.")

if __name__ == "__main__":
    main()
