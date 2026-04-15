"""Fetch all epoch metrics for open_sparse_lowrise and current status."""
from __future__ import annotations
import os, sys, json
try:
    import paramiko
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "paramiko"])
    import paramiko

HOST = "sert.ac.upc.edu"
USER = "gmoreno"
BASE = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68"

def r(c, cmd, timeout=25):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return (o.read() + e.read()).decode("utf-8", "replace").strip()

def main():
    pw = os.environ.get("SSH_PASSWORD", "")
    if not pw:
        print("Set SSH_PASSWORD"); sys.exit(2)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=pw, timeout=45)

    print("=== QUEUE ===")
    print(r(c, "squeue -u gmoreno"))
    print()

    print("=== OUT LOG ===")
    out = r(c, f"cat {BASE}/logs_train_try68_t68-open-sparse-lowrise_10027953.out 2>/dev/null")
    print(out[:4000])
    print()

    expert_dir = f"{BASE}/outputs/try68_expert_open_sparse_lowrise"
    files_raw = r(c, f"ls {expert_dir}/validate_metrics_epoch_*.json 2>/dev/null")
    if not files_raw:
        print("No validation files found yet")
        c.close()
        return

    files = sorted(files_raw.strip().split("\n"))
    print(f"=== EPOCH METRICS ({len(files)} epochs) ===")
    print(f"{'ep':>4} {'val_rmse':>9} {'train_rmse':>11} {'prior_rmse':>11} {'gain':>7} {'lr':>10} {'los_rmse':>9} {'nlos_rmse':>10}")
    for f in files:
        raw = r(c, f"cat '{f}' 2>/dev/null")
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        m = d["metrics"]
        ep = d["checkpoint"]["epoch"]
        vr = m["path_loss"]["rmse_physical"]
        tr = m.get("train_path_loss", {}).get("rmse_physical", 0)
        pr = m["prior_path_loss"]["rmse_physical"]
        gain = m["improvement_vs_prior"]["rmse_gain_db"]
        lr = d.get("runtime", {}).get("learning_rate", 0)
        los = d.get("focus", {}).get("regimes", {}).get("path_loss__los__LoS", {}).get("rmse_physical", 0)
        nlos = d.get("focus", {}).get("regimes", {}).get("path_loss__los__NLoS", {}).get("rmse_physical", 0)
        print(f"{ep:4d} {vr:9.2f} {tr:11.2f} {pr:11.2f} {gain:+7.2f} {lr:10.6f} {los:9.2f} {nlos:10.2f}")

    print()
    print("=== LATEST PROGRESS ===")
    prog = r(c, f"cat {expert_dir}/train_progress_latest.json 2>/dev/null")
    print(prog[:2000] if prog else "(none)")

    c.close()

if __name__ == "__main__":
    main()
