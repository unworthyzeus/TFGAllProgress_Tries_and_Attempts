import paramiko, os, json
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("sert.ac.upc.edu", username="gmoreno", password=os.environ["SSH_PASSWORD"], timeout=30)

out_dir = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68/outputs/try68_expert_open_sparse_lowrise"

# Get train progress (has LR)
_, o, _ = c.exec_command(f"cat {out_dir}/train_progress_latest.json", timeout=10)
prog = json.loads(o.read().decode().strip())
print(f"=== Train progress (epoch {prog.get('epoch')}, step {prog.get('step')}/{prog.get('total_steps')}) ===")
print(f"  LR = {prog.get('learning_rate')}")
print(f"  train RMSE = {prog.get('train_rmse_physical_running'):.3f}")
print(f"  loss = {prog.get('generator_loss_running'):.6f}")
print()

# Get val metrics for epochs 73-76
for ep in [72, 73, 74, 75, 76]:
    _, o, _ = c.exec_command(f"cat {out_dir}/validate_metrics_epoch_{ep}.json 2>/dev/null", timeout=10)
    raw = o.read().decode().strip()
    if not raw:
        continue
    d = json.loads(raw)
    m = d.get("metrics", {}).get("path_loss", {})
    lr = d.get("learning_rate", "?")
    rmse = m.get("rmse_physical", 0)
    mae = m.get("mae_physical", 0)
    los = d.get("metrics", {}).get("los_rmse_physical", "?")
    nlos = d.get("metrics", {}).get("nlos_rmse_physical", "?")
    print(f"ep {ep:>3}  RMSE={rmse:7.3f}  MAE={mae:6.3f}  LR={lr}  LoS={los}  NLoS={nlos}")

c.close()
