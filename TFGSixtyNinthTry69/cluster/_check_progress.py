import paramiko, os, json
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("sert.ac.upc.edu", username="gmoreno", password=os.environ["SSH_PASSWORD"], timeout=30)

out_dir = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtySixthTry66/outputs/try66_expert_open_sparse_lowrise"
cmd = f"ls -1 {out_dir}/val_metrics_epoch_*.json 2>/dev/null | sort -t_ -k4 -n | tail -15"
_, o, e = c.exec_command(cmd, timeout=15)
files = o.read().decode().strip().split("\n")

for f in files:
    if not f.strip():
        continue
    _, o2, _ = c.exec_command(f"cat {f.strip()}", timeout=10)
    data = json.loads(o2.read().decode().strip())
    m = data.get("metrics", {}).get("path_loss", {})
    ep = data.get("epoch", "?")
    rmse = m.get("rmse_physical", 0)
    mae = m.get("mae_physical", 0)
    lr = data.get("learning_rate", data.get("lr", "?"))
    los = data.get("metrics", {}).get("los_rmse_physical", "?")
    nlos = data.get("metrics", {}).get("nlos_rmse_physical", "?")
    print(f"ep {ep:>4}  RMSE={rmse:7.3f}  MAE={mae:6.3f}  LR={lr}  LoS={los}  NLoS={nlos}")

c.close()
