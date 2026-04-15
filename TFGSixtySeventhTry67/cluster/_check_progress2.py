import paramiko, os, json
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("sert.ac.upc.edu", username="gmoreno", password=os.environ["SSH_PASSWORD"], timeout=30)

out_dir = "/scratch/nas/3/gmoreno/TFGpractice/TFGSixtySixthTry66/outputs/try66_expert_open_sparse_lowrise"

# Find what json files exist
cmd = f"ls -1t {out_dir}/*.json 2>/dev/null | head -20"
_, o, e = c.exec_command(cmd, timeout=15)
print("=== JSON files ===")
print(o.read().decode().strip() or "(none)")
print()

# Also check the latest log for recent epoch metrics
cmd2 = f"grep '\"epoch\"' $(ls -t /scratch/nas/3/gmoreno/TFGpractice/TFGSixtySixthTry66/logs_train_try66_*.out 2>/dev/null | head -1) | tail -20"
_, o2, e2 = c.exec_command(cmd2, timeout=15)
print("=== Recent epochs from log ===")
lines = o2.read().decode().strip()
if lines:
    for line in lines.split("\n"):
        try:
            d = json.loads(line)
            ep = d.get("epoch", "?")
            rmse = d.get("path_loss.rmse_physical", "?")
            lr = d.get("learning_rate", "?")
            gloss = d.get("generator_loss", "?")
            print(f"  ep {ep}: RMSE={rmse}  LR={lr}  loss={gloss}")
        except:
            print(f"  {line[:120]}")
else:
    print("(none)")

c.close()
