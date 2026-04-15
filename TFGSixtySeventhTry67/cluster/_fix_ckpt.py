import paramiko, os
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("sert.ac.upc.edu", username="gmoreno", password=os.environ["SSH_PASSWORD"], timeout=30)
cmds = [
    "rm -f /scratch/nas/3/gmoreno/TFGpractice/TFGSixtySixthTry66/outputs/try66_expert_open_sparse_lowrise/epoch_73_model.pt",
    "ls -lhrt /scratch/nas/3/gmoreno/TFGpractice/TFGSixtySixthTry66/outputs/try66_expert_open_sparse_lowrise/*.pt 2>/dev/null",
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd, timeout=15)
    print(f"=== {cmd[:80]} ===")
    print(o.read().decode().strip() or e.read().decode().strip() or "(done)")
    print()
c.close()
