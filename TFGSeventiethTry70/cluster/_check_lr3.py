import paramiko, os
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("sert.ac.upc.edu", username="gmoreno", password=os.environ["SSH_PASSWORD"], timeout=30)
cmds = [
    "ls -t /scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68/logs_train_try68_*.out 2>/dev/null | head -3",
    "head -30 $(ls -t /scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68/logs_train_try68_*.out 2>/dev/null | head -1)",
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd, timeout=15)
    print(f"=== {cmd[:80]} ===")
    print(o.read().decode().strip() or e.read().decode().strip() or "(empty)")
    print()
c.close()
