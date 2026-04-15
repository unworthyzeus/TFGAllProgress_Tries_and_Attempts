import paramiko, os
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("sert.ac.upc.edu", username="gmoreno", password=os.environ["SSH_PASSWORD"], timeout=30)
cmds = [
    "grep -E 'learning_rate|resume|fast-forward|Scheduler|start_epoch' /scratch/nas/3/gmoreno/TFGpractice/TFGSeventySecondTry72/logs_train_try72_t72-open-sparse-lowrise-r1_10028617.out | tail -30",
    "grep 'learning_rate' /scratch/nas/3/gmoreno/TFGpractice/TFGSeventySecondTry72/logs_train_try72_t72-open-sparse-lowrise-r1_10028617.out | tail -5",
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd, timeout=15)
    print(f"=== {cmd[:80]} ===")
    print(o.read().decode().strip() or e.read().decode().strip() or "(empty)")
    print()
c.close()
