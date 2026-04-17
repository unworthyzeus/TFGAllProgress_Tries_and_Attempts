import paramiko, os
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("sert.ac.upc.edu", username="gmoreno", password=os.environ["SSH_PASSWORD"], timeout=30)
cmds = [
    "head -200 /scratch/nas/3/gmoreno/TFGpractice/TFGSeventyThirdTry73/logs_train_try73_t73-open-sparse-lowrise-r1_10028617.out",
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd, timeout=15)
    print(o.read().decode().strip() or e.read().decode().strip() or "(empty)")
c.close()
