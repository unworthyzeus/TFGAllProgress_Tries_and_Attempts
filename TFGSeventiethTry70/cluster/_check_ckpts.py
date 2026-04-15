import paramiko, os
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("sert.ac.upc.edu", username="gmoreno", password=os.environ["SSH_PASSWORD"], timeout=30)
cmd = "ls -lhrt /scratch/nas/3/gmoreno/TFGpractice/TFGSixtyEighthTry68/outputs/try68_expert_open_sparse_lowrise/*.pt 2>/dev/null"
_, o, e = c.exec_command(cmd, timeout=15)
print(o.read().decode().strip() or e.read().decode().strip() or "(empty)")
c.close()
