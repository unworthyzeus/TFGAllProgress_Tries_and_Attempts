import paramiko, os
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("sert.ac.upc.edu", username="gmoreno", password=os.environ["SSH_PASSWORD"], timeout=30)
cmds = [
    "squeue -w sert-2001",
    "scontrol show node sert-2001 2>/dev/null | grep -iE 'AllocTRES|Gres'",
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd)
    print(f"=== {cmd} ===")
    print(o.read().decode().strip() or e.read().decode().strip())
    print()
c.close()
