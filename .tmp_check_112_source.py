from __future__ import annotations

import paramiko

host = 'sert.ac.upc.edu'
user = 'gmoreno'
password = 'Slenderman,2004'
commands = [
    "squeue -u gmoreno -o '%i %t %M %l %R'",
    "echo '---'",
    "ls -lah /scratch/nas/3/gmoreno/TFGpractice/TFGFortyNinthTry49/outputs/fortyninthtry49_pmnet_prior_stage1_t49_stage1_w128_4gpu 2>/dev/null || true",
    "echo '---EPOCH32---'",
    "ls -lah /scratch/nas/3/gmoreno/TFGpractice/TFGFortyNinthTry49/outputs/fortyninthtry49_pmnet_prior_stage1_t49_stage1_w128_4gpu/epoch_32_cgan.pt 2>/dev/null || true",
    "echo '---BEST---'",
    "ls -lah /scratch/nas/3/gmoreno/TFGpractice/TFGFortyNinthTry49/outputs/fortyninthtry49_pmnet_prior_stage1_t49_stage1_w128_4gpu/best_cgan.pt 2>/dev/null || true",
]
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname=host, username=user, password=password, timeout=30)
stdin, stdout, stderr = client.exec_command('; '.join(commands), get_pty=True)
print(stdout.read().decode('utf-8', errors='replace'))
print(stderr.read().decode('utf-8', errors='replace'))
client.close()
