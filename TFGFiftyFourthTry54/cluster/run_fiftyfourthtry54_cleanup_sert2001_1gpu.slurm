#!/bin/bash
# Try 54 cleanup job on sert-2001 to clear stale user-side CUDA processes
# before launching the topology experts.
#SBATCH --job-name=ckm-t54-cleanup-sert2001
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:05:00
#SBATCH --output=logs_cleanup_fiftyfourthtry54_%j.out
#SBATCH --error=logs_cleanup_fiftyfourthtry54_%j.err

set -euo pipefail

echo "[cleanup] host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi || true

pkill -9 -u gmoreno -f 'train_partitioned_pathloss_expert.py' || true
pkill -9 -u gmoreno -f 'torchrun.*TFGFiftyFourthTry54' || true

sleep 5

echo "[cleanup] user processes after kill"
ps -u gmoreno -o pid,ppid,comm,args | egrep 'train_partitioned_pathloss_expert|torchrun.*TFGFiftyFourthTry54' || true

echo "[cleanup] nvidia-smi after"
nvidia-smi || true
echo "[cleanup] end=$(date --iso-8601=seconds)"
