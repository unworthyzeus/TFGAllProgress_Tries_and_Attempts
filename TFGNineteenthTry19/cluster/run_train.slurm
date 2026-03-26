#!/bin/bash
#SBATCH --job-name=ckm-firsttry-train
#SBATCH -p small_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs_train_%j.out
#SBATCH --error=logs_train_%j.err

set -e

source /scratch/nas/g/gmoreno/tf_env/bin/activate
export TF_FORCE_GPU_ALLOW_GROWTH=true

cd $SLURM_SUBMIT_DIR
python train.py --config configs/baseline.yaml
