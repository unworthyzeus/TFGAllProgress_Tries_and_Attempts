#!/bin/bash
#SBATCH --job-name=ckm-firsttry-eval
#SBATCH -p small_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs_eval_%j.out
#SBATCH --error=logs_eval_%j.err

set -e

source /scratch/nas/g/gmoreno/tf_env/bin/activate
export TF_FORCE_GPU_ALLOW_GROWTH=true

cd $SLURM_SUBMIT_DIR
python evaluate.py --config configs/baseline.yaml --checkpoint outputs/baseline_run/best.pt
