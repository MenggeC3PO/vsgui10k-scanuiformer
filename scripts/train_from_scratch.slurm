#!/bin/bash -l
#
# Example SLURM script. Edit account, partition, time limit, and environment
# activation according to your own cluster setup.

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=scanuiformer_scratch
#SBATCH --output=logs/scanuiformer_scratch_%j.out
#SBATCH --error=logs/scanuiformer_scratch_%j.err

set -euo pipefail
set -x

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date
nvidia-smi || true

# Activate your Python environment before submitting this job,
# or uncomment and edit these lines:
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

python src/scanuiformer/train.py \
  --config configs/train_from_scratch.json \
  --freeze_patch

date