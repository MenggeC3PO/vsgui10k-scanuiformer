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
#SBATCH --time=1-12:00:00
#SBATCH --job-name=scanuiformer_eval
#SBATCH --output=logs/scanuiformer_eval_%j.out
#SBATCH --error=logs/scanuiformer_eval_%j.err

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

MODEL_DIR="outputs/scanuiformer_continue_from_checkpoint"
EVAL_SCRIPT="src/scanuiformer/evaluate.py"

COMMON_ARGS=(
  --split test
  --device cuda
  --batch_size 8
  --num_workers 4
  --modes pure,hybrid
  --thresholds 0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60
  --margins 0.00,0.03,0.05,0.10
  --grid_size 8
  --multimatch_resample_n 20
  --save_per_sample
  --learned_min_stop_step 5
  --heuristic_min_steps 5
  --heuristic_target_margin 0.05
  --heuristic_stop_on_entry_count 1
  --heuristic_inside_patience 2
  --visualize
  --num_visuals 40
  --visual_seed 123
  --viz_thresholds 0.25,0.35,0.45,0.55
  --target_margin 0.05
)

run_one_eval () {
  SPLIT_LABEL="$1"
  CONFIG_PATH="$2"
  CKPT_LABEL="$3"
  CKPT_PATH="$4"

  if [ ! -f "${CKPT_PATH}" ]; then
    echo "SKIP missing checkpoint: ${CKPT_LABEL} ${CKPT_PATH}"
    return 0
  fi

  OUT_DIR="${MODEL_DIR}/eval_${SPLIT_LABEL}/${CKPT_LABEL}"
  mkdir -p "${OUT_DIR}"

  python "${EVAL_SCRIPT}" \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CKPT_PATH}" \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUT_DIR}"
}

run_split () {
  SPLIT_LABEL="$1"
  CONFIG_PATH="$2"

  run_one_eval "${SPLIT_LABEL}" "${CONFIG_PATH}" "best"      "${MODEL_DIR}/best.pt"
  run_one_eval "${SPLIT_LABEL}" "${CONFIG_PATH}" "best_dist" "${MODEL_DIR}/best_dist.pt"
  run_one_eval "${SPLIT_LABEL}" "${CONFIG_PATH}" "best_loss" "${MODEL_DIR}/best_loss.pt"
  run_one_eval "${SPLIT_LABEL}" "${CONFIG_PATH}" "last"      "${MODEL_DIR}/last.pt"
  run_one_eval "${SPLIT_LABEL}" "${CONFIG_PATH}" "epoch01"   "${MODEL_DIR}/epoch_01.pt"
  run_one_eval "${SPLIT_LABEL}" "${CONFIG_PATH}" "epoch09"   "${MODEL_DIR}/epoch_09.pt"
  run_one_eval "${SPLIT_LABEL}" "${CONFIG_PATH}" "epoch23"   "${MODEL_DIR}/epoch_23.pt"
}

run_split "seekui232" "configs/eval_seekui232.json"
run_split "internal_alltype" "configs/eval_internal_alltype.json"

echo "All evaluations finished."
date