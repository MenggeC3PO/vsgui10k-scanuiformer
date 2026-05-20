#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="outputs/scanuiformer_continue_from_checkpoint"

python src/scanuiformer/collect_results.py \
  --eval_root "${MODEL_DIR}/eval_internal_alltype" \
  --split test \
  --out_dir "${MODEL_DIR}/eval_internal_alltype/organized_results"

python src/scanuiformer/collect_results.py \
  --eval_root "${MODEL_DIR}/eval_seekui232" \
  --split test \
  --out_dir "${MODEL_DIR}/eval_seekui232/organized_results"