#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="outputs/scanuiformer_continue_from_checkpoint"

python src/scanuiformer/add_nss_metrics.py \
  --config configs/eval_internal_alltype.json \
  --eval_root "${MODEL_DIR}/eval_internal_alltype" \
  --split test \
  --out_dir "${MODEL_DIR}/eval_internal_alltype/nss_metrics"

python src/scanuiformer/add_nss_metrics.py \
  --config configs/eval_seekui232.json \
  --eval_root "${MODEL_DIR}/eval_seekui232" \
  --split test \
  --out_dir "${MODEL_DIR}/eval_seekui232/nss_metrics"