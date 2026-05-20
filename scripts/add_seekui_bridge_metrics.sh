#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="outputs/scanuiformer_continue_from_checkpoint"

python src/scanuiformer/add_seekui_bridge_metrics.py \
  --config configs/eval_seekui232.json \
  --eval_root "${MODEL_DIR}/eval_seekui232" \
  --split test \
  --out_dir "${MODEL_DIR}/eval_seekui232/seekui_bridge_metrics"