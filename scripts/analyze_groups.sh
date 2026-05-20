#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="outputs/scanuiformer_continue_from_checkpoint"
CHECKPOINT="epoch23"
MODE="pure"
THRESHOLD="0.35"

python src/scanuiformer/analyze_groups.py \
  --config configs/eval_internal_alltype.json \
  --per_sample_jsonl "${MODEL_DIR}/eval_internal_alltype/${CHECKPOINT}/test_${MODE}_thr${THRESHOLD}_per_sample_metrics.jsonl" \
  --nss_per_sample_csv "${MODEL_DIR}/eval_internal_alltype/nss_metrics/nss_per_sample_metrics.csv" \
  --checkpoint "${CHECKPOINT}" \
  --mode "${MODE}" \
  --threshold "${THRESHOLD}" \
  --split test \
  --out_dir "${MODEL_DIR}/group_analysis/internal_alltype_${CHECKPOINT}_${MODE}_thr${THRESHOLD}"