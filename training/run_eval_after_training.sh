#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR=${1:-outputs/qwen35-0.8b-valdoria-full-sft}
PRED_OUT=${2:-outputs/ft_predictions.jsonl}
DETAILS_OUT=${3:-outputs/ft_eval_details.json}

python training/scripts/generate_valdoria.py \
  --model "$MODEL_DIR" \
  --input data/eval/eval_prompts.jsonl \
  --output "$PRED_OUT"

python training/scripts/eval_predictions.py \
  --predictions "$PRED_OUT" \
  --expected data/eval/eval_expected.jsonl \
  --details "$DETAILS_OUT"
