#!/usr/bin/env bash
set -euo pipefail
python training/scripts/train_full_sft.py \
  --config training/configs/full_sft_qwen35_0_8b_12gb.yaml
