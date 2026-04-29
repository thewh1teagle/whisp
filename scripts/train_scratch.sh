#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/train_scratch.sh --num-speakers 123
#   scripts/train_scratch.sh --num-speakers 123 --resume outputs/whisp/step-500

RESUME=""
RESET_STEPS=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME="--resume $2"; shift 2 ;;
    --reset-steps) RESET_STEPS="--reset-steps"; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

uv run accelerate launch src/train.py \
  --train-dataset dataset/.cache/train.jsonl \
  --eval-dataset dataset/.cache/val.jsonl \
  --output-dir outputs/whisp \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --lr 3e-4 \
  --warmup-steps 100 \
  --logging-steps 10 \
  --save-steps 500 \
  ${RESUME} \
  ${RESET_STEPS} \
  "${EXTRA[@]+"${EXTRA[@]}"}"
