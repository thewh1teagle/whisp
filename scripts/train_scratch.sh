#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/train_scratch.sh
#   scripts/train_scratch.sh --resume outputs/whisp/step-500

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
  --train-dataset data/whisp-libriheavy-15k/data/train \
  --eval-dataset data/whisp-libriheavy-15k/data/validation \
  --speaker-refs-root data/whisp-libriheavy-15k/speaker_refs \
  --output-dir outputs/whisp \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --lr 3e-4 \
  --warmup-steps 100 \
  --logging-steps 10 \
  --eval-steps 500 \
  --save-steps 500 \
  --max-position-embeddings 4096 \
  ${RESUME} \
  ${RESET_STEPS} \
  "${EXTRA[@]+"${EXTRA[@]}"}"
