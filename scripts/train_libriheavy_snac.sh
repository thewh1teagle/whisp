#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/train_libriheavy_snac.sh --max-steps 100000
#   scripts/train_libriheavy_snac.sh --max-steps 100000 --resume outputs/whisp-libriheavy/step-500

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
  --dataset-format libriheavy-snac \
  --train-dataset libriheavy-snac \
  --eval-dataset libriheavy-snac \
  --output-dir outputs/whisp-libriheavy \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --lr 3e-4 \
  --warmup-steps 100 \
  --logging-steps 10 \
  --save-steps 500 \
  --shuffle-buffer-size 20000 \
  ${RESUME} \
  ${RESET_STEPS} \
  "${EXTRA[@]+"${EXTRA[@]}"}"
