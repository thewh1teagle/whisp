#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/infer.sh <checkpoint> <num-speakers> <speaker-id> <text> [output.wav]

CHECKPOINT=${1:?"Usage: $0 <checkpoint> <num-speakers> <speaker-id> <text> [output.wav]"}
NUM_SPEAKERS=${2:?"Usage: $0 <checkpoint> <num-speakers> <speaker-id> <text> [output.wav]"}
SPEAKER_ID=${3:?"Usage: $0 <checkpoint> <num-speakers> <speaker-id> <text> [output.wav]"}
TEXT=${4:?"Usage: $0 <checkpoint> <num-speakers> <speaker-id> <text> [output.wav]"}
OUTPUT=${5:-output.wav}

uv run src/infer.py \
  --checkpoint "${CHECKPOINT}" \
  --num-speakers "${NUM_SPEAKERS}" \
  --speaker-id "${SPEAKER_ID}" \
  --text "${TEXT}" \
  --output "${OUTPUT}"
