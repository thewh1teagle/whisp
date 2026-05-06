#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/infer.sh <checkpoint> <ref-speaker-embedding.pt> <text> [output.wav]

CHECKPOINT=${1:?"Usage: $0 <checkpoint> <ref-speaker-embedding.pt> <text> [output.wav]"}
REF_SPEAKER_EMBEDDING=${2:?"Usage: $0 <checkpoint> <ref-speaker-embedding.pt> <text> [output.wav]"}
TEXT=${3:?"Usage: $0 <checkpoint> <ref-speaker-embedding.pt> <text> [output.wav]"}
OUTPUT=${4:-output.wav}

uv run src/infer.py \
  --checkpoint "${CHECKPOINT}" \
  --ref-speaker-embedding "${REF_SPEAKER_EMBEDDING}" \
  --text "${TEXT}" \
  --output "${OUTPUT}"
