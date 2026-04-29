#!/usr/bin/env bash
set -euo pipefail

# Download, extract, SNAC-encode, phonemize, and split LibriTTS-R train_clean_360.
# This writes dataset/.cache/train.jsonl and dataset/.cache/val.jsonl so
# scripts/train_scratch.sh can be launched afterwards.

URL="${URL:-https://openslr.trmal.net/resources/141/train_clean_360.tar.gz}"
WORK_DIR="${WORK_DIR:-data/libritts-r}"
ARCHIVE="${ARCHIVE:-${WORK_DIR}/train_clean_360.tar.gz}"
EXTRACT_DIR="${EXTRACT_DIR:-${WORK_DIR}/extracted}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-dataset/.cache/train.jsonl}"
VAL_OUTPUT="${VAL_OUTPUT:-dataset/.cache/val.jsonl}"
SPEAKER_MAP="${SPEAKER_MAP:-dataset/.cache/speaker_map.json}"
VAL_COUNT="${VAL_COUNT:-1000}"
LIMIT="${LIMIT:-}"

mkdir -p "${WORK_DIR}" "${EXTRACT_DIR}" "$(dirname "${TRAIN_OUTPUT}")"

ensure_aria2() {
  if command -v aria2c >/dev/null 2>&1; then
    return
  fi

  if command -v apt >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y aria2
  fi
}

download_file() {
  local url="$1"
  local output="$2"
  local output_dir
  local output_name

  output_dir="$(dirname "${output}")"
  output_name="$(basename "${output}")"

  ensure_aria2
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x16 -s16 -k1M -c -d "${output_dir}" -o "${output_name}" "${url}"
  else
    wget -c "${url}" -O "${output}"
  fi
}

if [[ ! -f "${ARCHIVE}" ]]; then
  download_file "${URL}" "${ARCHIVE}"
else
  echo "archive exists: ${ARCHIVE}"
fi

if ! find "${EXTRACT_DIR}" -type f -name '*.wav' -print -quit | grep -q .; then
  tar -xzf "${ARCHIVE}" -C "${EXTRACT_DIR}"
else
  echo "already extracted: ${EXTRACT_DIR}"
fi

ARGS=(
  scripts/prepare_libritts_r.py
  "${EXTRACT_DIR}"
  --train-output "${TRAIN_OUTPUT}"
  --val-output "${VAL_OUTPUT}"
  --speaker-map "${SPEAKER_MAP}"
  --val-count "${VAL_COUNT}"
)

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

uv run "${ARGS[@]}"
