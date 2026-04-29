from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.codec import encode
from src.phonemize import phonemize_text


def prepare_one(task: tuple[str, str, str, int, bool, str]) -> tuple[int, dict | None, bool]:
    utt_id, value, audio_path, speaker_id, should_phonemize, language = task
    path = Path(audio_path)
    if not path.exists():
        return int(utt_id) if utt_id.isdigit() else 0, None, True

    text = value
    phonemes = phonemize_text(value, language=language) if should_phonemize else value
    encoded = encode(path)
    row = {
        "audio": str(path),
        "text": text,
        "phonemes": phonemes,
        "speaker_id": speaker_id,
        "audio_tokens": list(encoded.tokens),
    }
    return int(utt_id) if utt_id.isdigit() else 0, row, False


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Whisp semantic JSONL from metadata.csv + wav/")
    parser.add_argument("dataset_dir", type=Path, help="Folder containing metadata.csv and wav/")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--language", type=str, default="en-us")
    parser.add_argument("--phonemize", action="store_true", help="Treat metadata text as raw text and phonemize it")
    parser.add_argument("--delimiter", type=str, default="|")
    parser.add_argument("--audio-ext", type=str, default=".wav")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def read_metadata(path: Path, delimiter: str) -> list[tuple[str, str]]:
    rows = []
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"Expected at least 2 columns in {path}: id|phonemes_or_text")
            rows.append((row[0].strip(), row[1].strip()))
    return rows


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    metadata_path = dataset_dir / "metadata.csv"
    wav_dir = dataset_dir / "wav"
    output_path = args.output or dataset_dir / "dataset.jsonl"

    rows = read_metadata(metadata_path, args.delimiter)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            utt_id,
            value,
            str(wav_dir / f"{utt_id}{args.audio_ext}"),
            args.speaker_id,
            args.phonemize,
            args.language,
        )
        for utt_id, value in rows
    ]

    written = 0
    skipped = 0
    results = []

    if args.workers == 1:
        for task in tqdm(tasks, desc="Preparing", unit="utt"):
            _, row, missing = prepare_one(task)
            skipped += int(missing)
            if row is not None:
                results.append(row)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(prepare_one, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing", unit="utt"):
                _, row, missing = future.result()
                skipped += int(missing)
                if row is not None:
                    results.append(row)

    results.sort(key=lambda row: int(Path(row["audio"]).stem) if Path(row["audio"]).stem.isdigit() else row["audio"])
    with output_path.open("w") as out:
        for row in results:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote: {output_path}")
    print(f"rows: {written}")
    print(f"skipped_missing_audio: {skipped}")


if __name__ == "__main__":
    main()
