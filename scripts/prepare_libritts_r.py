from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prepare import build_row


TEXT_SUFFIXES = (".normalized.txt", ".original.txt", ".txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LibriTTS-R JSONL for Whisp training")
    parser.add_argument("root", type=Path, help="Extracted LibriTTS-R root or subset folder")
    parser.add_argument("--train-output", type=Path, default=Path("dataset/.cache/train.jsonl"))
    parser.add_argument("--val-output", type=Path, default=Path("dataset/.cache/val.jsonl"))
    parser.add_argument("--speaker-map", type=Path, default=Path("dataset/.cache/speaker_map.json"))
    parser.add_argument("--language", type=str, default="en-us")
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def find_text_path(wav_path: Path) -> Path | None:
    stem = wav_path.with_suffix("")
    for suffix in TEXT_SUFFIXES:
        path = stem.with_name(stem.name + suffix)
        if path.exists():
            return path
    return None


def speaker_key(wav_path: Path) -> str:
    # LibriTTS-R layout is normally subset/speaker/chapter/utterance.wav.
    if len(wav_path.parts) >= 3:
        return wav_path.parent.parent.name
    return "0"


def split_key(path: Path) -> int:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def main() -> None:
    args = parse_args()
    wav_paths = sorted(args.root.rglob("*.wav"))
    if args.limit is not None:
        wav_paths = wav_paths[: args.limit]
    if not wav_paths:
        raise FileNotFoundError(f"No wav files found under {args.root}")

    speaker_names = sorted({speaker_key(path) for path in wav_paths})
    speaker_to_id = {speaker: idx for idx, speaker in enumerate(speaker_names)}

    val_set = set(sorted(wav_paths, key=split_key)[: args.val_count])

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    args.speaker_map.parent.mkdir(parents=True, exist_ok=True)

    written_train = 0
    written_val = 0
    skipped_missing_text = 0

    with args.train_output.open("w") as train_out, args.val_output.open("w") as val_out:
        for wav_path in tqdm(wav_paths, desc="Preparing LibriTTS-R", unit="utt"):
            text_path = find_text_path(wav_path)
            if text_path is None:
                skipped_missing_text += 1
                continue

            text = text_path.read_text(encoding="utf-8").strip()
            if not text:
                skipped_missing_text += 1
                continue

            row = build_row(
                audio=wav_path,
                text=text,
                speaker_id=speaker_to_id[speaker_key(wav_path)],
                language=args.language,
            )

            if wav_path in val_set:
                val_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_val += 1
            else:
                train_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_train += 1

    args.speaker_map.write_text(
        json.dumps(
            {
                "speaker_to_id": speaker_to_id,
                "num_speakers": len(speaker_to_id),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"train: {args.train_output}")
    print(f"val: {args.val_output}")
    print(f"speaker_map: {args.speaker_map}")
    print(f"num_speakers: {len(speaker_to_id)}")
    print(f"train_rows: {written_train}")
    print(f"val_rows: {written_val}")
    print(f"skipped_missing_text: {skipped_missing_text}")
    print()
    print("train command:")
    print(f"  scripts/train_scratch.sh --num-speakers {len(speaker_to_id)}")


if __name__ == "__main__":
    main()
