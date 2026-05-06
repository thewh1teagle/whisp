from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phonemize import DEFAULT_LANGUAGE, phonemize_text


DEFAULT_ALLOWED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!;:'\"-()—"
DEFAULT_SENTENCE_END_CHARS = ".?!"
DEFAULT_TRAILING_CHARS = " \t\r\n'\"-()—,;:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a filtered SNAC Parquet dataset with normalized text and phonemes."
    )
    parser.add_argument("input_root", type=Path, help="Root containing .snac.parquet files.")
    parser.add_argument("output_root", type=Path, help="Output root for filtered .snac.parquet files.")
    parser.add_argument("--source-column", default="text_original", help="Text column to filter and phonemize.")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="espeak language passed to phonemizer.")
    parser.add_argument("--allowed-chars", default=DEFAULT_ALLOWED_CHARS, help="Exact allowed character set.")
    parser.add_argument(
        "--sentence-end-chars",
        default=DEFAULT_SENTENCE_END_CHARS,
        help="Effective final character must be one of these characters.",
    )
    parser.add_argument(
        "--allow-missing-sentence-end",
        action="store_true",
        help="Do not require text to end with sentence punctuation.",
    )
    parser.add_argument("--limit-files", type=int, default=None, help="Only process the first N files.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs that already exist.")
    return parser.parse_args()


def parquet_files(root: Path, limit: int | None) -> list[Path]:
    files = sorted(root.rglob("*.snac.parquet"))
    if limit is not None:
        files = files[:limit]
    return files


def output_path_for(input_root: Path, output_root: Path, input_path: Path) -> Path:
    return output_root / input_path.relative_to(input_root)


def is_allowed_text(text: str, allowed_chars: set[str]) -> bool:
    return all(char in allowed_chars for char in text)


def has_sentence_end(text: str, sentence_end_chars: set[str]) -> bool:
    stripped = text.rstrip(DEFAULT_TRAILING_CHARS)
    return bool(stripped) and stripped[-1] in sentence_end_chars


def filter_table(
    table: pa.Table,
    *,
    source_column: str,
    allowed_chars: set[str],
    sentence_end_chars: set[str],
    require_sentence_end: bool,
    language: str,
) -> tuple[pa.Table, int, int, float, float]:
    if source_column not in table.column_names:
        raise ValueError(f"Input table has no column named {source_column!r}")
    if "audio_duration" not in table.column_names:
        raise ValueError("Input table has no audio_duration column")

    source_values = table[source_column].to_pylist()
    durations = table["audio_duration"].to_pylist()

    keep_indices: list[int] = []
    text_normalized: list[str] = []
    phonemes_normalized: list[str] = []
    skipped_char_rows = 0
    skipped_end_rows = 0
    skipped_char_seconds = 0.0
    skipped_end_seconds = 0.0

    for index, (value, duration) in enumerate(zip(source_values, durations, strict=True)):
        text = "" if value is None else str(value)
        seconds = float(duration or 0.0)
        if not is_allowed_text(text, allowed_chars):
            skipped_char_rows += 1
            skipped_char_seconds += seconds
            continue
        if require_sentence_end and not has_sentence_end(text, sentence_end_chars):
            skipped_end_rows += 1
            skipped_end_seconds += seconds
            continue
        keep_indices.append(index)
        text_normalized.append(text)
        phonemes_normalized.append(phonemize_text(text, language=language))

    filtered = table.take(pa.array(keep_indices, type=pa.int64()))
    insert_at = filtered.column_names.index(source_column) + 1
    filtered = filtered.add_column(insert_at, "text_normalized", pa.array(text_normalized, type=pa.string()))
    filtered = filtered.add_column(
        insert_at + 1,
        "phonemes_normalized",
        pa.array(phonemes_normalized, type=pa.string()),
    )
    return filtered, skipped_char_rows, skipped_end_rows, skipped_char_seconds, skipped_end_seconds


def write_table(table: pa.Table, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    pq.write_table(table, tmp_path, compression="zstd")
    os.replace(tmp_path, output_path)


def copy_manifest(input_root: Path, output_root: Path, stats: dict[str, Any], args: argparse.Namespace) -> None:
    manifest_path = input_root / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    columns = list(manifest.get("columns", []))
    if args.source_column in columns:
        insert_at = columns.index(args.source_column) + 1
        for column in ["text_normalized", "phonemes_normalized"]:
            if column not in columns:
                columns.insert(insert_at, column)
                insert_at += 1

    manifest.update(
        {
            "filtered": True,
            "filter_source_column": args.source_column,
            "filter_allowed_chars": args.allowed_chars,
            "filter_require_sentence_end": not args.allow_missing_sentence_end,
            "filter_sentence_end_chars": args.sentence_end_chars,
            "phonemizer_language": args.language,
            "text_normalized": "same as source text for rows passing the allowed character filter",
            "phonemes_normalized": "src.phonemize.phonemize_text(text_normalized)",
            "columns": columns,
            "filtered_stats": stats,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {args.input_root}")

    files = parquet_files(args.input_root, args.limit_files)
    if not files:
        raise SystemExit(f"No .snac.parquet files found under: {args.input_root}")

    allowed_chars = set(args.allowed_chars)
    sentence_end_chars = set(args.sentence_end_chars)
    require_sentence_end = not args.allow_missing_sentence_end
    input_rows = output_rows = skipped_rows = skipped_char_rows = skipped_end_rows = 0
    input_seconds = output_seconds = skipped_seconds = skipped_char_seconds = skipped_end_seconds = 0.0
    written_files = skipped_existing_files = empty_files = 0

    for input_path in tqdm(files, desc="Filtering + phonemizing", unit="file"):
        output_path = output_path_for(args.input_root, args.output_root, input_path)
        if output_path.exists() and not args.overwrite:
            skipped_existing_files += 1
            continue

        table = pq.read_table(input_path)
        input_rows += table.num_rows
        input_seconds += float(pa.compute.sum(table["audio_duration"]).as_py() or 0.0)
        filtered, file_skipped_char_rows, file_skipped_end_rows, file_skipped_char_seconds, file_skipped_end_seconds = filter_table(
            table,
            source_column=args.source_column,
            allowed_chars=allowed_chars,
            sentence_end_chars=sentence_end_chars,
            require_sentence_end=require_sentence_end,
            language=args.language,
        )

        skipped_char_rows += file_skipped_char_rows
        skipped_end_rows += file_skipped_end_rows
        skipped_rows += file_skipped_char_rows + file_skipped_end_rows
        skipped_char_seconds += file_skipped_char_seconds
        skipped_end_seconds += file_skipped_end_seconds
        skipped_seconds += file_skipped_char_seconds + file_skipped_end_seconds
        output_rows += filtered.num_rows
        output_seconds += float(pa.compute.sum(filtered["audio_duration"]).as_py() or 0.0)

        if filtered.num_rows == 0:
            empty_files += 1
            continue

        write_table(filtered, output_path)
        written_files += 1

    stats = {
        "input_files": len(files),
        "written_files": written_files,
        "skipped_existing_files": skipped_existing_files,
        "empty_files": empty_files,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "skipped_rows": skipped_rows,
        "skipped_char_rows": skipped_char_rows,
        "skipped_missing_sentence_end_rows": skipped_end_rows,
        "input_hours": input_seconds / 3600.0,
        "output_hours": output_seconds / 3600.0,
        "skipped_hours": skipped_seconds / 3600.0,
        "skipped_char_hours": skipped_char_seconds / 3600.0,
        "skipped_missing_sentence_end_hours": skipped_end_seconds / 3600.0,
    }
    copy_manifest(args.input_root, args.output_root, stats, args)

    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
