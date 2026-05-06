from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phonemize import DEFAULT_LANGUAGE, phonemize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add a phonemes column to SNAC Parquet shards.")
    parser.add_argument("root", type=Path, help="Root containing .snac.parquet files.")
    parser.add_argument("--text-column", default="text_transcription", help="Text column to phonemize.")
    parser.add_argument("--output-root", type=Path, default=None, help="Write rewritten files under this root.")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="espeak language passed to phonemizer.")
    parser.add_argument("--limit-files", type=int, default=None, help="Only process the first N files.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute phonemes if the column already exists.")
    return parser.parse_args()


def parquet_files(root: Path, limit: int | None) -> list[Path]:
    files = sorted(root.rglob("*.snac.parquet"))
    if limit is not None:
        files = files[:limit]
    return files


def output_path_for(root: Path, output_root: Path | None, input_path: Path) -> Path:
    if output_root is None:
        return input_path
    return output_root / input_path.relative_to(root)


def add_phonemes_to_file(
    input_path: Path,
    output_path: Path,
    *,
    text_column: str,
    language: str,
    overwrite: bool,
) -> tuple[int, bool]:
    table = pq.read_table(input_path)
    if "phonemes" in table.column_names and not overwrite:
        if output_path != input_path and not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output_path, compression="zstd")
        return table.num_rows, True

    if text_column not in table.column_names:
        raise ValueError(f"{input_path} has no column named {text_column!r}")

    text_values = table[text_column].to_pylist()
    phonemes = [
        phonemize_text("" if text is None else str(text), language=language)
        for text in text_values
    ]
    phonemes_array = pa.array(phonemes, type=pa.string())

    if "phonemes" in table.column_names:
        column_index = table.column_names.index("phonemes")
        table = table.set_column(column_index, "phonemes", phonemes_array)
    else:
        insert_at = table.column_names.index(text_column) + 1
        table = table.add_column(insert_at, "phonemes", phonemes_array)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    pq.write_table(table, tmp_path, compression="zstd")
    os.replace(tmp_path, output_path)
    return table.num_rows, False


def main() -> None:
    args = parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"Root does not exist: {args.root}")

    files = parquet_files(args.root, args.limit_files)
    if not files:
        raise SystemExit(f"No .snac.parquet files found under: {args.root}")

    total_rows = 0
    skipped_files = 0
    for input_path in tqdm(files, desc="Adding phonemes", unit="file"):
        output_path = output_path_for(args.root, args.output_root, input_path)
        rows, skipped = add_phonemes_to_file(
            input_path,
            output_path,
            text_column=args.text_column,
            language=args.language,
            overwrite=args.overwrite,
        )
        total_rows += rows
        skipped_files += int(skipped)

    print(f"files: {len(files)}")
    print(f"rows: {total_rows}")
    print(f"skipped_existing: {skipped_files}")


if __name__ == "__main__":
    main()
