from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import sequence_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split filtered SNAC Parquets into final train/validation upload layout.")
    parser.add_argument("input_root", type=Path, help="Filtered Parquet root containing .snac.parquet files.")
    parser.add_argument("output_root", type=Path, help="Final dataset root to create.")
    parser.add_argument("--val-rows", type=int, default=250, help="Number of rows to write to data/validation/validation.parquet.")
    parser.add_argument("--max-sequence-length", type=int, default=4096, help="Drop rows longer than this training context.")
    parser.add_argument("--seed", type=str, default="whisp-libriheavy-v1", help="Stable split seed.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking them.")
    return parser.parse_args()


def split_key(path: Path, seed: str) -> int:
    digest = hashlib.sha1(f"{seed}:{path.as_posix()}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def length_keep_mask(table: pa.Table, max_sequence_length: int) -> pa.Array:
    return pa.array(
        [
            sequence_length(row) <= max_sequence_length
            for row in table.select(["phonemes_normalized", "snac_0", "snac_1", "snac_2"]).to_pylist()
        ]
    )


def write_validation_file(files: list[Path], output_path: Path, val_rows: int, max_sequence_length: int) -> set[str]:
    selected_ids: set[str] = set()
    batches = []
    for path in tqdm(files, desc="Writing validation", unit="file"):
        table = pq.read_table(path)
        table = table.filter(length_keep_mask(table, max_sequence_length))
        remaining = val_rows - len(selected_ids)
        if remaining <= 0:
            break
        take = min(remaining, table.num_rows)
        batch = table.slice(0, take)
        batches.append(batch)
        selected_ids.update(str(row_id) for row_id in batch["id"].to_pylist())

    if len(selected_ids) < val_rows:
        raise RuntimeError(f"Only found {len(selected_ids)} validation rows, requested {val_rows}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    pq.write_table(pa.concat_tables(batches), tmp_path, compression="zstd")
    tmp_path.replace(output_path)
    return selected_ids


def write_train_files(
    files: list[Path],
    selected_val_ids: set[str],
    input_root: Path,
    output_root: Path,
    *,
    max_sequence_length: int,
    copy: bool,
) -> tuple[int, float, int, int, float]:
    train_rows = 0
    train_seconds = 0.0
    skipped_length_rows = 0
    skipped_length_seconds = 0.0
    written_files = 0
    for src in tqdm(files, desc="Writing train", unit="file"):
        table = pq.read_table(src)
        length_mask = length_keep_mask(table, max_sequence_length)
        skipped = table.filter(pa.compute.invert(length_mask))
        skipped_length_rows += skipped.num_rows
        skipped_length_seconds += float(pa.compute.sum(skipped["audio_duration"]).as_py() or 0.0)

        keep_mask = pa.array(
            [
                bool(length_ok) and str(row_id) not in selected_val_ids
                for length_ok, row_id in zip(length_mask.to_pylist(), table["id"].to_pylist(), strict=True)
            ]
        )
        filtered = table.filter(keep_mask)
        if filtered.num_rows == 0:
            continue

        # If this file was untouched by the split or length filter, hardlink/copy it directly.
        if filtered.num_rows == table.num_rows:
            dst = output_root / "data" / "train" / src.relative_to(input_root)
            link_or_copy(src, dst, copy=copy)
        else:
            dst = output_root / "data" / "train" / src.relative_to(input_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = dst.with_suffix(dst.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            pq.write_table(filtered, tmp_path, compression="zstd")
            tmp_path.replace(dst)

        train_rows += filtered.num_rows
        train_seconds += float(pa.compute.sum(filtered["audio_duration"]).as_py() or 0.0)
        written_files += 1
    return train_rows, train_seconds, written_files, skipped_length_rows, skipped_length_seconds


def main() -> None:
    args = parse_args()
    files = sorted(args.input_root.rglob("*.snac.parquet"))
    if not files:
        raise SystemExit(f"No .snac.parquet files found under {args.input_root}")
    if args.val_rows <= 0:
        raise SystemExit("--val-rows must be positive")

    split_files = sorted(files, key=lambda path: split_key(path.relative_to(args.input_root), args.seed))
    selected_val_ids = write_validation_file(
        split_files,
        args.output_root / "data" / "validation" / "validation.parquet",
        args.val_rows,
        args.max_sequence_length,
    )
    train_rows, train_seconds, train_files, skipped_length_rows, skipped_length_seconds = write_train_files(
        files,
        selected_val_ids,
        args.input_root,
        args.output_root,
        max_sequence_length=args.max_sequence_length,
        copy=args.copy,
    )
    val_table = pq.read_table(args.output_root / "data" / "validation" / "validation.parquet", columns=["audio_duration"])
    val_rows = val_table.num_rows
    val_seconds = float(pa.compute.sum(val_table["audio_duration"]).as_py() or 0.0)
    manifest = {
        "format_version": 1,
        "layout": "whisp-libriheavy-parquet-v1",
        "source_root": str(args.input_root),
        "train_files": train_files,
        "validation_files": 1,
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "train_hours": train_seconds / 3600.0,
        "validation_hours": val_seconds / 3600.0,
        "max_sequence_length": args.max_sequence_length,
        "skipped_length_rows": skipped_length_rows,
        "skipped_length_hours": skipped_length_seconds / 3600.0,
        "columns": pq.read_schema(files[0]).names,
        "speaker_refs_root": "speaker_refs",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
