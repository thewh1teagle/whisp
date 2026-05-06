from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect character inventory for a Parquet text column.")
    parser.add_argument("root", type=Path, help="Root containing Parquet files.")
    parser.add_argument("--column", default="text_original", help="Text column to inspect.")
    parser.add_argument("--pattern", default="*.snac.parquet", help="Parquet glob pattern.")
    parser.add_argument("--limit-files", type=int, default=None, help="Only inspect the first N files.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write counts as JSON.")
    return parser.parse_args()


def display_char(char: str) -> str:
    if char == "\n":
        return "\\n"
    if char == "\r":
        return "\\r"
    if char == "\t":
        return "\\t"
    if char == " ":
        return "SPACE"
    return char


def main() -> None:
    args = parse_args()
    files = sorted(args.root.rglob(args.pattern))
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} under {args.root}")

    counts: Counter[str] = Counter()
    rows = 0
    for path in tqdm(files, desc=f"Inspecting {args.column}", unit="file"):
        table = pq.read_table(path, columns=[args.column])
        for value in table[args.column].to_pylist():
            if value is None:
                continue
            text = str(value)
            rows += 1
            counts.update(text)

    print(f"files: {len(files)}")
    print(f"rows: {rows}")
    print(f"unique_chars: {len(counts)}")
    print()
    print("char\tcodepoint\tcount")
    for char in sorted(counts):
        print(f"{display_char(char)}\tU+{ord(char):04X}\t{counts[char]}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"char": char, "codepoint": f"U+{ord(char):04X}", "count": counts[char]}
            for char in sorted(counts)
        ]
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
