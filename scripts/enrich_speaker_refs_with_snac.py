from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm


REF_COLUMNS = ["id", "snac_0", "snac_1", "snac_2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add ref SNAC columns to speaker_refs.parquet.")
    parser.add_argument("speaker_refs_parquet", type=Path)
    parser.add_argument("filtered_root", type=Path, help="Filtered SNAC Parquet root.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def read_matches(task: tuple[str, list[str]]) -> list[dict]:
    path_str, row_ids = task
    path_ids = set(row_ids)
    table = pq.read_table(path_str, columns=REF_COLUMNS)
    mask = pc.is_in(table["id"], value_set=pa.array(list(path_ids), type=pa.string()))
    matched = table.filter(mask)
    if matched.num_rows != len(path_ids):
        matched_ids = {str(row_id) for row_id in matched["id"].to_pylist()}
        missing = sorted(path_ids - matched_ids)
        raise RuntimeError(f"Missing {len(missing)} refs in {path_str}, first={missing[:5]}")
    return matched.to_pylist()


def main() -> None:
    args = parse_args()
    refs = pq.read_table(args.speaker_refs_parquet).to_pylist()
    wanted = {str(row["id"]) for row in refs}
    found: dict[str, dict] = {}

    manifest_path = args.speaker_refs_parquet.parent / "refs_manifest.jsonl"
    files_by_id: dict[str, Path] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row_id = str(row["id"])
            if row_id not in wanted:
                continue
            source_parquet = Path(row["source_parquet"])
            relative = Path(*source_parquet.parts[-2:])
            filtered_path = args.filtered_root / relative.with_name(relative.name.removesuffix(".parquet") + ".snac.parquet")
            files_by_id[row_id] = filtered_path

    ids_by_file: dict[Path, set[str]] = defaultdict(set)
    if files_by_id:
        for row_id, path in files_by_id.items():
            ids_by_file[path].add(row_id)
    else:
        for path in args.filtered_root.rglob("*.snac.parquet"):
            ids_by_file[path] = set(wanted)

    tasks = [(str(path), sorted(path_ids)) for path, path_ids in sorted(ids_by_file.items())]
    if args.workers <= 1:
        iterator = (read_matches(task) for task in tasks)
        for rows in tqdm(iterator, total=len(tasks), desc="Looking up ref SNAC", unit="file"):
            for row in rows:
                found[str(row["id"])] = row
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(read_matches, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Looking up ref SNAC", unit="file"):
                for row in future.result():
                    found[str(row["id"])] = row

    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} speaker refs in filtered dataset, first={missing[:5]}")

    enriched = []
    for row in refs:
        ref = found[str(row["id"])]
        enriched.append(
            {
                **row,
                "ref_snac_0": ref["snac_0"],
                "ref_snac_1": ref["snac_1"],
                "ref_snac_2": ref["snac_2"],
            }
        )

    schema = pa.schema(
        [
            ("speaker_id", pa.string()),
            ("ref_index", pa.int16()),
            ("embedding", pa.list_(pa.float32(), list_size=1024)),
            ("id", pa.string()),
            ("audio_duration", pa.float32()),
            ("source_audio_path", pa.string()),
            ("ref_snac_0", pa.list_(pa.uint16())),
            ("ref_snac_1", pa.list_(pa.uint16())),
            ("ref_snac_2", pa.list_(pa.uint16())),
        ]
    )
    tmp_path = args.speaker_refs_parquet.with_suffix(args.speaker_refs_parquet.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(enriched, schema=schema), tmp_path, compression="zstd")
    tmp_path.replace(args.speaker_refs_parquet)

    manifest_path = args.speaker_refs_parquet.with_suffix(".json")
    manifest = {
        "format_version": 2,
        "refs": len(enriched),
        "speakers": len({row["speaker_id"] for row in enriched}),
        "embedding_size": 1024,
        "has_ref_snac": True,
        "columns": schema.names,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
