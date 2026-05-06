from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack speaker ref .pt files into one Parquet file.")
    parser.add_argument("speaker_refs_root", type=Path, help="Root containing <speaker_id>/ref_*.pt files.")
    parser.add_argument("output_path", type=Path, help="Output speaker_refs.parquet path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.speaker_refs_root.glob("*/*.pt"))
    if not paths:
        raise SystemExit(f"No .pt speaker refs found under {args.speaker_refs_root}")

    rows = []
    for path in tqdm(paths, desc="Packing refs", unit="ref"):
        item = torch.load(path, map_location="cpu", weights_only=False)
        embedding = item["embedding"] if isinstance(item, dict) else item
        speaker_id = str(item.get("speaker_id", path.parent.name)) if isinstance(item, dict) else path.parent.name
        ref_index = int(path.stem.removeprefix("ref_"))
        rows.append(
            {
                "speaker_id": speaker_id,
                "ref_index": ref_index,
                "embedding": embedding.float().view(-1).tolist(),
                "id": item.get("id") if isinstance(item, dict) else None,
                "audio_duration": float(item.get("audio_duration", 0.0)) if isinstance(item, dict) else None,
                "source_audio_path": item.get("source_audio_path") if isinstance(item, dict) else None,
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
        ]
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp_path, compression="zstd")
    tmp_path.replace(args.output_path)

    manifest = {
        "format_version": 1,
        "refs": len(rows),
        "speakers": len({row["speaker_id"] for row in rows}),
        "embedding_size": 1024,
        "source": str(args.speaker_refs_root),
        "columns": schema.names,
    }
    args.output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
