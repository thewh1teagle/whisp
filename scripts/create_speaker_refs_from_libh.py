from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torchaudio
import torchaudio.functional as audio_F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qwen_speaker_model import load_qwen_speaker_encoder, mel_spectrogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Qwen speaker reference embeddings from original LibriHeavy audio bytes.")
    parser.add_argument("filtered_root", type=Path, help="Filtered SNAC Parquet root.")
    parser.add_argument("source_audio_root", type=Path, help="Original LibriHeavy Parquet root with audio bytes.")
    parser.add_argument("output_root", type=Path, help="Output speaker_refs root.")
    parser.add_argument("--weights", type=Path, default=Path("data/qwen_speaker/qwen3_tts_0_6b_speaker_encoder.pt"))
    parser.add_argument("--refs-per-speaker", type=int, default=10)
    parser.add_argument("--min-duration", type=float, default=4.0)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--target-duration", type=float, default=7.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_path_for(filtered_root: Path, source_audio_root: Path, filtered_path: Path) -> Path:
    relative = filtered_path.relative_to(filtered_root)
    name = relative.name.removesuffix(".snac.parquet") + ".parquet"
    return source_audio_root / relative.with_name(name)


def select_refs(args: argparse.Namespace) -> dict[Path, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    filtered_files = sorted(args.filtered_root.rglob("*.snac.parquet"))
    for filtered_path in tqdm(filtered_files, desc="Selecting refs", unit="file"):
        table = pq.read_table(filtered_path, columns=["id", "speaker_id", "audio_duration", "source_audio_path"])
        source_path = source_path_for(args.filtered_root, args.source_audio_root, filtered_path)
        for row_index, row in enumerate(table.to_pylist()):
            duration = float(row["audio_duration"] or 0.0)
            if duration < args.min_duration or duration > args.max_duration:
                continue
            candidates[str(row["speaker_id"])].append(
                {
                    "id": row["id"],
                    "speaker_id": str(row["speaker_id"]),
                    "audio_duration": duration,
                    "source_audio_path": row["source_audio_path"],
                    "source_parquet": source_path,
                    "score": abs(duration - args.target_duration),
                }
            )

    selected_by_source: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for speaker_id, rows in candidates.items():
        rows.sort(key=lambda row: (row["score"], row["id"]))
        for ref_index, row in enumerate(rows[: args.refs_per_speaker]):
            row["ref_index"] = ref_index
            selected_by_source[row["source_parquet"]].append(row)
    return selected_by_source


def load_audio_by_id(source_parquet: Path, wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    parquet_file = pq.ParquetFile(source_parquet)
    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_index, columns=["id", "audio"])
        for row in table.to_pylist():
            row_id = row["id"]
            if row_id in wanted_ids:
                found[row_id] = row["audio"]
        if len(found) == len(wanted_ids):
            break
    return found


def decode_audio(audio_bytes: bytes, sample_rate: int, device: str) -> tuple[torch.Tensor, int]:
    audio, source_sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if source_sample_rate != sample_rate:
        audio = audio_F.resample(audio, orig_freq=source_sample_rate, new_freq=sample_rate)
    return audio.to(device), source_sample_rate


def main() -> None:
    args = parse_args()
    selected_by_source = select_refs(args)
    total_refs = sum(len(rows) for rows in selected_by_source.values())
    print(f"selected_refs: {total_refs}")
    print(f"source_parquets: {len(selected_by_source)}")

    model, config = load_qwen_speaker_encoder(str(args.weights), device=args.device)
    written = 0
    manifest_rows = []

    for source_parquet, refs in tqdm(sorted(selected_by_source.items()), desc="Embedding refs", unit="file"):
        if not source_parquet.exists():
            raise FileNotFoundError(f"Missing source parquet: {source_parquet}")
        audio_by_id = load_audio_by_id(source_parquet, {row["id"] for row in refs})
        for row in refs:
            audio_info = audio_by_id.get(row["id"])
            if audio_info is None:
                raise ValueError(f"Missing audio bytes for {row['id']} in {source_parquet}")

            speaker_dir = args.output_root / f"{row['speaker_id']}"
            output_path = speaker_dir / f"ref_{row['ref_index']:03d}.pt"
            if output_path.exists() and not args.overwrite:
                manifest_rows.append({**row, "embedding": str(output_path.relative_to(args.output_root.parent))})
                continue

            audio, source_sample_rate = decode_audio(audio_info["bytes"], config.sample_rate, args.device)
            with torch.inference_mode():
                mel = mel_spectrogram(audio, config)
                embedding = model(mel).float().cpu().squeeze(0)

            speaker_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "embedding": embedding,
                    "id": row["id"],
                    "speaker_id": row["speaker_id"],
                    "audio_duration": row["audio_duration"],
                    "source_audio_path": row["source_audio_path"],
                    "source_sample_rate": source_sample_rate,
                    "sample_rate": config.sample_rate,
                    "source": "Qwen/Qwen3-TTS-12Hz-0.6B-Base speaker_encoder",
                },
                output_path,
            )
            written += 1
            manifest_rows.append({**row, "embedding": str(output_path.relative_to(args.output_root.parent))})

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "refs_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    print(f"written: {written}")
    print(f"manifest: {args.output_root / 'refs_manifest.jsonl'}")


if __name__ == "__main__":
    main()
