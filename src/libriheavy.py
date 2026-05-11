from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from src.snac_ordering import codes_to_depth_first


IGNORE_INDEX = -100
COLUMNS = ["id", "speaker_id", "phonemes", "snac_0", "snac_1", "snac_2"]


def parquet_files(root: str | Path) -> list[str]:
    files = sorted(str(path) for path in Path(root).glob("shard-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    return files


def speaker_map(root: str | Path) -> dict[str, int]:
    files = sorted(str(path) for path in (Path(root) / "speakers").glob("*.parquet"))
    ds = load_dataset("parquet", data_files=files, split="train", columns=["speaker_id"])
    speaker_ids = sorted({str(speaker_id) for speaker_id in ds["speaker_id"]}, key=int)
    return {speaker_id: idx for idx, speaker_id in enumerate(speaker_ids)}


def make_libriheavy_dataset(args, tokenizer: Tokenizer, *, split: str):
    speakers = speaker_map(args.train_dataset)
    audio_token_ids = [tokenizer.token_to_id(f"<audio_{idx}>") for idx in range(4096)]
    phoneme_ids = tokenizer.get_vocab()
    unk_id = tokenizer.token_to_id("<unk>")
    special = {token: tokenizer.token_to_id(token) for token in ["<s>", "</s>", "<speaker>", "</speaker>", "<text>", "</text>", "<audio>", "</audio>"]}

    root = args.train_dataset if split == "train" else args.eval_dataset
    ds = load_dataset(
        "parquet",
        data_files=parquet_files(root),
        split="train",
        streaming=True,
        columns=COLUMNS,
    )
    if split == "train" and args.shuffle_buffer_size > 0:
        ds = ds.shuffle(buffer_size=args.shuffle_buffer_size, seed=args.seed)

    skipped = {"missing_speaker": 0, "too_long": 0}

    def warn_skip(reason: str) -> None:
        skipped[reason] += 1
        count = skipped[reason]
        if count == 1 or count % 1000 == 0:
            print(f"warning: skipped {count} LibriHeavy {split} rows: {reason}", flush=True)

    def encode(row: dict) -> dict:
        speaker_id = speakers.get(str(row["speaker_id"]))
        if speaker_id is None:
            warn_skip("missing_speaker")
            return {"input_ids": [], "labels": []}

        prompt_ids = [
            special["<s>"],
            special["<speaker>"],
            tokenizer.token_to_id(f"<spk_{speaker_id}>"),
            special["</speaker>"],
            special["<text>"],
            *[phoneme_ids.get(char, unk_id) for char in str(row["phonemes"])],
            special["</text>"],
            special["<audio>"],
        ]
        audio_tokens = codes_to_depth_first(
            [
                torch.tensor(row["snac_0"], dtype=torch.long),
                torch.tensor(row["snac_1"], dtype=torch.long),
                torch.tensor(row["snac_2"], dtype=torch.long),
            ]
        )
        target_ids = [audio_token_ids[token] for token in audio_tokens] + [special["</audio>"], special["</s>"]]
        input_ids = prompt_ids + target_ids
        if len(input_ids) > args.max_sequence_length:
            warn_skip("too_long")
            return {"input_ids": [], "labels": []}
        return {
            "input_ids": input_ids,
            "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
            "speaker_id": speaker_id,
            "phonemes": row["phonemes"],
        }

    return ds.map(encode, remove_columns=COLUMNS).filter(lambda row: len(row["input_ids"]) > 0)
