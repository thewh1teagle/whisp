from __future__ import annotations

import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from src.snac_ordering import codes_to_depth_first
from src.tokenization import format_prompt, format_target


IGNORE_INDEX = -100
PARQUET_COLUMNS = ["id", "speaker_id", "text_normalized", "phonemes_normalized", "snac_0", "snac_1", "snac_2"]


def parquet_files(path: str | Path) -> list[str]:
    root = Path(path)
    if root.is_file():
        return [str(root)]
    files = sorted(str(file) for file in root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {root}")
    return files


def snac_token_count(row: dict) -> int:
    return len(row["snac_0"]) + len(row["snac_1"]) + len(row["snac_2"])


def sequence_length(row: dict, ref: dict | None = None) -> int:
    # Tokenizer is character-level except isolated special tokens:
    # <s>, <text>, </text>, <audio>, </audio>, </s>, and optional ref wrappers.
    total = len(str(row["phonemes_normalized"])) + 6 + snac_token_count(row)
    if ref is not None:
        total += 2 + len(ref["ref_snac_0"]) + len(ref["ref_snac_1"]) + len(ref["ref_snac_2"])
    return total


class WhispDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        speaker_refs_root: str | Path,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.speaker_refs_root = Path(speaker_refs_root)
        self.dataset = load_dataset(
            "parquet",
            data_files=parquet_files(self.path),
            split="train",
            columns=PARQUET_COLUMNS,
        )
        self.speaker_refs = self._load_speaker_refs()
        self.dataset = self._filter_dataset()

    def _load_speaker_refs(self) -> dict[str, list[Path | dict]]:
        parquet_path = self.speaker_refs_root / "speaker_refs.parquet"
        if parquet_path.exists():
            columns = [
                "speaker_id",
                "embedding",
                "ref_snac_0",
                "ref_snac_1",
                "ref_snac_2",
            ]
            schema_names = set(pq.read_schema(parquet_path).names)
            if all(column in schema_names for column in columns):
                table = pq.read_table(parquet_path, columns=columns)
            else:
                table = pq.read_table(parquet_path, columns=["speaker_id", "embedding"])

            refs: dict[str, list[dict]] = {}
            for row in table.to_pylist():
                row["embedding"] = torch.tensor(row["embedding"], dtype=torch.float32)
                refs.setdefault(str(row["speaker_id"]), []).append(row)
            if refs:
                return refs

        if not self.speaker_refs_root.is_dir():
            raise FileNotFoundError(f"Speaker refs root does not exist: {self.speaker_refs_root}")

        refs: dict[str, list[Path | dict]] = {}
        for speaker_dir in sorted(path for path in self.speaker_refs_root.iterdir() if path.is_dir()):
            paths = sorted(speaker_dir.glob("*.pt"))
            if paths:
                refs[speaker_dir.name] = paths
        if not refs:
            raise FileNotFoundError(f"No speaker ref .pt files found under {self.speaker_refs_root}")
        return refs

    def _filter_dataset(self):
        return self.dataset.filter(
            lambda row: str(row["speaker_id"]) in self.speaker_refs,
            desc=f"Filtering {self.path.name} speakers",
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        row = self.dataset[index]
        speaker_id = str(row["speaker_id"])
        refs = self.speaker_refs.get(speaker_id)
        if not refs:
            raise ValueError(f"Missing speaker refs for speaker_id={speaker_id!r}")

        target_audio_tokens = codes_to_depth_first(
            [
                torch.tensor(row["snac_0"], dtype=torch.long),
                torch.tensor(row["snac_1"], dtype=torch.long),
                torch.tensor(row["snac_2"], dtype=torch.long),
            ]
        )
        ref = random.choice(refs)
        ref_audio_tokens = None
        if isinstance(ref, dict) and "ref_snac_0" in ref:
            ref_audio_tokens = codes_to_depth_first(
                [
                    torch.tensor(ref["ref_snac_0"], dtype=torch.long),
                    torch.tensor(ref["ref_snac_1"], dtype=torch.long),
                    torch.tensor(ref["ref_snac_2"], dtype=torch.long),
                ]
            )

        phonemes = row["phonemes_normalized"]
        prompt = format_prompt(phonemes, ref_audio_tokens=ref_audio_tokens)
        target = format_target(target_audio_tokens)
        prompt_ids = self.tokenizer.encode(prompt).ids
        target_ids = self.tokenizer.encode(target).ids
        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "text": row["text_normalized"],
            "phonemes": phonemes,
            "ref_speaker_embedding": ref,
        }


class WhispDataCollator:
    def __init__(self, pad_token_id: int = 0, ignore_index: int = IGNORE_INDEX):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids, labels, attention_mask = [], [], []
        ref_speaker_embeddings = []

        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            labels.append(feature["labels"] + [self.ignore_index] * pad)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad)

            ref = feature["ref_speaker_embedding"]
            if isinstance(ref, dict):
                embedding = ref["embedding"]
            else:
                item = torch.load(ref, map_location="cpu", weights_only=False)
                embedding = item["embedding"] if isinstance(item, dict) else item
            ref_speaker_embeddings.append(embedding.float().view(-1))

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        batch["ref_speaker_embeddings"] = torch.stack(ref_speaker_embeddings)
        return batch


def make_dataloaders(args, tokenizer: Tokenizer) -> tuple[DataLoader, DataLoader]:
    collator = WhispDataCollator(pad_token_id=tokenizer.token_to_id("<pad>"))
    train_loader = DataLoader(
        WhispDataset(
            args.train_dataset,
            tokenizer,
            args.speaker_refs_root,
        ),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    eval_loader = DataLoader(
        WhispDataset(
            args.eval_dataset,
            tokenizer,
            args.speaker_refs_root,
        ),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    return train_loader, eval_loader
