from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from src.tokenization import format_prompt, format_target


IGNORE_INDEX = -100


class WhispDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: Tokenizer):
        self.path = Path(path)
        self.tokenizer = tokenizer
        with self.path.open() as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        if "input_ids" in row and "labels" in row:
            input_ids = [int(token) for token in row["input_ids"]]
            labels = [int(token) for token in row["labels"]]
        else:
            prompt = format_prompt(int(row["speaker_id"]), row["phonemes"])
            target = format_target([int(token) for token in row["audio_tokens"]])
            prompt_ids = self.tokenizer.encode(prompt).ids
            target_ids = self.tokenizer.encode(target).ids
            input_ids = prompt_ids + target_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "speaker_id": int(row["speaker_id"]),
            "text": row.get("text", ""),
            "phonemes": row.get("phonemes", ""),
        }


class WhispDataCollator:
    def __init__(self, pad_token_id: int = 0, ignore_index: int = IGNORE_INDEX):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids, labels, attention_mask = [], [], []

        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            labels.append(feature["labels"] + [self.ignore_index] * pad)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def make_dataloaders(args, tokenizer: Tokenizer) -> tuple[DataLoader, DataLoader]:
    collator = WhispDataCollator(pad_token_id=tokenizer.token_to_id("<pad>"))
    train_loader = DataLoader(
        WhispDataset(args.train_dataset, tokenizer),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    eval_loader = DataLoader(
        WhispDataset(args.eval_dataset, tokenizer),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    return train_loader, eval_loader
