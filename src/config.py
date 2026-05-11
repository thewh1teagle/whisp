from __future__ import annotations

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Whisp SNAC/Qwen3-MoE TTS model")
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--eval-dataset", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--dataset-format", choices=["auto", "jsonl", "libriheavy-snac"], default="auto")
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=100.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--max-eval-samples", type=int, default=150)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-position-embeddings", type=int, default=4096)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--flash-attention", action="store_true", help="Use Transformers flash_attention_2 attention implementation")
    parser.add_argument("--shuffle-buffer-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-steps", action="store_true", default=False)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    return parser.parse_args()
