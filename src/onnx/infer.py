from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import soundfile as sf
from tokenizers import Tokenizer

from src.codec import SAMPLE_RATE, decode
from src.tokenization import format_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio tokens with a static Whisp ONNX graph")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def _audio_id(token: str) -> int | None:
    if not token.startswith("<audio_") or not token.endswith(">"):
        return None
    return int(token.removeprefix("<audio_").removesuffix(">"))


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    row = rows[args.row]

    prompt = format_prompt(int(row["speaker_id"]), row["phonemes"])
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    graph_inputs = {item.name: item for item in session.get_inputs()}
    context_length = int(graph_inputs["input_ids"].shape[1])

    pad_id = tokenizer.token_to_id("<pad>") or 0
    eos_id = tokenizer.token_to_id("</s>")
    audio_end_id = tokenizer.token_to_id("</audio>")
    ids = tokenizer.encode(prompt).ids
    prompt_length = len(ids)
    if prompt_length >= context_length:
        raise ValueError(f"prompt has {prompt_length} tokens but ONNX context is {context_length}")

    audio_tokens: list[int] = []
    max_new_tokens = min(args.max_new_tokens, context_length - prompt_length)
    for _ in range(max_new_tokens):
        padded = ids + [pad_id] * (context_length - len(ids))
        input_ids = np.array([padded], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        feeds = {name: value for name, value in feeds.items() if name in graph_inputs}
        logits = session.run(None, feeds)[0]
        next_id = int(logits[0, len(ids) - 1].argmax())
        ids.append(next_id)

        if next_id in {eos_id, audio_end_id}:
            break
        token = tokenizer.id_to_token(next_id)
        audio_id = _audio_id(token)
        if audio_id is not None:
            audio_tokens.append(audio_id)

    usable = len(audio_tokens) - (len(audio_tokens) % 7)
    audio_tokens = audio_tokens[:usable]

    print(f"row: {args.row}")
    print(f"context_length: {context_length}")
    print(f"prompt_tokens: {prompt_length}")
    print(f"audio_tokens: {len(audio_tokens)}")
    if audio_tokens and args.output is not None:
        audio = decode(audio_tokens).detach().cpu().numpy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output, audio, SAMPLE_RATE)
        print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
