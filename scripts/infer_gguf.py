from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf
from tokenizers import Tokenizer

from src.codec import SAMPLE_RATE, decode
from src.tokenization import format_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a WAV from a Whisp GGUF model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--runner", type=Path, default=Path("tools/gguf_token_runner"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def audio_token_id(token: str) -> int | None:
    if not token.startswith("<audio_") or not token.endswith(">"):
        return None
    return int(token.removeprefix("<audio_").removesuffix(">"))


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    row = rows[args.row]

    prompt = format_prompt(int(row["speaker_id"]), row["phonemes"])
    prompt_ids = tokenizer.encode(prompt).ids
    prompt_arg = ",".join(str(token_id) for token_id in prompt_ids)

    result = subprocess.run(
        [str(args.runner), str(args.model), prompt_arg, str(args.max_new_tokens)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    generated_ids = [int(item) for item in result.stdout.strip().split(",") if item]
    audio_tokens: list[int] = []
    for token_id in generated_ids:
        token = tokenizer.id_to_token(token_id)
        if token in {"</s>", "</audio>"}:
            break
        value = audio_token_id(token)
        if value is not None:
            audio_tokens.append(value)

    usable = len(audio_tokens) - (len(audio_tokens) % 7)
    audio_tokens = audio_tokens[:usable]
    if not audio_tokens:
        raise RuntimeError(f"no audio tokens generated; runner stderr:\n{result.stderr}")

    audio = decode(audio_tokens).detach().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, SAMPLE_RATE)
    print(f"row: {args.row}")
    print(f"prompt_tokens: {len(prompt_ids)}")
    print(f"generated_ids: {len(generated_ids)}")
    print(f"audio_tokens: {len(audio_tokens)}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
