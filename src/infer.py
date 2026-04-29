from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tokenizers import Tokenizer

from src.checkpoint import load_checkpoint
from src.codec import SAMPLE_RATE, decode
from src.phonemize import phonemize_text
from src.tokenization import format_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate speech with a Whisp checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker-id", type=int, required=True)
    parser.add_argument("--num-speakers", type=int, required=True)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--language", type=str, default="en-us")
    parser.add_argument("--phonemes", action="store_true", help="Treat --text as already-phonemized input")
    return parser.parse_args()


def audio_tokens_from_ids(tokenizer: Tokenizer, ids: list[int]) -> list[int]:
    audio_tokens = []
    for token_id in ids:
        token = tokenizer.id_to_token(int(token_id))
        if token is None:
            continue
        if token.startswith("<audio_") and token.endswith(">"):
            audio_tokens.append(int(token.removeprefix("<audio_").removesuffix(">")))
        elif token in {"</audio>", "</s>"}:
            break
    usable = len(audio_tokens) - (len(audio_tokens) % 7)
    return audio_tokens[:usable]


def write_wav(path: str | Path, samples: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> None:
    samples_np = samples.detach().cpu().numpy()
    samples_i16 = (np.clip(samples_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples_i16.tobytes())


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    model = load_checkpoint(checkpoint_dir, num_speakers=args.num_speakers).to(device).eval()

    phonemes = args.text if args.phonemes else phonemize_text(args.text, language=args.language)
    prompt = format_prompt(args.speaker_id, phonemes)
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>"),
        )

    new_ids = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    audio_tokens = audio_tokens_from_ids(tokenizer, new_ids)
    if not audio_tokens:
        raise RuntimeError("Model did not generate usable audio tokens")

    audio = decode(audio_tokens)
    write_wav(args.output, audio, SAMPLE_RATE)
    print(f"audio_tokens: {len(audio_tokens)}")
    print(f"duration: {audio.numel() / SAMPLE_RATE:.2f}s")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
