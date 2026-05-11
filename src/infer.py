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
from src.tokenization import DEFAULT_AUDIO_VOCAB_SIZE, format_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate speech with a Whisp checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker-id", type=int, required=True)
    parser.add_argument("--num-speakers", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--language", default="en-us")
    parser.add_argument("--phonemes", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def generate_audio_tokens(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[int]:
    audio_start = tokenizer.token_to_id("<audio_0>")
    audio_last = tokenizer.token_to_id(f"<audio_{DEFAULT_AUDIO_VOCAB_SIZE - 1}>")
    stop_ids = {tokenizer.token_to_id("</audio>"), tokenizer.token_to_id("</s>")}
    if audio_start is None or audio_last is None or None in stop_ids:
        raise ValueError("Tokenizer is missing audio tokens")
    audio_end = audio_last + 1
    allowed_ids = list(range(audio_start, audio_end)) + list(stop_ids)
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        top_k=top_k,
        use_cache=True,
        pad_token_id=tokenizer.token_to_id("<pad>"),
        eos_token_id=list(stop_ids),
        prefix_allowed_tokens_fn=lambda _batch_id, _input_ids: allowed_ids,
    )
    new_ids = generated[0, input_ids.shape[1] :].tolist()
    audio_tokens = [token_id - audio_start for token_id in new_ids if audio_start <= token_id < audio_end]
    return audio_tokens[: len(audio_tokens) - (len(audio_tokens) % 7)]


def save_wav(path: Path, audio: torch.Tensor) -> None:
    samples = audio.detach().cpu().numpy()
    samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    model = load_checkpoint(args.checkpoint, num_speakers=args.num_speakers).to(device).eval()

    phonemes = args.text if args.phonemes else phonemize_text(args.text, language=args.language)
    prompt = format_prompt(args.speaker_id, phonemes)
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    max_new_tokens = args.max_new_tokens - (args.max_new_tokens % 7)

    with torch.inference_mode():
        audio_tokens = generate_audio_tokens(
            model,
            tokenizer,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    if not audio_tokens:
        raise RuntimeError("Model did not generate usable audio tokens")

    decode_device = torch.device("cuda" if torch.cuda.is_available() else device)
    audio = decode(audio_tokens, device=decode_device)
    save_wav(args.output, audio)
    print(f"phonemes: {phonemes}")
    print(f"audio_tokens: {len(audio_tokens)}")
    print(f"duration: {audio.numel() / SAMPLE_RATE:.2f}s")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
