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
from src.tokenization import audio_id_from_token, audio_stop_token_ids, audio_token_ids


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


def next_token(logits: torch.Tensor, *, temperature: float, top_p: float, top_k: int) -> int:
    if temperature <= 0:
        return int(logits.argmax().item())

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.numel())
        threshold = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < threshold, -torch.inf)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > top_p
        remove[0] = False
        logits[sorted_indices[remove]] = -torch.inf

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


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
    audio_ids = audio_token_ids(tokenizer)
    stop_ids = audio_stop_token_ids(tokenizer)
    if any(token_id is None for token_id in audio_ids + stop_ids):
        raise ValueError("Tokenizer is missing audio tokens")
    allowed_ids = audio_ids + stop_ids
    audio_tokens: list[int] = []

    for _ in range(max_new_tokens):
        logits = model(input_ids=input_ids).logits[0, -1]
        masked_logits = torch.full_like(logits, -torch.inf)
        masked_logits[allowed_ids] = logits[allowed_ids]

        token_id = next_token(masked_logits, temperature=temperature, top_p=top_p, top_k=top_k)
        if token_id in stop_ids:
            break

        audio_tokens.append(audio_id_from_token(tokenizer.id_to_token(token_id)))

        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([input_ids, token_tensor], dim=1)

    usable = len(audio_tokens) - (len(audio_tokens) % 7)
    return audio_tokens[:usable]


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
    prompt = f"<s><speaker><spk_{args.speaker_id}></speaker><text>{phonemes}</text><audio>"
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

    audio = decode(audio_tokens, device=device)
    save_wav(args.output, audio)
    print(f"phonemes: {phonemes}")
    print(f"audio_tokens: {len(audio_tokens)}")
    print(f"duration: {audio.numel() / SAMPLE_RATE:.2f}s")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
