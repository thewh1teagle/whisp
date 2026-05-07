from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from transformers import LogitsProcessor
from tokenizers import Tokenizer

from src.checkpoint import load_checkpoint
from src.codec import SAMPLE_RATE, decode
from src.phonemize import phonemize_text
from src.snac_ordering import codes_to_depth_first
from src.tokenization import DEFAULT_AUDIO_VOCAB_SIZE
from src.tokenization import format_prompt


SNAC_TOKENS_PER_SECOND = 87.5


class AudioTokenLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        prompt_length: int,
        min_audio_tokens: int,
        audio_vocab_size: int = DEFAULT_AUDIO_VOCAB_SIZE,
    ) -> None:
        self.prompt_length = prompt_length
        self.min_audio_tokens = min_audio_tokens

        allowed_tokens = []
        for audio_id in range(audio_vocab_size):
            token_id = tokenizer.token_to_id(f"<audio_{audio_id}>")
            if token_id is not None:
                allowed_tokens.append(token_id)

        self.stop_token_ids = [
            token_id
            for token in ("</audio>", "</s>")
            if (token_id := tokenizer.token_to_id(token)) is not None
        ]
        self.audio_start_id = tokenizer.token_to_id("<audio>")
        self.audio_token_ids = allowed_tokens
        self.audio_token_id_set = set(allowed_tokens)
        self.allowed_token_ids = allowed_tokens + self.stop_token_ids
        if not self.audio_token_ids:
            raise ValueError("Tokenizer does not contain audio tokens")
        if not self.stop_token_ids:
            raise ValueError("Tokenizer does not contain audio stop tokens")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        constrained = torch.full_like(scores, -torch.inf)
        for row_index, row in enumerate(input_ids):
            row_ids = [int(token_id) for token_id in row.detach().cpu().tolist()]
            if self.audio_start_id is not None and self.audio_start_id in row_ids:
                start = len(row_ids) - 1 - row_ids[::-1].index(self.audio_start_id)
                row_ids = row_ids[start + 1 :]
            audio_count = sum(token_id in self.audio_token_id_set for token_id in row_ids)
            allowed = self.allowed_token_ids if audio_count >= self.min_audio_tokens else self.audio_token_ids
            constrained[row_index, allowed] = scores[row_index, allowed]
        return constrained


def parse_args():
    parser = argparse.ArgumentParser(description="Generate speech with a Whisp checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument(
        "--ref-speaker-embedding",
        type=str,
        default=None,
        help="Path to a .pt speaker embedding. May also contain ref_snac_0/1/2.",
    )
    parser.add_argument(
        "--speaker-refs-root",
        type=str,
        default=None,
        help="Root containing speaker_refs.parquet with embedding and optional ref_snac_0/1/2.",
    )
    parser.add_argument("--speaker-id", type=str, default=None, help="Speaker id to load from speaker_refs.parquet.")
    parser.add_argument("--ref-index", type=int, default=0, help="Reference index for --speaker-id.")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional hard cap for generated audio duration. Overrides --max-new-tokens when set.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--language", type=str, default="en-us")
    parser.add_argument("--phonemes", action="store_true", help="Treat --text as already-phonemized input")
    parser.add_argument(
        "--constrained-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict generation to SNAC audio tokens plus audio/end stop tokens.",
    )
    parser.add_argument(
        "--min-audio-tokens",
        type=int,
        default=7,
        help="Minimum generated audio tokens before stop tokens are allowed.",
    )
    return parser.parse_args()


def _ref_audio_tokens_from_item(item: dict) -> list[int] | None:
    if not all(key in item for key in ("ref_snac_0", "ref_snac_1", "ref_snac_2")):
        return None

    return codes_to_depth_first(
        [
            torch.tensor(item["ref_snac_0"], dtype=torch.long),
            torch.tensor(item["ref_snac_1"], dtype=torch.long),
            torch.tensor(item["ref_snac_2"], dtype=torch.long),
        ]
    )


def load_ref_conditioning(args) -> tuple[torch.Tensor, list[int] | None, str]:
    if args.ref_speaker_embedding is not None:
        item = torch.load(args.ref_speaker_embedding, map_location="cpu", weights_only=False)
        if isinstance(item, dict):
            embedding = item["embedding"]
            ref_audio_tokens = _ref_audio_tokens_from_item(item)
            source = args.ref_speaker_embedding
        else:
            embedding = item
            ref_audio_tokens = None
            source = args.ref_speaker_embedding
        return embedding.float().view(1, -1), ref_audio_tokens, source

    if args.speaker_refs_root is None or args.speaker_id is None:
        raise ValueError("Provide either --ref-speaker-embedding or both --speaker-refs-root and --speaker-id")

    parquet_path = Path(args.speaker_refs_root) / "speaker_refs.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"speaker_refs.parquet not found: {parquet_path}")

    schema_names = set(pq.read_schema(parquet_path).names)
    columns = ["speaker_id", "embedding"]
    if "ref_index" in schema_names:
        columns.append("ref_index")
    if {"ref_snac_0", "ref_snac_1", "ref_snac_2"}.issubset(schema_names):
        columns.extend(["ref_snac_0", "ref_snac_1", "ref_snac_2"])

    table = pq.read_table(parquet_path, columns=columns)
    table = table.filter(pc.equal(table["speaker_id"], args.speaker_id))
    if "ref_index" in table.column_names:
        table = table.filter(pc.equal(table["ref_index"], args.ref_index))
    if table.num_rows == 0:
        raise ValueError(f"No speaker ref found for speaker_id={args.speaker_id!r} ref_index={args.ref_index}")

    item = table.slice(0, 1).to_pylist()[0]
    embedding = torch.tensor(item["embedding"], dtype=torch.float32).view(1, -1)
    ref_audio_tokens = _ref_audio_tokens_from_item(item)
    source = f"{parquet_path}:speaker_id={args.speaker_id}:ref_index={args.ref_index}"
    return embedding, ref_audio_tokens, source


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


def generated_audio_tokens(tokenizer: Tokenizer, generated_ids: list[int], prompt_token_count: int) -> list[int]:
    audio_start_id = tokenizer.token_to_id("<audio>")
    start = prompt_token_count

    if audio_start_id is not None:
        audio_starts = [idx for idx, token_id in enumerate(generated_ids) if int(token_id) == audio_start_id]
        if audio_starts:
            start = audio_starts[-1] + 1
        elif len(generated_ids) <= prompt_token_count:
            start = 0

    sliced = audio_tokens_from_ids(tokenizer, generated_ids[start:])
    if sliced:
        return sliced

    # Some HF generation paths with inputs_embeds return only generated ids,
    # while others return prompt-shaped placeholder ids plus generated ids.
    return audio_tokens_from_ids(tokenizer, generated_ids)


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
    model = load_checkpoint(checkpoint_dir).to(device).eval()

    ref_speaker_embeddings, ref_audio_tokens, ref_source = load_ref_conditioning(args)
    phonemes = args.text if args.phonemes else phonemize_text(args.text, language=args.language)
    prompt = format_prompt(phonemes, ref_audio_tokens=ref_audio_tokens)
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    ref_speaker_embeddings = ref_speaker_embeddings.to(device)
    max_new_tokens = args.max_new_tokens
    if args.max_seconds is not None:
        max_new_tokens = max(7, int(args.max_seconds * SNAC_TOKENS_PER_SECOND))
        max_new_tokens -= max_new_tokens % 7
    logits_processor = None
    if args.constrained_decode:
        min_audio_tokens = args.min_audio_tokens
        if ref_audio_tokens is not None:
            min_audio_tokens += len(ref_audio_tokens)
        logits_processor = [
            AudioTokenLogitsProcessor(
                tokenizer=tokenizer,
                prompt_length=input_ids.shape[1],
                min_audio_tokens=min_audio_tokens,
            )
        ]

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            ref_speaker_embeddings=ref_speaker_embeddings,
            max_new_tokens=max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>"),
        )

    generated_ids = generated[0].detach().cpu().tolist()
    audio_tokens = generated_audio_tokens(tokenizer, generated_ids, input_ids.shape[1])
    if not audio_tokens:
        raise RuntimeError("Model did not generate usable audio tokens")

    audio = decode(audio_tokens)
    write_wav(args.output, audio, SAMPLE_RATE)
    print(f"ref_source: {ref_source}")
    print(f"ref_audio_tokens: {0 if ref_audio_tokens is None else len(ref_audio_tokens)}")
    print(f"prompt_tokens: {input_ids.shape[1]}")
    print(f"generated_tokens: {len(generated_ids)}")
    print(f"audio_tokens: {len(audio_tokens)}")
    print(f"duration: {audio.numel() / SAMPLE_RATE:.2f}s")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
