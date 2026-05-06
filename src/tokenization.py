from __future__ import annotations

from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Split


# Extracted from phoneme_id_map keys in:
# https://huggingface.co/datasets/rhasspy/piper-checkpoints/blob/main/en/en_US/hfc_female/medium/config.json
PHONEMES = list(
    " !\"#$'(),-.0123456789:;?X^_abcdefghijklmnopqrstuvwxyz"
    "æçðøħŋœǀǁǂǃ"
    "ɐɑɒɓɔɕɖɗɘəɚɛɜɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾ"
    "ʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢʦʰʲ"
    "ˈˌːˑ˞ˤ̧̝̩̪̯̺̻̃̊βεθχᵻ↑↓ⱱ"
)

BASE_SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    # Whole training/generation sequence boundaries.
    "<s>",
    "</s>",
    # Phoneme text conditioning block.
    "<text>",
    "</text>",
    # Optional in-context reference audio conditioning block.
    "<ref_audio>",
    "</ref_audio>",
    # SNAC audio-token generation block.
    "<audio>",
    "</audio>",
]

DEFAULT_AUDIO_VOCAB_SIZE = 4096


def build_vocab(
    *,
    audio_vocab_size: int = DEFAULT_AUDIO_VOCAB_SIZE,
) -> dict[str, int]:
    tokens = []
    tokens.extend(BASE_SPECIAL_TOKENS)
    tokens.extend(PHONEMES)
    tokens.extend(f"<audio_{idx}>" for idx in range(audio_vocab_size))
    return {token: idx for idx, token in enumerate(tokens)}


def build_tokenizer(
    *,
    audio_vocab_size: int = DEFAULT_AUDIO_VOCAB_SIZE,
) -> Tokenizer:
    vocab = build_vocab(audio_vocab_size=audio_vocab_size)
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(Regex(r"<[^>]+>"), behavior="isolated"),
            Split(Regex(r""), behavior="isolated"),
        ]
    )
    tokenizer.add_special_tokens(
        [AddedToken(token, single_word=False, normalized=False) for token in BASE_SPECIAL_TOKENS]
    )
    tokenizer.add_special_tokens(
        [
            AddedToken(f"<audio_{idx}>", single_word=False, normalized=False)
            for idx in range(audio_vocab_size)
        ]
    )
    return tokenizer


def format_audio_tokens(audio_tokens: list[int] | tuple[int, ...]) -> str:
    return "".join(f"<audio_{token}>" for token in audio_tokens)


def format_prompt(
    phonemes: str,
    *,
    ref_audio_tokens: list[int] | tuple[int, ...] | None = None,
) -> str:
    ref = ""
    if ref_audio_tokens is not None:
        ref = f"<ref_audio>{format_audio_tokens(ref_audio_tokens)}</ref_audio>"
    return f"<s>{ref}<text>{phonemes}</text><audio>"


def format_target(audio_tokens: list[int] | tuple[int, ...]) -> str:
    return f"{format_audio_tokens(audio_tokens)}</audio></s>"


def save_tokenizer(
    path: str | Path,
    *,
    audio_vocab_size: int = DEFAULT_AUDIO_VOCAB_SIZE,
) -> None:
    tokenizer = build_tokenizer(audio_vocab_size=audio_vocab_size)
    tokenizer.save(str(path))
