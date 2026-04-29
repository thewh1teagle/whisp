from __future__ import annotations

from pathlib import Path

from src.codec import encode
from src.phonemize import phonemize_text


def build_row(
    *,
    audio: str | Path,
    text: str,
    speaker_id: int,
    language: str = "en-us",
) -> dict:
    phonemes = phonemize_text(text, language=language)
    audio_tokens = list(encode(audio).tokens)
    return {
        "audio": str(audio),
        "text": text,
        "phonemes": phonemes,
        "speaker_id": int(speaker_id),
        "audio_tokens": audio_tokens,
    }
