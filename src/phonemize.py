from __future__ import annotations

import espeakng_loader
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper


DEFAULT_LANGUAGE = "en-us"


_ESPEAK_READY = False


def _ensure_espeak() -> None:
    global _ESPEAK_READY
    if _ESPEAK_READY:
        return
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    _ESPEAK_READY = True


def phonemize_text(text: str, *, language: str = DEFAULT_LANGUAGE) -> str:
    _ensure_espeak()
    return phonemize(
        text,
        language=language,
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        strip=True,
    )
