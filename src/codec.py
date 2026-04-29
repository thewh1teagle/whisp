from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
from snac import SNAC

from src.snac_ordering import codes_to_depth_first, depth_first_to_codes


SAMPLE_RATE = 24_000
SNAC_REPO_ID = "hubertsiuzdak/snac_24khz"


@dataclass(frozen=True)
class SNACEncoding:
    tokens: tuple[int, ...]
    sample_rate: int = SAMPLE_RATE
    audio_length: int | None = None

    @property
    def lengths(self) -> tuple[int, ...]:
        frames = len(self.tokens) // 7
        return (frames, frames * 2, frames * 4)

    def to_depth_first(self) -> list[int]:
        return list(self.tokens)


_SNAC_MODEL_CACHE: dict[tuple[str, str], SNAC] = {}


def _device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _snac_model(device: str | torch.device | None = None) -> SNAC:
    resolved = _device(device)
    key = (SNAC_REPO_ID, str(resolved))
    if key not in _SNAC_MODEL_CACHE:
        model = SNAC.from_pretrained(SNAC_REPO_ID).eval().to(resolved)
        _SNAC_MODEL_CACHE[key] = model
    return _SNAC_MODEL_CACHE[key]


def _load_audio(audio: str | Path | np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(audio, str | Path):
        samples, _ = librosa.load(Path(audio), sr=SAMPLE_RATE, mono=True)
        tensor = torch.from_numpy(samples)
    elif isinstance(audio, np.ndarray):
        tensor = torch.from_numpy(audio)
    elif isinstance(audio, torch.Tensor):
        tensor = audio.detach()
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)!r}")

    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 3:
        raise ValueError(f"Expected audio with 1, 2, or 3 dimensions, got {tensor.ndim}")

    if tensor.shape[1] != 1:
        tensor = tensor.mean(dim=1, keepdim=True)

    return tensor.clamp(-1.0, 1.0).contiguous()


def encode(
    audio: str | Path | np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> SNACEncoding:
    """Encode 24 kHz mono audio into 2cent-style depth-first SNAC tokens."""
    audio_tensor = _load_audio(audio)
    audio_length = int(audio_tensor.shape[-1])
    model = _snac_model(device)

    with torch.inference_mode():
        codes = model.encode(audio_tensor.to(next(model.parameters()).device))

    return SNACEncoding(
        tokens=tuple(codes_to_depth_first(codes)),
        audio_length=audio_length,
    )


def decode(
    encoded: SNACEncoding | Iterable[int],
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Decode 2cent-style depth-first SNAC tokens to a mono waveform tensor."""
    if isinstance(encoded, SNACEncoding):
        tokens = encoded.tokens
        audio_length = encoded.audio_length
    else:
        tokens = tuple(int(token) for token in encoded)
        audio_length = None

    model = _snac_model(device)
    model_device = next(model.parameters()).device
    codes = depth_first_to_codes(tokens)
    device_codes = [code.to(model_device, dtype=torch.long) for code in codes]

    with torch.inference_mode():
        audio = model.decode(device_codes).detach().cpu()

    audio = audio.squeeze(0).squeeze(0).clamp(-1.0, 1.0)
    if audio_length is not None:
        audio = audio[:audio_length]
    return audio.contiguous()
