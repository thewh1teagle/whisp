from __future__ import annotations

import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.codec import SAMPLE_RATE, decode, encode


AUDIO_URL = "https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav"
PLAN_DIR = Path(__file__).resolve().parent
INPUT_WAV = PLAN_DIR / "female1.wav"
OUTPUT_WAV = PLAN_DIR / "reconstructed.wav"


def write_wav(path: Path, samples: torch.Tensor, sample_rate: int) -> None:
    samples_np = samples.detach().cpu().numpy()
    samples_i16 = (np.clip(samples_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples_i16.tobytes())


def main() -> None:
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_WAV.exists():
        urllib.request.urlretrieve(AUDIO_URL, INPUT_WAV)

    encoded = encode(INPUT_WAV)
    reconstructed = decode(encoded)
    write_wav(OUTPUT_WAV, reconstructed, SAMPLE_RATE)

    duration = reconstructed.numel() / SAMPLE_RATE
    print(f"input: {INPUT_WAV}")
    print(f"codes: {encoded.lengths}")
    print(f"depth-first tokens: {len(encoded.tokens)}")
    print(f"output: {OUTPUT_WAV}")
    print(f"duration: {duration:.2f}s")


if __name__ == "__main__":
    main()
