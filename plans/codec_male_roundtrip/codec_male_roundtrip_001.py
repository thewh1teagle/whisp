from __future__ import annotations

import sys
import urllib.request
import wave
from pathlib import Path

import librosa
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.codec import SAMPLE_RATE, decode, encode


AUDIO_URL = "https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/male1.wav"
PLAN_DIR = Path(__file__).resolve().parent
INPUT_WAV = PLAN_DIR / "male1.wav"
RECON_24K_WAV = PLAN_DIR / "male1_snac_reconstructed_24k.wav"
UPSAMPLED_44K_WAV = PLAN_DIR / "male1_snac_reconstructed_44k.wav"
UPSAMPLED_48K_WAV = PLAN_DIR / "male1_snac_reconstructed_48k.wav"


def write_wav(path: Path, samples: np.ndarray | torch.Tensor, sample_rate: int) -> None:
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    samples_i16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
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
    reconstructed_np = reconstructed.detach().cpu().numpy()

    reconstructed_44k = librosa.resample(
        reconstructed_np,
        orig_sr=SAMPLE_RATE,
        target_sr=44_100,
        res_type="soxr_hq",
    )
    reconstructed_48k = librosa.resample(
        reconstructed_np,
        orig_sr=SAMPLE_RATE,
        target_sr=48_000,
        res_type="soxr_hq",
    )

    write_wav(RECON_24K_WAV, reconstructed_np, SAMPLE_RATE)
    write_wav(UPSAMPLED_44K_WAV, reconstructed_44k, 44_100)
    write_wav(UPSAMPLED_48K_WAV, reconstructed_48k, 48_000)

    print(f"input: {INPUT_WAV}")
    print(f"codes: {encoded.lengths}")
    print(f"depth-first tokens: {len(encoded.tokens)}")
    print(f"24k: {RECON_24K_WAV}")
    print(f"44.1k: {UPSAMPLED_44K_WAV}")
    print(f"48k: {UPSAMPLED_48K_WAV}")


if __name__ == "__main__":
    main()
