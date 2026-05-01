from __future__ import annotations

import argparse
from pathlib import Path
import sys

import librosa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qwen_speaker_model import load_qwen_speaker_encoder, mel_spectrogram


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a frozen Qwen3-TTS speaker embedding from a wav file.")
    parser.add_argument("--weights", type=Path, default=Path("data/qwen_speaker/qwen3_tts_0_6b_speaker_encoder.pt"))
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/qwen_speaker/embedding.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, config = load_qwen_speaker_encoder(str(args.weights), device=args.device)
    wav, sr = librosa.load(args.audio, sr=config.sample_rate, mono=True)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(args.device)
    with torch.inference_mode():
        mel = mel_spectrogram(wav_tensor, config)
        embedding = model(mel).float().cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embedding": embedding.squeeze(0),
            "audio": str(args.audio),
            "sample_rate": config.sample_rate,
            "source": "Qwen/Qwen3-TTS-12Hz-0.6B-Base speaker_encoder",
        },
        args.output,
    )

    flat = embedding.flatten()
    print(f"audio: {args.audio}")
    print(f"weights: {args.weights}")
    print(f"output: {args.output}")
    print(f"mel_shape: {tuple(mel.shape)}")
    print(f"embedding_shape: {tuple(embedding.shape)}")
    print(f"embedding_mean: {flat.mean().item():.6f}")
    print(f"embedding_std: {flat.std(unbiased=False).item():.6f}")
    print(f"embedding_l2: {torch.linalg.vector_norm(flat).item():.6f}")


if __name__ == "__main__":
    main()
