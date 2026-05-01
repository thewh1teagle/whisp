from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Qwen3-TTS speaker_encoder.* weights.")
    parser.add_argument("--repo", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output", type=Path, default=Path("data/qwen_speaker/qwen3_tts_0_6b_speaker_encoder.pt"))
    args = parser.parse_args()

    model_path = hf_hub_download(args.repo, "model.safetensors")
    state = load_file(model_path, device="cpu")
    speaker_state = {
        key.removeprefix("speaker_encoder."): value.detach().cpu()
        for key, value in state.items()
        if key.startswith("speaker_encoder.")
    }
    if not speaker_state:
        raise RuntimeError(f"No speaker_encoder.* tensors found in {args.repo}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(speaker_state, args.output)

    params = sum(t.numel() for t in speaker_state.values())
    dtypes = sorted({str(t.dtype) for t in speaker_state.values()})
    print(f"repo: {args.repo}")
    print(f"output: {args.output}")
    print(f"tensors: {len(speaker_state)}")
    print(f"params: {params}")
    print(f"dtypes: {', '.join(dtypes)}")
    print(f"fp32_mb: {params * 4 / 1024 / 1024:.2f}")
    print(f"bf16_or_fp16_mb: {params * 2 / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
