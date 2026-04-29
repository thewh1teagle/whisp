from __future__ import annotations

import json
import shutil
from pathlib import Path

from transformers import Qwen3MoeForCausalLM

from src.tokenization import save_tokenizer


def save_checkpoint(
    model,
    output_dir: Path,
    *,
    step: int,
    loss: float,
    num_speakers: int,
    save_total_limit: int,
) -> None:
    ckpt_dir = output_dir / f"step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir), safe_serialization=True)
    save_tokenizer(ckpt_dir / "tokenizer.json", num_speakers=num_speakers)
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step, "loss": loss}, indent=2))

    checkpoints = sorted(output_dir.glob("step-*"), key=lambda path: int(path.name.split("-")[1]))
    while len(checkpoints) > save_total_limit:
        shutil.rmtree(checkpoints.pop(0))


def load_checkpoint(checkpoint_dir: str | Path, *, num_speakers: int):
    return Qwen3MoeForCausalLM.from_pretrained(str(checkpoint_dir))


def resume_step(checkpoint: str | Path, scheduler) -> int:
    state_path = Path(checkpoint) / "train_state.json"
    if not state_path.exists():
        return 0
    step = int(json.loads(state_path.read_text()).get("step", 0))
    for _ in range(step):
        scheduler.step()
    return step
