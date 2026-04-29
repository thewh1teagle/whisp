from __future__ import annotations

import argparse
import json
import sys
from types import MethodType
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer
from transformers import Qwen3MoeForCausalLM

from src.tokenization import format_prompt


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model: Qwen3MoeForCausalLM, sequence_length: int):
        super().__init__()
        self.model = model
        min_dtype = torch.finfo(torch.float32).min
        mask = torch.triu(torch.ones((sequence_length, sequence_length), dtype=torch.bool), diagonal=1)
        mask = torch.where(mask, min_dtype, 0.0).view(1, 1, sequence_length, sequence_length)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=self.causal_mask,
            use_cache=False,
        ).logits


def vectorized_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    selected_gate_up = self.gate_up_proj[top_k_index]
    gate_up = torch.einsum("nh,nkoh->nko", hidden_states, selected_gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    current = self.act_fn(gate) * up

    selected_down = self.down_proj[top_k_index]
    current = torch.einsum("nko,nkho->nkh", current, selected_down)
    current = current * top_k_weights.unsqueeze(-1)
    return current.sum(dim=1).to(hidden_states.dtype)


def patch_vectorized_experts(model: torch.nn.Module) -> None:
    for module in model.modules():
        if all(hasattr(module, name) for name in ("gate_up_proj", "down_proj", "act_fn")):
            module.forward = MethodType(vectorized_experts_forward, module)


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Whisp checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa"],
        default="eager",
        help="Attention implementation to use during export.",
    )
    parser.add_argument(
        "--experts-implementation",
        choices=["vectorized", "eager", "grouped_mm"],
        default="vectorized",
        help="MoE expert implementation to use during export. Vectorized is ONNX-friendly but computes selected experts directly.",
    )
    parser.add_argument(
        "--dynamo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the new torch.export ONNX path. The legacy tracer is currently friendlier to eager MoE.",
    )
    parser.add_argument(
        "--dynamic-axes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export dynamic batch/sequence axes. Static axes are currently more reliable for MoE export.",
    )
    return parser.parse_args()


def export_onnx(
    checkpoint: Path,
    output: Path,
    *,
    opset: int = 18,
    sequence_length: int = 64,
    tokenizer_path: Path | None = None,
    jsonl_path: Path | None = None,
    row_index: int = 0,
    attn_implementation: str = "eager",
    experts_implementation: str = "eager",
    dynamo: bool = False,
    dynamic_axes: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = Qwen3MoeForCausalLM.from_pretrained(
        str(checkpoint),
        dtype=torch.float32,
        attn_implementation=attn_implementation,
    )
    model.set_attn_implementation(attn_implementation)
    if experts_implementation == "vectorized":
        model.set_experts_implementation("eager")
        patch_vectorized_experts(model)
    else:
        model.set_experts_implementation(experts_implementation)
    model.eval()
    model.config.use_cache = False

    if jsonl_path is not None:
        tokenizer_path = tokenizer_path or checkpoint / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        row = rows[row_index]
        prompt = format_prompt(int(row["speaker_id"]), row["phonemes"])
        ids = tokenizer.encode(prompt).ids
        if sequence_length < len(ids):
            sequence_length = len(ids)
        pad_id = tokenizer.token_to_id("<pad>") or 0
        ids = ids + [pad_id] * (sequence_length - len(ids))
        input_ids = torch.tensor([ids], dtype=torch.long)
    else:
        input_ids = torch.ones((1, sequence_length), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    wrapper = LogitsWrapper(model, sequence_length).eval()

    with torch.inference_mode():
        axes = (
            {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            }
            if dynamic_axes
            else None
        )
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(output),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=dynamo,
        )


def main() -> None:
    args = parse_args()
    output = args.output or args.checkpoint / "onnx" / "model.onnx"
    export_onnx(
        args.checkpoint,
        output,
        opset=args.opset,
        sequence_length=args.sequence_length,
        tokenizer_path=args.tokenizer,
        jsonl_path=args.jsonl,
        row_index=args.row,
        attn_implementation=args.attn_implementation,
        experts_implementation=args.experts_implementation,
        dynamo=args.dynamo,
        dynamic_axes=args.dynamic_axes,
    )
    size_mib = output.stat().st_size / 1024 / 1024
    print(f"wrote: {output}")
    print(f"size: {size_mib:.1f} MiB")


if __name__ == "__main__":
    main()
