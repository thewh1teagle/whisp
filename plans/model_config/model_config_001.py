from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_model, count_parameters, estimate_active_parameters
from src.tokenization import build_tokenizer


def main() -> None:
    num_speakers = 3
    tokenizer = build_tokenizer(num_speakers=num_speakers)
    model = build_model(num_speakers=num_speakers)
    config = model.config

    print(f"vocab_size: {config.vocab_size}")
    print(f"tokenizer_vocab_size: {tokenizer.get_vocab_size()}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"intermediate_size: {config.intermediate_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {config.num_key_value_heads}")
    print(f"num_experts: {config.num_local_experts}")
    print(f"num_experts_per_tok: {config.num_experts_per_tok}")
    print(f"moe_intermediate_size: {config.moe_intermediate_size}")
    print(f"mlp_only_layers: {config.mlp_only_layers}")
    print(f"max_position_embeddings: {config.max_position_embeddings}")
    print(f"tie_word_embeddings: {config.tie_word_embeddings}")
    print(f"stored_parameters: {count_parameters(model):,}")
    print(f"active_parameters_rough: {estimate_active_parameters(model):,}")


if __name__ == "__main__":
    main()
