from __future__ import annotations

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from src.tokenization import build_vocab


MLP_ONLY_LAYERS = [0, 1, 2, 3, 4, 5]


def build_config(
    *,
    num_speakers: int,
    vocab_size: int | None = None,
    max_position_embeddings: int = 4096,
) -> Qwen3MoeConfig:
    if vocab_size is None:
        vocab_size = len(build_vocab(num_speakers=num_speakers))

    return Qwen3MoeConfig(
        vocab_size=vocab_size,
        hidden_size=640,
        intermediate_size=1280,
        num_hidden_layers=10,
        num_attention_heads=5,
        num_key_value_heads=5,
        max_position_embeddings=max_position_embeddings,
        mlp_only_layers=MLP_ONLY_LAYERS,
        moe_intermediate_size=1280,
        num_experts=20,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        rms_norm_eps=1e-6,
        rope_theta=500_000.0,
        router_aux_loss_coef=0.001,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )


def build_model(
    *,
    num_speakers: int,
    vocab_size: int | None = None,
    dtype: torch.dtype | None = None,
) -> Qwen3MoeForCausalLM:
    config = build_config(num_speakers=num_speakers, vocab_size=vocab_size)
    model = Qwen3MoeForCausalLM(config)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def count_parameters(model: torch.nn.Module, *, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def estimate_active_parameters(model: Qwen3MoeForCausalLM) -> int:
    config = model.config
    dense_total = 0
    expert_total = 0

    for name, parameter in model.named_parameters():
        if "experts." in name:
            expert_total += parameter.numel()
        else:
            dense_total += parameter.numel()

    active_expert_total = expert_total * config.num_experts_per_tok / config.num_local_experts
    return int(dense_total + active_expert_total)
