from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from src.tokenization import build_vocab


MLP_ONLY_LAYERS = [0, 1, 2, 3, 4, 5]
REF_SPEAKER_EMBEDDING_SIZE = 1024


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


class WhispForConditionalGeneration(nn.Module):
    def __init__(self, lm: Qwen3MoeForCausalLM, ref_speaker_embedding_size: int = REF_SPEAKER_EMBEDDING_SIZE):
        super().__init__()
        self.lm = lm
        self.ref_speaker_adapter = nn.Sequential(
            nn.LayerNorm(ref_speaker_embedding_size),
            nn.Linear(ref_speaker_embedding_size, lm.config.hidden_size),
        )
        self.config = lm.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        ref_speaker_embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if ref_speaker_embeddings is None:
            raise ValueError("ref_speaker_embeddings is required")

        token_embeddings = self.lm.get_input_embeddings()(input_ids)
        speaker_embeddings = self.ref_speaker_adapter(
            ref_speaker_embeddings.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        ).unsqueeze(1)

        inputs_embeds = torch.cat(
            [token_embeddings[:, :1], speaker_embeddings, token_embeddings[:, 1:]],
            dim=1,
        )

        if attention_mask is not None:
            speaker_mask = torch.ones(
                attention_mask.shape[0],
                1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask[:, :1], speaker_mask, attention_mask[:, 1:]], dim=1)

        if labels is not None:
            speaker_labels = torch.full(
                (labels.shape[0], 1),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([labels[:, :1], speaker_labels, labels[:, 1:]], dim=1)

        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        self.lm.save_pretrained(save_directory, **kwargs)
        torch.save(
            {
                "ref_speaker_adapter": self.ref_speaker_adapter.state_dict(),
                "ref_speaker_embedding_size": REF_SPEAKER_EMBEDDING_SIZE,
            },
            f"{save_directory}/ref_speaker_adapter.pt",
        )

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, ref_speaker_embedding_size: int = REF_SPEAKER_EMBEDDING_SIZE):
        lm = Qwen3MoeForCausalLM.from_pretrained(checkpoint_dir)
        model = cls(lm, ref_speaker_embedding_size=ref_speaker_embedding_size)
        adapter_path = f"{checkpoint_dir}/ref_speaker_adapter.pt"
        adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=False)
        model.ref_speaker_adapter.load_state_dict(adapter_state["ref_speaker_adapter"])
        return model


def build_model(
    *,
    num_speakers: int,
    vocab_size: int | None = None,
    dtype: torch.dtype | None = None,
) -> WhispForConditionalGeneration:
    config = build_config(num_speakers=num_speakers, vocab_size=vocab_size)
    model = WhispForConditionalGeneration(Qwen3MoeForCausalLM(config))
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
