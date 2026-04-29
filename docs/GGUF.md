# GGUF Notes

These notes document the current Whisp GGUF experiment so the next agent does
not need to rediscover the same export/runtime issues.

## Current State

The checkpoint at:

```text
outputs/michael-overfit/step-400
```

was exported to:

```text
outputs/michael-overfit/step-400/gguf/whisp-f16.gguf
```

Observed size:

```text
440 MiB
```

The exported GGUF loads and runs through a small raw-token llama.cpp runner.
The runner generated a row0 WAV at:

```text
outputs/michael-overfit/row0_gguf.wav
```

That file was about 2.39 seconds at 24 kHz.

## Why A Custom Runner Exists

Whisp uses a custom `tokenizers` WordLevel tokenizer with tokens like:

```text
<speaker><spk_0></speaker><text>...</text><audio>
<audio_123>...
```

llama.cpp text tokenization is not compatible with this tokenizer. For GGUF
inference, do not rely on `llama-cli` prompt text tokenization. Instead:

1. Use `tokenizers.Tokenizer.from_file(...)` in Python.
2. Format the prompt with `src.tokenization.format_prompt(...)`.
3. Encode to exact token IDs.
4. Pass those IDs directly into llama.cpp.

The helper binary is:

```text
tools/gguf_token_runner
```

Source:

```text
tools/gguf_token_runner.cpp
```

It accepts:

```console
tools/gguf_token_runner <model.gguf> <comma_token_ids> <max_new_tokens>
```

It returns generated token IDs as a comma-separated list. Python then maps
those IDs back through `tokenizer.id_to_token(...)`, keeps `<audio_N>` tokens,
trims to a multiple of 7, and decodes with `src.codec.decode(...)`.

## Inference Command

```console
uv run scripts/infer_gguf.py \
  --model outputs/michael-overfit/step-400/gguf/whisp-f16.gguf \
  --tokenizer outputs/michael-overfit/step-400/tokenizer.json \
  --jsonl data/michael-gold-v1/train_20.jsonl \
  --row 0 \
  --output outputs/michael-overfit/row0_gguf.wav \
  --max-new-tokens 240
```

## Building The Runner

After building llama.cpp, compile the runner with:

```console
g++ -std=c++17 -O2 tools/gguf_token_runner.cpp \
  -Itools/llama.cpp/include \
  -Itools/llama.cpp/ggml/include \
  -Ltools/llama.cpp/build/bin \
  -Wl,-rpath,$PWD/tools/llama.cpp/build/bin \
  -l:libllama.so.0.0.1 \
  -l:libggml.so.0.10.0 \
  -l:libggml-base.so.0.10.0 \
  -l:libggml-cpu.so.0.10.0 \
  -o tools/gguf_token_runner
```

The exact `.so` names may change when llama.cpp is updated. If the link step
fails, inspect:

```console
find tools/llama.cpp/build -maxdepth 4 -type f \( -name 'libllama*' -o -name 'libggml*' \)
```

## llama.cpp Patches Needed

Vanilla llama.cpp did not load this exported Qwen3-MoE correctly because the
model has leading dense MLP layers before MoE layers:

```text
mlp_only_layers = [0, 1, 2, 3, 4, 5]
```

The local `tools/llama.cpp` checkout was patched so Qwen3-MoE can preserve and
run those dense layers.

Important patched areas:

```text
tools/llama.cpp/convert_hf_to_gguf.py
tools/llama.cpp/gguf-py/gguf/constants.py
tools/llama.cpp/src/llama-model.cpp
tools/llama.cpp/src/models/qwen3moe.cpp
```

What the patches do:

- Write `leading_dense_block_count` from HF `mlp_only_layers`.
- Add dense FFN tensors to the Qwen3-MoE GGUF tensor map.
- Load dense FFN tensors for leading layers instead of expecting MoE gate
  tensors on every block.
- In the Qwen3-MoE graph, run dense `ffn_gate/ffn_up/ffn_down` for leading
  layers and MoE FFN for later layers.
- Read the local `tokenizer.json` directly to avoid AutoTokenizer injecting
  unrelated Qwen special tokens.

Without these patches, llama.cpp reported missing tensors such as:

```text
blk.0.ffn_gate_inp.weight
```

because it expected every layer to be an MoE layer.

## Export Notes

The GGUF tokenizer metadata is only enough for llama.cpp to load the model.
Actual inference bypasses llama.cpp tokenization. Do not treat the GGUF as a
self-contained text-tokenizable model yet.

The converter currently stores tokenizer metadata as a GPT-2-like tokenizer
with a dummy merge so llama.cpp initializes a BPE tokenizer. This is a loading
workaround, not the real Whisp tokenizer.

Use the checkpoint `tokenizer.json` for real prompt IDs.

## Sampling Notes

The current runner uses greedy argmax over the full vocabulary. That was enough
to produce audio for row0, but future quality work may need:

- temperature/top-k/top-p sampling,
- repetition controls,
- optional restriction to `<audio_N>` tokens after the audio prompt starts,
- stop on `</audio>` or `</s>`.

If generation returns no audio tokens, first inspect the raw generated IDs and
their tokenizer strings before changing the codec path.

## Codec Reminder

Generated `<audio_N>` values are SNAC code IDs in Whisp depth-first order:

```text
c, m0, f0, f1, m1, f2, f3
```

Always trim generated audio tokens to a multiple of 7 before decode.

