# ONNX Notes

These notes document the current Whisp ONNX experiment and its rough edges.

## Current State

ONNX export helpers exist at:

```text
src/onnx/export.py
src/onnx/infer.py
scripts/export_onnx.py
scripts/infer_onnx.py
```

ONNX is experimental. The GGUF path is currently more complete for this model.

## The Main Problem

The model is `Qwen3MoeForCausalLM`. PyTorch ONNX export does not cleanly lower
the default Qwen3-MoE expert implementation in this setup.

The core issue is the expert routing / expert matmul path. HF/PyTorch can run
it, but ONNX export has trouble tracing or lowering the MoE expert op in a
portable way.

## Export Wrapper

`src/onnx/export.py` wraps the model as a logits-only module:

```text
input_ids, attention_mask -> logits
```

It exports with `use_cache=False`. This avoids having to export KV-cache inputs
and outputs for the first prototype.

The wrapper also uses a static causal mask buffer for the chosen sequence
length. This made export simpler than trying to preserve every dynamic mask
path.

## Vectorized Expert Patch

The export script includes a local vectorized replacement for expert forward:

```text
patch_vectorized_experts(...)
```

It replaces the selected-expert implementation with a more direct tensor form
using gathers and `einsum`.

This is an export workaround only. It is not used by training code and should
not be moved into `src/model.py` unless it is validated for correctness and
speed.

## Command Shape

Example export command:

```console
uv run scripts/export_onnx.py \
  --checkpoint outputs/michael-overfit/step-400 \
  --output outputs/michael-overfit/step-400/onnx/model.onnx \
  --tokenizer outputs/michael-overfit/step-400/tokenizer.json \
  --jsonl data/michael-gold-v1/train_20.jsonl \
  --row 0 \
  --sequence-length 320 \
  --opset 18 \
  --attn-implementation eager \
  --experts-implementation vectorized
```

Important flags:

- `--sequence-length` controls the static context size.
- `--dynamic-axes` exists, but static axes are currently more reliable.
- `--dynamo` exists, but the legacy exporter has been friendlier so far.
- `--experts-implementation vectorized` uses the local ONNX-friendly expert
  replacement.

## ONNX Inference Shape

`src/onnx/infer.py` is simple static-context greedy inference:

1. Load `tokenizer.json`.
2. Format prompt with `format_prompt(...)`.
3. Pad to the exported ONNX context length.
4. Run ONNX Runtime.
5. Take argmax at the current final token position.
6. Append the generated token.
7. Extract `<audio_N>` tokens.
8. Trim to a multiple of 7.
9. Decode with `src.codec.decode(...)`.

Example:

```console
uv run scripts/infer_onnx.py \
  --model outputs/michael-overfit/step-400/onnx/model.onnx \
  --tokenizer outputs/michael-overfit/step-400/tokenizer.json \
  --jsonl data/michael-gold-v1/train_20.jsonl \
  --row 0 \
  --output outputs/michael-overfit/row0_onnx.wav \
  --max-new-tokens 240
```

## Gotchas

The ONNX graph is not an efficient autoregressive runtime yet:

- No KV-cache export.
- Static context means each generated token reruns the full graph.
- CPU ONNX Runtime is expected to be slow for longer samples.
- The export currently targets logits, not sampling.
- The SNAC decoder still runs outside ONNX through `src.codec.decode(...)`.

Do not compare ONNX runtime speed to llama.cpp/GGUF until KV-cache export and
proper runtime sampling are implemented.

## Dependency Notes

The repo currently includes ONNX-related dependencies in `pyproject.toml`:

```text
onnx
onnxruntime
onnxscript
```

`transformers` was upgraded during this work because Qwen3-MoE support and
export behavior depend on newer HF versions. If export starts failing after a
dependency update, first check the installed `transformers`, `torch`, and ONNX
versions.

## Recommended Next Steps

For ONNX to become a real deployment target:

1. Validate logits numerically against PyTorch for the same prompt.
2. Add a small parity script that compares top token IDs over several steps.
3. Export a KV-cache version.
4. Add sampling outside the ONNX graph.
5. Decide whether MoE should stay, or whether a dense model is better for ONNX.

For now, keep ONNX changes isolated in `src/onnx/` and scripts. Do not change
training/model code just to make ONNX export easier unless the model design
itself changes.

