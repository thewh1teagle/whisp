# Qwen Speaker Conditioning Handoff

## Goal

Move Whisp from closed-set speaker-ID conditioning to zero-shot reference-speaker conditioning using a frozen Qwen3-TTS speaker encoder.

Current target training row:

```json
{
  "phonemes": "həlˈoʊ",
  "audio_tokens": [2492, 426, 2825],
  "ref_speaker_embedding": "dataset/.cache/qwen_spk_emb/123/utt_456.pt"
}
```

`speaker_id` may exist in preparation metadata for pairing/evaluation, but the model should not receive speaker IDs.

## Conversation Decisions

- Use `Qwen/Qwen3-TTS-12Hz-0.6B-Base` speaker encoder first.
- Extract only `speaker_encoder.*` from Qwen model weights.
- Keep Qwen speaker encoder frozen.
- Precompute one `.pt` embedding per reference audio and store only the path in JSONL.
- Do not preserve old prompt compatibility in the new training path.
- Remove speaker-ID prompt conditioning:

```text
old: <s><speaker><spk_N></speaker><text>...</text><audio>
new: <s> + adapted speaker embedding token + <text>...</text><audio>
```

## Qwen Speaker Encoder Facts

Checked locally with inline `uv` and HF metadata.

- 0.6B Base speaker encoder:
  - output: `1024`
  - params: `8,854,336`
  - extracted weight file: about `17 MB` bf16
  - fp32 equivalent: about `33.78 MB`
- 1.7B Base speaker encoder:
  - output: `2048`
  - params: `12,001,088`
  - bf16 equivalent: about `22.9 MB`

Use 0.6B because Whisp hidden size is `640`, so the adapter is simple:

```text
LayerNorm(1024) -> Linear(1024, 640)
```

## Implemented Files

- `src/qwen_speaker_model.py`
  - Standalone ECAPA-TDNN-style Qwen speaker encoder.
  - Includes Qwen-style 24 kHz mel frontend.
  - `load_qwen_speaker_encoder(...)` loads extracted weights frozen.

- `scripts/extract_qwen_speaker_encoder.py`
  - Downloads `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
  - Extracts `speaker_encoder.*`.
  - Saves `data/qwen_speaker/qwen3_tts_0_6b_speaker_encoder.pt`.

- `scripts/infer_qwen_speaker_embedding.py`
  - Loads frozen extracted encoder.
  - Produces an embedding `.pt` from a wav.
  - Tested with:

```bash
wget -O data/qwen_speaker/female1.wav \
  https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav

uv run scripts/infer_qwen_speaker_embedding.py \
  --audio data/qwen_speaker/female1.wav \
  --output data/qwen_speaker/female1_embedding.pt
```

Observed result:

```text
mel_shape: (1, 1038, 128)
embedding_shape: (1, 1024)
embedding_mean: 0.000742
embedding_std: 0.337176
embedding_l2: 10.789660
```

Real warmed inference time on same file:

```text
CPU total:  ~95.6 ms
CUDA total: ~2.6 ms
```

The saved embedding `.pt` was about `5.8 KB`; raw `1024` float32 values are about `4 KB`.

## Training Wiring Implemented

- `src/tokenization.py`
  - Removed speaker tokens from vocab.
  - Prompt is now:

```text
<s><text>{phonemes}</text><audio>
```

- `src/data.py`
  - Requires `ref_speaker_embedding`.
  - Loads `.pt` embedding and returns `ref_speaker_embeddings: [B, 1024]`.
  - Labels still mask prompt tokens and train only audio/end tokens.

- `src/model.py`
  - Added `WhispForConditionalGeneration`.
  - Wraps `Qwen3MoeForCausalLM`.
  - Inserts adapted speaker vector after `<s>` as one conditioning token.
  - Adapter:

```python
nn.Sequential(
    nn.LayerNorm(1024),
    nn.Linear(1024, 640),
)
```

- `src/checkpoint.py`
  - Saves/loads `ref_speaker_adapter.pt` alongside HF model checkpoint.

## Validation Already Run

Compile:

```bash
uv run python -m py_compile \
  src/data.py src/model.py src/tokenization.py src/checkpoint.py src/train.py
```

Smoke forward:

```text
batch:
  input_ids: (1, 19)
  labels: (1, 19)
  attention_mask: (1, 19)
  ref_speaker_embeddings: (1, 1024)

model output:
  loss: finite
  logits: (1, 20, 4263)
```

Checkpoint round-trip:

```text
WhispForConditionalGeneration loaded successfully
ref_speaker_adapter exists in checkpoint
```

## LibriHeavy Prep Context

Dataset: `mythicinfinity/libriheavy`

Verified from dataset card / HF metadata:

- repo parquet size: about `1.395 TiB`
- `small`: `509h`, `417` speakers
- `medium`: `5,042h`, `1,531` speakers
- `large`: `50,794h`, `6,736` speakers

If `500h` took `20 min` to prepare, estimate:

- `small`: about `20 min`
- `medium`: about `3.36 h`
- `large`: about `33.9 h`

Do not download all large data unless there is enough NVMe. Prefer shard-by-shard processing:

```text
download/cache shard -> process -> write jsonl -> write .done -> optionally remove raw shard cache
```

Progress should be based on `audio_duration` hours, not bytes. Show examples, hours, rows/sec, and real-time factor.

## Next Work

1. Update/replace dataset preparation to create same-speaker different-utterance pairs.
2. Precompute Qwen speaker embeddings for reference utterances.
3. Emit JSONL rows with `ref_speaker_embedding` paths.
4. Run a small training job on LibriHeavy `small` or LibriTTS-R subset.
5. Add inference support for:
   - ref wav -> Qwen speaker embedding
   - text/phonemes -> Whisp generation with inserted speaker conditioning
6. Consider a warm-start conversion script only later. Directly resuming old checkpoints is not clean because the tokenizer/prompt changed.

