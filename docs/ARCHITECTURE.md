# Whisp Architecture

Whisp is a small codec-LM TTS experiment:

```text
text -> phonemes -> Qwen3-MoE LM -> SNAC audio tokens -> SNAC decoder -> wav
```

## Token Flow

Training rows are semantic JSONL:

```json
{
  "audio": "path.wav",
  "text": "Hello",
  "phonemes": "həlˈoʊ",
  "speaker_id": 0,
  "audio_tokens": [2492, 426, 2825]
}
```

`src/data.py` tokenizes on the fly:

```text
<s><speaker><spk_0></speaker><text>həlˈoʊ</text><audio>
<audio_2492><audio_426>...</audio></s>
```

Prompt tokens are masked with `-100`; only target audio/end tokens train the LM.

## Codec

`src/codec.py` uses `hubertsiuzdak/snac_24khz`.

SNAC native streams are converted to one canonical depth-first sequence in
`src/snac_ordering.py`:

```text
c, m0, f0, f1, m1, f2, f3
```

The audio vocab is:

```text
<audio_0> ... <audio_4095>
```

matching SNAC's 4096-entry codebooks.

## Tokenizer

`src/tokenization.py` builds a deterministic WordLevel tokenizer:

- Piper/espeak phoneme characters from `phoneme_id_map`
- speaker tokens: `<spk_N>`
- audio tokens: `<audio_N>`
- structure tokens: `<speaker>`, `<text>`, `<audio>`, etc.

## Model

`src/model.py` builds `Qwen3MoeForCausalLM`.

Current shape:

```text
hidden_size: 384
layers: 12
dense layers: 0-5
MoE layers: 6-11
experts: 32
experts per token: 1
stored params: ~245M with 6,190 speakers
active params: ~25M with 6,190 speakers
```

## Training

`src/train.py` follows the renikud-style Accelerate loop:

```text
JSONL -> WhispDataset -> collator -> Qwen3-MoE -> checkpoint
```

Checkpoints save:

- `model.safetensors`
- HF config
- `tokenizer.json`
- `train_state.json`

## Scripts

```text
scripts/prepare_overfit.sh
scripts/train_overfit.sh
scripts/train_scratch.sh
scripts/infer.sh
scripts/model_info.sh
```

Full LibriTTS-R preparation will add a speaker map and dataset scanner later.
