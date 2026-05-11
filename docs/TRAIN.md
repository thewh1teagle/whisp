# Training

## Prepare LibriTTS-R 360

Prepared LibriTTS-R SNAC data is available at:

```text
https://huggingface.co/datasets/thewh1teagle/whisp-libritts-r-snac
```

```bash
scripts/prepare_libritts_r_360.sh
```

This writes:

```text
dataset/.cache/train.jsonl
dataset/.cache/val.jsonl
dataset/.cache/speaker_map.json
```

## Sanity Check

```bash
wc -l dataset/.cache/train.jsonl dataset/.cache/val.jsonl
cat dataset/.cache/speaker_map.json | head
```

## Train

With `jq`:

```bash
scripts/train_scratch.sh --num-speakers "$(jq -r .num_speakers dataset/.cache/speaker_map.json)"
```

Without `jq`:

```bash
scripts/train_scratch.sh --num-speakers "$(uv run python -c "import json; print(json.load(open('dataset/.cache/speaker_map.json'))['num_speakers'])")"
```

Checkpoints are written to:

```text
outputs/whisp
```

Published checkpoints are available at:

```text
https://huggingface.co/thewh1teagle/whisper-snac-tts
```

Default training runs for `100` epochs. Stop/resume from checkpoints as needed.

## Train LibriHeavy SNAC

The local `libriheavy-snac/` parquet dataset is streamed with the Hugging Face
`datasets` library. It is not loaded into memory and it is not pretokenized on
disk.

```bash
scripts/train_libriheavy_snac.sh --max-steps 100000
```

By default, `torch` is resolved from the PyTorch CUDA 13.0 index through the
default `cu130` dependency group. On CUDA 12.8 hosts with older drivers, run
with the `cu128` group instead:

```bash
uv run --no-default-groups --group cu128 accelerate launch src/train.py ...
```

`--num-speakers` is inferred from `libriheavy-snac/speakers/manifest.json`.
Raw LibriHeavy speaker IDs are remapped to contiguous tokenizer IDs at startup
from the small `speakers/` parquet set.
