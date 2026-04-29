# Training

## Prepare LibriTTS-R 360

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

Default training runs for `100` epochs. Stop/resume from checkpoints as needed.
