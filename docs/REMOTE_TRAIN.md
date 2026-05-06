# Remote Train

```bash
rm -rf .venv

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone https://github.com/thewh1teagle/whisp.git
cd whisp
git checkout feature/qwen-speaker-conditioning

uv sync

uv run hf download thewh1teagle/whisp-libriheavy-snac-phonemes-15k \
  --repo-type dataset \
  --local-dir data/whisp-libriheavy-15k

uv run accelerate launch src/train.py \
  --train-dataset data/whisp-libriheavy-15k/data/train \
  --eval-dataset data/whisp-libriheavy-15k/data/validation \
  --speaker-refs-root data/whisp-libriheavy-15k/speaker_refs \
  --output-dir outputs/whisp \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --lr 3e-4 \
  --warmup-steps 100 \
  --logging-steps 10 \
  --eval-steps 500 \
  --eval-max-batches 32 \
  --save-steps 500 \
  --max-position-embeddings 4096
```

TensorBoard:

```bash
uv run tensorboard --logdir outputs/whisp/tensorboard --host 0.0.0.0 --port 6006
```
