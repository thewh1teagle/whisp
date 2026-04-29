# EXPERIMENTS

## 2026-04-29

### LibriTTS-R 360 SNAC/Qwen3-MoE Run

We trained the current Whisp codec-LM architecture on the prepared LibriTTS-R
360-hour SNAC dataset:

```text
text -> phonemes -> Qwen3-MoE LM -> SNAC audio tokens -> SNAC decoder -> wav
```

The run used the scratch training path from `scripts/train_scratch.sh` with
LibriTTS-R data prepared into `dataset/.cache/train.jsonl`,
`dataset/.cache/val.jsonl`, and `dataset/.cache/speaker_map.json`.

After about 8 hours of training, the model reached step `28500` and the loss
had moved from roughly `8.0` down to roughly `4.5`. At this point the speech was
clearly intelligible, which showed that the model had learned the basic
phoneme-to-SNAC-token mapping. The first subjective quality issue was that many
samples sounded robotic.

### Initial Interpretation

The loss drop was meaningful, but a loss around `4.5` is still high for
autoregressive codec-token generation. The model had learned enough coarse
structure for understandable speech, but not enough stable fine acoustic detail
for consistently natural output.

Because Whisp predicts a flattened SNAC stream:

```text
c, m0, f0, f1, m1, f2, f3
```

the later mid/fine codebook tokens matter a lot for timbre, texture, and
naturalness. A model can become intelligible before it becomes good at those
details.

The early conclusion was: do not assume the fix is simply "more data." First
verify codec quality, token ordering, inference constraints, speaker behavior,
and training convergence.

### Constrained Decoding

The first inference issue was that generation was using the full tokenizer
vocabulary. After the `<audio>` prompt, the model could still assign probability
mass to non-audio tokens such as structure, speaker, or text tokens.

We added constrained decoding in `src/infer.py`. During generation, logits are
restricted to:

```text
<audio_0> ... <audio_4095>
</audio>
</s>
```

Stop tokens are blocked until at least one full SNAC frame worth of audio tokens
has been generated. This made inference better behaved and removed one obvious
source of invalid continuations.

### Sampling and Duration Controls

After constrained decoding, we added practical inference controls for quality
and stability:

```text
--temperature
--top-p
--top-k
--repetition-penalty
--no-repeat-ngram-size
--max-seconds
```

The `--max-seconds` option converts a requested duration cap into the
corresponding number of SNAC tokens using the current depth-first 7-token frame
layout. This gives a hard upper bound on generated audio length and helps test
whether failures are caused by unconstrained over-generation.

Recommended long-sentence diagnostic settings were:

```bash
uv run src/infer.py \
  --checkpoint outputs/whisp/step-28500 \
  --num-speakers <N> \
  --speaker-id 307 \
  --text "<text>" \
  --output out.wav \
  --temperature 0.55 \
  --top-p 0.9 \
  --top-k 30 \
  --max-seconds 14
```

### Speaker 307 Result

Speaker `307` at checkpoint `step-28500` produced a strong short-form result
with constrained decoding.

Prompt:

```text
Every useful system starts as a rough experiment, but with patience, careful
testing, and steady improvements, it can become something people trust every
day.
```

Subjective result: sounded great.

This was an important result because it showed that the architecture, codec
path, token ordering, and checkpoint are capable of producing natural-sounding
speech for at least some speakers and utterance lengths.

### Long-Form Drift

The same speaker and checkpoint had problems on a longer prompt.

Prompt:

```text
When a project begins, the first results are often messy and uncertain, but
every experiment teaches us something useful. If we keep measuring carefully,
listening honestly, and improving the weak parts one by one, the system slowly
becomes clearer, stronger, and more reliable.
```

Subjective result: the model hallucinated after "listening honestly".

The likely failure mode is autoregressive drift. The model can stay aligned for
shorter utterances, but longer continuations accumulate token-level errors until
the generated audio diverges from the text. This is different from the earlier
"robotic speech" issue: speaker 307 proves the model can sound good, while the
longer prompt exposes sequence stability limits.

### Current Thoughts

The most likely current bottlenecks are:

- training is still early for high-quality codec-token generation;
- LibriTTS-R utterances are mostly short or medium length, so the model is not
  strongly trained for paragraph-length synthesis;
- different speakers have very different data quality and amount;
- the model has no explicit duration, alignment, or prosody conditioning;
- long-form generation compounds small audio-token mistakes.

More hours of short utterances should improve quality, but it may not fully
solve long-form hallucination. For paragraph synthesis, the model needs either
chunked inference, long-form training examples, or stronger alignment/duration
control.

### Practical Next Steps

Short term:

- continue training while validation loss is still falling;
- use constrained decoding by default;
- use lower temperature/top-p for long prompts;
- use `--max-seconds` to bound generation;
- synthesize long text sentence by sentence, then concatenate with small
  silences or crossfades.

Diagnostics:

- compare good and bad speakers by total clean minutes;
- run SNAC encode/decode roundtrips for good and bad speakers;
- compute or inspect per-speaker validation loss;
- test whether bad speakers improve with more training or remain poor because
  of data quality;
- compare short, medium, and long prompts for the same speaker and checkpoint.

Medium term:

- train on curated clean speakers first to establish a stronger quality ceiling;
- add longer examples if one-shot paragraph generation is required;
- consider duration or alignment conditioning;
- keep a stable set of subjective test prompts for regression testing.

### Published Artifacts

Prepared dataset:

```text
https://huggingface.co/datasets/thewh1teagle/whisp-libritts-r-snac
```

Published checkpoints:

```text
https://huggingface.co/thewh1teagle/whisper-snac-tts
```
