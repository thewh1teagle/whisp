# Whisp

Experimental codec-LM TTS stack.

```text
phonemes -> Qwen3-MoE LM -> SNAC audio tokens -> SNAC decoder -> wav
```

Current focus:

- SNAC 24 kHz codec encode/decode
- character phoneme tokenizer
- speaker-id conditioning
- Qwen3-MoE causal LM training
- PyTorch inference, with experimental ONNX/GGUF notes

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current structure.
