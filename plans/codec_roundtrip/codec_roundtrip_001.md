# Codec Round Trip 001

Validate the first SNAC codec wrapper.

Run:

```console
uv run plans/codec_roundtrip/codec_roundtrip_001.py
```

The script downloads `female1.wav`, encodes it with SNAC 24 kHz, decodes the
codes back to audio, and writes `plans/codec_roundtrip/reconstructed.wav`.
