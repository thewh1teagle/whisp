from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.codec import decode, encode
from src.snac_ordering import codes_to_depth_first, depth_first_to_codes


INPUT_WAV = ROOT / "plans" / "codec_roundtrip" / "female1.wav"


def main() -> None:
    encoded = encode(INPUT_WAV)
    tokens = encoded.to_depth_first()
    roundtrip_codes = depth_first_to_codes(tokens)
    roundtrip_tokens = codes_to_depth_first(roundtrip_codes)
    reconstructed = decode(tokens)

    print(f"native lengths: {encoded.lengths}")
    print(f"depth-first tokens: {len(tokens)}")
    print(f"first 14 depth-first tokens: {tokens[:14]}")
    print(f"ordering round-trip exact: {tokens == roundtrip_tokens}")
    print(f"decoded samples without trim metadata: {reconstructed.numel()}")


if __name__ == "__main__":
    main()
