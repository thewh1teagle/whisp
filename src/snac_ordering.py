from __future__ import annotations

from typing import Iterable

import torch


def codes_to_depth_first(codes: Iterable[torch.Tensor]) -> list[int]:
    """Flatten SNAC's 1:2:4 code streams in 2cent-style depth-first order."""
    code_tuple = tuple(code.detach().cpu().to(dtype=torch.long) for code in codes)
    if len(code_tuple) != 3:
        raise ValueError(f"Expected 3 SNAC code streams, got {len(code_tuple)}")

    coarse, mid, fine = [code.reshape(-1).tolist() for code in code_tuple]
    if len(mid) != len(coarse) * 2 or len(fine) != len(coarse) * 4:
        raise ValueError(
            "Expected SNAC code lengths in 1:2:4 ratio, got "
            f"{len(coarse)}:{len(mid)}:{len(fine)}"
        )

    tokens: list[int] = []
    for idx, coarse_code in enumerate(coarse):
        mid_idx = idx * 2
        fine_idx = idx * 4
        tokens.extend(
            [
                int(coarse_code),
                int(mid[mid_idx]),
                int(fine[fine_idx]),
                int(fine[fine_idx + 1]),
                int(mid[mid_idx + 1]),
                int(fine[fine_idx + 2]),
                int(fine[fine_idx + 3]),
            ]
        )
    return tokens


def depth_first_to_codes(tokens: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert 2cent-style depth-first SNAC tokens back into native code streams."""
    token_list = [int(token) for token in tokens]
    if len(token_list) % 7 != 0:
        raise ValueError(f"Depth-first SNAC token count must be divisible by 7, got {len(token_list)}")

    coarse: list[int] = []
    mid: list[int] = []
    fine: list[int] = []

    for offset in range(0, len(token_list), 7):
        c0, m0, f0, f1, m1, f2, f3 = token_list[offset : offset + 7]
        coarse.append(c0)
        mid.extend([m0, m1])
        fine.extend([f0, f1, f2, f3])

    return (
        torch.tensor(coarse, dtype=torch.long).unsqueeze(0),
        torch.tensor(mid, dtype=torch.long).unsqueeze(0),
        torch.tensor(fine, dtype=torch.long).unsqueeze(0),
    )
