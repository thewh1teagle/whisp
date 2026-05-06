from __future__ import annotations

import math

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader


def evaluate(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    accelerator: Accelerator,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.inference_mode():
        for index, batch in enumerate(eval_loader):
            if max_batches is not None and index >= max_batches:
                break

            with accelerator.autocast():
                loss = model(**batch).loss

            batch_size = int(batch["input_ids"].shape[0])
            losses = accelerator.gather_for_metrics(loss.detach().float().repeat(batch_size))
            total_loss += float(losses.sum().item())
            total_items += int(losses.numel())

    model.train()

    mean_loss = total_loss / max(total_items, 1)
    return {
        "eval/loss": mean_loss,
        "eval/perplexity": math.exp(min(mean_loss, 20.0)),
    }
