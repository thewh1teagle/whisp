from __future__ import annotations

import torch
from accelerate import Accelerator


def evaluate(model, eval_loader, accelerator: Accelerator, max_samples: int) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0

    for batch in eval_loader:
        batch_size = int(batch["input_ids"].shape[0])
        if sample_count >= max_samples:
            break
        if sample_count + batch_size > max_samples:
            keep = max_samples - sample_count
            batch = {key: value[:keep] for key, value in batch.items()}
            batch_size = keep

        with accelerator.autocast():
            with torch.no_grad():
                loss = model(**batch).loss
        loss = accelerator.reduce(loss.detach(), reduction="mean")
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size

    model.train()
    return loss_sum / max(sample_count, 1)
