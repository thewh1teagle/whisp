from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accelerate import Accelerator
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.checkpoint import load_checkpoint, resume_step, save_checkpoint
from src.config import parse_args
from src.data import make_dataloaders
from src.evaluate import evaluate
from src.model import build_model
from src.optimizer import build_optimizer, build_scheduler
from src.tokenization import build_tokenizer, save_tokenizer


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if accelerator.is_main_process else None

    tokenizer = build_tokenizer()
    if accelerator.is_main_process:
        save_tokenizer(output_dir / "tokenizer.json")

    train_loader, eval_loader = make_dataloaders(args, tokenizer)

    if args.resume:
        model = load_checkpoint(args.resume)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.resume}")
    else:
        model = build_model(
            max_position_embeddings=args.max_position_embeddings,
            attn_implementation=args.attn_implementation,
        )

    total_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    opt_step = 0
    if args.resume and not args.reset_steps:
        opt_step = resume_step(args.resume, scheduler)

    global_step = opt_step * args.gradient_accumulation_steps
    optimizer.zero_grad()

    for epoch in range(math.ceil(args.epochs)):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True, disable=not accelerator.is_main_process)
        loss_sum = 0.0
        loss_count = 0

        for batch in pbar:
            if opt_step >= total_steps:
                break

            with accelerator.autocast():
                out = model(**batch)
                loss = out.loss

            if not torch.isfinite(loss):
                lengths = batch["attention_mask"].sum(dim=1).detach().cpu().tolist()
                raise RuntimeError(
                    f"Non-finite loss at opt_step={opt_step} global_step={global_step} "
                    f"lengths={lengths} mixed_precision={args.mixed_precision}"
                )

            accelerator.backward(loss / args.gradient_accumulation_steps)
            loss_sum += loss.item()
            loss_count += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    if accelerator.is_main_process:
                        pbar.write(
                            f"Skipping optimizer step {opt_step + 1}: non-finite grad_norm={grad_norm.item()}"
                        )
                    optimizer.zero_grad()
                    loss_sum = 0.0
                    loss_count = 0
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                train_loss = loss_sum / loss_count
                pbar.set_postfix(step=opt_step, loss=f"{train_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if accelerator.is_main_process and writer and opt_step % args.logging_steps == 0:
                    writer.add_scalar("train/loss", train_loss, opt_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], opt_step)

                if opt_step % args.eval_steps == 0:
                    metrics = evaluate(
                        model,
                        eval_loader,
                        accelerator,
                        max_batches=args.eval_max_batches,
                    )
                    if accelerator.is_main_process and writer:
                        for name, value in metrics.items():
                            writer.add_scalar(name, value, opt_step)
                    if accelerator.is_main_process:
                        pbar.write(
                            " ".join(
                                [f"step={opt_step}"]
                                + [f"{name}={value:.4f}" for name, value in metrics.items()]
                            )
                        )

                if accelerator.is_main_process and opt_step % args.save_steps == 0:
                    save_checkpoint(
                        accelerator.unwrap_model(model),
                        output_dir,
                        step=opt_step,
                        loss=train_loss,
                        save_total_limit=args.save_total_limit,
                    )

    if accelerator.is_main_process:
        final_loss = loss_sum / max(loss_count, 1)
        save_checkpoint(
            accelerator.unwrap_model(model),
            output_dir,
            step=opt_step,
            loss=final_loss,
            save_total_limit=args.save_total_limit,
        )
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
