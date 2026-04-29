from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accelerate import Accelerator
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.checkpoint import resume_step, save_checkpoint
from src.config import parse_args
from src.data import make_dataloaders
from src.model import build_model
from src.optimizer import build_optimizer, build_scheduler
from src.tokenization import build_tokenizer, save_tokenizer


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if accelerator.is_main_process else None

    tokenizer = build_tokenizer(num_speakers=args.num_speakers)
    if accelerator.is_main_process:
        save_tokenizer(output_dir / "tokenizer.json", num_speakers=args.num_speakers)

    train_loader, eval_loader = make_dataloaders(args, tokenizer)
    model = build_model(num_speakers=args.num_speakers)

    if args.resume:
        state = load_file(str(Path(args.resume) / "model.safetensors"), device="cpu")
        model.load_state_dict(state)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.resume}")

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

            accelerator.backward(loss / args.gradient_accumulation_steps)
            loss_sum += loss.item()
            loss_count += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                train_loss = loss_sum / loss_count
                pbar.set_postfix(step=opt_step, loss=f"{train_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if accelerator.is_main_process and writer and opt_step % args.logging_steps == 0:
                    writer.add_scalar("train/loss", train_loss, opt_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], opt_step)

                if accelerator.is_main_process and opt_step % args.save_steps == 0:
                    save_checkpoint(
                        accelerator.unwrap_model(model),
                        output_dir,
                        step=opt_step,
                        loss=train_loss,
                        num_speakers=args.num_speakers,
                        save_total_limit=args.save_total_limit,
                    )

    if accelerator.is_main_process:
        final_loss = loss_sum / max(loss_count, 1)
        save_checkpoint(
            accelerator.unwrap_model(model),
            output_dir,
            step=opt_step,
            loss=final_loss,
            num_speakers=args.num_speakers,
            save_total_limit=args.save_total_limit,
        )
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
