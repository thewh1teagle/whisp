from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.checkpoint import load_checkpoint, resume_step, save_checkpoint
from src.config import parse_args
from src.data import make_dataloaders
from src.evaluate import evaluate
from src.model import build_model
from src.optimizer import build_optimizer, build_scheduler
from src.tokenization import build_tokenizer, save_tokenizer


def dataset_format(args) -> str:
    if args.dataset_format != "auto":
        return args.dataset_format
    return "libriheavy-snac" if Path(args.train_dataset).is_dir() else "jsonl"


def infer_num_speakers(args) -> int:
    if args.num_speakers is not None:
        return args.num_speakers

    if dataset_format(args) == "libriheavy-snac":
        manifest_path = Path(args.train_dataset) / "speakers" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return int(manifest["speakers_total"])

    raise ValueError("--num-speakers is required unless LibriHeavy speakers/manifest.json is present")


def total_training_steps(args, train_loader) -> int:
    if args.max_steps is not None:
        return args.max_steps
    try:
        return math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    except TypeError as exc:
        raise ValueError("--max-steps is required for streaming/iterable datasets") from exc


def dataset_stats(args) -> dict:
    if dataset_format(args) != "libriheavy-snac":
        return {}
    stats_path = Path(args.train_dataset) / "stats.json"
    if not stats_path.exists():
        return {}
    return json.loads(stats_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.eval_dataset is None:
        args.eval_dataset = args.train_dataset
    args.num_speakers = infer_num_speakers(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if accelerator.is_main_process else None

    tokenizer = build_tokenizer(num_speakers=args.num_speakers)
    if accelerator.is_main_process:
        save_tokenizer(output_dir / "tokenizer.json", num_speakers=args.num_speakers)

    train_loader, eval_loader = make_dataloaders(args, tokenizer)
    stats = dataset_stats(args)

    if args.resume:
        model = load_checkpoint(args.resume, num_speakers=args.num_speakers)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.resume}")
    else:
        model = build_model(
            num_speakers=args.num_speakers,
            max_position_embeddings=args.max_position_embeddings,
        )

    total_steps = total_training_steps(args, train_loader)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    opt_step = 0
    if args.resume and not args.reset_steps:
        opt_step = resume_step(args.resume, scheduler)

    global_step = opt_step * args.gradient_accumulation_steps
    samples_seen = 0
    audio_tokens_seen = 0
    optimizer.zero_grad()

    for epoch in range(math.ceil(args.epochs)):
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        elif hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        epoch_total = None
        if stats.get("rows"):
            epoch_total = math.ceil(int(stats["rows"]) / args.train_batch_size)
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}",
            total=epoch_total,
            dynamic_ncols=True,
            disable=not accelerator.is_main_process,
        )
        loss_sum = 0.0
        loss_count = 0

        for batch in pbar:
            if opt_step >= total_steps:
                break
            batch_size = int(batch["input_ids"].shape[0])
            samples_seen += batch_size
            audio_tokens_seen += max(0, int((batch["labels"] != -100).sum().item()) - (2 * batch_size))
            audio_hours_seen = audio_tokens_seen / float(stats.get("snac_tokens_per_second", 87.5)) / 3600.0

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
                pbar.set_postfix(
                    step=opt_step,
                    loss=f"{train_loss:.4f}",
                    samples=samples_seen,
                    audio_h=f"{audio_hours_seen:.2f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if accelerator.is_main_process and writer and opt_step % args.logging_steps == 0:
                    writer.add_scalar("train/loss", train_loss, opt_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], opt_step)

                if args.eval_steps > 0 and opt_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, eval_loader, accelerator, args.max_eval_samples)
                    if accelerator.is_main_process:
                        pbar.set_postfix(
                            step=opt_step,
                            loss=f"{train_loss:.4f}",
                            eval=f"{eval_loss:.4f}",
                            samples=samples_seen,
                            audio_h=f"{audio_hours_seen:.2f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        )
                        if writer:
                            writer.add_scalar("eval/loss", eval_loss, opt_step)

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
