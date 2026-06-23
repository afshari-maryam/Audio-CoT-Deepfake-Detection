"""Training script for COLMBO-DF (Section 4 – Experimental Setup).

Default configuration:
  - WavLM-base-plus encoder (frozen)
  - 6-layer QFormer projector (trainable)
  - Llama 3.2-1B-Instruct LLM (frozen; use --unfreeze_llm to train it too)
  - Supervised fine-tuning with next-token prediction loss
  - ShortCoT mode by default

Usage
-----
    python train.py \
        --manifest_train data/fakereason_train.json \
        --manifest_eval  data/fakereason_eval.json \
        --output_dir     checkpoints/shortcot \
        --mode           shortcot \
        --batch_size     4 \
        --grad_accum     4 \
        --epochs         3
"""

import argparse
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ModelConfig, TrainConfig, DRIVE_ROOT
from dataset import FAKEREASONDataset, collate_fn
from model import ColmboDF


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train COLMBO-DF")
    p.add_argument("--manifest_train", default=f"{DRIVE_ROOT}/fakereason_train.json")
    p.add_argument("--manifest_eval",  default=f"{DRIVE_ROOT}/fakereason_eval.json")
    p.add_argument("--output_dir",     default=f"{DRIVE_ROOT}/checkpoints")
    p.add_argument("--mode",           default="shortcot",
                   choices=["cot", "shortcot", "nocot"])
    p.add_argument("--encoder_name",   default="microsoft/wavlm-base-plus")
    p.add_argument("--llm_name",       default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--num_query_tokens", type=int, default=32)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--grad_accum",     type=int, default=4)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--epochs",         type=int, default=3)
    p.add_argument("--max_audio_len",  type=int, default=80000)
    p.add_argument("--max_text_len",   type=int, default=1024)
    p.add_argument("--warmup_ratio",   type=float, default=0.05)
    p.add_argument("--logging_steps",  type=int, default=50)
    p.add_argument("--save_steps",     type=int, default=500)
    p.add_argument("--unfreeze_llm",   action="store_true",
                   help="Also fine-tune the LLM weights (Unfreeze setting)")
    p.add_argument("--resume",         default=None,
                   help="Path to a checkpoint directory to resume from")
    return p.parse_args()


def save_checkpoint(model: ColmboDF, optimizer, scheduler, step: int, out_dir: str):
    ckpt_dir = Path(out_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector": model.projector.state_dict(),
            "llm":       model.llm.state_dict() if not all(
                not p.requires_grad for p in model.llm.parameters()
            ) else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step":      step,
        },
        ckpt_dir / "state.pt",
    )
    model.tokenizer.save_pretrained(ckpt_dir)
    print(f"  Saved checkpoint → {ckpt_dir}")


def load_checkpoint(model: ColmboDF, optimizer, scheduler, resume_dir: str) -> int:
    state = torch.load(Path(resume_dir) / "state.pt", map_location="cpu")
    model.projector.load_state_dict(state["projector"])
    if state.get("llm") is not None:
        model.llm.load_state_dict(state["llm"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"  Resumed from step {state['step']}")
    return int(state["step"])


def evaluate(model: ColmboDF, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            waveform1 = batch["waveform1"].to(device)
            waveform2 = batch["waveform2"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            out = model(waveform1, waveform2, input_ids, labels=labels)
            if out.loss is not None:
                total_loss += out.loss.item()
                n += 1
    model.train()
    return total_loss / max(n, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────────────
    print("Loading model …")
    model = ColmboDF(
        encoder_name=args.encoder_name,
        llm_name=args.llm_name,
        num_query_tokens=args.num_query_tokens,
        freeze_encoder=True,
        freeze_llm=not args.unfreeze_llm,
    ).to(device)

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("Loading datasets …")
    train_ds = FAKEREASONDataset(
        args.manifest_train,
        tokenizer=model.tokenizer,
        mode=args.mode,
        max_text_len=args.max_text_len,
        max_audio_len=args.max_audio_len,
    )
    eval_ds = FAKEREASONDataset(
        args.manifest_eval,
        tokenizer=model.tokenizer,
        mode=args.mode,
        max_text_len=args.max_text_len,
        max_audio_len=args.max_audio_len,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume:
        global_step = load_checkpoint(model, optimizer, scheduler, args.resume)

    # ── Mixed precision scaler ───────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Training loop ────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            waveform1 = batch["waveform1"].to(device)
            waveform2 = batch["waveform2"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                out = model(waveform1, waveform2, input_ids, labels=labels)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    pbar.set_postfix(loss=f"{out.loss.item():.4f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if global_step % args.save_steps == 0:
                    eval_loss = evaluate(model, eval_loader, device)
                    print(f"\n  Step {global_step} | eval loss: {eval_loss:.4f}")
                    save_checkpoint(
                        model, optimizer, scheduler, global_step, args.output_dir
                    )

    # ── Final checkpoint ─────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, scheduler, global_step, args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
