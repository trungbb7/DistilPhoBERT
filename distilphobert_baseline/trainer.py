import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import BaselineConfig
from .data import apply_mlm_masking, load_and_prepare_data
from .losses import DistillationLoss
from .modeling import build_student_model


log = logging.getLogger(__name__)


def train(config: BaselineConfig):
    log.info("=" * 60)
    log.info("DistilPhoBERT Baseline Training")
    log.info("=" * 60)
    log.info("Device: %s", config.device)
    log.info("Subset size: %s", config.subset_size)
    log.info("Batch size: %s", config.batch_size)
    log.info("Steps: %s", config.num_steps)

    os.makedirs(config.output_dir, exist_ok=True)

    log.info("\n[1/5] Loading teacher model: %s", config.teacher_name)
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_name)
    teacher = AutoModelForMaskedLM.from_pretrained(config.teacher_name)
    teacher = teacher.to(config.device)
    teacher.eval()

    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    log.info("Teacher loaded: %.1fM parameters", teacher_params)

    log.info("\n[2/5] Building student model...")
    teacher_base = AutoModel.from_pretrained(config.teacher_name)
    student = build_student_model(config, teacher_base)
    student = student.to(config.device)
    del teacher_base

    log.info("\n[3/5] Loading dataset...")
    dataset = load_and_prepare_data(config, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(config.device == "cuda"),
    )
    log.info("DataLoader ready: %s batches per epoch", len(dataloader))

    log.info("\n[4/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-6,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.num_steps,
    )

    criterion = DistillationLoss(
        temperature=config.temperature,
        alpha=config.alpha,
    )

    if config.use_wandb:
        import wandb

        wandb.init(
            project="distilphobert",
            config=vars(config),
            name=f"baseline-{config.subset_size}samples",
        )

    log.info("\n[5/5] Starting training...")
    log.info("-" * 60)

    student.train()
    global_step = 0
    total_loss_accum = 0.0
    start_time = time.time()

    while global_step < config.num_steps:
        for batch in dataloader:
            if global_step >= config.num_steps:
                break

            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            original_ids, masked_ids, labels = apply_mlm_masking(
                input_ids,
                tokenizer,
                config.mlm_probability,
            )

            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=original_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_outputs.logits

            student_outputs = student(
                input_ids=masked_ids,
                attention_mask=attention_mask,
            )
            student_logits = student_outputs.logits

            loss, loss_dict = criterion(
                student_logits,
                teacher_logits,
                labels,
                attention_mask,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)

            optimizer.step()
            scheduler.step()

            global_step += 1
            total_loss_accum += loss_dict["total"]

            if global_step % config.log_every == 0:
                avg_loss = total_loss_accum / config.log_every
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta_secs = (config.num_steps - global_step) / steps_per_sec

                log.info(
                    "Step %4d/%d | loss=%.4f (mlm=%.4f, kd=%.4f) | avg=%.4f | lr=%.2e | ETA=%.1fmin",
                    global_step,
                    config.num_steps,
                    loss_dict["total"],
                    loss_dict["mlm"],
                    loss_dict["kd"],
                    avg_loss,
                    scheduler.get_last_lr()[0],
                    eta_secs / 60,
                )
                total_loss_accum = 0.0

                if config.use_wandb:
                    import wandb

                    wandb.log(
                        {
                            "step": global_step,
                            "loss/total": loss_dict["total"],
                            "loss/mlm": loss_dict["mlm"],
                            "loss/kd": loss_dict["kd"],
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )

            if global_step % config.save_every == 0:
                ckpt_path = os.path.join(config.output_dir, f"step_{global_step}")
                student.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                log.info("Checkpoint saved: %s", ckpt_path)

    final_path = os.path.join(config.output_dir, "final")
    student.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    total_time = time.time() - start_time
    log.info("=" * 60)
    log.info("Training complete in %.1f minutes", total_time / 60)
    log.info("Final model saved: %s", final_path)
    log.info("=" * 60)

    return student, tokenizer