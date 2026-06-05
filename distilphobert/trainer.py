import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
import wandb
import logging

from config import BaselineConfig
from data import load_and_prepare_data
from models import build_student_model, DistillationLoss
from utils import sync_tensor_across_gpus

log = logging.getLogger(__name__)


def evaluate(config, student, teacher, val_dataloader, criterion):
    student.eval()
    total_val_loss_accum = torch.zeros(4, device=config.device)
    steps = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(config.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                config.device, non_blocking=True
            )
            labels = batch["labels"].to(config.device, non_blocking=True)

            with autocast("cuda"):
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                student_outputs = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                _, loss_dict = criterion(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    labels,
                    attention_mask,
                    student_outputs.hidden_states[-1],
                    teacher_outputs.hidden_states[-1],
                )

            total_val_loss_accum[0] += loss_dict["total"]
            total_val_loss_accum[1] += loss_dict["mlm"]
            total_val_loss_accum[2] += loss_dict["kd"]
            total_val_loss_accum[3] += loss_dict["cos"]
            steps += 1

    if steps > 0:
        total_val_loss_accum /= steps

    synced_val_losses = sync_tensor_across_gpus(total_val_loss_accum)
    student.train()

    return {
        "total": synced_val_losses[0].item(),
        "mlm": synced_val_losses[1].item(),
        "kd": synced_val_losses[2].item(),
        "cos": synced_val_losses[3].item(),
    }


def train(rank, world_size, config: BaselineConfig):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    config.device = torch.device(f"cuda:{rank}")

    if rank == 0:
        log.info("=" * 60)
        log.info("DistilPhoBERT Baseline Training")
        log.info("=" * 60)

    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.teacher_name)
    teacher = AutoModelForMaskedLM.from_pretrained(config.teacher_name).to(
        config.device
    )
    teacher.eval()

    student = build_student_model(config, teacher, rank=rank).to(config.device)
    student = DDP(student, device_ids=[rank])

    train_dataset, val_dataset = load_and_prepare_data(rank, world_size, config)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-6,
    )

    total_optimization_steps = (
        config.num_examples // config.effective_batch_size
    ) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_optimization_steps,
    )
    scaler = GradScaler("cuda")

    criterion = DistillationLoss(
        temperature=config.temperature,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
    )

    if rank == 0 and config.use_wandb:
        wandb.init(project="distilphobert", config=vars(config), name="Full training")

    if rank == 0:
        log.info("Starting training...")

    student.train()
    global_step = 0
    total_loss_accum = 0.0
    start_time = time.time()
    step_time = time.time()

    optimizer.zero_grad()
    for epoch in range(config.epochs):
        for local_step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(config.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                config.device, non_blocking=True
            )
            labels = batch["labels"].to(config.device, non_blocking=True)

            with autocast("cuda"):
                with torch.no_grad():
                    teacher_outputs = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                student_outputs = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                loss, loss_dict = criterion(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    labels,
                    attention_mask,
                    student_outputs.hidden_states[-1],
                    teacher_outputs.hidden_states[-1],
                )
                loss /= config.gradient_accumulation_steps

            is_accumulating = (local_step + 1) % config.gradient_accumulation_steps != 0

            if is_accumulating:
                with student.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            losses_to_sync = torch.stack(
                [
                    loss_dict["total"],
                    loss_dict["mlm"],
                    loss_dict["kd"],
                    loss_dict["cos"],
                ]
            )
            synced_losses = sync_tensor_across_gpus(losses_to_sync)

            total_loss_accum += synced_losses[0].item()

            if rank == 0 and (global_step + 1) % config.log_every == 0:
                current_time = time.time()
                interval_throughput = (
                    config.log_every * config.per_device_batch_size * world_size
                ) / (current_time - step_time)
                avg_loss = total_loss_accum / config.log_every

                log.info(
                    f"Step {local_step+1}/Epoch {epoch+1} | loss={synced_losses[0].item():.4f} | avg={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | throughput={interval_throughput:.4f}"
                )

                total_loss_accum = 0.0
                step_time = current_time

                if config.use_wandb:
                    wandb.log(
                        {
                            "step": global_step + 1,
                            "loss/total": synced_losses[0].item(),
                            "loss/mlm": synced_losses[1].item(),
                            "loss/kd": synced_losses[2].item(),
                            "loss/cosin": synced_losses[3].item(),
                            "lr": scheduler.get_last_lr()[0],
                        },
                        step=global_step + 1,
                    )

            if global_step % config.evaluate_every == 0:
                val_metrics = evaluate(
                    config, student, teacher, val_dataloader, criterion
                )
                if rank == 0:
                    log.info(
                        f"[EVAL] Step {global_step+1} | Val Loss: {val_metrics['total']:.4f}"
                    )
                    if config.use_wandb:
                        wandb.log(
                            {f"val/{k}": v for k, v in val_metrics.items()},
                            step=global_step + 1,
                        )

            if rank == 0 and global_step % config.save_every == 0:
                ckpt_path = os.path.join(config.output_dir, f"step_{global_step}")
                student.module.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                torch.save(
                    {
                        "global_step": global_step,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                    },
                    os.path.join(ckpt_path, "training_state.pt"),
                )

            global_step += 1

        if rank == 0:
            student.module.push_to_hub(
                "trungbb8/distilphobert-checkpoints",
                commit_message=f"Epoch: {epoch+1} - checkpoint",
            )

    if rank == 0:
        final_path = os.path.join(config.output_dir, "final")
        student.module.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        log.info(f"Training complete in {(time.time() - start_time)/60:.1f} minutes")

    return student, tokenizer
