import os
import torch
from dataclasses import dataclass


@dataclass
class BaselineConfig:
    # Dataset
    dataset_name: str = "trungbb8/vietnamese-news-corpus-tokenized"
    num_examples: int = 14_929_815
    max_length: int = 256

    # Model
    teacher_name: str = "vinai/phobert-base"
    student_num_layers: int = 6
    student_hidden_size: int = 768
    student_num_heads: int = 12
    student_intermediate_size: int = 3072

    # Distillation Loss
    temperature: float = 2.0
    alpha: float = 0.4
    beta: float = 0.2
    gamma: float = 0.4
    mlm_probability: float = 0.15

    # Training
    per_device_batch_size: int = int(os.environ.get("PER_DEVICE_BATCH_SIZE", 16))
    num_gpus: int = int(os.environ.get("NUM_GPUS", 2))
    effective_batch_size: int = int(os.environ.get("EFFECTIVE_BATCH_SIZE", 1024))
    gradient_accumulation_steps: int = effective_batch_size // (
        per_device_batch_size * num_gpus
    )
    epochs: int = int(os.environ.get("EPOCHS", 1))
    learning_rate: float = 5e-4
    warmup_steps: int = 1500
    grad_clip: float = 1.0
    log_every: int = 100
    evaluate_every: int = 10000
    save_every: int = 100000

    # Output
    output_dir: str = "./distilphobert_checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb: bool = True
