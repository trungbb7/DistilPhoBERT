from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BaselineConfig:
    # Dataset
    dataset_name: str = "ademax/binhvq-news-corpus"
    subset_size: int = 5_000
    text_column: str = "content"
    max_length: int = 128
    use_vncorenlp: bool = False
    vncorenlp_dir: Optional[str] = None
    vncorenlp_batch_size: int = 128

    # Model
    teacher_name: str = "vinai/phobert-base"
    student_num_layers: int = 6
    student_hidden_size: int = 768
    student_num_heads: int = 12
    student_intermediate_size: int = 3072

    # Distillation
    temperature: float = 2.0
    alpha: float = 0.5
    mlm_probability: float = 0.15

    # Training
    batch_size: int = 16
    num_steps: int = 500
    learning_rate: float = 5e-4
    warmup_steps: int = 50
    grad_clip: float = 1.0
    log_every: int = 50
    save_every: int = 200

    # Output
    output_dir: str = "./distilphobert_checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb: bool = False