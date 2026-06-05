import os
import logging
import torch.distributed as dist
from dotenv import load_dotenv
import wandb
import huggingface_hub

from config import BaselineConfig
from trainer import train
from utils import sanity_check

# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    load_dotenv()

    hf_access_token = os.environ.get("HF_ACCESS_TOKEN")
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    if hf_access_token:
        huggingface_hub.login(token=hf_access_token)
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    config = BaselineConfig()

    log.info("Config:")
    for k, v in vars(config).items():
        log.info(f"  {k}: {v}")
    log.info("")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Run training
    student, tokenizer = train(local_rank, world_size, config)

    if local_rank == 0:
        student_path = os.path.join(config.output_dir, "final")
        sanity_check(student_path)

        # Push to HuggingFace
        student.module.push_to_hub(
            "trungbb8/distilphobert-checkpoints", commit_message="final"
        )

    dist.destroy_process_group()
