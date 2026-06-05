from datasets import load_dataset
from config import BaselineConfig


def load_and_prepare_data(rank, world_size, config: BaselineConfig):
    # Load the full train split for streaming
    dataset = load_dataset(
        config.dataset_name,
        streaming=True,
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["test"].take(10000)

    # Shard cho cơ chế Distributed Data Parallel (DDP)
    train_dataset = train_dataset.shard(num_shards=world_size, index=rank)
    val_dataset = val_dataset.shard(num_shards=world_size, index=rank)

    return train_dataset, val_dataset
