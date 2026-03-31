import argparse
import logging
import os

from .config import BaselineConfig
from .logging_utils import setup_logging
from .sanity import sanity_check
from .trainer import train


log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DistilPhoBERT baseline trainer")
    parser.add_argument("--subset-size", type=int, default=None, help="Dataset subset size")
    parser.add_argument("--num-steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument(
        "--use-vncorenlp",
        action="store_true",
        help="Enable VnCoreNLP Vietnamese word segmentation before tokenization",
    )
    parser.add_argument(
        "--vncorenlp-dir",
        type=str,
        default=None,
        help="Directory containing VnCoreNLP model files",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip post-training sanity check")
    return parser.parse_args()


def _apply_overrides(config: BaselineConfig, args: argparse.Namespace) -> BaselineConfig:
    if args.subset_size is not None:
        config.subset_size = args.subset_size
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_length is not None:
        config.max_length = args.max_length
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.device is not None:
        config.device = args.device
    if args.use_vncorenlp:
        config.use_vncorenlp = True
    if args.vncorenlp_dir is not None:
        config.vncorenlp_dir = args.vncorenlp_dir
    if args.use_wandb:
        config.use_wandb = True
    return config


def main():
    setup_logging()
    args = _parse_args()

    config = BaselineConfig()
    config = _apply_overrides(config, args)

    log.info("Config:")
    for key, value in vars(config).items():
        log.info("  %s: %s", key, value)
    log.info("")

    train(config)

    if not args.skip_sanity:
        sanity_check(
            model_path=os.path.join(config.output_dir, "final"),
            teacher_name=config.teacher_name,
        )

    log.info("\nNext steps:")
    log.info("1. Scale subset_size len 100k, num_steps len 10k")
    log.info("2. Add VnCoreNLP word segmentation")
    log.info("3. Fine-tune cho Sentiment Analysis (VLSP 2016)")


if __name__ == "__main__":
    main()