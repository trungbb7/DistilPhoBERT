import logging
import os
from typing import Optional

import torch
from datasets import load_dataset

from .config import BaselineConfig


log = logging.getLogger(__name__)


class _VnCoreNLPSegmenter:
    def __init__(self, save_dir: Optional[str] = None):
        try:
            import py_vncorenlp
        except ImportError as exc:
            raise RuntimeError(
                "VnCoreNLP is enabled but py_vncorenlp is not installed. "
                "Install it with: pip install py_vncorenlp"
            ) from exc

        kwargs = {"annotators": ["wseg"]}
        if save_dir:
            kwargs["save_dir"] = save_dir

        self._segmenter = py_vncorenlp.VnCoreNLP(**kwargs)

    @staticmethod
    def _normalize(segmented_output) -> str:
        if isinstance(segmented_output, str):
            return segmented_output

        if isinstance(segmented_output, list):
            if not segmented_output:
                return ""

            if isinstance(segmented_output[0], str):
                return " ".join(segmented_output)

            if isinstance(segmented_output[0], list):
                return " ".join(" ".join(sentence_tokens) for sentence_tokens in segmented_output)

        return str(segmented_output)

    def segment(self, text: str) -> str:
        segmented = self._segmenter.word_segment(text)
        return self._normalize(segmented)


def load_and_prepare_data(config: BaselineConfig, tokenizer):
    log.info("Loading %s samples from %s...", config.subset_size, config.dataset_name)

    dataset = load_dataset(
        config.dataset_name,
        split=f"train[:{config.subset_size}]",
        trust_remote_code=True,
    )

    log.info("Dataset loaded: %s samples", len(dataset))
    log.info("Columns: %s", dataset.column_names)
    log.info("Sample text (100 chars): %s...", dataset[0][config.text_column][:100])

    if config.use_vncorenlp:
        if config.vncorenlp_dir and not os.path.isdir(config.vncorenlp_dir):
            raise RuntimeError(
                f"vncorenlp_dir does not exist: {config.vncorenlp_dir}. "
                "Download VnCoreNLP model files and pass the correct directory."
            )

        log.info("Applying VnCoreNLP segmentation before tokenization...")
        segmenter = _VnCoreNLPSegmenter(save_dir=config.vncorenlp_dir)

        def segment_fn(examples):
            raw_texts = examples[config.text_column]
            segmented_texts = [segmenter.segment(text) for text in raw_texts]
            return {config.text_column: segmented_texts}

        dataset = dataset.map(
            segment_fn,
            batched=True,
            batch_size=config.vncorenlp_batch_size,
            desc="VnCoreNLP segmentation",
        )

        log.info("Segmentation done. Sample segmented text: %s...", dataset[0][config.text_column][:100])

    def tokenize_fn(examples):
        return tokenizer(
            examples[config.text_column],
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
            return_tensors=None,
        )

    log.info("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=256,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    log.info("Tokenization done. Sample shape: %s", tokenized[0]["input_ids"].shape)

    return tokenized


def apply_mlm_masking(batch_input_ids, tokenizer, mlm_probability: float = 0.15):
    inputs = batch_input_ids.clone()
    labels = batch_input_ids.clone()

    prob_matrix = torch.full(labels.shape, mlm_probability)

    special_tokens_mask = (
        (batch_input_ids == tokenizer.cls_token_id)
        | (batch_input_ids == tokenizer.sep_token_id)
        | (batch_input_ids == tokenizer.pad_token_id)
    )
    prob_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100

    masked_input_ids = inputs.clone()
    masked_input_ids[masked_indices] = tokenizer.mask_token_id

    return inputs, masked_input_ids, labels