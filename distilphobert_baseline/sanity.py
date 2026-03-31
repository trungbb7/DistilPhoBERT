import logging

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


log = logging.getLogger(__name__)


def sanity_check(model_path: str, teacher_name: str = "vinai/phobert-base"):
    log.info("\n-- Sanity Check --")

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    student = AutoModelForMaskedLM.from_pretrained(model_path)
    student.eval()

    test_text = "Ha Noi la thu do cua Viet <mask> ."
    inputs = tokenizer(test_text, return_tensors="pt")

    with torch.no_grad():
        outputs = student(**inputs)
        logits = outputs.logits

    mask_pos = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    if len(mask_pos) > 0:
        mask_logits = logits[0, mask_pos[0], :]
        top5 = torch.topk(mask_logits, 5)
        top5_tokens = [tokenizer.decode([idx]) for idx in top5.indices]

        log.info("Input: '%s'", test_text)
        log.info("Top 5 predictions for [MASK]: %s", top5_tokens)
        if "Nam" in top5_tokens or "nam" in top5_tokens:
            log.info("Model working correctly.")
        else:
            log.info("Predictions seem off, consider checking training setup.")

    n_params = sum(p.numel() for p in student.parameters()) / 1e6
    log.info("Model size: %.1fM parameters", n_params)