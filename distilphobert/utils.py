import torch
import torch.distributed as dist
import logging

log = logging.getLogger(__name__)


def sync_tensor_across_gpus(tensor):
    cl = tensor.clone()
    dist.all_reduce(cl, op=dist.ReduceOp.SUM)
    cl /= dist.get_world_size()
    return cl


def sanity_check(model_path: str, teacher_name: str = "vinai/phobert-base"):
    log.info("\n── Sanity Check ──")
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from pyvi import ViTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    student = AutoModelForMaskedLM.from_pretrained(model_path)
    student.eval()

    test_text = "Hà Nội là thủ đô của Việt Nam."
    segmented_text = ViTokenizer.tokenize(test_text)
    masked_text = segmented_text.replace("Nam", "<mask>")

    inputs = tokenizer(masked_text, return_tensors="pt")

    with torch.no_grad():
        outputs = student(**inputs)
        logits = outputs.logits

    mask_pos = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[
        1
    ]

    if len(mask_pos) > 0:
        mask_logits = logits[0, mask_pos[0], :]
        top5 = torch.topk(mask_logits, 5)
        top5_tokens = [tokenizer.decode([idx]) for idx in top5.indices]

        log.info(f"Input: '{masked_text}'")
        log.info(f"Top 5 predictions for [MASK]: {top5_tokens}")
        log.info(
            "Model working correctly!"
            if "Nam" in top5_tokens or "nam" in top5_tokens
            else "Predictions seem off — check training."
        )
