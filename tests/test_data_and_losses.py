import unittest

import torch

from distilphobert_baseline.data import apply_mlm_masking
from distilphobert_baseline.losses import DistillationLoss


class _DummyTokenizer:
    cls_token_id = 0
    pad_token_id = 1
    sep_token_id = 2
    mask_token_id = 4


class DataAndLossesTest(unittest.TestCase):
    def test_apply_mlm_masking_respects_special_tokens(self):
        tokenizer = _DummyTokenizer()
        input_ids = torch.tensor([[0, 10, 11, 2, 1]], dtype=torch.long)

        original_ids, masked_ids, labels = apply_mlm_masking(
            input_ids,
            tokenizer,
            mlm_probability=1.0,
        )

        self.assertTrue(torch.equal(original_ids, input_ids))

        # Special tokens must remain unchanged and ignored in labels.
        self.assertEqual(masked_ids[0, 0].item(), 0)
        self.assertEqual(masked_ids[0, 3].item(), 2)
        self.assertEqual(masked_ids[0, 4].item(), 1)
        self.assertEqual(labels[0, 0].item(), -100)
        self.assertEqual(labels[0, 3].item(), -100)
        self.assertEqual(labels[0, 4].item(), -100)

        # Non-special tokens must be masked when probability is 1.0.
        self.assertEqual(masked_ids[0, 1].item(), tokenizer.mask_token_id)
        self.assertEqual(masked_ids[0, 2].item(), tokenizer.mask_token_id)
        self.assertEqual(labels[0, 1].item(), 10)
        self.assertEqual(labels[0, 2].item(), 11)

    def test_distillation_loss_returns_finite_values(self):
        criterion = DistillationLoss(temperature=2.0, alpha=0.5)
        student_logits = torch.randn(2, 3, 7, requires_grad=True)
        teacher_logits = torch.randn(2, 3, 7)
        labels = torch.tensor([[1, -100, 2], [0, 3, -100]])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

        total_loss, parts = criterion(student_logits, teacher_logits, labels, attention_mask)

        self.assertTrue(torch.isfinite(total_loss).item())
        self.assertIn("total", parts)
        self.assertIn("mlm", parts)
        self.assertIn("kd", parts)


if __name__ == "__main__":
    unittest.main()