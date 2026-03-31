import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def mlm_loss(self, student_logits, labels):
        return F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    def kd_loss(self, student_logits, teacher_logits, attention_mask):
        T = self.T

        s_logits = student_logits.view(-1, student_logits.size(-1))
        t_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        mask_flat = attention_mask.view(-1).bool()

        s_logits = s_logits[mask_flat]
        t_logits = t_logits[mask_flat]

        s_soft = F.log_softmax(s_logits / T, dim=-1)
        t_soft = F.softmax(t_logits / T, dim=-1)

        kd = F.kl_div(s_soft, t_soft, reduction="batchmean")
        return kd * (T ** 2)

    def forward(self, student_logits, teacher_logits, labels, attention_mask):
        loss_mlm = self.mlm_loss(student_logits, labels)
        loss_kd = self.kd_loss(student_logits, teacher_logits, attention_mask)

        total_loss = self.alpha * loss_mlm + (1 - self.alpha) * loss_kd

        return total_loss, {
            "total": total_loss.item(),
            "mlm": loss_mlm.item(),
            "kd": loss_kd.item(),
        }