import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaForMaskedLM
import logging

log = logging.getLogger(__name__)


def build_student_model(config, teacher_model, rank=0):
    if rank == 0:
        log.info("Building student model...")

    student_config = RobertaConfig(
        vocab_size=teacher_model.config.vocab_size,
        hidden_size=config.student_hidden_size,
        num_hidden_layers=config.student_num_layers,
        num_attention_heads=config.student_num_heads,
        intermediate_size=config.student_intermediate_size,
        max_position_embeddings=teacher_model.config.max_position_embeddings,
        type_vocab_size=1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    student = RobertaForMaskedLM(student_config)

    if rank == 0:
        log.info(
            "Initializing student weights from teacher (layer selection strategy)..."
        )

    teacher_layers = teacher_model.roberta.encoder.layer
    student_layers = student.roberta.encoder.layer

    layer_mapping = {0: 0, 1: 2, 2: 4, 3: 7, 4: 9, 5: 11}
    for s_idx, t_idx in layer_mapping.items():
        student_layers[s_idx].load_state_dict(teacher_layers[t_idx].state_dict())
        if rank == 0:
            log.info(f"  Student layer {s_idx} ← Teacher layer {t_idx}")

    student.roberta.embeddings.load_state_dict(
        teacher_model.roberta.embeddings.state_dict()
    )

    try:
        student.lm_head.load_state_dict(teacher_model.lm_head.state_dict())
    except Exception:
        if rank == 0:
            log.info("  LM head init: random (teacher architecture mismatch).")

    return student


class DistillationLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.4,
        beta: float = 0.2,
        gamma: float = 0.4,
    ):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def mlm_loss(self, student_logits, labels):
        return F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    def kd_loss(self, student_logits, teacher_logits, attention_mask):
        s_logits = student_logits.view(-1, student_logits.size(-1))
        t_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        mask_flat = attention_mask.view(-1).bool()

        s_logits = s_logits[mask_flat]
        t_logits = t_logits[mask_flat]

        s_soft = F.log_softmax(s_logits / self.T, dim=-1)
        t_soft = F.softmax(t_logits / self.T, dim=-1)

        kd = F.kl_div(s_soft, t_soft, reduction="batchmean")
        return kd * (self.T**2)

    def cosin_loss(
        self, student_last_hidden_states, teacher_last_hidden_states, attention_mask
    ):
        s_states = student_last_hidden_states.view(
            -1, student_last_hidden_states.size(-1)
        )
        t_states = teacher_last_hidden_states.view(
            -1, teacher_last_hidden_states.size(-1)
        )
        mask_flat = attention_mask.view(-1).bool()

        s_states = s_states[mask_flat]
        t_states = t_states[mask_flat]

        loss = 1 - F.cosine_similarity(s_states, t_states, dim=1).mean()
        return loss

    def forward(
        self,
        student_logits,
        teacher_logits,
        labels,
        attention_mask,
        student_last_hidden_state,
        teacher_last_hidden_state,
    ):
        loss_mlm = self.mlm_loss(student_logits, labels)
        loss_kd = self.kd_loss(student_logits, teacher_logits, attention_mask)
        loss_cos = self.cosin_loss(
            student_last_hidden_state, teacher_last_hidden_state, attention_mask
        )

        total_loss = (
            (self.alpha * loss_mlm) + (self.beta * loss_kd) + (self.gamma * loss_cos)
        )

        return total_loss, {
            "total": total_loss.detach().clone(),
            "mlm": loss_mlm.detach().clone(),
            "kd": loss_kd.detach().clone(),
            "cos": loss_cos.detach().clone(),
        }
