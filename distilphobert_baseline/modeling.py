import logging

from transformers import RobertaConfig, RobertaForMaskedLM

from .config import BaselineConfig


log = logging.getLogger(__name__)


def build_student_model(config: BaselineConfig, teacher_model):
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

    log.info("Initializing student weights from teacher...")
    teacher_layers = teacher_model.roberta.encoder.layer
    student_layers = student.roberta.encoder.layer

    # Distil-style mapping: sample teacher layers evenly into student depth.
    teacher_depth = len(teacher_layers)
    student_depth = len(student_layers)
    mapped_teacher_indices = [
        min((idx * teacher_depth) // student_depth, teacher_depth - 1)
        for idx in range(student_depth)
    ]

    for student_idx, teacher_idx in enumerate(mapped_teacher_indices):
        student_layers[student_idx].load_state_dict(teacher_layers[teacher_idx].state_dict())
        log.info("  Student layer %s <- Teacher layer %s", student_idx, teacher_idx)

    student.roberta.embeddings.load_state_dict(teacher_model.roberta.embeddings.state_dict())
    log.info("  Embeddings copied from teacher.")

    try:
        student.lm_head.load_state_dict(teacher_model.lm_head.state_dict())
        log.info("  LM head copied from teacher.")
    except Exception:
        log.info("  LM head init random due to architecture mismatch.")

    n_params = sum(p.numel() for p in student.parameters()) / 1e6
    log.info("Student model built: %.1fM parameters", n_params)

    return student