# DistilPhoBERT: A Compressed Variant of PhoBERT for Vietnamese NLP

This repository contains the implementation and pre-training pipeline for **DistilPhoBERT**, a distilled version of the [PhoBERT](https://github.com/VinAIResearch/PhoBERT) model. By leveraging knowledge distillation, we aim to create a faster, smaller, and more efficient transformer model specifically optimized for Vietnamese natural language tasks.
The model is available on HuggingFace [DistilPhoBERT](https://huggingface.co/trungbb8/distilphobert)

## Features

- **32% Size Reduction:** Reduced the number of parameters from 135 million (Teacher) to 92 million (Student), decreasing physical storage space from 515MB to 352MB.
- **Inference Speedup:** Processing speed is **1.88x to 1.9x** faster than the original model.
- **Excellent Performance Preservation:** Retains **96.3% - 99.8%** of PhoBERT's performance (F1-score) on standard benchmark datasets (VLSP 2016).
- **Advanced Pre-processing:** Comprehensive pipeline for cleaning, normalizing, and segmenting Vietnamese news data.
- **Modern Stack:** Built with `PyTorch`, `Hugging Face Transformers`, and `VnCoreNLP`.

---

## Model Architecture

The distillation process follows a task-agnostic **Teacher-Student** mechanism:

- **Teacher:** PhoBERT-base model (12 Transformer layers, hidden size 768, 12 attention heads).
- **Student:** DistilPhoBERT model (6 Transformer layers, hidden size 768, 12 attention heads).

To ensure rapid convergence of the Student model, a **Skip-layer Mapping** initialization strategy was applied. Specifically, the 6 Transformer layers of the Student were initialized directly from alternating layers of the Teacher network (Layers 0, 2, 4, 6, 8, 10) to capture the full spectrum of linguistic features from basic to abstract.

### Multi-task Loss Function

The system combines 3 loss functions (Triple Loss) to transfer knowledge:

1. **Masked Language Modeling (MLM) Loss:** Helps the Student maintain contextual understanding by predicting masked subwords based on actual labels.
2. **Knowledge Distillation (KD) Loss:** Forces the Student's prediction probability distribution to closely mimic the Teacher's (using KL Divergence and Softmax with Temperature $T=2.0$).
3. **Cosine Embedding Loss:** Aligns the direction of hidden state representations between the Student and Teacher.

---

## Dataset & Pre-processing

The model is pre-trained on a large-scale Vietnamese news corpus [ademax/binhvq-news-corpus](https://huggingface.co/datasets/ademax/binhvq-news-corpus) (~20GB). Quality data is the backbone of DistilPhoBERT, so we implemented a rigorous cleaning pipeline:

### Cleaning Pipeline Highlights:

1. **HTML & Noise Removal:** Stripped HTML tags and boilerplate text (ads, "See more" links).
2. **Standardization:** Normalized to **Unicode NFC** and standardized punctuation.
3. **Signature Stripping:** Heuristic-based removal of author names, locations, and journalist signatures.
4. **Filtering:** Retained high-quality articles (500 - 20,000 characters).
5. **Deduplication:** MD5-based exact match removal.
6. **Word Segmentation:** Applied `VnCoreNLP` to handle Vietnamese compound words (e.g., `trí_tuệ nhân_tạo`).

> **Dataset Availability:** The processed dataset is hosted on Hugging Face: [trungbb8/vietnamese-news-copus-segmented](https://huggingface.co/datasets/trungbb8/vietnamese-news-copus-segmented)

---

## Evaluation Results (VLSP 2016 Benchmark)

The model was fine-tuned and evaluated on three core tasks in direct comparison with PhoBERT-base and mBERT:

| Task                   | Model         |  F1-Score  | Inference Time | Performance Retained |
| :--------------------- | :------------ | :--------: | :------------: | :------------------: |
| **POS Tagging**        | PhoBERT-base  |   0.9486   |     40.26s     |         100%         |
|                        | DistilPhoBERT | **0.9472** |   **21.02s**   |      **99.85%**      |
| **NER**                | PhoBERT-base  |   0.9285   |     35.33s     |         100%         |
|                        | DistilPhoBERT | **0.9246** |   **17.97s**   |      **99.57%**      |
| **Sentiment Analysis** | PhoBERT-base  |   0.7790   |     5.78s      |         100%         |
|                        | DistilPhoBERT | **0.7505** |   **3.07s**    |      **96.34%**      |

---

## Development Team

- **Authors:** Nguyen Minh Trung & Nguyen Thanh Thuong
- **Institution:** Faculty of Information Technology - Nong Lam University (HCMUAF) (2022 - 2026 cohort).
