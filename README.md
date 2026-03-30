# DistilPhoBERT: A Compressed Variant of PhoBERT for Vietnamese NLP

This repository contains the implementation and pre-training pipeline for **DistilPhoBERT**, a distilled version of the [PhoBERT](https://github.com/VinAIResearch/PhoBERT) model. By leveraging knowledge distillation, we aim to create a faster, smaller, and more efficient transformer model specifically optimized for Vietnamese natural language tasks.

## 🚀 Features

- **Efficiency:** Significant reduction in model size and inference latency compared to PhoBERT.
- **Advanced Pre-processing:** Comprehensive pipeline for cleaning, normalizing, and segmenting Vietnamese news data.
- **Modern Stack:** Built with `PyTorch`, `Hugging Face Transformers`, and `VnCoreNLP`.

---

## 📊 Dataset & Pre-processing

The model is pre-trained on a large-scale Vietnamese news corpus (~20GB). Quality data is the backbone of DistilPhoBERT, so we implemented a rigorous cleaning pipeline:

### Cleaning Pipeline Highlights:

1. **HTML & Noise Removal:** Stripped HTML tags and boilerplate text (ads, "See more" links).
2. **Standardization:** Normalized to **Unicode NFC** and standardized punctuation.
3. **Signature Stripping:** Heuristic-based removal of author names, locations, and journalist signatures.
4. **Filtering:** Retained high-quality articles (500 - 20,000 characters).
5. **Deduplication:** MD5-based exact match removal.
6. **Word Segmentation:** Applied `VnCoreNLP` to handle Vietnamese compound words (e.g., `trí_tuệ nhân_tạo`).

> **Dataset Availability:** The processed dataset is hosted on Hugging Face: [trungbb8/vietnamese-news-copus-segmented](https://huggingface.co/datasets/trungbb8/vietnamese-news-copus-segmented)

---
