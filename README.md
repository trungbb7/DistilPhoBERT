# DistilPhoBERT: Knowledge Distillation for Vietnamese Language Models

This repository contains the implementation and data preprocessing pipeline for **DistilPhoBERT**, a lightweight, fast, and efficient version of the PhoBERT model. This project aims to provide a high-performance transformer model for Vietnamese NLP tasks with a significantly smaller footprint.

---

## 🏗 Project Overview

DistilPhoBERT leverages Knowledge Distillation to compress the original PhoBERT architecture while retaining maximum linguistic accuracy. This specific section of the repository focuses on the **Data Engineering Pipeline** required to prepare high-quality Vietnamese text corpora for the distillation process.

## 📊 Data Preprocessing Pipeline

To ensure the model learns from clean, high-signal data, we implement a multi-stage cleaning and filtering pipeline.

### 1. Cleaning & Normalization

- **HTML Stripping:** Removal of tags and noise from web-crawled data using `BeautifulSoup`.
- **Unicode Normalization (NFC):** Converting all Vietnamese text to NFC form to ensure consistency between composite and pre-composed characters.
- **Punctuation Standardizing:** Normalizing curly quotes, long dashes, and ellipses.
- **Boilerplate Removal:** Advanced Regex patterns to strip "Read more" links, image captions, and copyright notices.
- **Author Info Stripping:** Heuristic-based removal of journalist signatures and publication metadata at the end of articles.

### 2. Quality Filtering

- **Length Constraint:** Only documents between **500 and 10,000 characters** are retained to ensure sufficient context without introducing extreme outliers.
- **Control Character Removal:** Stripping non-printable ASCII/Unicode control characters.

### 3. Global Deduplication

- **Method:** MD5 Hashing.
- **Storage:** SQLite-backed hash tracking.
- **Efficiency:** This prevents the model from over-fitting on repeated news articles or syndicated content across different domains.
