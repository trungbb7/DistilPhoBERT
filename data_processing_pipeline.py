import os
import re
import unicodedata
import hashlib
from bs4 import BeautifulSoup
import py_vncorenlp
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# CLEANING FUNCTIONS


def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def normalize_nfc(text):
    return unicodedata.normalize("NFC", text)


def normalize_punctuation(text):
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-")
    text = text.replace("…", "...")
    return text


def remove_boilerplate(text):
    patterns = [
        r"Xem thêm:.*",
        r"Bản quyền.*",
        r"\s\(Ảnh: .*?\)\.?",
        r"Ảnh: .*",
        r"\s\(Ảnh minh họa: .*\).",
        r"\s\(Ảnh minh họa\)",
        r"Ảnh minh họa.",
        r"\(Nguồn: .*\)\.",
        r"\s\(Theo .*\)",
        r"\/\.",
        r">>",
    ]
    for p in patterns:
        text = re.sub(p, "", text)
    return text


def is_author_info(line):
    line = line.strip()

    patterns = [
        r"^[\W\s]+$",
        r"^[A-Z\sÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ-]+\.?$",
        r".*\(Tổng hợp\).*",
        r".*\(Theo .*\).*",
        r".*\(Tiếp tục .*\).*",
        r".*\(TTXVN\).*",
        r".*(Báo|Đón xem|Tổng hợp|Mời|Theo).*",
        r"^(PV|CTV|Bài và ảnh|Nguồn|Ảnh|Video|Từ khóa|Clip|Video):.*",
        r"^\w+\s\w+\s?\(.*\)$",
    ]

    for pattern in patterns:
        if re.match(pattern, line):
            return True

    words = line.split()
    if not (1 <= len(words) <= 4):
        return False
    is_capitalized = all(word[0].isupper() for word in words if word[0].isalpha())

    return is_capitalized


def clean_author_info(text):
    lines = text.strip().split("\n")
    if not text or not text.strip():
        return ""

    max_inspector_lines = min(3, len(lines) - 1)

    inspector_lines = lines[-max_inspector_lines:]
    remaining_inspectors_lines = []
    for line in inspector_lines:
        if not is_author_info(line):
            remaining_inspectors_lines.append(line)

    new_lines = lines[:-max_inspector_lines] + remaining_inspectors_lines
    new_text = "\n".join(new_lines).strip()
    sp = new_text.split(".")
    if len(sp) <= 1:
        return new_text
    target = sp[-2]
    if len(target.split()) >= 4:
        return new_text
    new_text = ".".join(sp[:-2] + sp[-1:]).strip()
    return new_text


def remove_control_chars(text):
    return re.sub(r"[\x00-\x1F\x7F]", " ", text)


def normalize_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(example):
    text = example["content"]

    text = remove_html(text)
    text = normalize_nfc(text)
    text = normalize_punctuation(text)
    text = remove_boilerplate(text)
    text = clean_author_info(text)
    text = remove_control_chars(text)
    text = normalize_whitespace(text)

    return {"text": text}


# FILTER FUNCTIONS
def filter_length(example, min_len=500, max_len=20000):
    return min_len <= len(example["text"]) <= max_len


seen = set()


def dedup(example):
    h = hashlib.md5(example["text"].encode()).hexdigest()
    if h in seen:
        return False
    seen.add(h)
    return True


# Word segmentation
segmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"],
    save_dir="/content/gdrive/MyDrive/workspace/Libraries/vncorenlp",
)


def segment(batch):
    segmented_texts = []
    for text in batch["text"]:
        if len(text.strip()) > 0:
            lines = segmenter.word_segment(text)
            segmented_texts.append(" ".join(lines))
        else:
            segmented_texts.append("")
    return {"text": segmented_texts}


login(token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"))

# Load dataset
dataset = load_dataset("ademax/binhvq-news-corpus", split="train")


# Apply pipeline

# 1. Clean text
dataset = dataset.map(clean_text, remove_columns=dataset.column_names, num_proc=4)

# 2. Filter
dataset = dataset.filter(filter_length, num_proc=4)

# 3. Dedup
dataset = dataset.filter(dedup, num_proc=1)

# Text segmentation
dataset = dataset.map(
    segment, batched=True, batch_size=50, num_proc=4, desc="Text segmenting"
)

dataset.save_to_disk(r"/content/gdrive/MyDrive/KLTN/datasets/segmented_ds")

# Push to hub

dataset.push_to_hub("trungbb8/news-demo", max_shard_size="500MB", num_proc=4)
