import os
import re
import unicodedata
import underthesea
import hashlib
import sqlite3
from datasets import load_dataset, Dataset
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup

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
def filter_length(example, min_len=500, max_len=10000):
    return min_len <= len(example["text"]) <= max_len


# DEDUPLICATION
conn = sqlite3.connect("dedup.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS hashes (hash TEXT PRIMARY KEY)")


def is_duplicated(text):
    hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    try:
        cursor.execute("INSERT INTO hashes(hash) VALUES(?)", (hash,))
        conn.commit()
        return False
    except sqlite3.IntegrityError:
        return True


def deduplicate(example):
    return not is_duplicated(example["text"])


# Load dataset
dataset = load_dataset(
    "parquet", data_files="datasets/temp/*.parquet", split="train", streaming=True
)

# small_dataset = dataset.take(10)


# Apply pipeline

# 1. Clean text
print("Cleaning text")
dataset = dataset.map(clean_text, remove_columns=dataset.column_names)

# 2. Filter
print("Filter text")
dataset = dataset.filter(filter_length)

# 3. Dedup
print("Dedup text")
dataset = dataset.filter(deduplicate)

# Save Output
print("Save output")
output_path = "datasets/temp_output/out.parquet"
batch_size = 10000

buffer = []
writer = None

for example in dataset:
    buffer.append({"text": example["text"]})
    if len(buffer) >= batch_size:
        table = pa.Table.from_pylist(buffer)

        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        buffer = []

if buffer:
    table = pa.Table.from_pylist(buffer)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
if writer:
    writer.close()


# Cleaning steps
cursor.close()
conn.close()
os.remove("dedup.db")
