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


def normalize_vietnamese_text(text):
    # NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Diacritics normalization
    return underthesea.text_normalize(text)


def normalize_punctuation(text):
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-")
    text = text.replace("…", "...")
    return text


def remove_boilerplate(text):
    patterns = [
        r"Xem thêm:.*",
        r"Bản quyền.*",
        r"\s\(Ảnh: .*?\)",
        r"Ảnh: .*",
        r"\s\(Ảnh minh họa: .*\).",
        r"Ảnh minh họa.",
        r"\(Nguồn: .*\)\.",
        r"\s\(Theo .*\)",
        r"\/\.",
    ]
    for p in patterns:
        text = re.sub(p, "", text)
    return text


def remove_control_chars(text):
    return re.sub(r"[\x00-\x1F\x7F]", " ", text)


def normalize_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(example):
    text = example["content"]

    text = remove_html(text)
    text = normalize_vietnamese_text(text)
    text = normalize_punctuation(text)
    text = remove_boilerplate(text)
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
