import re
import unicodedata
import hashlib
import sqlite3
from datasets import load_dataset
from bs4 import BeautifulSoup

# CLEANING FUNCTIONS


def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def normalize_unicode(text):
    return unicodedata.normalize("NFC", text)


def remove_control_chars(text):
    return re.sub(r"[\x00-\x1F\x7F]", " ", text)


def normalize_punctuation(text):
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-")
    text = text.replace("…", "...")
    return text


def remove_boilerplate(text):
    patterns = [
        r"Xem thêm.*",
        r"Theo .*",
        r"Bản quyền.*",
    ]
    for p in patterns:
        text = re.sub(p, "", text)
    return text


def normalize_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(example):
    text = example["text"]

    text = remove_html(text)
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = normalize_punctuation(text)
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)

    return {"text": text}


# FILTER FUNCTIONS
def filter_length(example, min_len=50, max_len=10000):
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
