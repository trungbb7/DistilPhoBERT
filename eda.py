# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset

# %%
dataset = load_dataset("ademax/binhvq-news-corpus", split="train", streaming=True)


sampled = dataset.shuffle(seed=42, buffer_size=1_000).take(2_000)

df = pd.DataFrame(list(sampled))


# %%
df["content_length"] = df["content"].apply(lambda x: len(str(x).split()))

# Content length historam
plt.figure(figsize=(10, 6))
sns.histplot(df["content_length"], kde=True, bins=1000)
plt.title("Content length distribution (Histogram & KDE)")
plt.xlabel("Content length")
plt.ylabel("Frequency")
plt.show()


# %%
# Content length plotbox
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["content_length"])
plt.title("Content length distribution (Boxplot)")
plt.xlabel("Content length")
plt.show()

# %%
