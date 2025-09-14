# %%
import patches
import re
from typing import Set, Dict, Any
import time
import pickle
import random
from tqdm.auto import tqdm
import dotenv
dotenv.load_dotenv()
import nest_asyncio
nest_asyncio.apply()
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from umap import UMAP
import openai
from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
import torch
import tiktoken
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import OpenAI

# %%
# Filter questions whose responses are too long;
# By standard we sample 1024 tokens from the policy
# Here we filter for at most 512 tokens

tokenizer = tiktoken.get_encoding("gpt2")

ds_iter = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

def add_prompt_length(batch):
    """
    Takes a batch of examples, tokenizes the prompts, 
    and adds a new 'prompt_length' column.
    """
    # Use the optimized 'encode_batch' for speed
    prompts = [conv[0]['content'] for conv in batch['conversation']]
    encodings = tokenizer.encode_batch(prompts, disallowed_special=())
    batch["prompt_length"] = [len(encoding) for encoding in encodings]
    return batch

ds_with_length = ds_iter.map(add_prompt_length, batched=True, batch_size=128)
ds_filtered = ds_with_length.filter(
    lambda ex: 
    ex["turn"] == 1
    and ex["language"] == "English" 
    and ex["prompt_length"] < 512
)

# %%
# 154s wtih batch size 128
start_time = time.time()
ds_50k = Dataset.from_list(list(ds_filtered.take(50000)))
print(f"Loaded 50k in {time.time() - start_time:.2f}s")

with open("data/wildchat/ds_50k.pkl", "wb") as f:
    pickle.dump(ds_50k, f)
    
# %%
# Basic cleaning: normalize, deduplicate, and filter low-quality prompts

def _normalize_text(text: str) -> str:
    # lowercase, trim, collapse internal whitespace
    return " ".join(text.lower().strip().split())

_STOPWORDS = {
    "the","is","at","which","on","and","a","an","to","of","in","for","it","this",
    "that","with","as","by","from","or","be","are","was","were","can","could","should",
    "would","do","does","did","have","has","had","but","if","than","then","so","such",
}

def _is_garbage(text: str) -> bool:
    # repeated single character sequences (>=6)
    if re.search(r"(.)\1{5,}", text):
        return True
    # high non-alnum (excluding whitespace) ratio
    total = len(text)
    if total == 0:
        return True
    non_alnum = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    if total > 0 and (non_alnum / total) > 0.5:
        return True
    # stopword-only prompts (after tokenization)
    tokens = [t for t in text.split() if t]
    if tokens and all(t in _STOPWORDS for t in tokens):
        return True
    return False

def _clean_records(
    ds: Dataset,
    min_words: int = 4,
    max_words: int = 1024
) -> Dataset:
    seen: Set[str] = set()
    cleaned: list[Dict[str, Any]] = []
    for item in tqdm(ds, desc="Filtering and deduplicating"):
        try:
            raw = item["conversation"][0]["content"]  # type: ignore[index]
            if not isinstance(raw, str):
                continue
            norm = _normalize_text(raw)
            wc = len(norm.split())
            if wc < min_words or wc > max_words:
                continue
            if _is_garbage(norm):
                continue
            if norm in seen:
                continue
            seen.add(norm)
            # keep original content; use normalized only for dedup/filtering
            new_item = dict(item)
            conv = list(item["conversation"])  # type: ignore[index]
            first_turn = dict(conv[0])
            first_turn["content"] = raw
            conv[0] = first_turn
            new_item["conversation"] = conv
            cleaned.append(new_item)
        except Exception:
            continue
    return Dataset.from_list(cleaned)

# %%
# 20s
start_time = time.time()
ds_50k_clean = _clean_records(ds_50k)
print(f"Data cleaning took {time.time() - start_time:.2f}s")

with open("data/wildchat/ds_50k_clean.pkl", "wb") as f:
    pickle.dump(ds_50k_clean, f)

# %%
with open("data/wildchat/ds_50k_clean.pkl", "rb") as f:
    ds_50k_clean = pickle.load(f)
# 41237 rows

print(ds_50k_clean)

# %%
client = openai.OpenAI()
CLUSTER_PROMPT_TEMPLATE = """Your task is to extract a short summary label from a list of given documents and keywords. Each document is a user prompt for a chatbot. The summary label should be a short phrase of at most 15 words, summarizing the common topic and intent of the user prompts.

*Sample documents:*
[DOCUMENTS]

*Keywords:*
[KEYWORDS]

Respond with the short summary label below.
"""

embedding_model = model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    low_memory=False,
)

representation_model = OpenAI(
    client,
    model="gpt-5-mini",
    prompt=CLUSTER_PROMPT_TEMPLATE,
    chat=True,
    nr_docs=10,
)
topic_model = BERTopic(
    min_topic_size=30,
    umap_model=umap_model,
    embedding_model=embedding_model,
    vectorizer_model=CountVectorizer(stop_words="english"),
    representation_model=representation_model,
)

# %%
prompts = [item["conversation"][0]["content"] for i, item in enumerate(ds_50k_clean)]  # type: ignore

# 1811s (30 minutes)
# Need to make this faster
print("Fitting BERTopic...")
start_time = time.time()
topics, probs = topic_model.fit_transform(prompts)
print(f"BERTopic fit_transform in {time.time() - start_time:.2f}s")

# %%
labels_df = topic_model.get_document_info(prompts)
labels_df.to_csv("data/wildchat/labels_50k.csv", index=True)

# %%
cluster_topics_df: pd.DataFrame = topic_model.get_topic_info()
cluster_topics_df.to_csv("data/wildchat/cluster_50k.csv", index=True)

# %%
# 1309s (22 minutes)
start_time = time.time()
hierarchical_topics_df: pd.DataFrame = topic_model.hierarchical_topics(prompts)
print(f"BERTopic hierarchical_topics in {time.time() - start_time:.2f}s")
hierarchical_topics_df.to_csv("data/wildchat/hierarch_50k.csv", index=True)


# %%
fig: go.Figure = topic_model.visualize_hierarchy()
fig.write_html("data/wildchat/hierarchy_visual.html")

# %%
# load the clusters

cluster_df: pd.DataFrame = pd.read_csv("data/wildchat/cluster.csv")
# %%
cluster_df.loc[cluster_df.index[3+1], "Representation"]

# %%
with pd.read_csv("data/wildchat/labels.csv", chunksize=10000) as reader:
    for chunk in reader:
        for index, row in chunk.iterrows():
            if int(row["Topic"]) == 2:
                print(row["Document"])
                print("\n")
        
        break

# %%
