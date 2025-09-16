# %%
import doctest
import patches
import re
from typing import Set, Dict, Any
import time
import pickle
import random
import asyncio
from tqdm.auto import tqdm
import dotenv
dotenv.load_dotenv()
import nest_asyncio
nest_asyncio.apply()
from collections import defaultdict
import plotly.graph_objects as go
from pathlib import Path

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

from client import get_universal_caller, sample_from_model_parallel

# %%
# Filter questions whose responses are too long;
# By standard we sample 1024 tokens from the policy
# Here we filter for at most 512 tokens

tokenizer = tiktoken.get_encoding("gpt2")

# ds_iter = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
ultrafeedback = load_dataset(
    "HuggingFaceH4/ultrafeedback_binarized", split="train_prefs"
)

# def add_prompt_length(batch):
#     """
#     Takes a batch of examples, tokenizes the prompts, 
#     and adds a new 'prompt_length' column.
#     """
#     # Use the optimized 'encode_batch' for speed
#     prompts = [conv[0]['content'] for conv in batch['conversation']]
#     encodings = tokenizer.encode_batch(prompts, disallowed_special=())
#     batch["prompt_length"] = [len(encoding) for encoding in encodings]
#     return batch

# ds_with_length = ds_iter.map(add_prompt_length, batched=True, batch_size=128)
# ds_filtered = ds_with_length.filter(
#     lambda ex: 
#     ex["turn"] == 1
#     and ex["language"] == "English" 
#     and ex["prompt_length"] < 512
# )
def preprocess_ultrafeedback(item):
    item["chosen"] = item["chosen"][1]["content"]
    item["rejected"] = item["rejected"][1]["content"]
    item["prompt_length"] = len(item["prompt"])
    item["chosen_length"] = len(item["chosen"])
    item["rejected_length"] = len(item["rejected"])
    return item
ultrafeedback = ultrafeedback.map(
    preprocess_ultrafeedback,
    num_proc=8,
).filter(
    lambda item: item["prompt_length"] < 1024 and item["chosen_length"] < 1024 and item["rejected_length"] < 1024,
    num_proc=8,
)
print("Ultrafeedback after filtering: ", len(ultrafeedback))


# %%
# 154s wtih batch size 128

# start_time = time.time()
# ds_50k = Dataset.from_list(list(ds_filtered.take(50000)))
# print(f"Loaded 50k in {time.time() - start_time:.2f}s")

Path("data/ultrafeedback").mkdir(parents=True, exist_ok=True)
with open("data/ultrafeedback/ds_20k.pkl", "wb") as f:
    pickle.dump(ultrafeedback, f)
    
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

with open("data/ultrafeedback/ds_20k.pkl", "rb") as f:
    ds_20k = pickle.load(f)

start_time = time.time()
ds_20k_clean = _clean_records(ds_20k)
print(f"Data cleaning took {time.time() - start_time:.2f}s")

with open("data/ultrafeedback/ds_20k_clean.pkl", "wb") as f:
    pickle.dump(ds_20k_clean, f)

# %%
# with open("data/ultrafeedback/ds_20k_clean.pkl", "rb") as f:
#     ds_20k_clean = pickle.load(f)
# 41237 rows

print(ds_20k_clean)

# %%
# docs = [item["conversation"][0]["content"] for i, item in enumerate(ds_50k_clean)]  # type: ignore
with open("data/ultrafeedback/ds_20k.pkl", "rb") as f:
    ds_20k = pickle.load(f)

docs = list(ds_20k["prompt"])
print(docs[0])

# %%
# client = openai.OpenAI()
# CLUSTER_PROMPT_TEMPLATE = """Your task is to extract a short summary label from a list of given documents and keywords. Each document is a user prompt for a chatbot. The summary label should be a short phrase of at most 15 words, summarizing the common topic and intent of the user prompts.

# *Sample documents:*
# [DOCUMENTS]

# *Keywords:*
# [KEYWORDS]

# Respond with the short summary label below.
# """

embedding_model = model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    low_memory=False,
)

# representation_model = OpenAI(
#     client,
#     model="gpt-5-mini",
#     prompt=CLUSTER_PROMPT_TEMPLATE,
#     chat=True,
#     nr_docs=30,
# )

topic_model = BERTopic(
    min_topic_size=30,
    umap_model=umap_model,
    embedding_model=embedding_model,
    vectorizer_model=CountVectorizer(stop_words="english"),
    # representation_model=representation_model,
)

# %%
# 1811s (30 minutes)
# Need to make this faster
print("Fitting BERTopic...")
start_time = time.time()
topics, probs = topic_model.fit_transform(docs)
print(f"BERTopic fit_transform in {time.time() - start_time:.2f}s")

# %%
labels_df = topic_model.get_document_info(docs)
labels_df.to_csv("data/ultrafeedback/labels_20k.csv", index=True)

# %%

labels_df = pd.read_csv("data/ultrafeedback/labels_20k.csv")

# %%
clusters = defaultdict(list)
representative = defaultdict(list)
for topic_id in tqdm(range(0, 86), desc="Processing topics"):
    for index, row in labels_df.iterrows():
        if int(row["Topic"]) == topic_id:
            clusters[topic_id].append((row["Document"], row["Probability"]))
            if bool(row["Representative_document"]):
                print(f"Found representative document for topic {topic_id}")
                representative[topic_id].append(row["Document"])

# %%
for topic_id in range(0, 86):
    print(f"Topic {topic_id} has {len(clusters[topic_id])} documents")

# %%

from llm_types import ChatHistory

CLUSTER_PROMPT_TEMPLATE = """Your task is to extract a short summary label from a list of given documents and keywords. Each document is a user prompt for a chatbot. The summary label should be a short phrase of 5-20 words, summarizing the common topic and intent of the user prompts.

## Representative documents (documents that lie close to the center of the cluster)

{representative_documents}

## 30 randomly sampled documents from the cluster

{sample_documents}

Think carefully, and only output the short summary label.
"""

to_send_chats = []

for topic_id in range(0, 86):
    sample_docs = random.sample(clusters[topic_id], 30)
    sample_docs_str = ("\n"+"-"*10+"\n").join([doc[0] for doc in sample_docs])

    representative_docs = representative[topic_id]
    representative_docs_str = ("\n"+"-"*10+"\n").join([doc for doc in representative_docs])

    to_send_chats.append(
        ChatHistory.from_system("You are an expert at summarizing documents.")
        .add_user(CLUSTER_PROMPT_TEMPLATE.format(
            representative_documents=representative_docs_str,
            sample_documents=sample_docs_str
        ))
    )

# %%
caller = get_universal_caller()
responses = asyncio.run(sample_from_model_parallel(to_send_chats, caller, max_par=64, model="openai/gpt-5", reasoning={"effort": "high"}, max_tokens=8192))

# %%
labels = [response.first_response for response in responses]

def convert_labels(label: str) -> str:
    label = int(label)
    if label == -1:
        return "N/A"
    return labels[label]

# add a new column of descriptions in the csv
labels_df["Topic_Summary"] = labels_df["Topic"].apply(convert_labels)
labels_df.to_csv("data/ultrafeedback/labels_20k.csv", index=True)

# %%
labels_df = pd.read_csv("data/ultrafeedback/labels_20k.csv")
labels_df

# %%
cluster_topics_df: pd.DataFrame = topic_model.get_topic_info()
cluster_topics_df.to_csv("data/ultrafeedback/cluster_20k.csv", index=True)

# %%
# 1309s (22 minutes)
print("Computing hierarchical topics...")
start_time = time.time()
hierarchical_topics_df: pd.DataFrame = topic_model.hierarchical_topics(docs)
print(f"BERTopic hierarchical_topics in {time.time() - start_time:.2f}s")
hierarchical_topics_df.to_csv("data/wildchat/hierarch_50k.csv", index=True)


# %%
fig: go.Figure = topic_model.visualize_hierarchy()
fig.write_html("data/wildchat/hierarchy_visual.html")

# %%
# Re-calculate topic representations

# load labels dataframe
labels_df: pd.DataFrame = pd.read_csv("data/wildchat/labels_50k.csv")
topics = labels_df["Topic"].tolist()
topic_model.topics_ = topics
topic_model.update_topics(
    docs,
    top_n_words=10,
    topics=topics,
    representation_model=representation_model,
)
# %%
