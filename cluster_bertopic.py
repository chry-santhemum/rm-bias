# %%
import patches
import re
from typing import Set, Dict, Any
import time
import pickle
import random
import asyncio
from tqdm.auto import tqdm
import dotenv
import nest_asyncio
from collections import defaultdict
import plotly.graph_objects as go
from pathlib import Path
from pprint import pprint

from umap import UMAP
from sentence_transformers import SentenceTransformer
import pandas as pd
import tiktoken
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from datasets import Dataset, load_dataset

from llm_types import ChatHistory
from client import get_universal_caller, sample_from_model_parallel

dotenv.load_dotenv()
nest_asyncio.apply()
tokenizer = tiktoken.get_encoding("gpt2")

# %%
# Filter questions whose responses are too long;
# By standard we sample 1024 tokens from the policy
# Here we filter for at most 512 tokens


def preprocess_ultrafeedback(item):
    item["chosen"] = item["chosen"][1]["content"]
    item["rejected"] = item["rejected"][1]["content"]
    item["prompt_length"] = len(tokenizer.encode(item["prompt"], disallowed_special=()))
    item["chosen_length"] = len(tokenizer.encode(item["chosen"], disallowed_special=()))
    item["rejected_length"] = len(
        tokenizer.encode(item["rejected"], disallowed_special=())
    )
    return item


start_time = time.time()
ultrafeedback = (
    load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    .map(
        preprocess_ultrafeedback,
        num_proc=8,  # type: ignore
    )
    .filter(
        lambda item: item["prompt_length"] < 512
        and item["chosen_length"] < 512
        and item["rejected_length"] < 512,
        num_proc=8,
    )
)
print(f"Preprocessing in {time.time() - start_time:.2f}s")
print("Ultrafeedback after filtering: ", len(ultrafeedback))  # 40477

# %%
Path("data/ultrafeedback").mkdir(parents=True, exist_ok=True)
with open("data/ultrafeedback/ds_filtered.pkl", "wb") as f:
    pickle.dump(ultrafeedback, f)

# %%


def preprocess_wildchat(item):
    item["prompt"] = item["conversation"][0]["content"]
    item["prompt_length"] = len(tokenizer.encode(item["prompt"], disallowed_special=()))
    return item


wildchat_iter = load_dataset(
    "allenai/WildChat-1M", split="train", streaming=True
).filter(
    lambda ex: ex["turn"] == 1 and ex["language"] == "English",
)

print("Taking 50k...")
wildchat = Dataset.from_list(list(wildchat_iter.take(50000)))

Path("data/wildchat").mkdir(parents=True, exist_ok=True)
with open("data/wildchat/ds_pre_filter.pkl", "wb") as f:
    pickle.dump(wildchat, f)

# %%
with open("data/wildchat/ds_pre_filter.pkl", "rb") as f:
    wildchat = pickle.load(f)

print("Preprocessing...")
start_time = time.time()
wildchat = wildchat.map(
    preprocess_wildchat,
    num_proc=8,
    remove_columns=wildchat.column_names,
).filter(
    lambda ex: ex["prompt_length"] < 512,
    num_proc=8,
)

print(f"Preprocessing in {time.time() - start_time:.2f}s")
print("Wildchat after filtering: ", len(wildchat))  # 41850

with open("data/wildchat/ds_filtered.pkl", "wb") as f:
    pickle.dump(wildchat, f)

# %%
# Basic cleaning: normalize, deduplicate, and filter low-quality prompts


def _normalize_text(text: str) -> str:
    # lowercase, trim, collapse internal whitespace
    return " ".join(text.lower().strip().split())


_STOPWORDS = {
    "the",
    "is",
    "at",
    "which",
    "on",
    "and",
    "a",
    "an",
    "to",
    "of",
    "in",
    "for",
    "it",
    "this",
    "that",
    "with",
    "as",
    "by",
    "from",
    "or",
    "be",
    "are",
    "was",
    "were",
    "can",
    "could",
    "should",
    "would",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "but",
    "if",
    "than",
    "then",
    "so",
    "such",
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


def _clean_records(ds: Dataset, min_words: int = 4, max_words: int = 1024) -> Dataset:
    seen_norms: Set[str] = set()
    seen_shingles: Set[str] = set()
    shingle_n: int = 5  # word n-gram size
    dup_overlap_ratio: float = 0.5  # consider duplicate if >=80% shingles already seen
    min_shingles_for_check: int = 5
    cleaned: list[Dict[str, Any]] = []
    for item in tqdm(ds, desc="Filtering and deduplicating"):
        try:
            raw = item["prompt"]  # type: ignore[index]
            if not isinstance(raw, str):
                continue
            norm = _normalize_text(raw)
            wc = len(norm.split())
            if wc < min_words or wc > max_words:
                continue
            if _is_garbage(norm):
                continue
            # Fast exact dedup first
            if norm in seen_norms:
                continue

            # Shingle-based fuzzy dedup (simple and fast)
            tokens = [t for t in norm.split() if t]
            if len(tokens) >= shingle_n:
                shingles = {
                    " ".join(tokens[i : i + shingle_n])
                    for i in range(len(tokens) - shingle_n + 1)
                }
            else:
                # fallback to unigrams if too short
                shingles = set(tokens)

            # If we have enough shingles, compute overlap ratio against seen_shingles
            is_dup = False
            if len(shingles) >= min_shingles_for_check:
                overlap = sum((1 for s in shingles if s in seen_shingles))
                overlap_ratio = overlap / max(1, len(shingles))
                if overlap_ratio >= dup_overlap_ratio:
                    is_dup = True

            if is_dup:
                continue

            # Accept: record exact norm and add shingles to global set
            seen_norms.add(norm)
            seen_shingles.update(shingles)
            # keep original content; use normalized only for dedup/filtering
            cleaned.append(dict(item))
        except Exception:
            continue
    return Dataset.from_list(cleaned)


# %%

with open("data/ultrafeedback/ds_filtered.pkl", "rb") as f:
    ultrafeedback = pickle.load(f)
print("Ultrafeedback before cleaning: ", len(ultrafeedback))

start_time = time.time()
ultrafeedback = _clean_records(ultrafeedback)
print(f"Data cleaning took {time.time() - start_time:.2f}s")
print("Ultrafeedback after cleaning: ", len(ultrafeedback))  # 31786

with open("data/ultrafeedback/ds_cleaned.pkl", "wb") as f:
    pickle.dump(ultrafeedback, f)

# %%

with open("data/wildchat/ds_filtered.pkl", "rb") as f:
    wildchat = pickle.load(f)
print("Wildchat before cleaning: ", len(wildchat))

start_time = time.time()
wildchat = _clean_records(wildchat)
print(f"Data cleaning took {time.time() - start_time:.2f}s")
print("Wildchat after cleaning: ", len(wildchat))  # 27025

with open("data/wildchat/ds_cleaned.pkl", "wb") as f:
    pickle.dump(wildchat, f)

# %%
# docs = [item["conversation"][0]["content"] for i, item in enumerate(ds_50k_clean)]  # type: ignore
with open("data/ultrafeedback/ds_20k.pkl", "rb") as f:
    ds_20k = pickle.load(f)

docs = list(ds_20k["prompt"])
print(docs[0])

# %%
embedding_model = model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

umap_model = UMAP(
    n_neighbors=15,
    n_components=10,
    min_dist=0.0,
    metric="cosine",
    low_memory=False,
)

# %%
with open("data/wildchat/ds_cleaned.pkl", "rb") as f:
    wildchat = pickle.load(f)
docs = list(wildchat["prompt"])
print(docs[10086])

topic_model = BERTopic(
    min_topic_size=30,
    umap_model=umap_model,
    embedding_model=embedding_model,
    vectorizer_model=CountVectorizer(stop_words="english"),
)

print("Fitting BERTopic...")
start_time = time.time()
topics, probs = topic_model.fit_transform(docs)
print(f"BERTopic fit_transform in {time.time() - start_time:.2f}s")

labels_df = topic_model.get_document_info(docs)
labels_df.to_csv("data/wildchat/labels.csv", index=True)  # type: ignore

# %%
with open("data/ultrafeedback/ds_cleaned.pkl", "rb") as f:
    ultrafeedback = pickle.load(f)
docs = list(ultrafeedback["prompt"])
print(docs[10086])

topic_model = BERTopic(
    min_topic_size=30,
    umap_model=umap_model,
    embedding_model=embedding_model,
    vectorizer_model=CountVectorizer(stop_words="english"),
)

print("Fitting BERTopic...")
start_time = time.time()
topics, probs = topic_model.fit_transform(docs)
print(f"BERTopic fit_transform in {time.time() - start_time:.2f}s")

labels_df = topic_model.get_document_info(docs)
labels_df.to_csv("data/ultrafeedback/labels.csv", index=True)  # type: ignore

# %%

labels_df = pd.read_csv("data/wildchat/labels.csv")

num_topics = len(set(labels_df["Topic"].tolist()))  # type: ignore
clusters = defaultdict(list)
representative = defaultdict(list)

for index, row in labels_df.iterrows():
    topic_id = int(row["Topic"])
    if topic_id >= 0:
        clusters[topic_id].append((row["Document"], row["Probability"]))
        if bool(row["Representative_document"]):
            representative[topic_id].append(row["Document"])

for topic_id in range(0, num_topics - 1):
    print(f"Topic {topic_id} has {len(clusters[topic_id])} documents")

# %%
NUM_SAMPLED_DOCS = 30
CLUSTER_PROMPT_TEMPLATE = """Your task is to extract a short summary label from a list of given documents and keywords. Each document is a user prompt for a chatbot. The summary label should be a short phrase of at most one sentence, summarizing the common *topic* and *intent* of the user prompts *on a high level*. Make sure to summarize both the topic (what the user is asking about) and the intent (what kind of response the user is looking for).

## Three representative documents (documents that lie close to the center of the cluster)

{representative_documents}

## A list of {num_sampled_docs} randomly sampled documents from the cluster

{sample_documents}

Think carefully, and only output the short summary label.
"""

to_send_chats = []

for topic_id in range(0, num_topics - 1):
    sample_docs = random.sample(clusters[topic_id], NUM_SAMPLED_DOCS)
    sample_docs_str = ("\n" + "-" * 10 + "\n").join([doc[0] for doc in sample_docs])

    representative_docs = representative[topic_id]
    representative_docs_str = ("\n" + "-" * 10 + "\n").join(
        [doc for doc in representative_docs]
    )

    to_send_chats.append(
        ChatHistory.from_system("You are an expert at summarizing documents.").add_user(
            CLUSTER_PROMPT_TEMPLATE.format(
                representative_documents=representative_docs_str,
                sample_documents=sample_docs_str,
                num_sampled_docs=NUM_SAMPLED_DOCS,
            )
        )
    )

# %%
caller = get_universal_caller()
responses = asyncio.run(
    sample_from_model_parallel(
        to_send_chats,
        caller,
        max_par=64,
        desc="Summarizing topics",
        model="openai/gpt-5",
        reasoning={"effort": "medium"},
        max_tokens=8192,
    )
)
summaries = [response.first_response for response in responses]

# %%

pprint(summaries)


# %%
def convert_labels(label: str) -> str:
    label = int(label)  # type: ignore
    if label == -1:
        return "N/A"
    return summaries[label]  # type: ignore


# add a new column of descriptions in the csv
labels_df["Topic_Summary"] = labels_df["Topic"].apply(convert_labels)  # type: ignore
labels_df.to_csv("data/wildchat/labels_summaries.csv", index=True)  # type: ignore


# %%
# Filtering out prompts that are not well-described by the cluster summary

labels_df = pd.read_csv("data/ultrafeedback/labels_summaries.csv")

num_topics = len(set(labels_df["Topic"].tolist()))  # type: ignore
summaries = defaultdict(str)
clusters = defaultdict(list)

for index, row in labels_df.iterrows():
    topic_id = int(row["Topic"])
    if topic_id >= 0:
        clusters[topic_id].append(row)
        if topic_id not in summaries:
            summaries[topic_id] = row["Topic_Summary"]  # type: ignore

start, end = 21, 31

for topic_id in range(start, end):
    print(
        f"Topic {topic_id}: {len(clusters[topic_id])} docs, summary: {summaries[topic_id]}"
    )
    sorted_cluster = sorted(
        clusters[topic_id], key=lambda row: float(row["Probability"]), reverse=True
    )
    clusters[topic_id] = sorted_cluster[:200]

    # for row in clusters[topic_id][:10]:
    #     print(row["Document"])
    #     print("-"*80)

# %%
VERIFICATION_PROMPT_TEMPLATE = """Your task is to decide whether a given summary accurately reflects the *topic* and *intent* of a given user prompt.

## User Prompt

{user_prompt}

## Summary

{summary}

As a useful heuristic, you can consider the summary to accurately reflect the user prompt, if any generic response to the summary can be adapted to be a response to the specific user prompt.

Think carefully, and only output "Yes" or "No" in your response: "Yes" if the summary accurately reflects the user prompt, "No" otherwise.
"""


to_send_chats = []
chat_topic_ids = []
rows_list = []

for topic_id in range(start, end):
    for row in clusters[topic_id]:
        to_send_chats.append(
            ChatHistory.from_system(
                "You are an expert at summarizing documents and understanding user prompts."
            ).add_user(
                VERIFICATION_PROMPT_TEMPLATE.format(
                    summary=summaries[topic_id],
                    user_prompt=row["Document"],
                )
            )
        )
        chat_topic_ids.append(topic_id)
        rows_list.append(row)

print(f"Sending {len(to_send_chats)} chats to the model")

# %%
caller = get_universal_caller()
responses = asyncio.run(
    sample_from_model_parallel(
        to_send_chats,
        caller,
        max_par=32,
        desc="Filtering prompts",
        model="openai/gpt-5-mini",
        reasoning={"effort": "high"},
        max_tokens=4096,
    )
)
answers = [response.first_response for response in responses]

# %%
pprint(answers)

# %%
filtered_stats = defaultdict(int)
for i, answer in enumerate(answers):
    if answer.lower() == "yes":
        rows_list[i]["Verified"] = True
        filtered_stats[chat_topic_ids[i]] += 1
    else:
        rows_list[i]["Verified"] = False
    rows_list[i]["Verified_Reasoning"] = responses[i].reasoning_content

for topic_id in range(start, end):
    print(f"Topic {topic_id} has {filtered_stats[topic_id]} documents")

# %%
# Put rows into a df

filtered_rows_df = pd.DataFrame(rows_list)
filtered_rows_df.to_csv("data/wildchat/ds_verified_test.csv", index=True)

# %%

responses[0].reasoning_content


# %%
# cluster_topics_df: pd.DataFrame = topic_model.get_topic_info()
# cluster_topics_df.to_csv("data/ultrafeedback/cluster_20k.csv", index=True)

# # 1309s (22 minutes)
# print("Computing hierarchical topics...")
# start_time = time.time()
# hierarchical_topics_df: pd.DataFrame = topic_model.hierarchical_topics(docs)
# print(f"BERTopic hierarchical_topics in {time.time() - start_time:.2f}s")
# hierarchical_topics_df.to_csv("data/wildchat/hierarch_50k.csv", index=True)

# fig: go.Figure = topic_model.visualize_hierarchy()
# fig.write_html("data/wildchat/hierarchy_visual.html")
# %%
