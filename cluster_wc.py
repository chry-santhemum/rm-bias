"""Cluster prompts in WildChat"""
# %%
import time
import pickle
import random
from tqdm.auto import tqdm
import dotenv
dotenv.load_dotenv()
import nest_asyncio
nest_asyncio.apply()
import plotly.graph_objects as go

import openai
import numpy as np
import pandas as pd
import torch
# import hdbscan
# from datasets import Dataset, load_dataset
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import MiniBatchKMeans
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import OpenAI

# %%
# ds_iter = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
# ds_iter = ds_iter.filter(
#     lambda ex: 
#     ex["turn"] == 1
#     and ex["language"] == "English" 
#     and len(ex["conversation"][0]["content"]) < 10000
# )

# ds_50k = Dataset.from_list(list(ds_iter.take(50000)))

# %%
with open("data/wildchat/ds_50k_clean.pkl", "rb") as f:
    ds_50k = pickle.load(f)
# %%
# # Basic cleaning: normalize, deduplicate, and filter low-quality prompts
# import re
# from typing import Set, Dict, Any

# def _normalize_text(text: str) -> str:
#     # lowercase, trim, collapse internal whitespace
#     return " ".join(text.lower().strip().split())

# _STOPWORDS = {
#     "the","is","at","which","on","and","a","an","to","of","in","for","it","this",
#     "that","with","as","by","from","or","be","are","was","were","can","could","should",
#     "would","do","does","did","have","has","had","but","if","than","then","so","such",
# }

# def _is_garbage(text: str) -> bool:
#     # repeated single character sequences (>=6)
#     if re.search(r"(.)\1{5,}", text):
#         return True
#     # high non-alnum (excluding whitespace) ratio
#     total = len(text)
#     if total == 0:
#         return True
#     non_alnum = sum(1 for c in text if not (c.isalnum() or c.isspace()))
#     if total > 0 and (non_alnum / total) > 0.5:
#         return True
#     # stopword-only prompts (after tokenization)
#     tokens = [t for t in text.split() if t]
#     if tokens and all(t in _STOPWORDS for t in tokens):
#         return True
#     return False

# def _clean_records(ds: Dataset,
#                    min_words: int = 4,
#                    max_words: int = 1024) -> Dataset:
#     seen: Set[str] = set()
#     cleaned: list[Dict[str, Any]] = []
#     for item in tqdm(ds):
#         try:
#             raw = item["conversation"][0]["content"]  # type: ignore[index]
#             if not isinstance(raw, str):
#                 continue
#             norm = _normalize_text(raw)
#             wc = len(norm.split())
#             if wc < min_words or wc > max_words:
#                 continue
#             if _is_garbage(norm):
#                 continue
#             if norm in seen:
#                 continue
#             seen.add(norm)
#             # keep original content; use normalized only for dedup/filtering
#             new_item = dict(item)
#             conv = list(item["conversation"])  # type: ignore[index]
#             first_turn = dict(conv[0])
#             first_turn["content"] = raw
#             conv[0] = first_turn
#             new_item["conversation"] = conv
#             cleaned.append(new_item)
#         except Exception:
#             continue
#     return Dataset.from_list(cleaned)

# ds_50k = _clean_records(ds_50k)

# with open("data/wildchat/ds_50k_clean.pkl", "wb") as f:
#     pickle.dump(ds_50k, f)

# # %%
# print(ds_50k[0])
# # %%
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # %%
# batch_size = 128
# embeddings = torch.zeros(len(ds_50k), 384, dtype=torch.float32)

# for i in tqdm(range(0, len(ds_50k), batch_size)):
#     batch = ds_50k.select(range(i, min(i + batch_size, len(ds_50k))))
#     prompts = [
#         item["conversation"][0]["content"]  # type: ignore
#         for item in batch
#     ]
#     batch_emb = embedding_model.encode(prompts)
#     embeddings[i:i+batch_size] = torch.tensor(batch_emb, dtype=torch.float32)


# # %%
# with open("data/wildchat/emb_50k_clean.pkl", "wb") as f:
#     pickle.dump(embeddings, f)

# %%
with open("data/wildchat/emb_50k_clean.pkl", "rb") as f:
    embeddings = pickle.load(f)

# %%
# inertias = []
# cluster_range = range(50, 301, 10)
# for k in tqdm(cluster_range):
#     model = MiniBatchKMeans(n_clusters=k, random_state=42)
#     model.fit(embeddings)
#     inertias.append(model.inertia_)

# # %%

# with open("data/temp_inertias.pkl", "wb") as f:
#     pickle.dump(inertias, f)

# # %%

# with open("data/temp_inertias.pkl", "rb") as f:
#     inertias = pickle.load(f)
# cluster_range = range(50, 301, 10)

# import plotly.express as px

# fig = px.line(x=cluster_range, y=inertias, markers=True)
# fig.show()

# # %%

# # use HDBSCAN to cluster embeddings
# emb_np = torch.nn.functional.normalize(embeddings, p=2, dim=1).numpy()
# emb_np = emb_np.astype(np.float64)

# start_time = time.time()
# min_cluster_size = 10
# min_samples = 5
# clusterer = hdbscan.HDBSCAN(
#     min_cluster_size=min_cluster_size,
#     min_samples=min_samples,
#     metric="cosine",
#     cluster_selection_method="eom",
#     algorithm="generic",
# )
# clusterer.fit(emb_np)

# labels = clusterer.labels_
# probabilities = clusterer.probabilities_
# n = emb_np.shape[0]
# num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# noise = int((labels == -1).sum())

# print(
#     f"HDBSCAN min_cluster_size={min_cluster_size}, min_samples={min_samples} -> "
#     f"{num_clusters} clusters, noise {noise}/{n} in {time.time() - start_time:.2f}s"
# )

# with open("data/hdbscan_labels.pkl", "wb") as f:
#     pickle.dump({
#         "min_cluster_size": min_cluster_size,
#         "min_samples": min_samples,
#         "labels": labels,
#         "probabilities": probabilities,
#     }, f)
# # %%
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 8))
# clusterer.condensed_tree_.plot()
# plt.show()

# # %%
# # print some clusters

# cluster_ids = sorted(random.sample(range(num_clusters), 10))
# for cluster_id in cluster_ids:
#     print("=" * 80)
#     print(f"Cluster {cluster_id}:")
#     indices = np.where(labels == cluster_id)[0]

#     for index in indices:
#         index = int(index)
#         print(ds_50k[index]["conversation"][0]["content"])
#         print("\n")


# %%
client = openai.OpenAI()
CLUSTER_PROMPT_TEMPLATE = "You will extract a short topic label from given documents and keywords.\nHere are two examples of topics you created before:\n\n# Example 1\nSample texts from this topic:\n- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.\n- Meat, but especially beef, is the worst food in terms of emissions.\n- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.\n\nKeywords: meat beef eat eating emissions steak food health processed chicken\ntopic: Environmental impacts of eating meat\n\n# Example 2\nSample texts from this topic:\n- I have ordered the product weeks ago but it still has not arrived!\n- The website mentions that it only takes a couple of days to deliver but I still have not received mine.\n- I got a message stating that I received the monitor but that is not true!\n- It took a month longer to deliver than was advised...\n\nKeywords: deliver weeks product shipping long delivery received arrived arrive week\ntopic: Shipping and delivery issues\n\n# Your task\nSample texts from this topic:\n[DOCUMENTS]\n\nKeywords: [KEYWORDS]\n\nBased on the information above, extract a short topic label (a short phrase of at most 10 words) in the following format:\ntopic: <topic_label>\n"

representation_model = OpenAI(
    client,
    model="gpt-4o",
    prompt=CLUSTER_PROMPT_TEMPLATE,
    chat=True,
    nr_docs=10,
)
topic_model = BERTopic(
    vectorizer_model=CountVectorizer(stop_words="english"),
    representation_model=representation_model,
)

# %%
prompts = [item["conversation"][0]["content"] for i, item in enumerate(ds_50k)]  # type: ignore
emb_np = torch.nn.functional.normalize(embeddings, p=2, dim=1).numpy()


start_time = time.time()
topics, probs = topic_model.fit_transform(prompts, emb_np)
print(f"BERTopic fit_transform in {time.time() - start_time:.2f}s")

# %%
labels_df = topic_model.get_document_info(prompts)
labels_df.to_csv("data/wildchat/labels.csv", index=True)

# %%
cluster_topics_df: pd.DataFrame = topic_model.get_topic_info()
cluster_topics_df.to_csv("data/wildchat/cluster.csv", index=True)

# %%
start_time = time.time()
hierarchical_topics_df: pd.DataFrame = topic_model.hierarchical_topics(prompts)
print(f"BERTopic hierarchical_topics in {time.time() - start_time:.2f}s")
hierarchical_topics_df.to_csv("data/wildchat/hierarchical.csv", index=True)


# %%
fig: go.Figure = topic_model.visualize_hierarchy()
fig.write_html("data/wildchat/hierarchy_visual.html")

# %%