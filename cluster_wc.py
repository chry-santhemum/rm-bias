"""Cluster prompts in WildChat"""
# %%
import time
import pickle
import json
import random
import logging
from tqdm.auto import tqdm
from itertools import islice
import asyncio
import nest_asyncio
nest_asyncio.apply()

import numpy as np
import torch
import hdbscan
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

logging.getLogger(__name__)

# %%

from datasets import load_dataset, Dataset

ds_iter = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
ds_iter = ds_iter.filter(
    lambda ex: 
    ex["turn"] == 1
    and ex["language"] == "English" 
    and len(ex["conversation"][0]["content"]) < 8192
)

ds_10k = Dataset.from_list(list(ds_iter.take(10000)))
# %%
with open("data/wildchat_10k.pkl", "rb") as f:
    ds_10k = pickle.load(f)

# %%

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
batch_size = 128
embeddings = torch.zeros(len(ds_10k), 384, dtype=torch.float32)

for i in tqdm(range(0, len(ds_10k), batch_size)):
    batch = ds_10k.select(range(i, min(i + batch_size, len(ds_10k))))
    prompts = [
        item["conversation"][0]["content"]  # type: ignore
        for item in batch
    ]
    batch_emb = embedding_model.encode(prompts)
    embeddings[i:i+batch_size] = torch.tensor(batch_emb, dtype=torch.float32)


# %%
with open("data/wildchat_10k_emb.pkl", "rb") as f:
    embeddings = pickle.load(f)

# %%
inertias = []
cluster_range = range(50, 301, 10)
for k in tqdm(cluster_range):
    model = MiniBatchKMeans(n_clusters=k, random_state=42)
    model.fit(embeddings)
    inertias.append(model.inertia_)

# %%

with open("data/temp_inertias.pkl", "wb") as f:
    pickle.dump(inertias, f)

# %%

with open("data/temp_inertias.pkl", "rb") as f:
    inertias = pickle.load(f)
cluster_range = range(50, 301, 10)

import plotly.express as px

fig = px.line(x=cluster_range, y=inertias, markers=True)
fig.show()

# %%

# use HDBSCAN to cluster embeddings
emb_np = torch.nn.functional.normalize(embeddings, p=2, dim=1).numpy()
emb_np = emb_np.astype(np.float64)

start_time = time.time()
min_cluster_size = 10
min_samples = 5
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="cosine",
    cluster_selection_method="eom",
    algorithm="generic",
)
clusterer.fit(emb_np)

labels = clusterer.labels_
probabilities = clusterer.probabilities_
n = emb_np.shape[0]
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise = int((labels == -1).sum())

print(
    f"HDBSCAN min_cluster_size={min_cluster_size}, min_samples={min_samples} -> "
    f"{num_clusters} clusters, noise {noise}/{n} in {time.time() - start_time:.2f}s"
)

with open("data/hdbscan_labels.pkl", "wb") as f:
    pickle.dump({
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "labels": labels,
        "probabilities": probabilities,
    }, f)
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
clusterer.condensed_tree_.plot()
plt.show()

# %%
# print some clusters

cluster_ids = sorted(random.sample(range(num_clusters), 10))
for cluster_id in cluster_ids:
    print("=" * 80)
    print(f"Cluster {cluster_id}:")
    indices = np.where(labels == cluster_id)[0]

    for index in indices:
        index = int(index)
        print(ds_10k[index]["conversation"][0]["content"])
        print("\n")


# %%
