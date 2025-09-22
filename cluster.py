# %%
import patches
import time
import pickle
import random
import asyncio
import dotenv
import nest_asyncio
from collections import defaultdict
from pprint import pprint

import numpy as np
from umap import UMAP
from sentence_transformers import SentenceTransformer
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from llm_types import ChatHistory
from client import get_universal_caller, sample_from_model_parallel

dotenv.load_dotenv()
nest_asyncio.apply()

# %%
embedding_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device_map": "auto"},
    tokenizer_kwargs={"padding_side": "left"},
)

def embed(ds_name: str, prompt: str):
    with open(f"data/{ds_name}/ds_cleaned.pkl", "rb") as f:
        ds_cleaned = pickle.load(f)
    docs = list(ds_cleaned["prompt"])

    print(f"{ds_name} has {len(docs)} docs.")
    print(f"First 5 docs from {ds_name}:\n")
    for doc in docs[:5]:
        print(doc)
        print("-"*70)
        
    start_time = time.time()
    embeddings = embedding_model.encode(
        docs, 
        prompt=prompt,
        show_progress_bar=True,
    )
    print(f"Embedding took {time.time() - start_time:.2f}s")
    np.save(f"data/{ds_name}/embeddings.npy", embeddings)
    print(f"Saved embeddings to data/{ds_name}/embeddings.npy")

# %%
CLUSTER_PROMPT = "Instruct: Given a user prompt, summarize the topic and user intent of the prompt.\n\nUser prompt:"

embed("wildchat", prompt=CLUSTER_PROMPT)

# %%

MIN_TOPIC_SIZE = 30

def fit_bertopic(
    ds_name: str,
    n_neighbors: int = 30,
    n_components: int = 8,
):
    with open(f"data/{ds_name}/ds_cleaned.pkl", "rb") as f:
        ds_cleaned = pickle.load(f)
    docs = list(ds_cleaned["prompt"])

    embeddings = np.load(f"data/{ds_name}/embeddings.npy")

    print(f"{ds_name} has {len(docs)} docs.")
    print(f"Embeddings shape: {embeddings.shape}")

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        low_memory=False,
    )

    topic_model = BERTopic(
        min_topic_size=MIN_TOPIC_SIZE,
        umap_model=umap_model,
        vectorizer_model=CountVectorizer(stop_words="english"),
    )

    print("Fitting BERTopic...")
    start_time = time.time()
    topic_model.fit_transform(
        docs,
        embeddings=embeddings,
    )
    print(f"BERTopic fit_transform in {time.time() - start_time:.2f}s")

    labels_df = topic_model.get_document_info(docs)
    labels_df.to_csv(f"data/{ds_name}/labels.csv", index=True)  # type: ignore
    print(f"Saved labels to data/{ds_name}/labels.csv")

    # Filter for rows whose probability is at least 0.8
    labels_df_likely = labels_df[labels_df["Probability"] >= 0.8]
    labels_df_likely.to_csv(f"data/{ds_name}/labels_likely.csv", index=True)  # type: ignore
    print(f"Saved labels with probability at least 0.8 to data/{ds_name}/labels_likely.csv")

    return topic_model

# %%
topic_model = fit_bertopic("wildchat")

# %%
# Look at some clusters
labels = pd.read_csv("data/wildchat/labels_likely.csv")
clusters = defaultdict(list)
representative = defaultdict(list)

for index, row in labels.iterrows():
    topic_id = int(row["Topic"])
    if topic_id >= 0:
        clusters[topic_id].append((row["Document"], row["Probability"]))
        if bool(row["Representative_document"]):
            representative[topic_id].append(row["Document"])

for topic_id in sorted(list(set(labels["Topic"].tolist()))):
    print(f"Topic {topic_id} has {len(clusters[topic_id])} documents")

    for doc in clusters[topic_id][:10]:
        print("-"*70)
        print(doc[0])

    print("="*70)

# Print number of documents with certain probability

def get_num_docs_with_probability(cluster: list[tuple[str, float]], probability: float) -> int:
    return len([doc for doc in cluster if doc[1] >= probability])

p = 0.8
for topic_id in sorted(list(set(labels["Topic"].tolist()))):
    print(f"Topic {topic_id} has {get_num_docs_with_probability(clusters[topic_id], p)} out of {len(clusters[topic_id])} documents with probability >= {p}")

# %%
NUM_SAMPLED_DOCS = 20

TOPIC_MODELING_PROMPT = """Your task is to extract a short summary label from a list of given documents. Each document is a user prompt for a chatbot. The summary label should be a short phrase of at most one sentence, summarizing the common *topic* and *intent* of the user prompts *on a high level*. Make sure to summarize both the topic (what the user is asking about) and the intent (what kind of response the user is looking for).

## Three representative documents (documents that lie close to the center of the cluster)

{representative_documents}

## A list of {num_sampled_docs} randomly sampled documents from the cluster

{sample_documents}

Think carefully, and then in your output field only output the short summary."""


def topic_modeling(ds_name: str):
    labels = pd.read_csv(f"data/{ds_name}/labels_likely.csv")
    clusters = defaultdict(list)
    representative = defaultdict(list)

    for _, row in labels.iterrows():
        topic_id = int(row["Topic"])
        if topic_id >= 0:
            clusters[topic_id].append(row)
            if bool(row["Representative_document"]):
                representative[topic_id].append(row["Document"])

    to_send_chats = []
    summary_to_topic_id = []

    for topic_id in set(labels["Topic"].tolist()):
        if len(clusters[topic_id]) < MIN_TOPIC_SIZE:
            continue

        sample_docs = random.sample(clusters[topic_id], NUM_SAMPLED_DOCS)
        sample_docs_str = ("\n" + "-" * 10 + "\n").join([row["Document"] for row in sample_docs])

        representative_docs = representative[topic_id]
        representative_docs_str = ("\n" + "-" * 10 + "\n").join(
            [doc for doc in representative_docs]
        )

        to_send_chats.append(
            ChatHistory.from_system("You are an expert at summarizing documents.").add_user(
                TOPIC_MODELING_PROMPT.format(
                    representative_documents=representative_docs_str,
                    sample_documents=sample_docs_str,
                    num_sampled_docs=NUM_SAMPLED_DOCS,
                )
            )
        )
        summary_to_topic_id.append(topic_id)

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

    pprint(summaries)

    def convert_labels(label: str) -> str:
        label = int(label)  # type: ignore
        if label in summary_to_topic_id:
            return summaries[summary_to_topic_id.index(label)]  # type: ignore
        return "N/A"

    # add a new column of descriptions in the csv
    labels["Topic_Summary"] = labels["Topic"].apply(convert_labels)  # type: ignore
    labels.to_csv(f"data/{ds_name}/labels_summaries.csv", index=True)  # type: ignore
    print(f"Saved to data/{ds_name}/labels_summaries.csv")

# %%
topic_modeling("wildchat")

# %%
# Look at some clusters

labels = pd.read_csv("data/wildchat/labels_summaries.csv")
summaries = defaultdict(str)
clusters = defaultdict(list)

for _, row in labels.iterrows():
    topic_id = int(row["Topic"])
    if topic_id >= 0:
        clusters[topic_id].append(row)
        if topic_id not in summaries:
            summaries[topic_id] = row["Topic_Summary"]  # type: ignore

for topic_id in sorted(list(set(labels["Topic"].tolist()))):
    print(
        f"Topic {topic_id}: {len(clusters[topic_id])} docs, summary: {summaries[topic_id]}"
    )
    for row in clusters[topic_id][:10]:
        print("-"*80)
        print(row["Document"])

    print("="*80)
        

# %%
# Filtering out prompts that are not well-described by the cluster summary

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

start, end = 15, 20

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
    rows_list[i]["Verified_Reasoning"] = str(responses[i].reasoning_content)

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
