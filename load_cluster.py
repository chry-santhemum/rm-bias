"""Helpers for loading user prompt clusters from disk."""

# %%
import json
import random
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

from synthetic import PromptCluster
from state import Cluster, SeedState
from utils import set_seed_all


CLUSTER_DATASETS = ["alpaca", "ultrafeedback", "wildchat"]


def print_clusters(ds_name: str, print_prompts: bool = False) -> None:
    assert ds_name in CLUSTER_DATASETS
    labels = pd.read_csv(f"data/{ds_name}/labels_summaries.csv")
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
        if print_prompts:
            for row in clusters[topic_id][:10]:
                print("-" * 80)
                print(row["Document"])
            print("=" * 80)


def load_clusters(
    ds_name: str,
    topic_ids: list[int] = [],  # only for datasets in CLUSTER_DATASETS
    min_prompts_per_cluster: int = 48,
    max_prompts_per_cluster: int = 80,
    seed: int = 10086,
) -> dict[int, PromptCluster]:
    """
    Deterministically load prompts into clusters.
    """
    set_seed_all(seed)
    id_to_cluster: dict[int, PromptCluster] = {}

    if ds_name in CLUSTER_DATASETS:
        labels = pd.read_csv(f"data/{ds_name}/labels_summaries.csv")

        summaries = defaultdict(str)
        clusters = defaultdict(list)

        for _, row in labels.iterrows():
            topic_id = int(row["Topic"])
            if topic_id in topic_ids:
                clusters[topic_id].append(row)
                if topic_id not in summaries:
                    summaries[topic_id] = row["Topic_Summary"]  # type: ignore

        for topic_id in topic_ids:
            if len(clusters[topic_id]) < min_prompts_per_cluster:
                raise ValueError(
                    f"Not enough prompts for cluster {topic_id} in {ds_name}: {len(clusters[topic_id])}"
                )
            random.shuffle(clusters[topic_id])
            if len(clusters[topic_id]) > max_prompts_per_cluster:
                clusters[topic_id] = clusters[topic_id][:max_prompts_per_cluster]

            id_to_cluster[topic_id] = PromptCluster(
                summary=summaries[topic_id],
                prompts=[item["Document"] for item in clusters[topic_id]],
            )

    elif ds_name in ["instruction-dataset", "agent-harm"]:

        if ds_name == "instruction-dataset":
            instruction_test = load_dataset(
                "HuggingFaceH4/instruction-dataset", split="test"
            )
            prompts = list(instruction_test["prompt"])

            if len(prompts) < min_prompts_per_cluster:
                raise ValueError(f"Not enough prompts for {ds_name}: {len(prompts)}")
            random.shuffle(prompts)
            if len(prompts) > max_prompts_per_cluster:
                prompts = prompts[:max_prompts_per_cluster]

            id_to_cluster[0] = PromptCluster(
                summary="Any user prompt from a general instruction dataset.",
                prompts=prompts,
            )

        elif ds_name == "agent-harm":
            pass

    elif ds_name.startswith("synthetic"):
        for json_file in Path(f"data/{ds_name}").glob("*.json"):
            try:
                cluster_id = int(json_file.name.split(".")[0])
            except ValueError:
                continue
            if cluster_id in topic_ids:
                with open(json_file, "r") as f:
                    data = json.load(f)
                prompts = data["prompts"]

                if len(prompts) < min_prompts_per_cluster:
                    raise ValueError(
                        f"Not enough prompts for {ds_name}: {len(prompts)}"
                    )

                random.shuffle(prompts)
                if len(prompts) > max_prompts_per_cluster:
                    prompts = prompts[:max_prompts_per_cluster]

                id_to_cluster[cluster_id] = PromptCluster(
                    summary=data["summary"],
                    prompts=prompts,
                )

    for id, cluster in id_to_cluster.items():
        print(
            f"\n\nCluster {id}:\n"
            f"Summary: {cluster.summary}\n"
            f"Number of prompts: {len(cluster.prompts)}\n"
            f"First 5 prompts:\n"
            f"======================="
        )

        for prompt in cluster.prompts[:5]:
            print(prompt)
            print("-" * 40)

    return id_to_cluster


def load_initial_seed_states(
    ds_name: str,
    topic_ids: list[int] = [],  # only for datasets in CLUSTER_DATASETS
    train_batch_size: int = 0,
    val_split_size: int = 0,
) -> list[SeedState]:
    initial_seed_states = []
    id_to_cluster = load_clusters(ds_name, topic_ids=topic_ids)

    for id in topic_ids:
        cluster_dict = id_to_cluster[id]
        prompts = cluster_dict.prompts
        train_prompts = prompts[:-val_split_size] if val_split_size > 0 else prompts
        val_prompts = prompts[-val_split_size:] if val_split_size > 0 else []

        if train_batch_size > len(train_prompts):
            raise ValueError(
                f"Train batch size {train_batch_size} is greater than the number of train prompts {len(train_prompts)}"
            )

        cluster = Cluster(
            summary=cluster_dict.summary,
            train_prompts=train_prompts,
            val_prompts=val_prompts,
        )
        initial_seed_states.append(
            SeedState(
                index=id,
                dataset=ds_name,
                cluster=cluster,
                state={},
                history=[],
            )
        )

    print(f"\n\nLoaded {len(initial_seed_states)} seed states")
    for state in initial_seed_states:
        print(
            f"  - Seed {state.index}, {len(state.cluster.train_prompts)} train prompts:\n"
            f"    {state.cluster.summary}"
        )
    print("\n\n\n\n")

    return initial_seed_states
