# %%
import json
import random
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

from state import PromptCluster
from rater import (
    PolicyModel,
    RatingFunction,
    RewardModel,
    prompt_to_hash_path,
    prompt_rollout, 
    prompt_rating,
)
from standard_prompts import set_seed_all


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
                print("-"*80)
                print(row["Document"])
            print("="*80)


def load_clusters(
    ds_name: str,
    topic_ids: list[int] = [],  # only for datasets in CLUSTER_DATASETS
    min_prompts_per_cluster: int = 40,
    max_prompts_per_cluster: int = 160,
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
                raise ValueError(f"Not enough prompts for cluster {topic_id} in {ds_name}: {len(clusters[topic_id])}")
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

    elif ds_name == "synthetic":
        for json_file in Path("data/synthetic").glob("*.json"):
            with open(json_file, "r") as f:
                data = json.load(f)
            id_to_cluster[int(json_file.name.split(".")[0])] = PromptCluster(
                summary=data["summary"],
                prompts=data["prompts"],
            )

    for id, cluster in id_to_cluster.items():
        print(
            f"Cluster {id}:\n"
            f"Summary: {cluster.summary}\n"
            f"Number of prompts: {len(cluster.prompts)}\n"
        )

    return id_to_cluster


def initialize_prompt_stats(
    target_dir: Path,
    id_to_cluster: dict[int, PromptCluster],
    policy: PolicyModel,
    raters: list[RatingFunction] = [],
    rating_only: bool = False,
):
    all_user_prompts = []
    for cluster in id_to_cluster.values():
        all_user_prompts.extend(cluster.prompts)

    if not rating_only:
        prompt_rollout(
            prompts=all_user_prompts,
            target_dir=target_dir,
            policy_model=policy,
            n_samples=16,
        )

    for rater in raters:
        prompt_rating(
            prompts=all_user_prompts,
            target_dir=target_dir,
            rater=rater,
            policy_model=policy,
        )

    for id, cluster in tqdm(
        id_to_cluster.items(), desc="Adding dataset info to prompt stats"
    ):
        for prompt in cluster.prompts:
            file_path = prompt_to_hash_path(prompt, target_dir)
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            json_data["topic_label"] = id
            json_data["topic_name"] = cluster.summary
            json_data["dataset"] = target_dir.name

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4)


# %%
if __name__ == "__main__":
    id_to_cluster = load_clusters(
        "alpaca", 
        topic_ids=[0, 2, 4, 6, 9, 11, 15, 18, 21, 53, 71, 83]  # alpaca
        # topic_ids=[4, 5, 6, 10, 14, 16, 17, 18, 19, 24, 26, 29, 32, 36]  # wildchat
    )
    # initialize_prompt_stats(
    #     target_dir=Path("data/prompt_stats/wildchat"),
    #     id_to_cluster=id_to_cluster,
    #     policy=PolicyModel(model_name="meta-llama/llama-3.1-8b-instruct"),
    #     raters=[RewardModel(reward_model_name="skywork-v2")],
    # )
# %%
