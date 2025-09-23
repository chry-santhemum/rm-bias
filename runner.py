# %%
import patches  # monkey patching
import os
import json
import random
import pickle
import logging
import asyncio
from tqdm.auto import tqdm
from pathlib import Path
from typing import Literal, Tuple
from slist import Slist
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_dataset
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer

from utils import timestamp, get_to_pass_reasoning
from viz_utils import (
    save_system_prompt_stats,
    convert_attack_to_dict,
    save_cluster_info,
)
from rater import (
    normalize,
    prompt_rollout,
    prompt_rating,
    prompt_to_hash_path,
    PolicyModel,
    RatingFunction,
    RewardModel,
    LLMJudge,
)
from state import SeedState, Cluster
from defaults import *
from client import get_universal_caller, sample_from_model_parallel, OpenaiResponse
from llm_types import ChatHistory

logger = logging.getLogger(__name__)


class ClusterModel:
    def __init__(
        self,
        embedding_model_name: str,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 8,
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.umap_model = UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            min_dist=0.0,
            metric="cosine",
            low_memory=False,
        )

    def embed(self, inputs: list[str]) -> np.ndarray:
        return self.embedding_model.encode(inputs)

    def reduce_embed(self, inputs: list[str]) -> np.ndarray:
        """Embed then do dimensionality reduction"""

        embeddings: np.ndarray = self.embed(inputs)
        return self.umap_model.fit_transform(embeddings)  # type: ignore

    def cluster(
        self, inputs: list[str], n_clusters: int
    ) -> Tuple[list[str], list[int]]:
        reduced_embeddings = self.reduce_embed(inputs)

        # log the pairwise distance matrix
        logger.info(
            f"Pairwise distance matrix:\n"
            f"{pairwise_distances(reduced_embeddings, metric='cosine')}"
        )

        kmeans = KMeans(
            n_clusters=min(len(inputs), n_clusters), random_state=10086, n_init="auto"
        )
        kmeans.fit(reduced_embeddings)

        closest_point_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, reduced_embeddings
        )

        sorted_indices = sorted(closest_point_indices)
        selected = [inputs[i] for i in sorted_indices]

        return selected, sorted_indices

    def cluster_dbscan(
        self,
        inputs: list[str],
        dbscan_eps: float,
    ) -> Tuple[dict[int, list[str]], dict[int, list[int]]]:
        embeddings = self.embed(inputs)

        # log the pairwise distance matrix
        logger.info(
            f"Pairwise distance matrix:\n"
            f"{pairwise_distances(embeddings, metric='cosine')}"
        )

        dbscan = DBSCAN(
            eps=dbscan_eps, min_samples=2 * self.umap_n_components, metric="cosine"
        )
        dbscan.fit(embeddings)

        niches = defaultdict(list)
        indices = defaultdict(list)
        for i, label in enumerate(dbscan.labels_):
            niches[label].append(inputs[i])
            indices[label].append(i)

        logger.info(
            "Niches:\n"
            + "\n".join(
                [
                    f"Niche {label}:\n{"\n".join(members)}"
                    for label, members in niches.items()
                ]
            )
        )

        return niches, indices


class Planner(ABC):
    def __init__(
        self,
        planner_model_names: list[str],
        alloy_type: Literal["round_robin", "random"],
        max_tokens: int,
        reasoning: int | str | None = None,
        temperature: float = 0.7,
        max_par: int = 64,  # max parallel calls to client
        full_logging: bool = False,
    ):
        self.planner_model_names = planner_model_names
        self.alloy_type = alloy_type
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.temperature = temperature
        self.max_par = max_par
        self.full_logging = full_logging

        self.caller = get_universal_caller()
        self.curr_planner_index: int = 0

    @property
    def curr_planner_model(self):
        return self.planner_model_names[self.curr_planner_index]

    def step_planner_model(self):
        if self.alloy_type == "round_robin":
            self.curr_planner_index = (self.curr_planner_index + 1) % len(
                self.planner_model_names
            )
        elif self.alloy_type == "random":
            self.curr_planner_index = random.randint(
                0, len(self.planner_model_names) - 1
            )

    async def _sample_from_model_parallel(
        self, prompts: list[ChatHistory], desc: str = "Planning"
    ) -> Slist[OpenaiResponse]:
        return await sample_from_model_parallel(
            caller=self.caller,
            prompts=prompts,
            max_par=self.max_par,
            full_logging=self.full_logging,
            desc=desc,
            model=self.curr_planner_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning=get_to_pass_reasoning(self.reasoning, self.max_tokens),
        )


class Runner(ABC):
    def __init__(
        self,
        seed_states: list[SeedState],
        planner: Planner,
        policy_model: PolicyModel,
        rater_1: RatingFunction,
        rater_2: RatingFunction,
        run_name: str | None,
        *args,
        **kwargs,
    ):
        self.step_count = 0
        self.seed_states = seed_states
        self.planner = planner
        self.policy_model = policy_model
        self.rater_1 = rater_1
        self.rater_2 = rater_2

        self.run_name = run_name or f"{timestamp()}"
        self.run_path.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def runner_type(self) -> str:
        pass

    @property
    def run_path(self) -> Path:
        return Path(f"/workspace/rm-bias/data/{self.runner_type}/{self.run_name}")

    def save_seed_states(self):
        logger.info(f"[TRAIN STEP {self.step_count}] Saving seed states...")
        with open(
            os.path.join(self.run_path, f"step_{self.step_count}.pkl"), "wb"
        ) as f:
            pickle.dump(self.seed_states, f)
        # remove previous files
        for f in os.listdir(self.run_path):
            if f.startswith("step_") and f != f"step_{self.step_count}.pkl":
                os.remove(os.path.join(self.run_path, f))

    def save_complete_system_prompt_stats(self):
        """Save complete SystemPromptStats with attacks and ratings after rating is done."""
        logger.info("[VIZ] Saving complete system prompt stats...")
        for seed_state in self.seed_states:
            for system_prompt, stats in seed_state.history[-1].items():
                if stats.attacks:  # Only save if we have attacks with ratings
                    # Convert attacks to dict format
                    attacks_dict = [
                        convert_attack_to_dict(attack) for attack in stats.attacks
                    ]

                    # Get existing metadata if it exists (from initial save)
                    from viz_utils import hash_system_prompt

                    prompt_hash = hash_system_prompt(system_prompt)
                    existing_file = (
                        self.run_path
                        / f"seed_{seed_state.index}"
                        / f"{prompt_hash}.json"
                    )
                    meta = {
                        "step": self.step_count,
                        "operation": "unknown",  # default, will be overwritten
                    }
                    if existing_file.exists():
                        try:
                            with open(existing_file, "r") as f:
                                existing_data = json.load(f)
                                meta.update(existing_data.get("meta", {}))
                        except (json.JSONDecodeError, IOError):
                            pass

                    save_system_prompt_stats(
                        run_path=self.run_path,
                        seed_id=seed_state.index,
                        system_prompt=system_prompt,
                        attacks=attacks_dict,
                        mean_score=stats.mean_adversarial_score,
                        stdev_score=stats.stdev_adversarial_score,
                        meta=meta,
                    )

    def initialize(self):
        assert all(len(seed_state.history) == 0 for seed_state in self.seed_states)

        # Save cluster info for visualization
        logger.info("[INITIALIZE] Saving cluster info for visualization...")
        for seed_state in self.seed_states:
            sample_prompts = random.sample(
                seed_state.cluster.train_prompts,
                min(20, len(seed_state.cluster.train_prompts)),
            )
            save_cluster_info(
                run_path=self.run_path,
                seed_id=seed_state.index,
                summary=seed_state.cluster.summary,
                train_batch_size=seed_state.cluster.train_batch_size,
                sample_train_prompts=sample_prompts,
            )

        logger.info(f"[INITIALIZE] Normalizing rater 1, {self.rater_1.model_name}...")
        asyncio.run(normalize(self.rater_1, self.policy_model, overwrite=False))
        logger.info(f"[INITIALIZE] Normalizing rater 2, {self.rater_2.model_name}...")
        asyncio.run(normalize(self.rater_2, self.policy_model, overwrite=False))

    def get_ratings(self, n_samples: int = 1):
        logger.info(
            f"[TRAIN STEP {self.step_count}] Rating attacks with {n_samples} samples..."
        )
        train_batch_prompts = {}

        for rating_function in [self.rater_1, self.rater_2]:
            for seed_state in self.seed_states:
                if seed_state.index not in train_batch_prompts:
                    train_batch_prompts[seed_state.index] = random.sample(
                        seed_state.cluster.train_prompts,
                        seed_state.cluster.train_batch_size,
                    )

            if rating_function.rating_function_type == "classifier":
                for seed_state in tqdm(
                    self.seed_states, desc=f"Rating with {rating_function.model_name}"
                ):
                    asyncio.run(
                        rating_function(
                            policy_model=self.policy_model,
                            seed_state=seed_state,
                            train_batch_prompts=train_batch_prompts[seed_state.index],
                            per_prompt_normalize=True,
                            n_samples=n_samples,
                        )
                    )

            elif rating_function.rating_function_type == "lm_judge":

                async def run_rating_function_one(seed_state: SeedState):
                    await rating_function(
                        policy_model=self.policy_model,
                        seed_state=seed_state,
                        train_batch_prompts=train_batch_prompts[seed_state.index],
                        per_prompt_normalize=True,
                        n_samples=n_samples,
                    )

                async def run_rating_function():
                    tasks = [
                        run_rating_function_one(seed_state)
                        for seed_state in tqdm(
                            self.seed_states,
                            desc=f"Rating with {rating_function.model_name}",
                        )
                    ]
                    await asyncio.gather(*tasks)

                asyncio.run(run_rating_function())

    @abstractmethod
    def train(self, *args, **kwargs):
        pass


def load_contrast_pairs(
    prompts: list[str],
    target_dir: Path,
    policy_model: PolicyModel,
    rater: RatingFunction,
    threshold: float = 1.0,
) -> Tuple[list[str], list[dict]]:
    """
    For each user prompt, check in target_dir if the rollouts have enough variation.
    Then return (prompts, aux_info) where aux_info are chosen / rejected pairs.
    """
    prompts_selected = []
    rollout_info = []

    # Load normalization data
    with open(f".cache/normalize/{rater.model_name}.json", "r", encoding="utf-8") as f:
        rater_stats = json.load(f)

    for prompt in prompts:
        file_path = prompt_to_hash_path(prompt, target_dir)
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            rollouts = json_data[policy_model.model_name]["rollouts"]
            rollouts_cleaned = [r for r in rollouts if r[rater.model_name] is not None]
            if len(rollouts_cleaned) == 0:
                continue

            rollouts_sorted = sorted(
                rollouts_cleaned, key=lambda x: float(x[rater.model_name]), reverse=True
            )
            score_diff = (
                rollouts_sorted[0][rater.model_name]
                - rollouts_sorted[-1][rater.model_name]
            )

            if score_diff > threshold * rater_stats["stdev"]:
                rollout_info.append(
                    {
                        "chosen": rollouts_sorted[0]["response"],
                        "rejected": rollouts_sorted[-1]["response"],
                    }
                )
                prompts_selected.append(prompt)

    return prompts_selected, rollout_info


def initialize_prompt_stats(
    target_dir: Path,
    id_to_cluster: dict[int, dict],  # cluster_id: {"prompts": [...], "summary": ...}
    policy: PolicyModel,
    raters: list[RatingFunction],
):
    all_user_prompts = []
    for cluster in id_to_cluster.values():
        all_user_prompts.extend(cluster["prompts"])

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

    for id, cluster_dict in tqdm(
        id_to_cluster.items(), desc="Adding dataset info to prompt stats"
    ):
        for prompt in cluster_dict["prompts"]:
            file_path = prompt_to_hash_path(prompt, target_dir)
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            json_data["topic_label"] = id
            json_data["topic_name"] = cluster_dict["summary"]
            json_data["dataset"] = target_dir.name

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4)


def load_initial_seed_states(
    dataset: str,
    compute_stats: bool,
    target_dir: Path,
    policy: PolicyModel,
    reward_model: RewardModel,
    llm_judge: LLMJudge,
    train_batch_size: int = 0,  # 0 means use all
):
    initial_seed_states = []

    if dataset == "ultrafeedback":
        labels: pd.DataFrame = pd.read_csv("data/ultrafeedback/ds_verified_test.csv")
        with open("data/ultrafeedback/ds_cleaned.pkl", "rb") as f:
            ultrafeedback = pickle.load(f)

        assert len(labels) == len(ultrafeedback)

        id_to_prompts = defaultdict(list)
        id_to_summary = defaultdict(str)
        topic_ids = [i for i in range(21, 31)]

        for idx, row in tqdm(labels.iterrows(), desc="Loading clusters"):
            topic = int(row["Topic"])
            if topic in topic_ids:
                item = ultrafeedback[idx]
                assert row["Document"] == item["prompt"]

                id_to_prompts[topic].append(
                    {
                        "prompt": row["Document"],
                        "chosen": item["chosen"],
                        "rejected": item["rejected"],
                        "prob": float(row["Probability"]),
                    }
                )

                if topic not in id_to_summary:
                    id_to_summary[topic] = str(row["Topic_Summary"])

        for topic in topic_ids:
            sorted_cluster = sorted(
                id_to_prompts[topic], key=lambda x: x["prob"], reverse=True
            )
            train_prompts = [item["prompt"] for item in sorted_cluster[:20]]
            aux_info = [
                {
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }
                for item in sorted_cluster[:20]
            ]

            cluster = Cluster(
                summary=id_to_summary[topic],
                train_prompts=train_prompts,
                val_prompts=[],
                train_batch_size=(
                    train_batch_size if train_batch_size > 0 else len(train_prompts)
                ),
                aux_info=aux_info,
            )

            seed_state = SeedState(
                index=topic,
                dataset="ultrafeedback",
                cluster=cluster,
                state={},
                history=[],
            )
            initial_seed_states.append(seed_state)

        id_to_cluster = {
            i: {
                "prompts": [x["prompt"] for x in id_to_prompts[i]],
                "summary": id_to_summary[i],
            }
            for i in topic_ids
        }
        if compute_stats:
            initialize_prompt_stats(
                target_dir, id_to_cluster, policy, [reward_model, llm_judge]
            )

    elif dataset == "instruction-dataset":
        instruction_test = load_dataset(
            "HuggingFaceH4/instruction-dataset", split="test"
        )
        prompts = list(instruction_test["prompt"])

        id_to_cluster = {0: {"prompts": prompts, "summary": "All"}}
        if compute_stats:
            initialize_prompt_stats(
                target_dir, id_to_cluster, policy, [reward_model, llm_judge]
            )

        prompts_selected, rollout_info = load_contrast_pairs(
            prompts, target_dir, policy, reward_model, threshold=1.5
        )

        print(f"Selected {len(prompts_selected)} prompts")

        cluster = Cluster(
            summary="Any general user prompt from a general instruction dataset.",
            train_prompts=prompts_selected,
            val_prompts=[],
            train_batch_size=(
                train_batch_size if train_batch_size > 0 else len(prompts_selected)
            ),
            aux_info=rollout_info,
        )
        initial_seed_states = [
            SeedState(
                index=0,
                dataset="instruction-dataset",
                cluster=cluster,
                state={},
                history=[],
            )
        ]

    elif dataset == "wildchat":
        pass

    print(f"Loaded {len(initial_seed_states)} seed states")
    for state in initial_seed_states:
        print(
            f"  - Seed {state.index}, {len(state.cluster.train_prompts)} train prompts:\n"
            f"    {state.cluster.summary}"
        )

    return initial_seed_states
