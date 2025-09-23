# %%
import patches  # monkey patching
import os
import json
import random
import pickle
import logging
import asyncio
import multiprocessing
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
from prompt_stats import CLUSTER_DATASETS, load_clusters
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


    @staticmethod
    def classifier_worker(task_queue, result_queue):
        """
        Worker process for running GPU-intensive classifier ratings.
        """
        while True:
            task_args = task_queue.get()
            if task_args is None: # Sentinel value to exit
                break
            
            # Unpack arguments
            rating_function, policy_model, seed_state, prompts, n_samples = task_args

            print(f"GPU Worker: Rating for seed {seed_state.index}...")
            # Since the original function was async, we use asyncio.run here.
            asyncio.run(rating_function(
                policy_model=policy_model,
                seed_state=seed_state,
                train_batch_prompts=prompts,
                per_prompt_normalize=True,
                n_samples=n_samples,
                use_tqdm=False,
            ))
            result_queue.put(f"Completed GPU rating for seed {seed_state.index}")


    @staticmethod
    def lm_judge_worker(task_queue, result_queue):
        """Worker for I/O-intensive LM Judge ratings."""
        while True:
            task_args = task_queue.get()
            if task_args is None:
                break

            rating_function, policy_model, seed_state, prompts, n_samples = task_args
            print(f"API Worker: Starting rating for seed {seed_state.index}...")

            asyncio.run(rating_function(
                policy_model=policy_model,
                seed_state=seed_state,
                train_batch_prompts=prompts,
                per_prompt_normalize=False,
                n_samples=n_samples,
                use_tqdm=False,
            ))
            result_queue.put(f"Completed API rating for seed {seed_state.index}")


    def get_ratings(self, n_samples: int = 1):
        logger.info(
            f"[TRAIN STEP {self.step_count}] Rating attacks with {n_samples} samples..."
        )
        train_batch_prompts = {}
        classifier_tasks = []
        lm_judge_tasks = []

        for seed_state in self.seed_states:
            if seed_state.index not in train_batch_prompts:
                train_batch_prompts[seed_state.index] = random.sample(
                    seed_state.cluster.train_prompts,
                    seed_state.cluster.train_batch_size,
                )

        for rating_function in [self.rater_1, self.rater_2]:
            for seed_state in self.seed_states:
                task_args = (
                    rating_function,
                    self.policy_model,
                    seed_state,
                    train_batch_prompts[seed_state.index],
                    n_samples,
                )
                if rating_function.rating_function_type == "classifier":
                    classifier_tasks.append(task_args)
                elif rating_function.rating_function_type == "lm_judge":
                    lm_judge_tasks.append(task_args)

        # Run classifier and LM judge simultaneously
        multiprocessing.set_start_method("spawn", force=True)
        gpu_queue = multiprocessing.Queue()
        api_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        gpu_process = multiprocessing.Process(target=Runner.classifier_worker, args=(gpu_queue, result_queue))
        api_process = multiprocessing.Process(target=Runner.lm_judge_worker, args=(api_queue, result_queue))

        gpu_process.start()
        api_process.start()

        print(f"Distributing {len(classifier_tasks)} GPU tasks and {len(lm_judge_tasks)} API tasks...")
        for task in classifier_tasks:
            gpu_queue.put(task)
        for task in lm_judge_tasks:
            api_queue.put(task)

        # Collect results
        total_tasks = len(classifier_tasks) + len(lm_judge_tasks)
        for _ in tqdm(range(total_tasks), desc="Rating attacks"):
            result = result_queue.get()
            logger.info(result)

        gpu_queue.put(None)
        api_queue.put(None)
        gpu_process.join()
        api_process.join()
        
        logger.info("All ratings completed.")


    @abstractmethod
    def train(self, *args, **kwargs):
        pass


def load_contrast_pairs(
    prompts: list[str],
    target_dir: Path,
    policy_model: PolicyModel,
    rater: RatingFunction,
    threshold: float = 1.0,
) -> list[dict]:
    """
    For each user prompt, check in target_dir if the rollouts have enough variation,
    according to the given rater.

    Returns {"prompt": ..., "chosen": ..., "rejected": ...}
    """
    contrast_pairs = []

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
                contrast_pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": rollouts_sorted[0]["response"],
                        "rejected": rollouts_sorted[-1]["response"],
                    }
                )

    return contrast_pairs


def load_initial_seed_states(
    target_dir: Path,
    dataset: str,
    topic_ids: list[int] = [],  # only for datasets in CLUSTER_DATASETS
    train_batch_size: int = 0, 
):
    initial_seed_states = []
    id_to_cluster = load_clusters(dataset, topic_ids=topic_ids)
    
    for id, cluster_dict in id_to_cluster.items():
        prompts = cluster_dict["prompts"]
        train_size = len(prompts) * 4 // 5
        train_prompts = prompts[:train_size]
        val_prompts = prompts[train_size:]

        if train_batch_size > len(train_prompts):
            raise ValueError(f"Train batch size {train_batch_size} is greater than the number of train prompts {len(train_prompts)}")

        cluster = Cluster(
            summary=cluster_dict["summary"],
            train_prompts=train_prompts,
            val_prompts=val_prompts,
            train_batch_size=(
                train_batch_size if train_batch_size > 0 else len(train_prompts)
            ),
        )
        initial_seed_states.append(
            SeedState(
                index=id,
                dataset=target_dir.name,
                cluster=cluster,
                state={},
                history=[],
            )
        )

    print(f"Loaded {len(initial_seed_states)} seed states")
    for state in initial_seed_states:
        print(
            f"  - Seed {state.index}, {len(state.cluster.train_prompts)} train prompts:\n"
            f"    {state.cluster.summary}"
        )

    return initial_seed_states
