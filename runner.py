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

from utils import timestamp, get_to_pass_reasoning
from viz_utils import (
    save_system_prompt_stats,
    convert_attack_to_dict,
    save_cluster_info,
)
from raters import (
    normalize,
    prompt_to_hash_path,
    PolicyModel,
    RatingFunction,
    RewardModel,
    LLMJudge,
)
from prompt_stats import CLUSTER_DATASETS, load_clusters
from state import SeedState, Cluster
from client import get_universal_caller, sample_from_model_parallel, OpenaiResponse
from llm_types import ChatHistory

logger = logging.getLogger(__name__)


class Runner(ABC):
    def __init__(
        self,
        seed_states: list[SeedState],
        policy_model: PolicyModel,
        rewrite_model: RewriteModel,
        reward_model: RewardModel,
        
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
            f"[TRAIN STEP {self.step_count}] Rating attacks..."
        )
        


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
