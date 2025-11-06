"""
Cost estimate:

[per seed state]
Rewrites: train_batch_size * n_pop * n_rollouts (~4096 tokens per call)
"""

# %%
import patches
import copy
import json
import dotenv
import random
import textwrap
import logging
import asyncio
import nest_asyncio
from pathlib import Path
from typing import Literal, Optional
from collections import defaultdict

from caller import ChatHistory
from state import SeedState, AttributeStats, Cluster
from utils import (
    timestamp,
    parse_json_response,
    ClusterModel,
    set_seed_all,
    logging_setup,
)
from load_cluster import load_initial_seed_states
from models import PolicyModel, RewriteModel, JudgeModel
from planner import Planner, NaivePlanner, ContrastPlanner
from reward_model import RewardModel
from runner import Runner

dotenv.load_dotenv()
nest_asyncio.apply()
set_seed_all(10086)

logger = logging.getLogger(__name__)




class OneTurnRunner(Runner):
    def __init__(
        self,
        seed_states: list[SeedState],
        planner: Planner,
        policy_model: PolicyModel,
        rewrite_model: RewriteModel,
        reward_model: RewardModel,
        judge_model: JudgeModel,
        cluster_model: ClusterModel,
        n_new: int,
        n_pop: int,
        train_batch_size: int,
        n_rollouts: int,
        run_name: str | None = None,
    ):
        super().__init__(
            seed_states=seed_states,
            policy_model=policy_model,
            rewrite_model=rewrite_model,
            reward_model=reward_model,
            judge_model=judge_model,
            run_name=run_name,
            n_rollouts=n_rollouts,
        )
        self.cluster_model = cluster_model
        self.planner = planner
        self.n_new = n_new
        self.n_pop = n_pop
        self.train_batch_size = train_batch_size

    @property
    def runner_type(self) -> str:
        return "one_turn"

    def train(self):
        self.planner.load_contrast_pairs(runner=self)

        self.planner.plan(
            runner=self,
            n_new=self.n_new,
            n_pop=self.n_pop,
            cluster_model=self.cluster_model,
        )

        for seed_state in self.seed_states:
            sample_user_prompts = random.sample(
                seed_state.cluster.train_prompts, self.train_batch_size
            )
            rewrite_results = self.evaluate_attributes(
                user_prompts=sample_user_prompts,
                attributes=list(seed_state.history[-1].keys()),
                save_dir=self.run_path
                / f"step_{self.step_count}_seed_{seed_state.index}",
            )
            for attribute, rollouts in rewrite_results.items():
                if attribute == "":
                    continue
                seed_state.history[-1][attribute].rollouts = rollouts  # type: ignore

        self.judge_attributes()
        top_attributes = self.save_attribute_stats(top_k=8)

        self.get_val_baselines()
        self.validate(final_attributes=top_attributes)


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_new", type=int, default=5)
    parser.add_argument("--n_pop", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_split_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    cluster_model = ClusterModel(
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        umap_n_neighbors=5,
        umap_n_components=5,
    )

    if args.dataset == "alpaca":
        topic_ids = [0, 2, 4, 6, 9, 11, 15, 21, 34, 35, 83]
    elif args.dataset == "wildchat":
        topic_ids = [4, 5, 6, 10, 14, 16, 17, 18, 19, 24, 26, 29, 32, 36]
    elif args.dataset == "synthetic":
        topic_ids = [0]
    elif args.dataset == "synthetic_1":
        # topic_ids = [0, 1, 3, 6, 7, 8, 9, 10, 11, 12, 14]
        # topic_ids = [3, 6, 7, 8, 9, 10, 11, 14]
        topic_ids = [8, 9, 10, 11]

    initial_seed_states = load_initial_seed_states(
        ds_name=args.dataset,
        topic_ids=topic_ids,
        train_batch_size=args.train_batch_size,
        val_split_size=args.val_split_size,
    )

    run_name = f"{timestamp()}-n_pop{args.n_pop}-{args.dataset}"
    Path(f"logs/one_turn").mkdir(parents=True, exist_ok=True)
    Path(f"data/one_turn").mkdir(parents=True, exist_ok=True)
    logging_setup(filename=f"logs/one_turn/{run_name}.log", level=logging.INFO)

    planner = NaivePlanner(
        model_names=["anthropic/claude-opus-4.1", "google/gemini-2.5-pro"],
        alloy_type="round_robin",
        max_tokens=8192,
        reasoning=6000,
        temperature=1.0,
        max_par=128,
    )

    runner = OneTurnRunner(
        seed_states=initial_seed_states,
        planner=planner,
        policy_model=PolicyModel(
            model_name="meta-llama/llama-3.1-8b-instruct", temperature=0.9
        ),
        rewrite_model=RewriteModel(model_name="openai/gpt-5-nano", max_par=512),
        reward_model=RewardModel(model_name="skywork-v2", batch_size=64),
        judge_model=JudgeModel(),
        cluster_model=cluster_model,
        n_new=args.n_new,
        n_pop=args.n_pop,
        n_rollouts=16,
        train_batch_size=args.train_batch_size,
        run_name=run_name,
    )

    runner.get_baselines()

    try:
        runner.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Full traceback: ", exc_info=True)
        raise
