"""
Cost estimate:

[per seed state]
Rewrites: train_batch_size * n_pop * n_rollouts (~4096 tokens per call)
"""

# %%

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
from state import SeedState, AttributeStats, Cluster, Rollout
from utils import (
    timestamp,
    parse_json_response,
    ClusterModel,
    async_gather,
    set_seed_all,
    logging_setup,
)
from load_cluster import load_initial_seed_states
from models import PolicyModel, RewriteModel, JudgeModel
from planner import Planner, ListPlanner, PairPlanner
from reward_models import RewardModel
from runner import Runner
from bias_evaluator import BiasEvaluator

dotenv.load_dotenv()
nest_asyncio.apply()
set_seed_all(10086)

logger = logging.getLogger(__name__)


class OneTurnRunner(Runner):
    def __init__(
        self,
        seed_states: list[SeedState],
        planner: ListPlanner,
        policy_model: PolicyModel,
        bias_evaluator: BiasEvaluator,
        judge_model: JudgeModel,
        cluster_model: ClusterModel,
        n_new: int,
        n_pop: int,
        n_traj_in_context: int,
        n_per_user_prompt: int,
        train_batch_size: int,
        n_baseline_rollouts: int,
        n_rewrite_rollouts: int,
        run_name: str | None = None,
    ):
        super().__init__(
            seed_states=seed_states,
            policy_model=policy_model,
            bias_evaluator=bias_evaluator,
            judge_model=judge_model,
            run_name=run_name,
            n_baseline_rollouts=n_baseline_rollouts,
        )
        self.planner = planner
        self.cluster_model = cluster_model

        self.n_new = n_new
        self.n_pop = n_pop

        self.n_traj_in_context = n_traj_in_context
        self.n_per_user_prompt = n_per_user_prompt
        self.n_rewrite_rollouts = n_rewrite_rollouts
        self.train_batch_size = train_batch_size

    @property
    def runner_type(self) -> str:
        return "one_turn"

    def train(self):
        self.planner.plan(
            runner=self,
            n_new=self.n_new,
            n_pop=self.n_pop,
            n_traj_in_context=self.n_traj_in_context,
            n_per_user_prompt=self.n_per_user_prompt,
            cluster_model=self.cluster_model,
            max_num_train_prompts=4,  # DEBUG
        )

        evaluate_tasks = []
        seed_state_indices = []

        for seed_state_idx, seed_state in enumerate(self.seed_states):
            user_prompts = random.sample(
                seed_state.cluster.train_prompts, self.train_batch_size
            )

            for attribute in seed_state.history[-1]:
                seed_state_indices.append(seed_state_idx)
                evaluate_tasks.append(
                    self.bias_evaluator.evaluate_attributes(
                        user_prompts=user_prompts,
                        attributes=[attribute],
                        baseline_rollouts=self.baselines,
                    )
                )

        evaluate_results = asyncio.run(async_gather(evaluate_tasks))

        for result, seed_state_idx in zip(evaluate_results, seed_state_indices):
            (key,) = result
            val = result[key]
            self.seed_states[seed_state_idx].history[-1][key].rollouts = val

        final_attributes = self.save_attribute_stats(
            top_k=8, save_dir=self.run_path / f"step_{self.step_count}_stats"
        )

        asyncio.run(self.validate(final_attributes=final_attributes))


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_new", type=int, default=16)
    parser.add_argument("--n_pop", type=int, default=32)
    parser.add_argument("--n_traj_in_context", type=int, default=16)
    parser.add_argument("--n_per_user_prompt", type=int, default=1)
    parser.add_argument("--n_baseline_rollouts", type=int, default=32)
    parser.add_argument("--n_rewrite_rollouts", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_split_size", type=int, default=16)
    args = parser.parse_args()

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
    elif args.dataset == "synthetic_2":
        # topic_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # topic_ids = [1, 3, 4, 6, 8, 9, 12, 14, 16]
        topic_ids = [1]

    initial_seed_states = load_initial_seed_states(
        ds_name=args.dataset,
        topic_ids=topic_ids,
        train_batch_size=args.train_batch_size,
        val_split_size=args.val_split_size,
    )

    run_name = f"{timestamp()}-naive-{args.dataset}"
    Path(f"logs/one_turn").mkdir(parents=True, exist_ok=True)
    Path(f"data/one_turn").mkdir(parents=True, exist_ok=True)
    logging_setup(filename=f"logs/one_turn/{run_name}.log", level=logging.INFO)

    planner = ListPlanner(
        model_names=["anthropic/claude-sonnet-4.5"],
        alloy_type="round_robin",
        max_tokens=8192,
        reasoning=6000,
        max_par=128,
    )

    cluster_model = ClusterModel(
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
    )

    bias_evaluator = BiasEvaluator(
        rewrite_model=RewriteModel(model_name="openai/gpt-5-nano", max_par=512),
        reward_model=RewardModel(model_name="skywork-v2", batch_size=32),
    )

    runner = OneTurnRunner(
        seed_states=initial_seed_states,
        planner=planner,
        policy_model=PolicyModel(
            model_name="meta-llama/llama-3.1-8b-instruct", temperature=0.95
        ),
        bias_evaluator=bias_evaluator,
        cluster_model=cluster_model,
        judge_model=JudgeModel(
            model_name="anthropic/claude-haiku-4.5", max_tokens=2048, reasoning=2000
        ),
        n_new=args.n_new,
        n_pop=args.n_pop,
        n_traj_in_context=args.n_traj_in_context,
        n_per_user_prompt=args.n_per_user_prompt,
        n_baseline_rollouts=args.n_baseline_rollouts,
        n_rewrite_rollouts=args.n_rewrite_rollouts,
        train_batch_size=args.train_batch_size,
        run_name=run_name,
    )

    # runner.get_baselines()

    with open(
        f"data/one_turn/20251107-075750-naive-synthetic_2/train_baselines/baseline_results.json",
        "r",
    ) as f:
        train_baselines = json.load(f)

    runner.baselines = {}
    for user, rollouts in train_baselines.items():
        runner.baselines[user] = [
            Rollout(response=rollout["response"], score=rollout["score"])
            for rollout in rollouts
        ]

    try:
        runner.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Full traceback: ", exc_info=True)
        raise
