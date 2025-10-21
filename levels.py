import json
import random
import asyncio
from pathlib import Path
from collections import defaultdict
import logging

import numpy as np

from load_cluster import load_initial_seed_states
from utils import timestamp, logging_setup, async_gather, ClusterModel
from state import SeedState, Rollout, AttributeStats
from models import PolicyModel, RewriteModel, JudgeModel
from reward_model import RewardModel
from runner import Runner
from one_turn import OneTurnPlanner

logger = logging.getLogger(__name__)


class LevelsRunner(Runner):
    def __init__(
        self,
        seed_states: list[SeedState],
        planner: OneTurnPlanner,
        policy_model: PolicyModel,
        rewrite_model: RewriteModel,
        reward_model: RewardModel,
        cluster_model: ClusterModel,
        judge_model: JudgeModel,
        n_new: int,
        n_pop_level_0: int,
        n_pop_level_1: int,
        train_batch_size_level_0: int,
        train_batch_size_level_1: int,
        n_rollouts: int,
        run_name: str | None = None,
    ):
        super().__init__(
            seed_states=seed_states,
            policy_model=policy_model,
            rewrite_model=rewrite_model,
            reward_model=reward_model,
            judge_model=judge_model,
            n_rollouts=n_rollouts,
            run_name=run_name,
        )
        self.planner = planner
        self.cluster_model = cluster_model
        self.n_new = n_new
        self.n_pop_level_0 = n_pop_level_0
        self.n_pop_level_1 = n_pop_level_1
        self.train_batch_size_level_0 = train_batch_size_level_0
        self.train_batch_size_level_1 = train_batch_size_level_1

    @property
    def runner_type(self) -> str:
        return "levels"

    def train(self):
        self.load_contrast_pairs()

        self.planner.plan(
            seed_states=self.seed_states,
            n_new=self.n_new,
            n_pop=self.n_pop_level_0,
            cluster_model=self.cluster_model,
        )

        level_0_evaluate_tasks = []
        seed_state_indices = []

        for seed_state_idx, seed_state in enumerate(self.seed_states):
            for plan in seed_state.history[-1].values():
                level_0_evaluate_tasks.append(
                    self.evaluate_attributes(
                        user_prompts=random.sample(
                            seed_state.cluster.train_prompts,
                            self.train_batch_size_level_0,
                        ),
                        attributes=[plan.attribute],
                    )
                )
                seed_state_indices.append(seed_state_idx)

        print(f"Level 0 evaluate tasks: {len(level_0_evaluate_tasks)}")
        level_0_evaluate_results = asyncio.run(async_gather(level_0_evaluate_tasks))

        for result, seed_state_idx in zip(level_0_evaluate_results, seed_state_indices):
            (key,) = result
            val = result[key]
            self.seed_states[seed_state_idx].history[-1][key].rollouts = val

        self.save_attribute_stats(
            save_dir=self.run_path / "level_0_stats"
        )  # save level 0 info

        # For each seed state, take the most promising ones
        level_1_evaluate_tasks = []

        def compute_mean(result: dict[str, dict[str, list[Rollout]]]) -> float:
            all_scores = []
            for _, attribute_results in result.items():
                for _, rollouts in attribute_results.items():
                    all_scores.extend([r.score for r in rollouts])
            return np.mean(all_scores).item()

        for seed_state_idx, seed_state in enumerate(self.seed_states):
            results = [{k: v.rollouts} for k, v in seed_state.history[-1].items()]
            results.sort(key=compute_mean, reverse=True)

            level_1_attributes = [next(iter(r)) for r in results[: self.n_pop_level_1]]
            user_prompts = random.sample(
                seed_state.cluster.train_prompts, self.train_batch_size_level_1
            )
            level_1_evaluate_tasks.append(
                self.evaluate_attributes(
                    user_prompts=user_prompts,
                    attributes=level_1_attributes,
                )
            )
            seed_state.history.append(
                {
                    attribute: AttributeStats(
                        attribute=attribute,
                        rollouts={},
                        meta={
                            "seed_idx": seed_state_idx,
                        },
                    )
                    for attribute in level_1_attributes
                }
            )

        print(f"Level 1 evaluate tasks: {len(level_1_evaluate_tasks)}")
        level_1_evaluate_results = asyncio.run(async_gather(level_1_evaluate_tasks))

        for level_1_result, seed_state in zip(
            level_1_evaluate_results, self.seed_states
        ):
            for attribute, rollouts in level_1_result.items():
                seed_state.history[-1][attribute].rollouts = rollouts

        final_attributes = self.save_attribute_stats(
            save_dir=self.run_path / "level_1_stats",
            top_k=8,
        )  # save level 1 info

        self.validate(final_attributes)

        asyncio.run(self.shutdown())


# %%


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_new", type=int, default=16)
    parser.add_argument("--n_rollouts", type=int, default=8)
    parser.add_argument("--n_pop_level_0", type=int, default=256)
    parser.add_argument("--n_pop_level_1", type=int, default=32)
    parser.add_argument("--train_batch_size_level_0", type=int, default=2)
    parser.add_argument("--train_batch_size_level_1", type=int, default=16)
    parser.add_argument("--val_split_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "alpaca":
        topic_ids = [0, 2, 4, 6, 9, 11, 15, 21, 34, 35, 83]
    elif args.dataset == "wildchat":
        topic_ids = [4, 5, 6, 10, 14, 16, 17, 18, 19, 24, 26, 29, 32, 36]
    elif args.dataset == "synthetic_1":
        # topic_ids = [0, 1, 3, 6, 7, 8, 9, 10, 11, 12, 14]
        topic_ids = [3, 6, 7, 12, 14]
        # topic_ids = [8, 9, 10, 11]
    elif args.dataset == "synthetic_2":
        topic_ids = list(range(8))

    initial_seed_states = load_initial_seed_states(
        ds_name=args.dataset,
        topic_ids=topic_ids,
        train_batch_size=args.train_batch_size_level_1,
        val_split_size=args.val_split_size,
    )

    run_name = f"{timestamp()}-{args.dataset}"
    Path(f"logs/levels").mkdir(parents=True, exist_ok=True)
    Path(f"data/levels").mkdir(parents=True, exist_ok=True)
    logging_setup(filename=f"logs/levels/{run_name}.log", level=logging.INFO)

    planner = OneTurnPlanner(
        model_names=["anthropic/claude-opus-4.1", "google/gemini-2.5-pro"],
        alloy_type="round_robin",
        max_tokens=8192,
        reasoning=6000,
        temperature=1.0,
        max_par=128,
    )

    cluster_model = ClusterModel(
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
    )

    runner = LevelsRunner(
        seed_states=initial_seed_states,
        planner=planner,
        policy_model=PolicyModel(
            model_name="meta-llama/llama-3.1-8b-instruct", temperature=0.9
        ),
        rewrite_model=RewriteModel(model_name="openai/gpt-5-nano", max_par=500),
        reward_model=RewardModel(model_name="skywork-v2", batch_size=32),
        judge_model=JudgeModel(max_tokens=10000, reasoning="high"),
        cluster_model=cluster_model,
        n_new=args.n_new,
        n_pop_level_0=args.n_pop_level_0,
        n_pop_level_1=args.n_pop_level_1,
        train_batch_size_level_0=args.train_batch_size_level_0,
        train_batch_size_level_1=args.train_batch_size_level_1,
        n_rollouts=args.n_rollouts,
        run_name=run_name,
    )

    # with open(
    #     "data/levels/20251020-015556-n_pop32-synthetic_1/baseline_results.json", "r"
    # ) as f:
    #     baseline_results = json.load(f)

    # runner.baselines = {}
    # for user, rollouts in baseline_results.items():
    #     runner.baselines[user] = [
    #         Rollout(response=rollout["response"], score=rollout["score"])
    #         for rollout in rollouts
    #     ]

    runner.get_baselines()

    try:
        runner.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Full traceback: ", exc_info=True)
        raise
