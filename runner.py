# %%
import os
import json
import time
import pickle
import logging
import asyncio
from pathlib import Path
from typing import Any
from collections import defaultdict
from dataclasses import replace, asdict
from abc import ABC, abstractmethod

from state import SeedState, Rollout
from utils import timestamp, logging_setup, async_gather
from models import PolicyModel, RewriteModel, JudgeModel
from reward_models import RewardModel
from load_cluster import load_initial_seed_states
from bias_workers import evaluate_baselines
from bias_evaluator import BiasEvaluator

logger = logging.getLogger(__name__)


# %%


class Runner(ABC):
    def __init__(
        self,
        seed_states: list[SeedState],
        policy_model: PolicyModel,
        bias_evaluator: BiasEvaluator,
        judge_model: JudgeModel,
        run_name: str | None,
        n_baseline_rollouts: int = 16,
        *args,
        **kwargs,
    ):
        self.step_count = 0
        self.seed_states = seed_states
        self.policy_model = policy_model
        self.bias_evaluator = bias_evaluator
        self.judge_model = judge_model
        self.n_baseline_rollouts = n_baseline_rollouts

        self.run_name = run_name or f"{timestamp()}"
        self.run_path.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def runner_type(self) -> str:
        pass

    @property
    def run_path(self) -> Path:
        return Path(f"./data/{self.runner_type}/{self.run_name}")

    @property
    def all_train_prompts(self) -> list[str]:
        all_prompts = []
        for seed_state in self.seed_states:
            all_prompts.extend(seed_state.cluster.train_prompts)
        return all_prompts

    @property
    def all_val_prompts(self) -> list[str]:
        all_prompts = []
        for seed_state in self.seed_states:
            all_prompts.extend(seed_state.cluster.val_prompts)
        return all_prompts

    def get_baselines(self):
        # get baseline rollouts and rewards
        start_time = time.time()
        self.baselines: dict[str, list[Rollout]] = asyncio.run(
            evaluate_baselines(
                user_prompts=self.all_train_prompts,
                policy_model=self.policy_model,
                reward_model=self.bias_evaluator.reward_model,
                save_dir=self.run_path / "train_baselines",
                n_rollouts=self.n_baseline_rollouts,
            )
        )
        print(f"Baseline rollouts taken: {(time.time() - start_time):.2f} seconds")
        logging.info(
            f"Baseline rollouts taken: {(time.time() - start_time):.2f} seconds"
        )

    def get_val_baselines(self):
        # get baseline rollouts and rewards
        start_time = time.time()
        self.val_baselines: dict[str, list[Rollout]] = asyncio.run(
            evaluate_baselines(
                user_prompts=self.all_val_prompts,
                policy_model=self.policy_model,
                reward_model=self.bias_evaluator.reward_model,
                save_dir=self.run_path / "val_baselines",
                n_rollouts=self.n_baseline_rollouts,
            )
        )
        print(
            f"Validation baseline rollouts taken: {(time.time() - start_time):.2f} seconds"
        )
        logging.info(
            f"Validation baseline rollouts taken: {(time.time() - start_time):.2f} seconds"
        )

    # async def _judge_attribute_helper(self) -> list[dict[str, int]]:
    #     tasks = []
    #     for seed_state in self.seed_states:
    #         attributes = list(seed_state.history[-1].keys())
    #         cluster_summary = seed_state.cluster.summary

    #         tasks.append(self.judge_model.judge_attribute(attributes, cluster_summary))

    #     return await asyncio.gather(*tasks)

    # def judge_attributes(self):
    #     """
    #     Judge all attributes in the latest history of each seed state.
    #     """
    #     results = asyncio.run(self._judge_attribute_helper())
    #     for seed_state, judge_scores in zip(self.seed_states, results):
    #         for attribute, judge_score in judge_scores.items():
    #             seed_state.history[-1][attribute].judge_score = judge_score

    def save_attribute_stats(
        self, top_k: int = 8, save_dir: Path | None = None
    ) -> dict[int, list[str]]:
        """
        Save a condensed version of previous step's attribute stats for each seed state,
        ordered by mean reward difference from baseline.

        Returns seed_state_index -> top k list of attributes
        """
        if save_dir is None:
            save_dir = self.run_path / "final_stats"
        save_dir.mkdir(parents=True, exist_ok=True)

        top_attributes = dict()

        for seed_state in self.seed_states:
            all_attributes = []
            for attribute, attribute_stats in seed_state.history[-1].items():

                all_attributes.append(
                    {
                        "attribute": attribute,
                        "judge_score": attribute_stats.judge_score,
                        "mean_reward_diff": attribute_stats.mean_reward_diff(
                            self.baselines
                        ),
                        "all_rollouts": attribute_stats.all_rollouts,
                        "meta": attribute_stats.meta,
                    }
                )

            all_attributes = sorted(
                all_attributes, key=lambda x: x["mean_reward_diff"], reverse=True
            )

            # overwrites if already exists
            seed_save_dir = save_dir / f"seed_{seed_state.index}.json"
            with open(seed_save_dir, "w") as f:
                json.dump(all_attributes, f, indent=4)

            top_attributes[seed_state.index] = [
                attr["attribute"] for attr in all_attributes[:top_k]
            ]

        return top_attributes

    def save_seed_states(self):
        logger.info(f"[TRAIN STEP {self.step_count}] Saving seed states...")
        with open(
            os.path.join(self.run_path, f"step_{self.step_count}.pkl"), "wb"
        ) as f:
            pickle.dump(self.seed_states, f)
        # remove previous files
        for f in os.listdir(self.run_path):
            if (
                f.startswith("step_")
                and f.endswith(".pkl")
                and f != f"step_{self.step_count}.pkl"
            ):
                os.remove(os.path.join(self.run_path, f))

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    async def validate(
        self, final_attributes: dict[int, list[str]], get_val_baselines: bool = True
    ):
        """
        final_attributes: seed_state_index -> list of attributes
        """
        if get_val_baselines:
            self.get_val_baselines()

        validation_results = []

        async with self.bias_evaluator as evaluator:
            for seed_state in self.seed_states:
                stats = await evaluator.evaluate_attributes(
                    user_prompts=seed_state.cluster.val_prompts,
                    attributes=final_attributes[seed_state.index],
                    save_dir=self.run_path
                    / "validate"
                    / f"seed_{seed_state.index}_validate",
                    baselines=self.val_baselines,
                )
                validation_results.append(stats)

        self.judge(validation_results=validation_results)  # type: ignore

    def judge(self, validation_results: list[dict[str, dict[str, list[Rollout]]]]):
        # use judge model
        print(f"Using judge model {self.judge_model.model_name}...")
        NUM_TRIALS = 2

        judge_tasks = []
        judge_tasks_info = []
        for seed_state_idx in range(len(self.seed_states)):
            validation_result_seed = validation_results[seed_state_idx]
            for attribute, attribute_stats in validation_result_seed.items():
                for user_prompt, rollouts in attribute_stats.items():
                    baseline_rollouts = self.val_baselines[user_prompt]
                    for rollout_idx, rollout in enumerate(rollouts[:4]):
                        judge_tasks.append(
                            self.judge_model.compare_responses(
                                user_prompt=user_prompt,
                                response_1=rollout.response,
                                response_2=baseline_rollouts[rollout_idx].response,
                                num_trials=NUM_TRIALS,
                            )
                        )
                        judge_tasks_info.append(
                            {
                                "seed_state_idx": seed_state_idx,
                                "attribute": attribute,
                                "user_prompt": user_prompt,
                                "rollout_idx": rollout_idx,
                            }
                        )
        
        logger.info(f"Running {len(judge_tasks)} judge tasks...")
        judge_tasks_results = asyncio.run(
            async_gather(
                judge_tasks, max_parallel=self.judge_model.max_par // NUM_TRIALS
            )
        )
        judge_results = {
            seed_state_idx: {
                attribute: defaultdict(list)
                for attribute in validation_results[seed_state_idx]
            }
            for seed_state_idx in range(len(self.seed_states))
        }

        for judge_task_result, judge_task_info in zip(
            judge_tasks_results, judge_tasks_info
        ):
            seed_state_idx = judge_task_info["seed_state_idx"]
            attribute = judge_task_info["attribute"]
            user_prompt = judge_task_info["user_prompt"]
            rollout_idx = judge_task_info["rollout_idx"]
            judge_results[seed_state_idx][attribute][user_prompt].append(
                judge_task_result
            )

        for seed_state_idx, seed_state in enumerate(self.seed_states):
            with open(
                self.run_path / "validate" / f"seed_{seed_state.index}_judge.json", "w"
            ) as f:
                json.dump(judge_results[seed_state_idx], f, indent=4)

        return judge_results


class TestRunner(Runner):
    @property
    def runner_type(self) -> str:
        return "test"

    def train(self):
        pass
