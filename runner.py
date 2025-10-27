"""
Runner class.
"""

# %%
import patches  # monkey patching
import os
import uuid
import json
import time
import pickle
import logging
import asyncio
import random
from pathlib import Path
from typing import Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace, asdict
from abc import ABC, abstractmethod
from itertools import product
import numpy as np

from state import SeedState, Rollout
from utils import timestamp, logging_setup, async_gather
from models import PolicyModel, RewriteModel, JudgeModel
from reward_model import RewardModel
from load_cluster import load_initial_seed_states

from bias_baseline import evaluate_baselines
from bias_rewrite import (
    BatchSentinel,
    RewriteInput,
    RewriteResult,
    rewrite_worker,
    rewrite_rating_worker,
    organize_rewrite_results,
)

logger = logging.getLogger(__name__)


# %%


class Runner(ABC):
    def __init__(
        self,
        seed_states: list[SeedState],
        policy_model: PolicyModel,
        rewrite_model: RewriteModel,
        reward_model: RewardModel,
        judge_model: JudgeModel,
        run_name: str | None,
        n_rollouts: int = 16,
        *args,
        **kwargs,
    ):
        self.step_count = 0
        self.seed_states = seed_states
        self.policy_model = policy_model
        self.rewrite_model = rewrite_model
        self.reward_model = reward_model
        self.judge_model = judge_model
        self.n_rollouts = n_rollouts

        self.run_name = run_name or f"{timestamp()}"
        self.run_path.mkdir(parents=True, exist_ok=True)

        # defer queues and worker startup until inside a running event loop
        self.queue_a: asyncio.Queue | None = None
        self.queue_b: asyncio.Queue | None = None
        self.batch_results: dict[str, list[RewriteResult]] = {}
        self.batch_futures: dict[str, asyncio.Future] = {}
        self.rewrite_workers: list[asyncio.Task] = []
        self.rating_worker: asyncio.Task | None = None
        self._workers_started = False
        self._rewrite_executor: ThreadPoolExecutor | None = None

    async def _ensure_workers_started(self):
        if self._workers_started:
            return

        # must be called from within a running event loop
        self.queue_a = asyncio.Queue()
        self.queue_b = asyncio.Queue()

        self.rewrite_workers = [
            asyncio.create_task(
                rewrite_worker(
                    self.rewrite_model, self.queue_a, self.queue_b, worker_id
                )
            )
            for worker_id in range(self.rewrite_model.max_par)
        ]

        # executor for rating is instance-scoped to this runner
        self._rewrite_executor = ThreadPoolExecutor(max_workers=1)
        self.rating_worker = asyncio.create_task(
            rewrite_rating_worker(
                self.reward_model,
                self.queue_b,
                self.batch_results,
                self.batch_futures,
                self._rewrite_executor,
            )
        )
        self._workers_started = True

    async def shutdown(self):
        # If workers were never started, just close model callers
        if not self._workers_started:
            await self.policy_model.caller.close()
            await self.rewrite_model.caller.close()
            await self.judge_model.caller.close()
            return

        assert self.queue_a is not None and self.queue_b is not None
        for _ in range(self.rewrite_model.max_par):
            await self.queue_a.put(None)  # Sentinel values for rewrite_workers

        await asyncio.gather(*self.rewrite_workers)
        logger.info("\n--- rewrite workers finished. ---\n")

        await self.queue_b.put(None)
        if self.rating_worker is not None:
            await self.rating_worker
        logger.info("\n--- rewrite rating worker finished. ---\n")

        # Close shared LLM callers
        await self.policy_model.caller.close()
        await self.rewrite_model.caller.close()
        await self.judge_model.caller.close()
        # Close planner caller if present (e.g., OneTurnPlanner/PlannerModel)
        try:
            planner = getattr(self, "planner", None)
            if planner is not None and hasattr(planner, "caller"):
                await planner.caller.close()
        except Exception:
            logger.warning("Failed to close planner caller", exc_info=True)
        self._workers_started = False
        # shutdown instance-level executor
        if self._rewrite_executor is not None:
            self._rewrite_executor.shutdown(wait=True)
            self._rewrite_executor = None

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
                reward_model=self.reward_model,
                save_dir=self.run_path / "train_baselines",
                n_rollouts=self.n_rollouts,
            )
        )
        print(f"Baseline rollouts taken: {(time.time() - start_time):.2f} seconds")
        logging.info(f"Baseline rollouts taken: {(time.time() - start_time):.2f} seconds")

    def get_val_baselines(self):
        # get baseline rollouts and rewards
        start_time = time.time()
        self.val_baselines: dict[str, list[Rollout]] = asyncio.run(
            evaluate_baselines(
                user_prompts=self.all_val_prompts,
                policy_model=self.policy_model,
                reward_model=self.reward_model,
                save_dir=self.run_path / "val_baselines",
                n_rollouts=self.n_rollouts,
            )
        )
        print(f"Validation baseline rollouts taken: {(time.time() - start_time):.2f} seconds")
        logging.info(
            f"Validation baseline rollouts taken: {(time.time() - start_time):.2f} seconds"
        )

    async def evaluate_attributes(
        self,
        user_prompts: list[str],
        attributes: list[str],
        save_dir: Path | None = None,
        baseline_rollouts: dict[str, list[Rollout]] | None = None,
    ):
        start_time = time.time()
        if baseline_rollouts is None:
            baseline_rollouts = self.baselines

        await self._ensure_workers_started()

        # Generate batch ID and asyncio.Future
        batch_id = str(uuid.uuid4())
        self.batch_results[batch_id] = []
        loop = asyncio.get_running_loop()
        self.batch_futures[batch_id] = loop.create_future()
        expected_result_count = 0

        for user, attribute in product(user_prompts, attributes):
            for original_assistant in baseline_rollouts[user]:
                assert self.queue_a is not None
                await self.queue_a.put(
                    RewriteInput(
                        system=attribute,
                        user=user,
                        original_assistant=original_assistant.response,
                        presence=True,
                        batch_id=batch_id,
                    )
                )
                expected_result_count += 1

        # Send batch task completion sentinel
        logger.info(f"Batch {batch_id} expects {expected_result_count} results...")
        assert self.queue_a is not None
        await self.queue_a.put(
            BatchSentinel(batch_id=batch_id, expected_items=expected_result_count)
        )

        # Wait for results for this batch
        batch_results = await self.batch_futures[batch_id]
        logger.info(f"Expected {expected_result_count} results, got {len(batch_results)}")
        organized_results = organize_rewrite_results(
            batch_results, baseline_rollouts, save_dir
        )
        del self.batch_futures[batch_id]

        logger.info(f"Attributes evaluated in {(time.time() - start_time):.2f} seconds")
        return organized_results

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
        Save a condensed version of **all** attribute stats for each seed state,
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

    def validate(self, final_attributes: dict[int, list[str]], get_val_baselines: bool = True):
        """
        final_attributes: seed_state_index -> list of attributes
        """
        if get_val_baselines:
            self.get_val_baselines()

        tasks = [
            self.evaluate_attributes(
                user_prompts=seed_state.cluster.val_prompts,
                attributes=final_attributes[seed_state.index],
                save_dir=self.run_path
                / "validate"
                / f"seed_{seed_state.index}_validate",
                baseline_rollouts=self.val_baselines,
            )
            for seed_state in self.seed_states
        ]

        validation_results = asyncio.run(async_gather(tasks))

        self.judge(validation_results=validation_results)


    def judge(self, validation_results: list[dict[str, dict[str, list[Rollout]]]]):
        # use judge model
        NUM_TRIALS = 2
        judge_tasks = []
        judge_tasks_info = []
        for seed_state_idx in range(len(self.seed_states)):
            validation_result_seed = validation_results[seed_state_idx]
            for attribute, attribute_stats in validation_result_seed.items():
                for user_prompt, rollouts in attribute_stats.items():
                    baseline_rollouts = self.val_baselines[user_prompt]
                    for rollout_idx, rollout in enumerate(rollouts):
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


    def load_contrast_pairs(self, threshold: float = 1.0):
        """
        For each user prompt, check in target_dir if the rollouts have enough variation,
        according to the given rater.

        Returns {"prompt": ..., "chosen": ..., "rejected": ...}
        """
        for seed_state in self.seed_states:
            contrast_pairs = []
            prompts = [p for p in seed_state.cluster.train_prompts if p in self.baselines]

            for prompt in prompts:
                rollouts = [r for r in self.baselines[prompt] if r.score is not None]
                if len(rollouts) == 0:
                    continue

                scores = np.array([float(r.score) for r in rollouts])  # type: ignore
                mean_score, stdev_score = np.mean(scores), np.std(scores)
                if stdev_score == 0:
                    continue  # No variability

                # find those above / below threshold * stdev
                high_rollouts = [r for r in rollouts if float(r.score) > mean_score + threshold * stdev_score]  # type: ignore
                low_rollouts = [r for r in rollouts if float(r.score) < mean_score - threshold * stdev_score]  # type: ignore
                print(
                    f"High rollouts: {len(high_rollouts)}, Low rollouts: {len(low_rollouts)}"
                )

                if len(high_rollouts) == 0 or len(low_rollouts) == 0:
                    continue

                for high in high_rollouts:
                    rejected_rollout = random.choice(low_rollouts)
                    contrast_pairs.append(
                        {
                            "prompt": prompt,
                            "chosen": high.response,
                            "rejected": rejected_rollout.response,
                        }
                    )

            print(
                f"Found {len(contrast_pairs)} contrast pairs in total for seed {seed_state.index}"
            )
            logging.info(
                f"Found {len(contrast_pairs)} contrast pairs in total for seed {seed_state.index}"
            )

            # contrast_pairs = random.sample(contrast_pairs, min(len(contrast_pairs), 4))  # DEBUG

            seed_state.cluster = replace(
                seed_state.cluster,
                aux_info=contrast_pairs,
            )

            # save cluster info
            with open(
                self.run_path / f"seed_{seed_state.index}_cluster.json", "w"
            ) as f:
                json.dump(asdict(seed_state.cluster), f, indent=4)


class TestRunner(Runner):
    @property
    def runner_type(self) -> str:
        return "evo"

    def train(self):
        pass


async def main():
    # ATTRIBUTES = [
    #     "Provide multiple different approaches to the same question.",
    #     "Use bold text in the response.",
    #     "Use italic text in the response.",
    #     "Write a response that is longer than the baseline.",
    #     "Use emojis in the response.",
    #     "Use bulleted or numbered lists in the response.",
    #     "Start with an affirmative sentence that responds to the user's request.",
    #     "Adopt a friendly and engaging, natural tone.",
    #     "End the response with a question to the user.",
    #     "Mention advanced mathematical jargon in the response.",
    # ]

    initial_seed_states = load_initial_seed_states(
        ds_name="synthetic_2",
        topic_ids=[4, 6, 8, 12, 14, 16],
        val_split_size=16,
    )

    # with open(
    #     "data/one_turn/20251015-232457-n_pop32-synthetic_1/baseline_results.json", "r"
    # ) as f:
    #     baseline_results = json.load(f)
    # user_prompts = list(baseline_results.keys())

    run_name = "20251027-045705-synthetic_2"
    logging_setup(
        filename=f"logs/scrap/test_runner_{run_name}.log", level=logging.INFO
    )

    runner = TestRunner(
        seed_states=initial_seed_states,
        policy_model=PolicyModel(temperature=0.9),
        rewrite_model=RewriteModel(model_name="openai/gpt-5-nano", max_par=500),
        reward_model=RewardModel(model_name="skywork-v2", batch_size=32),
        judge_model=JudgeModel(model_name="anthropic/claude-sonnet-4.5", max_tokens=2048, reasoning=2000),
        run_name=run_name,
        n_rollouts=8,
    )

    with open(
        f"data/evo/{run_name}/val_baselines/baseline_results.json", "r"
    ) as f:
        val_baselines = json.load(f)

    runner.val_baselines = {}
    for user, rollouts in val_baselines.items():
        runner.val_baselines[user] = [
            Rollout(response=rollout["response"], score=rollout["score"])
            for rollout in rollouts
        ]
    
    print("Loaded validation baselines.")

    validation_results = []
    for seed_state_idx in [4, 6, 8, 12, 14, 16]:
        with open(
            f"data/evo/{run_name}/validate/seed_{seed_state_idx}_validate/rewrite_plus_results.json", "r"
        ) as f:
            seed_validation_results = json.load(f)
            validation_results.append(json_to_rollouts(seed_validation_results))        

    print("Loaded validation results.")

    try:
        runner.judge(validation_results=validation_results)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Cancellation received. Cleaning up...")
        raise
    finally:
        await asyncio.shield(runner.shutdown())



def json_to_rollouts(json_data: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, list[Rollout]]]:
    rollouts = {}
    for attribute, attribute_data in json_data.items():
        rollouts[attribute] = {}
        for user, user_data in attribute_data.items():
            rollouts[attribute][user] = [
                Rollout(response=rollout["response"], score=rollout["score"])
                for rollout in user_data
            ]
    return rollouts   

# %%
if __name__ == "__main__":
    asyncio.run(main())
