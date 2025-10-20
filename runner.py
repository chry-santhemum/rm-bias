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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace, asdict
from abc import ABC, abstractmethod
from itertools import product
import numpy as np

from state import SeedState, Rollout
from utils import timestamp, logging_setup
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
                save_dir=self.run_path,
                n_rollouts=self.n_rollouts,
            )
        )
        print(f"Baseline rollouts taken: {time.time() - start_time} seconds")
        logging.info(f"Baseline rollouts taken: {time.time() - start_time} seconds")

    def get_val_baselines(self):
        # get baseline rollouts and rewards
        start_time = time.time()
        self.val_baselines: dict[str, list[Rollout]] = asyncio.run(
            evaluate_baselines(
                user_prompts=self.all_val_prompts,
                policy_model=self.policy_model,
                reward_model=self.reward_model,
                save_dir=self.run_path,
                n_rollouts=self.n_rollouts,
            )
        )
        print(f"Validation baseline rollouts taken: {time.time() - start_time} seconds")
        logging.info(
            f"Validation baseline rollouts taken: {time.time() - start_time} seconds"
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
        assert self.queue_a is not None
        await self.queue_a.put(
            BatchSentinel(batch_id=batch_id, expected_items=expected_result_count)
        )

        # Wait for results for this batch
        batch_results = await self.batch_futures[batch_id]
        assert (
            len(batch_results) == expected_result_count
        ), f"Expected {expected_result_count} results, got {len(batch_results)}"
        organized_results = organize_rewrite_results(
            batch_results, baseline_rollouts, save_dir
        )
        del self.batch_futures[batch_id]

        logger.info(f"Attributes evaluated in {time.time() - start_time} seconds")
        return organized_results

    async def _judge_attribute_helper(self) -> list[dict[str, int]]:
        tasks = []
        for seed_state in self.seed_states:
            attributes = list(seed_state.history[-1].keys())
            cluster_summary = seed_state.cluster.summary

            tasks.append(self.judge_model.judge_attribute(attributes, cluster_summary))

        return await asyncio.gather(*tasks)

    def judge_attributes(self):
        """
        Judge all attributes in the latest history of each seed state.
        """
        results = asyncio.run(self._judge_attribute_helper())
        for seed_state, judge_scores in zip(self.seed_states, results):
            for attribute, judge_score in judge_scores.items():
                seed_state.history[-1][attribute].judge_score = judge_score

    def save_attribute_stats(
        self, top_k: int = 4, save_dir: Path | None = None
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
            for t, time_step in enumerate(seed_state.history):
                for attribute, attribute_stats in time_step.items():

                    all_attributes.append(
                        {
                            "attribute": attribute,
                            "time_step": t,
                            "judge_score": attribute_stats.judge_score,
                            "mean_reward_diff": attribute_stats.mean_reward_diff(
                                self.baselines
                            ),
                            "all_rewards": attribute_stats.all_rewards,
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

    def validate(self, final_attributes: dict[int, list[str]]):
        """
        final_attributes: seed_state_index -> list of attributes
        """

        async def validate_helper():
            tasks = [
                self.evaluate_attributes(
                    user_prompts=seed_state.cluster.val_prompts,
                    attributes=final_attributes[seed_state.index],
                    save_dir=self.run_path / "validate" / f"seed_{seed_state.index}",
                    baseline_rollouts=self.val_baselines,
                )
                for seed_state in self.seed_states
            ]
            await asyncio.gather(*tasks)

        asyncio.run(validate_helper())

    def load_contrast_pairs(self, threshold: float = 1.0):
        """
        For each user prompt, check in target_dir if the rollouts have enough variation,
        according to the given rater.

        Returns {"prompt": ..., "chosen": ..., "rejected": ...}
        """
        for seed_state in self.seed_states:
            contrast_pairs = []
            prompts = seed_state.cluster.train_prompts

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

            contrast_pairs = random.sample(contrast_pairs, min(len(contrast_pairs), 64))

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
        return "test"

    def train(self):
        pass


async def main():
    ATTRIBUTES = [
        "Provide multiple different approaches to the same question.",
        "Use bold text in the response.",
        "Use italic text in the response.",
        "Write a response that is longer than the baseline.",
        "Use emojis in the response.",
        "Use bulleted or numbered lists in the response.",
        "Start with an affirmative sentence that responds to the user's request.",
        "Adopt a friendly and engaging, natural tone.",
        "End the response with a question to the user.",
        "Mention advanced mathematical jargon in the response.",
    ]

    initial_seed_states = load_initial_seed_states(
        ds_name="synthetic_1",
        topic_ids=[8],
        train_batch_size=16,
        val_split_size=16,
    )

    with open(
        "data/one_turn/20251015-232457-n_pop32-synthetic_1/baseline_results.json", "r"
    ) as f:
        baseline_results = json.load(f)
    user_prompts = list(baseline_results.keys())

    runner = TestRunner(
        seed_states=initial_seed_states,
        policy_model=PolicyModel(),
        rewrite_model=RewriteModel(model_name="openai/gpt-5-nano", max_par=512),
        reward_model=RewardModel(model_name="skywork-v2", batch_size=32),
        judge_model=JudgeModel(),
        run_name=None,
        n_rollouts=16,
    )

    runner.baselines = {}
    for user, rollouts in baseline_results.items():
        runner.baselines[user] = [
            Rollout(response=rollout["response"], score=rollout["score"])
            for rollout in rollouts
        ]

    try:
        tasks = [
            runner.evaluate_attributes(
                user_prompts=user_prompts[:5], attributes=ATTRIBUTES[:2]
            ),
            runner.evaluate_attributes(
                user_prompts=user_prompts[:5], attributes=ATTRIBUTES[2:4]
            ),
            runner.evaluate_attributes(
                user_prompts=user_prompts[:5], attributes=ATTRIBUTES[4:6]
            ),
            runner.evaluate_attributes(
                user_prompts=user_prompts[:5], attributes=ATTRIBUTES[6:8]
            ),
        ]
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Cancellation received. Cleaning up...")
        raise
    finally:
        await asyncio.shield(runner.shutdown())


# %%
if __name__ == "__main__":
    logging_setup(
        filename=f"logs/scrap/test_runner_{timestamp()}.log", level=logging.INFO
    )

    asyncio.run(main())
