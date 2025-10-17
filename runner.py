# %%
import patches  # monkey patching
import os
import uuid
import json
import time
import pickle
import logging
import asyncio
from asyncio import Queue
import random
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import replace, asdict
from typing import Literal, Any
from abc import ABC, abstractmethod
from itertools import product
from dataclasses import dataclass
import numpy as np

from state import SeedState, Rollout
from utils import timestamp
from models import PolicyModel, RewriteModel, JudgeModel
from reward_model import RewardModel
from bias import (
    PromptResult,
    RewriteInput,
    RewriteResult,
    rewrite_plus_worker,
    rating_worker,
    evaluate_baselines,
    organize_rewrite_plus_results,
)


logger = logging.getLogger(__name__)


@dataclass
class BatchTask:
    batch_id: str
    data: Any


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

        # initialize queues and rewrite workers
        self.queue_a = asyncio.Queue()
        self.queue_b = asyncio.Queue()
        self.batch_results = {}  # Dict[batch_id, List[results]]
        self.batch_completions = {}  # Dict[batch_id, asyncio.Event]
        self.rewrite_tasks = [
            asyncio.create_task(
                rewrite_plus_worker(rewrite_model, self.queue_a, self.queue_b, worker_id)
            )
            for worker_id in range(rewrite_model.max_par)
        ]

        self.rating_worker_task = asyncio.create_task(
            rating_worker(reward_model, self.queue_b, self.rewrite_results)
        )
    
    async def shutdown(self):
        for _ in range(self.rewrite_model.max_par):
            await self.queue_a.put(None)  # Sentinel values for rewrite_workers

        await asyncio.gather(*self.rewrite_tasks)
        logger.info("\n--- rewrite workers finished. ---\n")

        await self.queue_b.put(None)
        await self.rating_worker_task
        logger.info("\n--- rating worker finished. ---\n")


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

        # Generate unique batch ID
        batch_id = str(uuid.uuid4())
        self.batch_results[batch_id] = []
        self.batch_completions[batch_id] = asyncio.Event()
        expected_result_count = 0

        for user, attribute in product(user_prompts, attributes):
            for original_assistant in baseline_rollouts[user]:
                await self.queue_a.put(
                    RewriteInput(
                        system=attribute,
                        user=user,
                        original_assistant=original_assistant.response,
                        presence=True,
                    )
                )
                expected_result_count += 1

        # Send batch task completion sentinel
        await self.queue_a.put(BatchTask(batch_id=batch_id, data=("BATCH_COMPLETE", expected_result_count)))
        
        # Get results for this batch
        batch_results = self.batch_results[batch_id]
        del self.batch_results[batch_id]
        del self.batch_completions[batch_id]

        organized_results = organize_rewrite_plus_results(batch_results, baseline_rollouts, save_dir)

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

    def save_attribute_stats(self, top_k: int = 4) -> dict[int, list[str]]:
        """
        Save a condensed version of all attribute stats for each seed state,
        ordered by adversarial score.

        Returns seed_state_index -> top k list of attributes
        """

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
                            "adversarial_score": attribute_stats.adversarial_score(
                                self.baselines
                            ),
                            "all_rewards": attribute_stats.all_rewards,
                        }
                    )

            all_attributes = sorted(
                all_attributes, key=lambda x: x["adversarial_score"], reverse=True
            )

            # overwrites if already exists
            with open(
                self.run_path / f"final_stats_seed_{seed_state.index}.json", "w"
            ) as f:
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

    async def validate(self, final_attributes: dict[int, list[str]]):
        """
        final_attributes: seed_state_index -> list of attributes
        """
        for seed_state in self.seed_states:
            await self.evaluate_attributes(
                user_prompts=seed_state.cluster.val_prompts,
                attributes=final_attributes[seed_state.index],
                save_dir=self.run_path / "validate" / f"seed_{seed_state.index}",
                baseline_rollouts=self.val_baselines,
            )

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
                f"Found {len(contrast_pairs)}/{len(prompts)} contrast pairs for seed {seed_state.index}"
            )
            logging.info(
                f"Found {len(contrast_pairs)}/{len(prompts)} contrast pairs for seed {seed_state.index}"
            )

            seed_state.cluster = replace(
                seed_state.cluster,
                aux_info=contrast_pairs,
            )

            # save cluster info
            with open(
                self.run_path / f"seed_{seed_state.index}_cluster.json", "w"
            ) as f:
                json.dump(asdict(seed_state.cluster), f, indent=4)
