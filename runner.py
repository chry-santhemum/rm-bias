# %%
import patches  # monkey patching
import os
import json
import time
import pickle
import logging
import asyncio
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import replace
from typing import Literal
from abc import ABC, abstractmethod

from state import SeedState, Rollout
from utils import timestamp
from models import PolicyModel, RewriteModel, JudgeModel
from reward_model import RewardModel
from bias import (
    evaluate_baselines,
    evaluate_attributes_conditional,
    evaluate_attributes_rewrite,
    evaluate_attributes_half,
)


logger = logging.getLogger(__name__)


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

    @property
    @abstractmethod
    def runner_type(self) -> str:
        pass

    @property
    def run_path(self) -> Path:
        return Path(f"/workspace/rm-bias/data/{self.runner_type}/{self.run_name}")


    @property
    def all_train_prompts(self) -> list[str]:
        all_prompts = []
        for seed_state in self.seed_states:
            all_prompts.extend(seed_state.cluster.train_prompts)
        return all_prompts


    def get_baselines(self):
        # get baseline rollouts and rewards
        start_time = time.time()
        self.baselines: dict[str, list[Rollout]] = asyncio.run(evaluate_baselines(
            user_prompts=self.all_train_prompts,
            policy_model=self.policy_model,
            rater=self.reward_model,
            save_dir=self.run_path,
            n_rollouts=self.n_rollouts,
        ))
        print(f"Baseline rollouts taken: {time.time() - start_time} seconds")


    def evaluate_attributes(
        self,
        user_prompts: list[str],
        attributes: list[str],
        method: Literal["conditional", "rewrite", "half"],
        save_dir: Path | None = None,
    ):
        start_time = time.time()
        if method == "rewrite":
            results = asyncio.run(evaluate_attributes_rewrite(
                user_prompts=user_prompts,
                policy_model=None,
                baseline_rollouts=self.baselines,
                rater=self.reward_model,
                attributes=attributes,
                rewrite_model=self.rewrite_model,
                save_dir=save_dir,
                n_rollouts=self.n_rollouts,
                n_rewrites=1,
            ))
        elif method == "conditional":
            results = asyncio.run(evaluate_attributes_conditional(
                user_prompts=user_prompts,
                policy_model=self.policy_model,
                rater=self.reward_model,
                attributes=attributes,
                save_dir=save_dir,
                n_rollouts=self.n_rollouts,
            ))
        elif method == "half":
            results = asyncio.run(evaluate_attributes_half(
                user_prompts=user_prompts,
                policy_model=self.policy_model,
                rater=self.reward_model,
                attributes=attributes,
                rewrite_model=self.rewrite_model,
                save_dir=save_dir,
                n_rollouts=self.n_rollouts,
                n_rewrites=1,
            ))
        print(f"Attributes evaluated in {time.time() - start_time} seconds")
        return results


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


    @abstractmethod
    def train(self, *args, **kwargs):
        pass


    def load_contrast_pairs(self, threshold: float = 1.0):
        """
        For each user prompt, check in target_dir if the rollouts have enough variation,
        according to the given rater.

        Returns {"prompt": ..., "chosen": ..., "rejected": ...}
        """
        contrast_pairs = []

        # Load normalization data
        with open(f".cache/normalize/{self.reward_model.model_name}.json", "r", encoding="utf-8") as f:
            rater_stats = json.load(f)

        for seed_state in self.seed_states:
            prompts = seed_state.cluster.train_prompts

            for prompt in prompts:
                rollouts = [r for r in self.baselines[prompt] if r.score is not None]
                if len(rollouts) == 0:
                    continue

                rollouts_sorted = sorted(
                    rollouts, key=lambda x: float(x.score), reverse=True  # type: ignore
                )
                score_diff = (
                    rollouts_sorted[0].score
                    - rollouts_sorted[-1].score
                )  # type: ignore

                if score_diff > threshold * rater_stats["stdev"]:
                    contrast_pairs.append(
                        {
                            "prompt": prompt,
                            "chosen": rollouts_sorted[0].response,
                            "rejected": rollouts_sorted[-1].response,
                        }
                    )

            print(f"Found {len(contrast_pairs)}/{len(prompts)} contrast pairs for seed {seed_state.index}")

            seed_state.cluster = replace(
                seed_state.cluster,
                aux_info=contrast_pairs,
            )

