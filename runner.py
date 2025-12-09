import os
import json
import time
import pickle
from loguru import logger
from pathlib import Path
from typing import Literal
from abc import ABC, abstractmethod

from state import SeedState, Rollout
from utils import timestamp
from api_models import GenerationModel
from reward_models import RewardModel
from bias_workers import evaluate_baselines
from bias_evaluator import BiasEvaluator

class Runner(ABC):
    def __init__(
        self,
        seed_states: list[SeedState],
        policy_model: GenerationModel,
        bias_evaluator: BiasEvaluator,
        teacher_model: RewardModel,
        run_name: str | None,
        n_baseline_rollouts: int = 16,
        *args,
        **kwargs,
    ):
        self.step_count = 0
        self.seed_states = seed_states
        self.policy_model = policy_model
        self.bias_evaluator = bias_evaluator
        self.teacher_model = teacher_model
        self.n_baseline_rollouts = n_baseline_rollouts

        self.run_name = run_name or f"{timestamp()}"
        self.run_path.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def runner_type(self) -> str:
        pass

    @property
    def run_path(self) -> Path:
        return Path(f"data/{self.runner_type}/{self.run_name}")

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

    async def get_baselines(self):
        # get baseline rollouts and rewards
        start_time = time.time()
        self.baselines: dict[str, list[Rollout]] = await evaluate_baselines(
            user_prompts=self.all_train_prompts,
            policy_model=self.policy_model,
            reward_model=self.bias_evaluator.reward_model,
            save_dir=self.run_path / "train_baselines",
            n_rollouts=self.n_baseline_rollouts,
        )
        print(f"Baseline rollouts taken: {(time.time() - start_time):.2f} seconds")
        logger.info(
            f"Baseline rollouts taken: {(time.time() - start_time):.2f} seconds"
        )


    async def get_val_baselines(self):
        start_time = time.time()
        self.val_baselines: dict[str, list[Rollout]] = await evaluate_baselines(
            user_prompts=self.all_val_prompts,
            policy_model=self.policy_model,
            reward_model=self.bias_evaluator.reward_model,
            save_dir=self.run_path / "val_baselines",
            n_rollouts=self.n_baseline_rollouts,
        )
        duration = time.time() - start_time
        print(f"Validation baseline rollouts taken: {duration:.2f} seconds")
        logger.info(f"Validation baseline rollouts taken: {duration:.2f} seconds")

    
    def get_references(self, seed_state_idx: int, attribute: str) -> dict|None:
        # get reference pair if exists (PairPlanner)
        # if not, write None
        if attribute not in self.seed_states[seed_state_idx].history[-1]:
            raise KeyError("Given attribute is not found in the last timestep.")

        att_meta = self.seed_states[seed_state_idx].history[-1][attribute].meta
        if "response_A" in att_meta:
            return {
                "user_prompt": att_meta["user_prompt"],
                "response_A": att_meta["response_A"],
                "response_B": att_meta["response_B"],
            }
        else:
            return None

    def save_attribute_stats(
        self, direction: Literal["plus", "minus"], save_dir: Path | None = None
    ):
        """
        Save a condensed version of previous step's attribute stats for each seed state,
        ordered by mean reward difference from baseline.

        Returns seed_state_index -> top k list of attributes
        """
        if save_dir is None:
            save_dir = self.run_path / "final_stats"
        save_dir.mkdir(parents=True, exist_ok=True)

        for seed_state in self.seed_states:
            all_attributes = []
            for attribute, attribute_stats in seed_state.history[-1].items():

                all_attributes.append(
                    {
                        "attribute": attribute,
                        "mean_reward_diff": attribute_stats.mean_reward_diff(
                            self.baselines  # type: ignore
                        ),
                        "all_rollouts": attribute_stats.all_rollouts,
                        "meta": attribute_stats.meta,
                    }
                )

            if direction == "plus":
                all_attributes = sorted(all_attributes, key=lambda x: x["mean_reward_diff"], reverse=True)
            else:
                all_attributes = sorted(all_attributes, key=lambda x: x["mean_reward_diff"], reverse=False)

            # overwrites if already exists
            seed_save_dir = save_dir / f"seed_{seed_state.index}.json"
            with open(seed_save_dir, "w") as f:
                json.dump(all_attributes, f, indent=4)


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
        if not hasattr(self, "val_baselines"):
            await self.get_val_baselines()

        validation_results = []

        for seed_state in self.seed_states:
            async with self.bias_evaluator as evaluator:
                stats = await evaluator.evaluate_attributes(
                    user_prompts=seed_state.cluster.val_prompts,
                    attributes=final_attributes[seed_state.index],
                    save_dir=self.run_path
                    / "validate"
                    / f"seed_{seed_state.index}_validate",
                    baselines=self.val_baselines,
                )
            validation_results.append(stats)
        
        judge_results = await self.teacher_model.judge_validation_results(
            validation_results=validation_results,
            val_baselines=self.val_baselines,  # type: ignore
        )

        for i, seed_state in enumerate(self.seed_states):
            with open(
                self.run_path / "validate" / f"seed_{seed_state.index}_judge.json", "w"
            ) as f:
                json.dump(judge_results[i], f, indent=4)


class TestRunner(Runner):
    @property
    def runner_type(self) -> str:
        return "test"

    def train(self):
        pass
