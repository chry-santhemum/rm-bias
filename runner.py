import os
import json
import time
import pickle
from loguru import logger
from pathlib import Path
from typing import Literal
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

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
        n_baseline_rollouts: int,
        n_validate_rollouts: int = 8,
        *args,
        **kwargs,
    ):
        self.step_count = 0
        self.seed_states = seed_states
        self.policy_model = policy_model
        self.bias_evaluator = bias_evaluator
        self.teacher_model = teacher_model
        self.n_baseline_rollouts = n_baseline_rollouts
        self.n_validate_rollouts = n_validate_rollouts
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
            n_rollouts=self.n_validate_rollouts,
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
        ordered by student winrate (reward diff).

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
                        "student_winrate": attribute_stats.winrate("student"),
                        "teacher_winrate": attribute_stats.winrate("teacher"),
                        "all_rollouts": attribute_stats.to_dict(),
                        "meta": attribute_stats.meta,
                    }
                )

            if direction == "plus":
                all_attributes = sorted(
                    all_attributes,
                    key=lambda x: x["student_winrate"] if x["student_winrate"] is not None else float("-inf"),
                    reverse=True
                )
            else:
                all_attributes = sorted(
                    all_attributes,
                    key=lambda x: x["student_winrate"] if x["student_winrate"] is not None else float("inf"),
                    reverse=False
                )

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

        # Populate teacher_score on rollouts in place
        await self.teacher_model.judge_rollouts(
            evaluate_results=validation_results,
            baselines=self.val_baselines,  # type: ignore
            first_n_rollouts=4,  # increased
            first_n_user_prompts=16,  # increased
        )

        # Save validation results with teacher scores
        for i, seed_state in enumerate(self.seed_states):
            # Extract teacher winrates from the rollouts
            teacher_results = {}
            for attribute, rollouts_by_prompt in validation_results[i].items():
                teacher_results[attribute] = {}
                for user_prompt, rollouts in rollouts_by_prompt.items():
                    teacher_winrates = [r.teacher_score.score for r in rollouts if r is not None and r.teacher_score is not None]
                    if len(teacher_winrates) > 0:
                        teacher_results[attribute][user_prompt] = teacher_winrates

            with open(
                self.run_path / "validate" / f"seed_{seed_state.index}_validate/teacher_diffs.json", "w"
            ) as f:
                json.dump(teacher_results, f, indent=4)

        # Save validation stats and plot for each seed
        for i, seed_state in enumerate(self.seed_states):
            candidate_stats = []
            for attribute, rollouts_by_prompt in validation_results[i].items():
                student_scores = []
                teacher_scores = []
                for user_prompt, rollouts in rollouts_by_prompt.items():
                    for r in rollouts:
                        if r is None:
                            continue
                        if r.student_score is not None and r.student_score.score is not None:
                            student_scores.append(r.student_score.score)
                        if r.teacher_score is not None and r.teacher_score.score is not None:
                            teacher_scores.append(r.teacher_score.score)

                student_winrate = float(np.mean(student_scores)) if student_scores else None
                teacher_winrate = float(np.mean(teacher_scores)) if teacher_scores else None

                candidate_stats.append({
                    "attribute": attribute,
                    "student_winrate": student_winrate,
                    "teacher_winrate": teacher_winrate,
                })

            # Save candidate stats as JSON
            with open(
                self.run_path / "validate" / f"seed_{seed_state.index}_validate/candidate_stats.json", "w"
            ) as f:
                json.dump(candidate_stats, f, indent=4)

            # Create scatter plot
            valid_points = [(s["student_winrate"], s["teacher_winrate"])
                           for s in candidate_stats
                           if s["student_winrate"] is not None and s["teacher_winrate"] is not None]

            if valid_points:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter([p[0] for p in valid_points], [p[1] for p in valid_points],
                          c='blue', alpha=0.7, marker='o')

                ax.set_xlabel('Student Winrate')
                ax.set_ylabel('Teacher Winrate')
                ax.set_title(f'Validation Results: Seed {seed_state.index}')
                ax.grid(True, alpha=0.3)

                fig.savefig(self.run_path / "validate" / f"seed_{seed_state.index}_validate/validation_scatter.pdf")
                plt.close(fig)
        
        # Save validation baselines results with updated teacher scores (if exist)
        print("Saving validation baselines with updated teacher scores...")
        with open(self.run_path / "val_baselines/rollouts.json", "w") as f:
            json_data = {k: [
                {
                    "response": r.response,
                    "model": r.model,
                    "student_score": r.student_score.raw_score,
                    "teacher_score": r.teacher_score.raw_score if r.teacher_score is not None else None,
                } 
                for r in v
            ] for k, v in self.val_baselines.items()}
            json.dump(json_data, f, indent=4, sort_keys=True)


class TestRunner(Runner):
    @property
    def runner_type(self) -> str:
        return "test"

    def train(self):
        pass
