"""
Cost estimate:

[per seed state]
Rewrites: train_batch_size * n_pop * n_rollouts (~4096 tokens per call)
"""

# %%

import dotenv
import random
from typing import Literal
from loguru import logger

from state import SeedState
from utils import ClusterModel, set_seed_all
from models import GenerationModel, JudgeModel
from planner import Planner
from runner import Runner
from bias_evaluator import BiasEvaluator

dotenv.load_dotenv()
set_seed_all(10086)


class OneTurnRunner(Runner):
    def __init__(
        self,
        seed_states: list[SeedState],
        hypothesis_planner: Planner,
        cluster_model: ClusterModel,
        policy_model: GenerationModel,
        bias_evaluator: BiasEvaluator,
        judge_model: JudgeModel,
        train_batch_size: int,
        n_baseline_rollouts: int,
        n_rewrite_rollouts: int,
        direction: Literal["plus", "minus"],
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
        self.planner = hypothesis_planner
        self.cluster_model = cluster_model
        self.bias_evaluator = bias_evaluator

        self.n_rewrite_rollouts = n_rewrite_rollouts
        self.train_batch_size = train_batch_size
        self.direction: Literal["plus", "minus"] = direction

    @property
    def runner_type(self) -> str:
        return "one_turn"

    async def train(self, validate: bool = True):
        self.planner.plan(
            runner=self,
            direction=self.direction,
            cluster_model=self.cluster_model,
        )

        evaluate_results = dict()

        for seed_state_idx, seed_state in enumerate(self.seed_states):
            async with self.bias_evaluator as evaluator:
                user_prompts = random.sample(
                    seed_state.cluster.train_prompts, self.train_batch_size
                )
                attributes = list(seed_state.history[-1].keys())
                references = [
                    self.get_references(seed_state_idx, att)
                    for att in attributes
                ]

                stats = await evaluator.evaluate_attributes(
                    user_prompts=user_prompts,
                    attributes=attributes,
                    references=references,
                    baselines=self.baselines,
                    n_rollouts=self.n_rewrite_rollouts,
                )
            evaluate_results[seed_state_idx] = stats

        for seed_state_idx, stats in evaluate_results.items():
            for attribute, rollouts in stats.items():
                self.seed_states[seed_state_idx].history[-1][attribute].rollouts = rollouts

        final_attributes = self.save_attribute_stats(
            direction=self.direction,
            top_k=8,
            save_dir=self.run_path / f"step_{self.step_count}_stats",
        )

        if validate:
            await self.validate(final_attributes=final_attributes)
