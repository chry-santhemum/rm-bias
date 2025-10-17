import math
from typing import Any
from functools import cached_property
from dataclasses import dataclass, field
import numpy as np


# def adversariality(
#     z_score_1: float,
#     z_score_2: float,
# ) -> float:
#     """
#     Motivation: lines of (x - c)(- y - c) = 0.25
#     """
#     return 0.5 * (z_score_1 - z_score_2 - math.sqrt((z_score_1 + z_score_2) ** 2 + 1))


def adversariality(reward_diff: float, judge_score: float) -> float:
    return reward_diff - judge_score / 2.5


@dataclass
class PromptCluster:
    summary: str
    prompts: list[str]


@dataclass(frozen=True)
class Cluster:
    summary: str
    train_prompts: list[str]
    val_prompts: list[str]
    train_batch_size: int
    aux_info: Any = None


@dataclass(frozen=True)
class Rating:
    score: float
    rater_model_name: str

    def __repr__(self):
        if self.score is None:
            score_str = "None"
        else:
            score_str = f"{self.score:.2f}"
        return f"Rating(score={score_str}, rater={self.rater_model_name})"


@dataclass
class Rollout:
    response: str
    score: float | None


@dataclass
class AttributeStats:
    attribute: str
    judge_score: float | None = None
    rollouts: dict[str, list[Rollout]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent", None)

    @property
    def mean_rewards(self) -> dict[str, float]:
        mean_results = {}
        for user_prompt, rollouts in self.rollouts.items():
            rollouts = [r for r in rollouts if r.score is not None]
            mean_results[user_prompt] = np.mean([r.score for r in rollouts]).item()  # type: ignore
        return mean_results

    @property
    def all_rewards(self) -> dict[str, list[float]]:
        all_results = {}
        for user_prompt, rollouts in self.rollouts.items():
            rollouts = [r for r in rollouts if r.score is not None]
            all_results[user_prompt] = [r.score for r in rollouts]
        return all_results

    def mean_reward_diff(self, baselines: dict[str, list[Rollout]]) -> float | None:
        mean_rewards = self.mean_rewards
        if len(mean_rewards) == 0:
            return None

        baseline_scores = []
        for user_prompt in self.rollouts.keys():
            baseline_scores.extend([r.score for r in baselines[user_prompt]])
        if len(baseline_scores) == 0:
            return None
        return (
            np.mean(list(mean_rewards.values())).item()
            - np.mean(baseline_scores).item()
        )

    def adversarial_score(self, baselines: dict[str, list[Rollout]]) -> float | None:
        reward_diff = self.mean_reward_diff(baselines)
        if reward_diff is None or self.judge_score is None:
            return None
        return adversariality(reward_diff, self.judge_score)

    @cached_property
    def bootstrap_CI(self, confidence: float = 0.95) -> dict[str, float]: ...


@dataclass
class SeedState[T]:
    index: int
    dataset: str
    cluster: Cluster
    state: T  # information to persist, e.g. current population
    history: list[dict[str, AttributeStats]]

    def __repr__(self):
        return (
            f"SeedState(\n"
            f"index={self.index},\n"
            f"cluster_summary={self.cluster.summary},\n"
            f"num_steps={len(self.history)}"
        )
