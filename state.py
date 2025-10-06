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
class PlusMinusRollout:
    plus: str
    minus: str
    plus_score: float | None
    minus_score: float | None

@dataclass
class Rollout:
    response: str
    score: float | None


@dataclass
class AttributeStats:
    attribute: str
    judge_score: float|None = None
    rollouts: dict[str, list[PlusMinusRollout]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent", None)

    @property
    def mean_rewards(self) -> dict[str, dict[str, float]]:
        mean_results = {}
        for user_prompt, rollouts in self.rollouts.items():
            rollouts = [r for r in rollouts if r.plus_score is not None and r.minus_score is not None]
            mean_results[user_prompt] = {
                "plus": np.mean([r.plus_score for r in rollouts]).item(),  # type: ignore
                "minus": np.mean([r.minus_score for r in rollouts]).item(),  # type: ignore
            }
        return mean_results

    
    @property
    def all_rewards(self) -> dict[str, dict[str, list[float]]]:
        all_results = {}
        for user_prompt, rollouts in self.rollouts.items():
            rollouts = [r for r in rollouts if r.plus_score is not None and r.minus_score is not None]
            all_results[user_prompt] = {
                "plus": [r.plus_score for r in rollouts], 
                "minus": [r.minus_score for r in rollouts], 
            }
        return all_results
    

    @property
    def mean_reward_diff(self) -> float|None:
        mean_rewards = self.mean_rewards
        if len(mean_rewards) == 0:
            return None
        return np.mean([mean_rewards[k]["plus"] - mean_rewards[k]["minus"] for k in mean_rewards]).item()

    @property
    def adversarial_score(self) -> float | None:
        if self.judge_score is None or self.mean_reward_diff is None:
            return None
        return adversariality(self.mean_reward_diff, self.judge_score)
        

    @cached_property
    def bootstrap_CI(self, confidence: float = 0.95) -> dict[str, float]:
        ...




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
