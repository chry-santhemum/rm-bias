import math
from typing import Any
from functools import cached_property
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass(frozen=True)
class Cluster:
    summary: str
    train_prompts: list[str]
    val_prompts: list[str]
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
    score: float


@dataclass
class AttributeStats:
    attribute: str
    rollouts: dict[str, list[Rollout|None]] = field(default_factory=dict)
    judge_scores: dict[str, list[float|None]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent", None)

    @property
    def mean_rewards(self) -> dict[str, float]:
        mean_results = {}
        for user_prompt, rollouts in self.rollouts.items():
            rollouts = [r for r in rollouts if r is not None and r.score is not None]
            mean_results[user_prompt] = np.mean([r.score for r in rollouts]).item()  # type: ignore
        return mean_results

    @property
    def all_rewards(self) -> dict[str, list[float]]:
        all_results = {}
        for user_prompt, rollouts in self.rollouts.items():
            rollouts = [r for r in rollouts if r is not None and r.score is not None]
            all_results[user_prompt] = [r.score for r in rollouts]
        return all_results

    @property
    def all_rollouts(self) -> dict[str, list[Any]]:
        all_results = {}
        for user_prompt, rollouts in self.rollouts.items():
            all_results[user_prompt] = [asdict(r) if r is not None else None for r in rollouts]
        return all_results

    def mean_reward_diff(self, baselines: dict[str, list[Rollout|None]]) -> float | None:
        mean_rewards = self.mean_rewards
        if len(mean_rewards) == 0:
            return None

        baseline_scores = []
        for user_prompt in self.rollouts.keys():
            baseline_scores.extend([r.score for r in baselines[user_prompt] if r is not None])
        if len(baseline_scores) == 0:
            return None
        return (
            np.mean(list(mean_rewards.values())).item()
            - np.mean(baseline_scores).item()
        )
    
    def reward_winrate(self, baselines: dict[str, list[Rollout|None]]) -> float|None:
        all_diffs = []
        for user_prompt, rollouts in self.rollouts.items():
            baseline_rollouts = baselines[user_prompt]
            for rewrite_rollout, baseline_rollout in zip(rollouts, baseline_rollouts):
                if rewrite_rollout is None or baseline_rollout is None:
                    continue
                all_diffs.append(rewrite_rollout.score - baseline_rollout.score)
        if len(all_diffs) == 0:
            return None
        return sum(1 for d in all_diffs if d > 0) / len(all_diffs)
    
    def judge_winrate(self) -> float|None:
        all_winrates = []
        for user_prompt, judge_scores in self.judge_scores.items():
            winrates_clean = [wr for wr in judge_scores if wr is not None]
            all_winrates.extend(winrates_clean)
        if len(all_winrates) == 0:
            return None
        return np.mean(all_winrates).item()

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
