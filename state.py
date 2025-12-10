from typing import Any, Literal
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass(frozen=True)
class Cluster:
    summary: str
    train_prompts: list[str]
    val_prompts: list[str]
    aux_info: Any = None


@dataclass(kw_only=True, slots=True)
class Score:
    score: float | None      # reward diff if it is RM, winrate if it is judge
    raw_score: float | None  # only exists when it is a reward model
    reasoning: str | None    # only exists when it is a LLM judge
    model_name: str


@dataclass(kw_only=True, slots=True)
class Rollout:
    response: str
    student_score: Score
    teacher_score: Score | None = None


@dataclass
class AttributeStats:
    attribute: str
    rollouts: dict[str, list[Rollout|None]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent", None)

    @property
    def all_rollouts(self) -> dict[str, list]:
        all_results = {}
        for user_prompt, rollouts in self.rollouts.items():
            all_results[user_prompt] = [asdict(r) if r is not None else None for r in rollouts]
        return all_results

    def winrate(self, rater: Literal["student", "teacher"]) -> float | None:
        """
        Computes the mean score across all rollouts for the specified rater.
        For student RM, score is the reward diff (rewritten - baseline).
        For teacher, score is the preference winrate (or reward diff if teacher is RM).
        """
        all_scores = []
        for user_prompt, rollouts in self.rollouts.items():
            for rollout in rollouts:
                if rollout is None:
                    continue
                if rater == "student":
                    if rollout.student_score is not None and rollout.student_score.score is not None:
                        all_scores.append(rollout.student_score.score)
                else:  # teacher
                    if rollout.teacher_score is not None and rollout.teacher_score.score is not None:
                        all_scores.append(rollout.teacher_score.score)
        if len(all_scores) == 0:
            return None
        return np.mean(all_scores).item()

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
