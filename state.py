from typing import Any, Literal
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Cluster:
    summary: str
    train_prompts: list[str]
    val_prompts: list[str]
    aux_info: Any = None


@dataclass(kw_only=True, slots=True)
class RewriteScore:
    score: float | None      # reward diff if it is RM, winrate if it is judge
    raw_score: float | None  # only exists when it is a reward model
    reasoning: str | None    # only exists when it is a LLM judge
    model_name: str


@dataclass(kw_only=True, slots=True)
class Rollout:
    response: str
    student_score: RewriteScore
    teacher_score: RewriteScore | None = None


@dataclass
class AttributeStats:
    attribute: str
    rollouts: dict[str, list[Rollout|None]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent", None)

    def to_dict(self) -> dict[str, list[dict|None]]:
        all_results = {}
        for user_prompt, rollouts in self.rollouts.items():
            user_prompt_results = []
            for r in rollouts:
                if r is None:
                    user_prompt_results.append(None)
                else:
                    user_prompt_results.append({
                        "response": r.response,
                        "student_score": r.student_score.score,
                        "teacher_score": r.teacher_score.score if r.teacher_score is not None else None,
                    })
            all_results[user_prompt] = user_prompt_results
        return all_results

    def winrate(self, rater: Literal["student", "teacher"]) -> float | None:
        all_scores = []
        for _, rollouts in self.rollouts.items():
            for r in rollouts:
                if r is None:
                    continue
                if rater == "student":
                    if r.student_score.score is not None:
                        all_scores.append(r.student_score.score)
                elif rater == "teacher":
                    if r.teacher_score is not None and r.teacher_score.score is not None:
                        all_scores.append(r.teacher_score.score)

        if len(all_scores) == 0:
            return None
        return np.mean(all_scores).item()

    def __repr__(self):
        return (
            f"AttributeStats(\n"
            f"  attribute={self.attribute[:40]!r},\n"
            f"  num_user_prompts={len(self.rollouts)},\n"
            f"  student_winrate={self.winrate('student')},\n"
            f"  teacher_winrate={self.winrate('teacher')},\n"
            f"  meta={self.meta!r},\n"
            f")"
        )

@dataclass
class SeedState:
    index: int
    dataset: str
    cluster: Cluster
    state: dict
    history: list[dict[str, AttributeStats]]
