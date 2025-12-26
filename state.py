from typing import Any, Literal
from dataclasses import dataclass, field
import math
import numpy as np

from utils import remove_outliers


@dataclass
class Cluster:
    summary: str
    train_prompts: list[str]
    val_prompts: list[str]
    aux_info: Any = None

    def to_dict(self) -> dict[str, Any]:
        """ignores aux_info"""
        return {
            "summary": self.summary,
            "train_prompts": self.train_prompts,
            "val_prompts": self.val_prompts,
        }


@dataclass(kw_only=True, slots=True)
class RewriteScore:
    score: float | None      # reward diff if RM, winrate if judge, None if baseline
    raw_score: float | None  # only exists when it is a reward model
    reasoning: str | None    # only exists when it is a LLM judge
    model_name: str

@dataclass(kw_only=True, slots=True)
class Rollout:
    response: str
    student_score: RewriteScore
    teacher_score: RewriteScore | None = None
    model: str | None = None


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
                    results_dict = {
                        "response": r.response,
                        "student_score": r.student_score.score,
                    }
                    if r.teacher_score is not None:
                        results_dict["teacher_score"] = r.teacher_score.score
                        if r.teacher_score.reasoning is not None:
                            results_dict["teacher_reasoning"] = r.teacher_score.reasoning

                    user_prompt_results.append(results_dict)
            all_results[user_prompt] = user_prompt_results
        return all_results

    def winrate(self, rater: Literal["student", "teacher"]) -> float | None:
        all_scores = []
        for _, rollouts in self.rollouts.items():
            for r in rollouts:
                if r is None or r.teacher_score is None:
                    continue
                if rater == "student":
                    if r.student_score.score is not None:
                        all_scores.append(r.student_score.score)
                elif rater == "teacher":
                    if r.teacher_score.score is not None:
                        all_scores.append(r.teacher_score.score)

        if len(all_scores) == 0:
            return None

        # Check if all scores are -1, 0, or 1 (allowing for small floating point error).
        unique_rounded = set()
        for s in all_scores:
            if math.isclose(s, 1, abs_tol=1e-6):
                unique_rounded.add(1)
            elif math.isclose(s, 0, abs_tol=1e-6):
                unique_rounded.add(0)
            elif math.isclose(s, -1, abs_tol=1e-6):
                unique_rounded.add(-1)
            else:
                unique_rounded.add('other')

        if unique_rounded <= {-1, 0, 1}:
            filtered_scores = all_scores
        else:
            filtered_scores = remove_outliers(all_scores)

        return np.mean(filtered_scores).item()

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
