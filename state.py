import math
from typing import Any
from dataclasses import dataclass, field
import numpy as np
from pydantic_core.core_schema import NoneSchema
from llm_types import ChatHistory


def adversariality(
    z_score_1: float,
    z_score_2: float,
) -> float:
    """
    Motivation: lines of (x - c)(- y - c) = 0.25
    """
    return 0.5 * (z_score_1 - z_score_2 - math.sqrt((z_score_1 + z_score_2) ** 2 + 1))


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
    plus_score: float
    minus_score: float

@dataclass
class Rollout:
    response: str
    score: float


@dataclass
class AttributeStats:
    attribute: str
    judge_score: float|None = None
    rollouts: dict[str, list[PlusMinusRollout|Rollout]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent", None)

    @property
    def mean_reward(self) -> dict[str, float]:
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
