import math
from typing import Any
from dataclasses import dataclass, field
import numpy as np
from llm_types import ChatHistory


def adversariality(
    z_score_1: float,
    z_score_2: float,
) -> float:
    """
    Motivation: lines of (x - c)(- y - c) = 0.25
    """
    return 0.5 * (z_score_1 - z_score_2 - math.sqrt((z_score_1 + z_score_2) ** 2 + 1))


@dataclass(frozen=True)
class Cluster:
    summary: str
    train_prompts: list[str]
    val_prompts: list[str]
    train_batch_size: int = 0  # 0 means use all
    aux_info: Any = None  # any information about train prompts, e.g. responses from dataset


@dataclass(frozen=True)
class Rater:
    model_name: str
    rating_function_type: str


@dataclass(frozen=True)
class Rating:
    raw_score: float  # unnormalized score
    rater: Rater
    aux_info: dict[str, Any] = field(
        default_factory=dict
    )  # info such as reasoning, normalized score, number of comparisons, etc.

    def __repr__(self):
        return f"Rating(score={self.raw_score:.2f}, rater_model_name={self.rater.model_name}, aux_info={self.aux_info})"


@dataclass(frozen=True)
class Attack:
    chat_history: ChatHistory  # system, user, assistant
    ratings: list[Rating]
    aux_info: dict[str, Any] = field(
        default_factory=dict
    )  # info such as model names, reasoning, etc.

    @property
    def system(self) -> str:
        return self.chat_history.get_first("system")

    @property
    def user(self) -> str:
        return self.chat_history.get_first("user")

    @property
    def assistant(self) -> str:
        return self.chat_history.get_first("assistant")

    @property
    def normalized_reward(self) -> float | None:
        """Get the first rating by a reward model, and take the normalized_score in aux_info"""
        for rating in self.ratings:
            if rating.rater.rating_function_type == "classifier":
                return rating.aux_info.get("normalized_score", None)
        return None

    @property
    def normalized_lm_judge(self) -> float | None:
        """Get the first rating by a LLM judge, and take the normalized_score in aux_info"""
        for rating in self.ratings:
            if rating.rater.rating_function_type == "lm_judge":
                return rating.aux_info.get("normalized_score", None)
        return None

    @property
    def adversarial_score(self) -> float | None:
        if self.normalized_reward is None or self.normalized_lm_judge is None:
            return None
        return adversariality(
            z_score_1=self.normalized_reward,
            z_score_2=self.normalized_lm_judge,
        )

    def __repr__(self):
        return (
            f"Attack(\n"
            f"system={self.system[:50]+'...'},\n"
            f"user={self.user[:50]+'...'},\n"
            f"assistant={self.assistant[:50]+'...'},\n"
            f"ratings={self.ratings}\n"
            f")"
        )


@dataclass
class SystemPromptStats:
    system_prompt: str
    system_prompt_dir: str
    meta: dict[str, Any] = field(
        default_factory=dict
    )  # such as parent / island id / other state info
    attacks: list[Attack] = field(default_factory=list)

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent", None)

    @property
    def scores(self) -> list[float]:
        return [
            attack.adversarial_score
            for attack in self.attacks
            if attack.adversarial_score is not None
        ]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores))

    @property
    def stdev_score(self) -> float:
        return float(np.std(self.scores))


@dataclass
class SeedState[T]:
    index: int
    dataset: str
    cluster: Cluster
    state: T  # information to persist, e.g. current population
    history: list[dict[str, SystemPromptStats]]

    def __repr__(self):
        return (
            f"SeedState(\n"
            f"index={self.index},\n"
            f"cluster_summary={self.cluster.summary},\n"
            f"num_steps={len(self.history)}"
        )
