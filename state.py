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
    aux_info: Any = (
        None  # any information about train prompts, e.g. responses from dataset
    )


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
        return f"Rating(score={self.raw_score:.2f}, rater={self.rater.model_name})"


@dataclass(frozen=True)
class RatedResponse:
    assistant: str
    ratings: list[Rating]

    def raw_score(self, rater_model_name: str) -> float | None:
        for rating in self.ratings:
            if rating.rater.model_name == rater_model_name:
                return rating.raw_score
        return None

    def normalized_score(self, rater_model_name: str) -> float | None:
        for rating in self.ratings:
            if rating.rater.model_name == rater_model_name:
                return rating.aux_info.get("normalized_score", None)
        return None


@dataclass(frozen=True)
class Attack:
    system: str
    user: str
    responses: list[RatedResponse]
    aux_info: dict[str, Any] = field(
        default_factory=dict
    )  # info such as policy model names, reasoning, etc.

    def mean_raw_score(self, rater_model_name: str) -> float | None:
        raw_scores = []
        for response in self.responses:
            raw_score = response.raw_score(rater_model_name)
            if raw_score is not None:
                raw_scores.append(raw_score)
        if len(raw_scores) == 0:
            return None
        return float(np.mean(raw_scores))


    def mean_normalized_score(self, rater_model_name: str) -> float | None:
        normalized_scores = []
        for response in self.responses:
            normalized_score = response.normalized_score(rater_model_name)
            if normalized_score is not None:
                normalized_scores.append(normalized_score)
        if len(normalized_scores) == 0:
            return None
        return float(np.mean(normalized_scores))


    def adversarial_score(self, rater_1_model_name: str|None=None, rater_2_model_name: str|None=None) -> float | None:
        if rater_1_model_name is None:
            rater_1_model_name = self.responses[0].ratings[0].rater.model_name
        if rater_2_model_name is None:
            rater_2_model_name = self.responses[0].ratings[1].rater.model_name
        score_1 = self.mean_normalized_score(rater_1_model_name)
        score_2 = self.mean_normalized_score(rater_2_model_name)
        if score_1 is None or score_2 is None:
            return None
        return adversariality(
            z_score_1=score_1,
            z_score_2=score_2,
        )

    def __repr__(self):
        return (
            f"Attack(\n"
            f"\tsystem={self.system[:50]+'...'},\n"
            f"\tuser={self.user[:50]+'...'},\n"
            f"\tnum_responses={len(self.responses)},\n"
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

    def adversarial_scores(self, rater_1_model_name: str|None=None, rater_2_model_name: str|None=None) -> list[float]:
        return [
            score for attack in self.attacks
            if (score := attack.adversarial_score(rater_1_model_name, rater_2_model_name)) is not None
        ]

    @property
    def mean_adversarial_score(self, rater_1_model_name: str|None=None, rater_2_model_name: str|None=None) -> float:
        return float(np.mean(self.adversarial_scores(rater_1_model_name, rater_2_model_name)))

    @property
    def stdev_adversarial_score(self, rater_1_model_name: str|None=None, rater_2_model_name: str|None=None) -> float:
        return float(np.std(self.adversarial_scores(rater_1_model_name, rater_2_model_name)))


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
