from typing import Any
from dataclasses import dataclass, field
import numpy as np
from art2.llm_utils import ChatHistory


@dataclass(frozen=True)
class Cluster:
    summary: str
    train_prompts: list[str]
    val_prompts: list[str]

@dataclass(frozen=True)
class Rater:
    model_name: str
    rating_function_type: str

@dataclass(frozen=True)
class Rating:
    score: float  # unnormalized score
    rater: Rater
    aux_info: dict[str, Any] = field(default_factory=dict)  # info such as reasoning, normalized score, number of comparisons, etc.

    def __repr__(self):
        return f"Rating(score={self.score:.2f}, rater_model_name={self.rater.model_name}, aux_info={self.aux_info})"

@dataclass(frozen=True)
class Attack:
    chat_history: ChatHistory  # system, user, assistant
    ratings: list[Rating]
    aux_info: dict[str, Any] = field(default_factory=dict)  # info such as model names, reasoning, etc.

    @property
    def system(self) -> str:
        return self.chat_history.get_first('system')

    @property
    def user(self) -> str:
        return self.chat_history.get_first('user')
    
    @property
    def assistant(self) -> str:
        return self.chat_history.get_first('assistant')

    def __repr__(self):
        return (
            f"Attack(\n"
            f"system={self.system[:50]+'...' if self.system else 'N/A'},\n"
            f"user={self.user[:50]+'...' if self.user else 'N/A'},\n"
            f"assistant={self.assistant[:50]+'...' if self.assistant else 'N/A'},\n"
            f"ratings={self.ratings}\n"
            f")"
        )

@dataclass
class SeedState:
    cluster: Cluster
    history: list[dict[str, Any]]  # must have "attacks" key

    def __repr__(self):
        return (
            f"SeedState(\n"
            f"cluster_summary={self.cluster.summary},\n"
            f"num_steps={len(self.history)},\n"
            f"last_attack={self.history[-1]['attacks'][-1]}\n"
            f")"
        )


@dataclass
class SystemPromptStats:
    system_prompt: str
    attacks: list[Attack] = field(default_factory=list)

    @property
    def scores(self) -> list[float]:
        return [
            attack.aux_info["adversarial_score"]
            for attack in self.attacks
            if "adversarial_score" in attack.aux_info
        ]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores))

    @property
    def stdev_score(self) -> float:
        return float(np.std(self.scores))


@dataclass
class EvoSeedState:
    cluster: Cluster
    current_pop: dict[str, int]  # system prompt -> step index
    history: list[dict[str, SystemPromptStats]]

    def __repr__(self):
        return (
            f"EvoSeedState(\n"
            f"cluster_summary={self.cluster.summary},\n"
            f"num_steps={len(self.history)},\n"
            f"current_pop_size={len(self.current_pop)},\n"
            f"current_pop=[\n"
            f"{",\n".join([f"Step {step_idx}: {sp}" for sp, step_idx in self.current_pop.items()])}\n"
            f"]\n)"
        )