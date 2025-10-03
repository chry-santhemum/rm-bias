"""Prompt statistics caching and rating function types."""

# %%
import patches

import os
import hashlib
import asyncio
import json
import torch
import numpy as np
import logging
from tqdm.auto import tqdm
from pathlib import Path
from slist import Slist
import random
from abc import ABC, abstractmethod
from dataclasses import replace, dataclass
from functools import partial
from typing import Any, Tuple
import nest_asyncio

import pandas as pd
from datasets import load_dataset

from llm_types import ChatHistory
from state import (
    SystemPromptStats,
    Attack,
    RatedResponse,
    Rating,
    Rater,
    SeedState,
)
from models import PolicyModel
from utils import load_model, get_to_pass_reasoning, parse_json_response, REWARD_MODELS
from standard_prompts import set_seed_all, make_prompt_mix
from client import (
    is_thinking_model,
    get_universal_caller,
    sample_from_model,
    sample_from_model_parallel,
)

# %%
logger = logging.getLogger(__name__)
nest_asyncio.apply()


@dataclass(frozen=True)
class RatingResult:
    score: float | None
    reasoning: str | None


def prompt_to_hash_path(prompt: str, target_dir: Path) -> Path:
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    return target_dir / f"{prompt_hash}.json"


class RatingFunction(ABC):
    def __init__(self):
        # Default scaling values
        self.mean: float = 0.0
        self.stdev: float = 1.0

    @property
    def rater(self) -> Rater:
        return Rater(
            model_name=self.model_name, rating_function_type=self.rating_function_type
        )

    @property
    @abstractmethod
    def rating_function_type(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    async def async_rate(self, chat_histories: list[ChatHistory], use_tqdm: bool=True) -> list[RatingResult]:
        pass

    async def __call__(
        self,
        policy_model: PolicyModel,
        seed_state: SeedState,
        chat_histories: list[ChatHistory],
        chat_histories_to_attack_idx: list[Tuple[int, int]],
        per_prompt_normalize: bool,  # whether to normalize per-prompt
        use_tqdm: bool=True,
    ):
        """
        Modifies seed_state in-place.
        Applies rating function to each system prompt in the latest step of seed_state.history.
        """

        
        rating_results = await self.async_rate(chat_histories, use_tqdm=use_tqdm)
        scores: list[float | None] = [
            result.score for result in rating_results
        ]
        reasonings: list[str | None] = [
            result.reasoning for result in rating_results
        ]

        per_prompt_means = []
        if per_prompt_normalize:
            for attack_idx, attack in enumerate(attacks):
                sps = system_prompt_stats[attack_to_sps_idx[attack_idx]]
                # assert attack.system == sps.system_prompt
                file_path = prompt_to_hash_path(
                    attack.user, Path(f"data/prompt_stats/{sps.system_prompt_dir}")
                )
                with open(file_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    per_prompt_means.append(
                        json_data[policy_model.model_name]["summary_stats"][
                            self.model_name
                        ]["mean"]
                    )

        # Normalize scores
        if per_prompt_normalize:
            normalized_scores = [
                (
                    (scores[i] - per_prompt_means[chat_histories_to_attack_idx[i][0]])
                    / (self.stdev + 1e-6)
                    if scores[i] is not None
                    else None
                )
                for i in range(len(scores))
            ]
        else:
            normalized_scores = [
                (
                    (scores[i] - self.mean) / (self.stdev + 1e-6)  # type: ignore
                    if scores[i] is not None
                    else None
                )
                for i in range(len(scores))
            ]

        for i in range(len(chat_histories)):
            attack_idx, response_idx = chat_histories_to_attack_idx[i]

            if self.rater not in [
                rating.rater
                for rating in attacks[attack_idx].responses[response_idx].ratings
            ]:
                attacks[attack_idx].responses[response_idx].ratings.append(
                    Rating(
                        raw_score=scores[i],  # type: ignore
                        rater=self.rater,
                        aux_info={
                            "normalized_score": normalized_scores[i],
                            "reasoning": reasonings[i],
                            **(
                                {"per_prompt_mean": per_prompt_means[attack_idx]}
                                if per_prompt_normalize
                                else {}
                            ),
                        },
                    )
                )



class RewardModel(RatingFunction):
    """
    Wrapper around reward models; __init__ kwargs (e.g. device) are passed to load_model
    """

    def __init__(self, reward_model_name: str, batch_size: int = 32, **kwargs):
        assert (
            reward_model_name in REWARD_MODELS
        ), f"Model {reward_model_name} not local!!!"
        self._model_name = reward_model_name
        self.batch_size = batch_size
        self.model, self.tokenizer = load_model(reward_model_name, **kwargs)
        self.device = self.model.device
        super().__init__()

    @property
    def rating_function_type(self) -> str:
        return "classifier"

    @property
    def model_name(self) -> str:
        return self._model_name

    def rate(self, chat_histories: list[ChatHistory], use_tqdm: bool=True) -> list[RatingResult]:
        rewards = []

        pbar = tqdm(range(0, len(chat_histories), self.batch_size), desc="Rating responses") if use_tqdm else range(0, len(chat_histories), self.batch_size)
        for i in pbar:
            batch = chat_histories[i : i + self.batch_size]
            inputs = [chat.remove_system().to_openai_messages() for chat in batch]
            input_ids = self.tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                padding_side="right",
            ).to(self.model.device)

            attn_mask = input_ids.ne(self.tokenizer.pad_token_id)

            with torch.no_grad():
                scores = self.model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits.squeeze(-1)

                rewards.extend([RatingResult(score=s, reasoning=None) for s in scores.tolist()])

        return rewards


    async def async_rate(self, chat_histories: list[ChatHistory], use_tqdm: bool=True) -> list[RatingResult]:
        return await asyncio.to_thread(self.rate, chat_histories, use_tqdm)



# %%
# from pathlib import Path
# import json
# from tqdm.auto import tqdm

# # iterate over each folder in this path
# run_path = Path("/workspace/rm-bias/data/prompt_stats")
# for dataset_dir in run_path.iterdir():
#     if not dataset_dir.is_dir():
#         continue

#     if dataset_dir.name == "agent-harm":
#         print("Skipping agent-harm")
#         continue

#     print(f"Processing dataset folder: {dataset_dir}")
#     for json_file in tqdm(dataset_dir.glob("*.json"), desc="Processing JSON files"):
#         with open(json_file, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         summary_stats = data["meta-llama/llama-3.1-8b-instruct"]["summary_stats"]
#         for rater_name, rater_stats in summary_stats.items():
#             rater_stats["scores_raw"] = rater_stats.pop("rewards_raw")
#             rater_stats["scores_winsorized"] = rater_stats.pop("rewards_winsorized")

#         with open(json_file, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=4)

#                 all_adversarial_scores = []

#                 for attack in data["attacks"]:
#                     # for rating in attack["ratings"]:
#                     #     if rating["rater"]["model_name"] == "openai/gpt-5-nano":
#                     #         rating["aux_info"]["normalized_score"] = (rating["raw_score"] - 5.0) / (lm_judge_loaded_stats["stdev"] + 1e-6)
#                     #     elif rating["rater"]["model_name"] == "skywork-v2":
#                     #         rating["aux_info"]["normalized_score"] = rating["aux_info"]["normalized_score"] / (reward_loaded_stats["stdev"] + 1e-6)

#                     attack["aux_info"]["normalized_lm_judge"] = attack["ratings"][1]["aux_info"]["normalized_score"]
#                     attack["aux_info"]["normalized_reward"] = attack["ratings"][0]["aux_info"]["normalized_score"]
#                     attack["aux_info"]["adversarial_score"] = adversariality(
#                         z_score_1=attack["aux_info"]["normalized_reward"],
#                         z_score_2=attack["aux_info"]["normalized_lm_judge"],
#                     )
#                     all_adversarial_scores.append(attack["aux_info"]["adversarial_score"])

#                 mean = float(np.mean(all_adversarial_scores))
#                 stdev = float(np.std(all_adversarial_scores, ddof=1))

#                 data["mean_score"] = mean
#                 data["stdev_score"] = stdev

#                 with open(json_file, "w", encoding="utf-8") as f:
#                     json.dump(data, f, indent=4)
#             except Exception as e:
#                 print(f"    Error reading {json_file}: {e}")

# for prompt in tqdm(prompts, desc="Processing prompts"):
#     file_path = prompt_to_hash_path(
#         prompt, Path("data/prompt_stats/instruction-dataset")
#     )
#     if file_path.exists():
#         with open(file_path, "r", encoding="utf-8") as f:
#             json_data = json.load(f)
#         assert json_data["prompt"] == prompt

#         # compute correlation between scores
#         rewards = json_data["summary_stats"]["skywork-v2"]["rewards_raw"]
#         lm_judge = json_data["summary_stats"]["openai/gpt-5-nano"]["rewards_raw"]

#         cleaned_rewards, cleaned_lm_judge = [], []

#         for i in range(len(rewards)):
#             if rewards[i] is not None and lm_judge[i] is not None:
#                 cleaned_rewards.append(rewards[i])
#                 cleaned_lm_judge.append(lm_judge[i])

#         correlation = np.corrcoef(cleaned_rewards, cleaned_lm_judge)[0, 1]

#         json_data["correlation"] = float(correlation)

#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(json_data, f, indent=4)
#     else:
#         print("Prompt not found: ", prompt)

# ultrafeedback = pd.read_csv("data/ultrafeedback/labels_20k.csv")
# prompts = ultrafeedback["Document"].tolist()

# %%
if __name__ == "__main__":
    set_seed_all(10086)

    # agent_harm = load_dataset("ai-safety-institute/AgentHarm", name="chat", split="test_public")
    # prompts = list(agent_harm["prompt"])

    instruction_test = load_dataset("HuggingFaceH4/instruction-dataset", split="test")
    prompts = list(instruction_test["prompt"])

    target_dir = Path("data/prompt_stats/instruction-dataset")

    policy = PolicyModel(
        model_name="meta-llama/llama-3.1-70b-instruct",
        max_tokens=1024,
        temperature=0.8,
    )
    prompt_rollout(
        prompts=prompts,
        target_dir=target_dir,
        policy_model=policy,
        n_samples=16,
    )

    rater_1 = RewardModel(
        reward_model_name="skywork-v2",
        batch_size=32,
    )

    # rater_2 = LLMJudge(
    #     judge_model_name="openai/gpt-5-nano",
    #     rubric=HANDWRITTEN_RUBRIC,
    #     max_par=256,
    # )

    prompt_rating(
        prompts=prompts,
        target_dir=target_dir,
        rater=rater_1,
        policy_model=policy,
    )
    # prompt_rating(
    #     prompts=prompts,
    #     target_dir=target_dir,
    #     rater=rater_2,
    #     policy_model=policy,
    # )

    # for prompt in tqdm(prompts, desc="Post-processing prompts"):
    #     file_path = prompt_to_hash_path(prompt, target_dir)
    #     if file_path.exists():
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             json_data = json.load(f)

    #         json_data["topic_label"] = 0
    #         json_data["topic_name"] = "All"
    #         json_data["dataset"] = "instruction-dataset"

    #         with open(file_path, "w", encoding="utf-8") as f:
    #             json.dump(json_data, f, indent=4)
    #     else:
    #         print("Prompt not found: ", prompt)


# %%
# cluster_df: pd.DataFrame = pd.read_csv("data/wildchat/cluster_50k.csv")
# labels_df: pd.DataFrame = pd.read_csv("data/wildchat/labels_50k.csv")
# prompts_to_sample = []

# for topic_id in tqdm(range(1, 30), desc="Processing topics"):
#     topic = cluster_df.loc[cluster_df.index[topic_id+1], "Name"].split('_', maxsplit=1)[-1]  # description
#     all_user_prompts = []

#     with pd.read_csv("data/wildchat/labels_50k.csv", chunksize=10000) as reader:
#         for chunk in reader:
#             for index, row in chunk.iterrows():
#                 if int(row["Topic"]) == topic_id:
#                     all_user_prompts.append(row["Document"])

#     topic_prompts = random.sample(all_user_prompts, min(100, len(all_user_prompts)))

#     for prompt in tqdm(topic_prompts, desc="Processing prompts"):
#         prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
#         file_path = Path("data/prompt_stats") / f"{prompt_hash}.json"
#         if file_path.exists():
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 json_data = json.load(f)
#                 assert json_data["prompt"] == prompt

#                 if "skywork-v2" not in json_data["summary_stats"]:
#                     new_dict = {}
#                     key_names = list(json_data["summary_stats"].keys())

#                     for key in key_names:
#                         val = json_data["summary_stats"][key]
#                         new_dict[key] = val
#                         del json_data["summary_stats"][key]

#                     json_data["summary_stats"]["skywork-v2"] = new_dict

#                 else:
#                     print("Already in correct format")
#                 with open(file_path, 'w', encoding='utf-8') as f:
#                     json.dump(json_data, f, indent=4)
#         else:
#             print("Prompt not found: ", prompt)

#     print("=" * 80)
#     print(f"Topic {topic_id}: {topic} with {len(all_user_prompts)} user prompts")
#     print("\nExample prompts:\n")
#     for prompt in all_user_prompts[:10]:
#         print("-" * 80)
#         print(prompt)


# %%
