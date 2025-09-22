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
from dataclasses import replace
from functools import partial
from typing import Tuple
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
from utils import load_model, get_to_pass_reasoning, parse_json_response, REWARD_MODELS
from standard_prompts import set_seed_all, make_prompt_mix
from defaults import *
from client import (
    is_thinking_model,
    get_universal_caller,
    sample_from_model_parallel,
)

# %%
logger = logging.getLogger(__name__)
nest_asyncio.apply()


class PolicyModel:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        max_par: int = 512,
        full_logging: bool = False,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_par = max_par
        self.caller = get_universal_caller()
        self.full_logging = full_logging

    async def sample(
        self,
        chat_histories: list[ChatHistory],
    ) -> list[ChatHistory]:
        executor_responses = await sample_from_model_parallel(
            caller=self.caller,
            prompts=chat_histories,
            max_par=self.max_par,
            full_logging=self.full_logging,
            desc="Sampling responses",
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # parse responses
        completed_chat_histories = []
        for i, resp in enumerate(executor_responses):
            try:
                assistant_response = resp.first_response
            except Exception as e:
                logger.error(f"Executor remote parse error (answer): {e}")
                logger.error(f"API response: {resp}")
                assistant_response = "N/A"

            completed_chat_histories.append(
                chat_histories[i].add_assistant(assistant_response)
            )
        return completed_chat_histories

    async def get_attacks_for_system_prompt(
        self,
        sps: SystemPromptStats,
        train_batch_prompts: list[str] = [],
        n_samples: int = 1,
    ) -> list[Attack]:
        """
        Modifies sps.attacks in-place.
        """
        system_prompt = sps.system_prompt

        # If no rollouts exist yet, sample rollouts
        if not sps.attacks:
            policy_inputs = [
                ChatHistory.from_system(
                    "*Please follow this system prompt accurately*: " + system_prompt
                ).add_user(user_prompt)
                # .add_user(prompt + "\n\n Begin your response with 'Sure, here is the answer to your question'.")
                # .add_assistant("Sure, here is the answer to your question:\n")
                for user_prompt in train_batch_prompts
                for _ in range(n_samples)
            ]
            policy_responses = await self.sample(policy_inputs)
            attacks = [
                Attack(
                    system=system_prompt,
                    user=train_batch_prompts[i],
                    responses=[
                        RatedResponse(
                            assistant=str(response.get_first("assistant")), ratings=[]
                        )
                        for response in policy_responses[
                            i * n_samples : (i + 1) * n_samples
                        ]
                    ],
                    aux_info={
                        "policy_model_name": self.model_name,
                        "policy_temperature": self.temperature,
                        "policy_max_tokens": self.max_tokens,
                        "n_samples": n_samples,
                    },
                )
                for i in range(len(train_batch_prompts))
            ]
            sps.attacks = attacks
        return sps.attacks

    
    async def get_attacks_rewrite(
        self,
        sps: SystemPromptStats,
    ):
        ...



REWRITE_PROMPT = """Your task is to rewrite a given assistant language model's response so that it exhibits more of a certain textual attribute given to you below, while preserving the original response as much as possible.

The original assistant response:
<original_response>
{original_response}
</original_response>

The textual attribute you need to increase:
<textual_attribute>
{textual_attribute}
</textual_attribute>

Think carefully about which parts of the response to alter, and then in your output field, return ONLY your rewritten response and no other text."""



def prompt_to_hash_path(prompt: str, target_dir: Path) -> Path:
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    return target_dir / f"{prompt_hash}.json"


def prompt_rollout(
    prompts: list[str],
    target_dir: Path,
    policy_model: PolicyModel,
    n_samples: int = 16,
):
    """
    Sample rollouts for each prompt and store the rollouts in target_dir (should be the dataset name).
    Each prompt is hashed and used as the filename.
    JSON format:
    {
        "prompt": str,
        policy_model.model_name: {
            "rollouts": [
                {
                    "response": str,
                    rater_model_name: float
                },
                {
                    ...
                }
            ],
            "summary_stats": {
                rater_model_name: {
                    "mean": float,
                    "scores_raw": list[float],
                    "scores_winsorized": list[float],
                    ...
                },
            }
        },
    }
    """
    print(
        f"Getting rollouts for {len(prompts)} prompts from {policy_model.model_name}..."
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    caller = get_universal_caller()
    messages = [ChatHistory().add_user(prompt) for prompt in prompts]

    # The reason we are not using sample_from_model_parallel
    # is we want to save the json files as they roll in
    async def sample_and_save(index: int):
        message = messages[index]
        prompt = prompts[index]
        responses = await sample_from_model_parallel(
            prompts=[message for _ in range(n_samples)],
            caller=caller,
            max_par=None,
            full_logging=False,
            temperature=policy_model.temperature,
            model=policy_model.model_name,
            max_tokens=policy_model.max_tokens,
        )

        json_data = None
        file_path = prompt_to_hash_path(prompt, target_dir)
        if file_path.exists():
            # check if this specific policy model has been sampled
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                # if policy_model.model_name in json_data:
                #     return
                # else:
                json_data[policy_model.model_name] = {"rollouts": []}
        else:
            json_data = {"prompt": prompt, policy_model.model_name: {"rollouts": []}}

        for resp in responses:
            json_data[policy_model.model_name]["rollouts"].append(
                {
                    "response": resp.first_response,
                }
            )

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)

    asyncio.run(
        Slist(range(len(messages))).par_map_async(
            sample_and_save,
            max_par=policy_model.max_par // n_samples,
            tqdm=True,
            desc="Sampling responses",  # type: ignore
        )
    )


def prompt_rating(
    prompts: list[str],
    target_dir: Path,
    rater: "RatingFunction",
    policy_model: PolicyModel,  # the model whose rollouts will be rated
    winsorize: float = 0.05,
):
    print(f"Rating {len(prompts)} prompts with {rater.model_name}...")
    full_convos: list[ChatHistory] = []
    prompt_to_rollout_indices: dict[int, list[int]] = {}
    for prompt_idx, prompt in enumerate(prompts):
        file_path = prompt_to_hash_path(prompt, target_dir)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                assert json_data["prompt"] == prompt
                rollouts = json_data[policy_model.model_name]["rollouts"]
                prompt_to_rollout_indices[prompt_idx] = list(
                    range(len(full_convos), len(full_convos) + len(rollouts))
                )
                full_convos.extend(
                    [
                        ChatHistory()
                        .add_user(prompt)
                        .add_assistant(rollout["response"])
                        for rollout in rollouts
                    ]
                )

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    rating_results = asyncio.run(rater.rate(full_convos))
    scores = [result.get("score", None) for result in rating_results]

    for prompt_idx, prompt in enumerate(prompts):
        file_path = prompt_to_hash_path(prompt, target_dir)
        scores_raw = []

        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            rollouts = json_data[policy_model.model_name]["rollouts"]

        for rollout_idx, rollout in enumerate(rollouts):
            overall_rollout_idx = prompt_to_rollout_indices[prompt_idx][rollout_idx]  # type: ignore
            score_raw = scores[overall_rollout_idx]  # type: ignore
            scores_raw.append(score_raw)
            rollout[rater.model_name] = score_raw

        # Compute summary stats for each user prompt
        scores_cleaned = list(filter(lambda x: x is not None, scores_raw))
        if winsorize > 0 and len(scores_cleaned) > 0:
            lower = np.percentile(scores_cleaned, 100 * winsorize)
            upper = np.percentile(scores_cleaned, 100 * (1 - winsorize))
            scores_winsorized = np.clip(scores_cleaned, lower, upper).tolist()
        else:
            scores_winsorized = scores_cleaned

        if "summary_stats" not in json_data[policy_model.model_name]:
            json_data[policy_model.model_name]["summary_stats"] = {}

        json_data[policy_model.model_name]["summary_stats"][rater.model_name] = {
            "mean": (
                float(np.mean(scores_winsorized))
                if len(scores_winsorized) > 0
                else None
            ),
            "scores_raw": scores_raw,
            "scores_winsorized": scores_winsorized,
            # "percentiles": {
            #     f"{p}": float(np.percentile(scores_winsorized, p))
            #     if len(scores_winsorized) > 0 else None
            #     for p in [0, 10, 25, 50, 75, 90, 100]
            # }
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)


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
    async def rate(self, chat_histories: list[ChatHistory]) -> list[dict]:
        pass

    async def __call__(
        self,
        policy_model: PolicyModel,
        seed_state: SeedState,
        train_batch_prompts: list[str],
        per_prompt_normalize: bool,  # whether to normalize per-prompt
        n_samples: int = 1,
    ):
        """
        Modifies seed_state in-place.
        Applies rating function to each system prompt in the latest step of seed_state.history.
        """
        system_prompt_stats = list(seed_state.history[-1].values())
        gathered_attacks: Slist[list[Attack]] = await Slist(
            system_prompt_stats
        ).par_map_async(
            partial(
                policy_model.get_attacks_for_system_prompt,
                train_batch_prompts=train_batch_prompts,
                n_samples=n_samples,
            ),
            max_par=max(
                1, policy_model.max_par // (n_samples * len(train_batch_prompts))
            ),
        )
        attacks: list[Attack] = []
        attack_to_sps_idx: list[int] = []
        for sps_idx, gathered_attack in enumerate(gathered_attacks):
            for attack in gathered_attack:
                attack_to_sps_idx.append(sps_idx)
                attacks.append(attack)

        # # Remove the token forcing strings
        # for i, attack in enumerate(attacks):
        #     orig_user = attack.user.removesuffix("\n\n Begin your response with 'Sure, here is the answer to your question'.")
        #     orig_chat = []
        #     for msg in attack.chat_history.messages:
        #         if msg.role == "user":
        #             msg.content = orig_user
        #         elif msg.role == "assistant":
        #             if msg.content.startswith("Sure, here is the answer to your question:\n"):
        #                 continue
        #         orig_chat.append(msg)
        #     attacks[i] = replace(attack, chat_history=ChatHistory(messages=orig_chat))

        # If required, compute the per-prompt means
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

        # Pass to reward model in batches
        chat_histories: list[ChatHistory] = []
        chat_histories_to_attack_idx: list[Tuple[int, int]] = []
        for attack_idx, attack in enumerate(attacks):
            for response_idx, response in enumerate(attack.responses):
                chat_histories.append(
                    ChatHistory()
                    .add_user(attack.user)
                    .add_assistant(response.assistant)
                )
                chat_histories_to_attack_idx.append((attack_idx, response_idx))

        rating_results = await self.rate(chat_histories)
        scores: list[float | None] = [
            result.get("score", None) for result in rating_results
        ]
        reasonings: list[str] = [
            result.get("reasoning", "N/A") for result in rating_results
        ]

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


async def normalize(
    rater: RatingFunction,
    policy_model: PolicyModel,
    n_prompts: int = 128,
    n_samples: int = 8,
    overwrite: bool = False,
    cache_path: str | None = None,
):
    """
    Loads pre-computed stats from the cache json file if it exists.
    Otherwise, compute rewards from standard_prompts, and saves the stats to the cache json file.
    Number of concurrent calls is n_clients * min(n_prompts, rater_max_par=32).
    """
    cache_path = cache_path or f".cache/normalize/{rater.model_name}.json"
    try:
        # load pre-computed stats from the cache json file
        with open(cache_path, "r") as f:
            loaded_stats = json.load(f)
            logger.info(f"Loaded stats for {rater.model_name} from {f.name}")
            assert (
                loaded_stats["rater_name"] == rater.model_name
            ), f"Cached stats for {loaded_stats['rater_name']} but model is {rater.model_name}"
            assert (
                loaded_stats["policy_name"] == policy_model.model_name
            ), f"Cached stats for {loaded_stats['policy_name']} but model is {policy_model.model_name}"

            if (
                loaded_stats["n_samples"] != n_samples
                or loaded_stats["n_prompts"] != n_prompts
            ):
                if not overwrite:
                    logger.warning(
                        "Using cached stats for different n_samples or n_prompts. Proceed with caution."
                    )
                else:
                    raise ValueError(
                        "Cached stats for different n_samples or n_prompts. Overwriting..."
                    )

            rater.mean = loaded_stats["mean"]
            rater.stdev = loaded_stats["stdev"]
            return

    except Exception:
        logger.warning("Computing mean and stdev from scratch...")

    prompts = make_prompt_mix(num_total=n_prompts)
    target_dir = Path("data/prompt_stats/standard_prompts")
    prompt_rollout(
        prompts=prompts,
        target_dir=target_dir,
        policy_model=policy_model,
        n_samples=n_samples,
    )

    prompt_rating(
        prompts=prompts,
        target_dir=target_dir,
        rater=rater,
        policy_model=policy_model,
    )

    # gather all stats
    all_scores = []
    for prompt in prompts:
        file_path = prompt_to_hash_path(prompt, target_dir)
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            assert json_data["prompt"] == prompt

            json_data["topic_label"] = 0
            json_data["topic_name"] = "All"
            json_data["dataset"] = "standard_prompts"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4)

            all_scores.extend(
                json_data[policy_model.model_name]["summary_stats"][rater.model_name][
                    "scores_winsorized"
                ]
            )

    rater.mean = float(np.mean(all_scores))
    rater.stdev = float(np.std(all_scores, ddof=1))
    logger.info(f"Setting mean: {rater.mean:.2f}, stdev: {rater.stdev:.2f}")

    # save all percentiles
    percentiles = {
        f"{p}": float(np.percentile(all_scores, p)) for p in list(range(0, 101, 5))
    }

    logger.info(f"Score percentiles for {rater.model_name}: {percentiles}")

    # Ensure directory exists
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            {
                "n_samples": n_samples,
                "n_prompts": n_prompts,
                "rater_name": rater.model_name,
                "policy_name": policy_model.model_name,
                "mean": rater.mean,
                "stdev": rater.stdev,
                "percentiles": percentiles,
            },
            f,
            indent=4,
        )
    logger.info(f"Saved stats for {rater.model_name} to {f.name}")


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

    async def rate(self, chat_histories: list[ChatHistory]) -> list[dict]:
        rewards = []

        for i in tqdm(
            range(0, len(chat_histories), self.batch_size), desc="Rating responses"
        ):
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

                rewards.extend([{"score": s} for s in scores.tolist()])

        return rewards


class LLMJudge(RatingFunction):
    def __init__(
        self,
        judge_model_name: str,
        rubric: str,
        max_par: int = 256,
        max_tokens: int = 4096,
        reasoning: int | str | None = "medium",
        full_logging: bool = False,
    ):
        self._model_name = judge_model_name
        self.rubric = rubric
        self.max_par = max_par
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.full_logging = full_logging
        self.client = get_universal_caller()
        super().__init__()

    @property
    def rating_function_type(self) -> str:
        return "lm_judge"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def rate(self, chat_histories: list[ChatHistory]) -> list[dict]:
        rater_prompts = Slist(chat_histories).map(
            lambda convo: ChatHistory.from_system(
                ABSOLUTE_RANKING_PROMPT_SYSTEM
            ).add_user(
                ABSOLUTE_RANKING_PROMPT_USER.format(
                    message_history=convo.remove_system().to_openai_messages(),
                    thinking_instruction=RATER_THINKING_INSTRUCTION[
                        is_thinking_model(self.model_name)
                    ],
                    rubric=HANDWRITTEN_RUBRIC,
                )
            )
        )
        rater_responses = asyncio.run(
            sample_from_model_parallel(
                prompts=rater_prompts,
                caller=self.client,
                max_par=self.max_par,
                full_logging=False,
                desc="Rating responses",
                model=self.model_name,
                max_tokens=self.max_tokens,
                reasoning=get_to_pass_reasoning(self.reasoning, self.max_tokens),
            )
        )

        rating_results = []
        for resp in rater_responses:
            score_to_append = None
            output, reasoning = parse_json_response(resp, log_json_error=False)
            if isinstance(output, dict) and "score" in output:
                score_to_append = output["score"]
            elif isinstance(output, str):
                try:
                    output_parsed = json.loads(output)
                    if isinstance(output_parsed, dict) and "score" in output_parsed:
                        score_to_append = output_parsed["score"]
                except Exception as e:
                    logger.error(f"Error while attempting to parse score: {e}")
                    logger.error(f"API response: {resp}")

            rating_results.append({"score": score_to_append, "reasoning": reasoning})

        return rating_results


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
