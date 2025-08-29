# %%
import sys

sys.path.append("/workspace/pm-bias")

import dotenv

dotenv.load_dotenv("/workspace/pm-bias/.env")

import time
import asyncio
import nest_asyncio

nest_asyncio.apply()

import json
from typing import Any
import matplotlib.pyplot as plt

import math
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from datasets import Dataset
import functools
import hashlib
import pickle
from pathlib import Path
from slist import Slist
from absl import logging

from state import SeedState, Attack
from client import ChatHistory, get_universal_caller, sample_from_model_parallel
from llm_types import is_local_model, is_thinking_model
from prompts import *
from standard_prompts import make_prompt_mix
from utils import load_model, REWARD_MODELS, is_notebook


# %%
def get_effort_from_tokens(reasoning_tokens: int, max_tokens: int) -> str:
    ratio: float = reasoning_tokens / max_tokens
    assert 0 <= ratio <= 1, f"Invalid reasoning to max_tokens ratio: {ratio}"
    if ratio < 0.3:
        return "low"
    elif ratio < 0.7:
        return "medium"
    else:
        return "high"

def get_tokens_from_effort(effort: str, max_tokens: int) -> int:
    match effort:
        case "low":
            return int(max_tokens * 0.2)
        case "medium":
            return int(max_tokens * 0.5)
        case "high":
            return int(max_tokens * 0.8)
        case _:
            raise ValueError(f"Invalid effort: {effort}")

def get_to_pass_reasoning(reasoning: int | str | None, max_tokens: int) -> dict|None:
    if isinstance(reasoning, str):
        to_pass_reasoning = {
            "max_tokens": get_tokens_from_effort(reasoning, max_tokens),
            "effort": reasoning,
        }
    elif isinstance(reasoning, int):
        to_pass_reasoning = {
            "max_tokens": reasoning,
            "effort": get_effort_from_tokens(reasoning, max_tokens),
        }
    else:
        to_pass_reasoning = None
    return to_pass_reasoning


def adversariality(
    z_score_1: float,
    z_score_2: float,
) -> float:
    """
    Motivation: lines of (x - c)(y - c) = 0.25
    """

    # if z_score_1 > 0 and z_score_2 < 0:
    #     return z_score_1 - z_score_2
    # else:
    #     return min(z_score_1, -z_score_2)

    return 0.5 * (z_score_1 - z_score_2 - math.sqrt((z_score_1 + z_score_2)**2 + 1))



def custom_cache(cache_dir: str, model_arg_names: list[str] | None = None):
    """Decorator that caches function results to disk based on arguments hash.

    Args:
        cache_dir: Directory to store cache files
        model_arg_names: List of argument names that contain RewardModel instances
                        which should be replaced with model name for hashing
    """
    model_arg_names = model_arg_names or []

    def decorator(func):
        cache_path = Path(cache_dir) / func.__name__
        cache_path.mkdir(parents=True, exist_ok=True)

        def get_cache_key(args, kwargs):
            # Replace RewardModel instances with their names for hashing
            cache_kwargs = {}
            for key, value in kwargs.items():
                if key in model_arg_names:
                    # Handle tuples/lists of models
                    if isinstance(value, (tuple, list)):
                        cache_value = tuple(
                            v.model_name if hasattr(v, "model_name") else v
                            for v in value
                        )
                    elif hasattr(value, "model_name"):
                        cache_value = value.model_name
                    else:
                        cache_value = value
                    cache_kwargs[key] = cache_value
                else:
                    cache_kwargs[key] = value

            try:
                # Try to serialize for hashing
                cache_data = (args, sorted(cache_kwargs.items()))
                return hashlib.md5(
                    pickle.dumps(cache_data, protocol=pickle.HIGHEST_PROTOCOL)
                ).hexdigest()
            except (TypeError, pickle.PicklingError) as e:
                # Fall back to string representation if pickling fails
                logging.warning(f"Cache key generation using str fallback due to: {e}")
                cache_str = f"{func.__name__}_{args}_{sorted(cache_kwargs.items())}"
                return hashlib.md5(cache_str.encode()).hexdigest()

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache_key = get_cache_key(args, kwargs)
                cache_file = cache_path / f"{cache_key}.pkl"

                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        logging.info(f"Loading cached result from {cache_file}")
                        return pickle.load(f)

                logging.info(f"Computing {func.__name__} for {cache_key}")
                result = await func(*args, **kwargs)
                # Write to temp file first to avoid corruption
                temp_file = cache_file.with_suffix(".tmp")
                with open(temp_file, "wb") as f:
                    pickle.dump(result, f)
                temp_file.rename(cache_file)

                return result

            async_wrapper.clear_cache = lambda: [
                f.unlink() for f in cache_path.glob("*.pkl")
            ]
            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache_key = get_cache_key(args, kwargs)
                cache_file = cache_path / f"{cache_key}.pkl"

                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        logging.info(f"Loading cached result from {cache_file}")
                        return pickle.load(f)

                logging.info(f"Computing {func.__name__} for {cache_key}")
                result = func(*args, **kwargs)
                # Write to temp file first to avoid corruption
                temp_file = cache_file.with_suffix(".tmp")
                with open(temp_file, "wb") as f:
                    pickle.dump(result, f)
                temp_file.rename(cache_file)

                return result

            sync_wrapper.clear_cache = lambda: [
                f.unlink() for f in cache_path.glob("*.pkl")
            ]
            return sync_wrapper

    return decorator


# %%
class RewardModel:
    """
    Wrapper around reward models; __init__ kwargs are passed to load_model (e.g. device)
    """

    def __init__(
        self,
        model_name: str,
        stats: dict[str, float] | None = None,
        **kwargs,
    ):
        assert model_name in REWARD_MODELS, f"Model {model_name} not local!!!"
        self.model_name = model_name
        self.model, self.tokenizer_info = load_model(model_name, **kwargs)

        self.device = self.model.device
        self.tokenizer = self.tokenizer_info.tokenizer

        # load stats
        if stats is None:
            logging.info(
                f"Using default stats for {self.model_name}. Run .normalize() if you want to compute the normalized rewards."
            )
            self.mean = 0.0
            self.stdev = 1.0
        else:
            logging.info(f"Using provided stats for {self.model_name}")
            self.mean = stats["mean"]
            self.stdev = stats["stdev"]

    def __call__(
        self,
        inputs: list[list[dict]] | list[ChatHistory] | Tensor,
        normalized: bool = True,
    ) -> Tensor:
        """
        Call the reward model and tokenize if needed.
        """
        if isinstance(inputs, list):
            # inputs are messages
            if isinstance(inputs[0], ChatHistory):
                inputs = [input.to_openai_messages() for input in inputs]
            input_ids = self.tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                padding_side="right",
            ).to(self.device)
        elif isinstance(inputs, Tensor):
            input_ids = inputs.to(self.device)

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # logging.info(f"Input IDs first example: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

        with torch.no_grad():
            scores = self.model(
                input_ids=input_ids, attention_mask=attn_mask
            ).logits.squeeze(-1)

        if normalized:
            scores = (scores - self.mean) / self.stdev

        return scores

    def normalize(
        self,
        policy_model_name: str,
        n_prompts: int = 2048,
        n_samples: int = 16,
        n_clients: int = 16,
        overwrite: bool = False,
        cache_path: str | None = None,
    ):
        """
        Loads pre-computed stats from the cache json file if it exists.
        Otherwise, compute rewards from standard_prompts, then set self.mean and self.stdev
        Number of concurrent calls is n_clients * min(n_prompts, rater_max_par=32)
        and saves the stats to the cache json file.
        """
        cache_path = cache_path or f".rwdcache/per_model/{self.model_name}.json"
        try:
            # load pre-computed stats from the cache json file
            with open(cache_path, "r") as f:
                loaded_stats = json.load(f)
                logging.info(f"Loaded stats for {self.model_name} from {f.name}")
                assert (
                    loaded_stats["reward_model_name"] == self.model_name
                ), f"Cached stats for {loaded_stats['reward_model_name']} but model is {self.model_name}"
                assert (
                    loaded_stats["policy_model_name"] == policy_model_name
                ), f"Cached stats for {loaded_stats['policy_model_name']} but model is {policy_model_name}"

                if (
                    loaded_stats["n_samples"] != n_samples
                    or loaded_stats["n_prompts"] != n_prompts
                ):
                    if not overwrite:
                        logging.warning(
                            "Using cached stats for different n_samples or n_prompts. Proceed with caution. Use overwrite=True to overwrite."
                        )
                    else:
                        raise ValueError(
                            f"Cached stats for different n_samples or n_prompts. Overwriting..."
                        )

                self.mean = loaded_stats["mean"]
                self.stdev = loaded_stats["stdev"]
                return
        except Exception as e:
            logging.error(f"Computing from scratch.")
            logging.error(f"Error: {e}")

        prompts = make_prompt_mix(num_total=n_prompts)
        stats = asyncio.run(
            Slist(prompts["prompt"]).par_map_async(
                func=lambda prompt: per_prompt_reward_stats(
                    prompt=prompt,
                    raters=(self,),  # Pass the RewardModel instance
                    policy_model_name=policy_model_name,
                    N=n_samples,
                ),
                max_par=n_clients,
                tqdm=True,
            )
        )

        # gather all stats
        all_rewards_raw = []
        for stat in stats:
            all_rewards_raw.extend(stat[self.model_name]["rewards_raw"])
        all_rewards = list(filter(lambda x: x is not None, all_rewards_raw))

        self.mean = float(np.mean(all_rewards))
        self.stdev = float(np.std(all_rewards, ddof=1))
        logging.info(f"Setting mean: {self.mean:.2f}, stdev: {self.stdev:.2f}")

        # save all percentiles
        percentiles = {
            f"{p}": float(np.percentile(all_rewards, p)) for p in list(range(0, 101, 5))
        }

        logging.info(f"Reward percentiles for {self.model_name}: {percentiles}")

        # Ensure directory exists
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "n_samples": n_samples,
                    "n_prompts": n_prompts,
                    "reward_model_name": self.model_name,
                    "policy_model_name": policy_model_name,
                    "mean": self.mean,
                    "stdev": self.stdev,
                    "percentiles": percentiles,
                },
                f,
                indent=4,
            )

        logging.info(f"Saved stats for {self.model_name} to {f.name}")


class PolicyModel:
    """
    Wrapper around text generation models. __init__ kwargs are passed to load_model (e.g. device)
    """

    def __init__(self, model_name: str, **kwargs):
        assert model_name in MODEL_NAMES, f"Model {model_name} not local!!!"

        self.model_name = model_name
        self.model, self.tokenizer_info = load_model(model_name, **kwargs)
        self.device = self.model.device
        self.tokenizer = self.tokenizer_info.tokenizer

    def __call__(
        self, inputs: list[list[dict]] | list[ChatHistory] | Tensor, **kwargs
    ) -> list[str]:
        """
        Model generation wrapper. kwargs are passed to model.generate
        """
        if isinstance(inputs, list):
            # inputs are messages
            if isinstance(inputs[0], ChatHistory):
                inputs = [input.to_openai_messages() for input in inputs]
            input_ids = self.tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                padding_side="left",  # generation pads on the left
            ).to(self.device)
        elif isinstance(inputs, Tensor):
            input_ids = inputs.to(self.device)

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # logging.info(f"Input IDs first example: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                **kwargs,
            )

        # logging.info(f"Outputs first example: {self.tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=False)}")
        return [
            self.tokenizer.decode(
                gen_id[input_ids.shape[1] :], skip_special_tokens=True
            )
            for gen_id in gen_ids
        ]


@custom_cache(cache_dir=".rwdcache", model_arg_names=["raters"])
async def per_prompt_reward_stats(
    prompt: str,
    raters: tuple[str | RewardModel, ...],  # local RM instance or API model name
    policy_model_name: str,  # currently API models
    N: int = 16,
    temperature: float = 1.0,
    policy_max_tokens: int = 512,
    policy_max_par: int = 64,  # max parallel sampling from policy
    rater_max_par: int = 32,  # batch size / max parallel for rater
    winsorize_pct: float = 0.0,
    policy_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict:
    """
    Take N samples from the policy model and compute reward statistics.

    In most cases, use only one rater to ease caching.
    """
    caller = get_universal_caller()
    message = ChatHistory.from_system(policy_system_prompt).add_user(prompt)
    policy_responses: Slist[ChatHistory] = await sample_from_model_parallel(
        prompts=[message for _ in range(N)],
        caller=caller,
        max_par=policy_max_par,
        temperature=temperature,
        model=policy_model_name,
        max_tokens=policy_max_tokens,
    )
    # print("\n".join([resp.first_response for resp in policy_responses]))

    # Compose full conversations with assistant responses
    full_convos = policy_responses.map(
        lambda resp: message.add_assistant(resp.first_response)
    )

    all_outputs = {}

    for rater in raters:
        # Handle both RewardModel instances and string model names
        if isinstance(rater, RewardModel):
            reward_model = rater
            rater_name = rater.model_name
            rewards = []
            for i in range(0, N, rater_max_par):
                batch = full_convos[i : i + rater_max_par]
                rewards.extend(reward_model(batch, normalized=False).tolist())

        elif is_local_model(rater):
            reward_model = RewardModel(rater)
            rater_name = rater
            rewards = []
            for i in range(0, N, rater_max_par):
                batch = full_convos[i : i + rater_max_par]
                rewards.extend(reward_model(batch, normalized=False).tolist())

        else:
            rater_name = rater
            rater_prompts = full_convos.map(
                lambda convo: ChatHistory.from_system(
                    ABSOLUTE_RANKING_PROMPT_SYSTEM
                ).add_user(
                    ABSOLUTE_RANKING_PROMPT_USER.format(
                        message_history=convo.remove_history().to_openai_messages(),
                        thinking_instruction=RATER_THINKING_INSTRUCTION[
                            is_thinking_model(rater_name)
                        ],
                        rubric=HANDWRITTEN_RUBRIC,
                    )
                )
            )
            rater_responses = await sample_from_model_parallel(
                prompts=rater_prompts,
                caller=caller,
                max_par=rater_max_par,
                model=rater,
                max_tokens=2048,
                reasoning={"max_tokens": 2000, "effort": "high"},
            )

            rewards = []
            for i, resp in enumerate(rater_responses):
                try:
                    raw_text = resp.first_response
                    try:
                        block = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
                    except Exception:
                        block = raw_text
                    parsed_resp = json.loads(block)
                    rewards.append(parsed_resp["score"])
                except Exception as e:
                    logging.error(
                        f"Failed to parse rater response: {resp.first_response}"
                    )
                    logging.error(f"Error: {e}")
                    rewards.append(None)

        rewards_cleaned = np.array([r for r in rewards if r is not None], dtype=float)

        if is_notebook():
            draw_reward_histogram(rewards_cleaned, title=rater_name)
        logging.info(
            f"Reward percentiles for {rater_name}: {np.percentile(rewards_cleaned, [0, 10, 25, 50, 75, 90, 100])}"
        )

        # Winsorize
        if winsorize_pct > 0:
            lower = np.percentile(rewards_cleaned, 100 * winsorize_pct)
            upper = np.percentile(rewards_cleaned, 100 * (1 - winsorize_pct))
            rewards_winsorized = np.clip(rewards_cleaned, lower, upper)
        else:
            rewards_winsorized = rewards_cleaned

        output = {
            "mean": float(np.mean(rewards_winsorized)),
            "stdev": float(np.std(rewards_winsorized, ddof=1)),
            "N": int(N),
            "rewards_raw": rewards,
            "rewards_winsorized": rewards_winsorized.tolist(),
        }

        all_outputs[rater_name] = output
    return all_outputs


def draw_reward_histogram(rewards: np.ndarray, title: str):
    """
    Draw a histogram of the rewards and mark the 5th, 25th, 50th, 75th, and 95th percentiles.
    """
    if len(rewards) == 0:
        print("No rewards to plot.")
        return

    percentiles = [5, 25, 50, 75, 95]
    perc_values = np.percentile(rewards, percentiles)

    plt.figure(figsize=(8, 4))
    plt.hist(rewards, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    for p, v in zip(percentiles, perc_values):
        plt.axvline(v, color="r", linestyle="--", linewidth=1)
        plt.text(
            v,
            plt.ylim()[1] * 0.95,
            f"{p}%",
            rotation=90,
            color="r",
            va="top",
            ha="right",
            fontsize=8,
        )
    plt.title(title)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def seed_states_to_dataframe(seed_states: list[SeedState]) -> pd.DataFrame:
    """
    Create a DataFrame from a list of seed states where each row represents an attack.

    Args:
        seed_states: List of seed states to convert

    Returns:
        DataFrame with columns: seed, attack_index, prompt, response, and rating columns
    """
    rows = []

    for seed_state in seed_states:
        seed = seed_state.seed

        for attack_index, attack in enumerate(seed_state.attacks):
            row = {
                "seed": seed,
                "iteration": attack_index,
                "response": attack.response,
            }

            if attack.ratings:
                for rating in attack.ratings:
                    row[f"{rating.rater.model_name}_rating"] = rating.score
                    for key, value in rating.aux_info.items():
                        row[f"{rating.rater.model_name}_{key}"] = value

            if attack.aux_info:
                for key, value in attack.aux_info.items():
                    row[key] = value

            rows.append(row)

    return pd.DataFrame(rows)


async def time_operation(operation_name, coroutine):
    """Time an async operation and print the duration."""
    start_time = time.time()
    result = await coroutine
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"  {operation_name} completed in {duration:.2f} seconds")
    return result


def pareto_sort(points: dict[Any, tuple[float, float]], top_k: int | None) -> list[Any]:
    """
    Sort points by distance to closest point in the Pareto frontier.
    If top_k is None, return only the points on the Pareto frontier.
    """
    if not points or (top_k is not None and top_k <= 0):
        return []

    indices = list(points.keys())
    pareto = []
    for i in indices:
        xi, yi = points[i]
        dominated = False
        for j in indices:
            if i == j:
                continue
            xj, yj = points[j]
            if xj >= xi and yj >= yi and (xj > xi or yj > yi):
                dominated = True
                break
        if not dominated:
            pareto.append(i)

    # For each point, compute distance to closest Pareto frontier point
    def dist_to_pareto(idx):
        x, y = points[idx]
        return (
            min(
                ((x - points[p][0]) ** 2 + (y - points[p][1]) ** 2) ** 0.5
                for p in pareto
            )
            if pareto
            else 0.0
        )

    # Sort by distance to Pareto frontier (ascending), then by sum of coordinates (descending)
    sorted_indices = sorted(
        indices,
        key=lambda idx: (dist_to_pareto(idx), -(points[idx][0] + points[idx][1])),
    )

    if top_k is None:
        return pareto
    elif len(sorted_indices) <= top_k:
        return sorted_indices
    else:
        return sorted_indices[:top_k]


def pareto_get_attack_indices(
    attacks: list[Attack], top_k: int | None = 10
) -> list[int]:
    """
    Choose the attacks to include in the attacker context using Pareto optimization.

    Args:
        attacks: List of attacks to select from
        top_k: Number of attacks to select (None = all pareto optimal)
    """
    rating_points = {}

    for i, attack in enumerate(attacks):
        if not attack.ratings:
            ## not rated yet
            continue

        # TODO: smarter selection by clustering similar responses
        # Scaling here is eyeballed obviously (lm_judge is from 0 to 100)
        classifier_score, lm_judge_score = 0.0, 100.0
        for rating in attack.ratings:
            if rating.rater.rating_function_type == "classifier":
                classifier_score = rating.score
            elif rating.rater.rating_function_type in ["elo_lm", "absolute_lm"]:
                lm_judge_score = rating.score

        # Want to minimize lm_judge_score and maximize classifier_score
        rating_points[i] = (classifier_score, -lm_judge_score / 30)

    selected_indices = pareto_sort(rating_points, top_k=top_k)
    return selected_indices


def weighted_get_attack_indices(
    attacks: list[Attack],
    rho: float,
    top_k: int = 10,
    sample: bool = True,
) -> list[int]:
    """
    Select attacks based on weighted combination of scores.

    Args:
        attacks: List of attacks to select from
        rho: Weight for linear combination (0 = pure lm_judge, 1 = pure classifier)
        top_k: Number of attacks to select
        sample: If True, use probabilistic sampling; if False, take top-k deterministically
    """
    scores = {}

    for i, attack in enumerate(attacks):
        if not attack.ratings:
            continue

        classifier_score, lm_judge_score = 0.0, 10.0
        for rating in attack.ratings:
            if rating.rater.rating_function_type == "classifier":
                classifier_score = rating.score
            elif rating.rater.rating_function_type in ["elo_lm", "absolute_lm"]:
                lm_judge_score = rating.score

        # Higher score is better: high classifier, low lm_judge
        scores[i] = rho * classifier_score - (1 - rho) * lm_judge_score

    if not scores:
        return []

    indices = list(scores.keys())
    values = list(scores.values())

    if not sample:
        # Deterministic: take top-k by score
        sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)
        return sorted_indices[: min(top_k, len(sorted_indices))]

    # Probabilistic sampling
    values_array = np.array(values)

    # Shift to make all positive (softmax-like)
    exp_values = np.exp(values_array - np.max(values_array))
    probs = exp_values / exp_values.sum()

    # Sample without replacement
    num_samples = min(top_k, len(indices))
    selected = np.random.choice(indices, size=num_samples, replace=False, p=probs)

    return selected.tolist()
# %%
