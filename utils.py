# %%
import json
import time
import logging
import datetime
import functools
import re
import os
from slist import Slist
from IPython.core.getipython import get_ipython
from pathlib import Path
import hashlib
import pickle
import asyncio
from typing import Any, Tuple

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from llm_types import ChatHistory
from state import Attack
from client import (
    OpenaiResponse,
    is_thinking_model,
    get_universal_caller,
    sample_from_model_parallel,
)
from defaults import *

logger = logging.getLogger(__name__)

# TODO: add attention_mask every time we call the model

# %%
REWARD_MODELS = {
    "skywork": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    "skywork-v2": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "tulu3": "allenai/Llama-3.1-Tulu-3-8B-RM",
    "llamarb2": "allenai/Llama-3.1-8B-Instruct-RM-RB2",
    "skywork-v2-qwen-8b": "Skywork/Skywork-Reward-V2-Qwen3-8B",
}

POLICY_MODELS = {
    "llama-3.1-8b-instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "tulu3-sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "qwen-3-8b-instruct": "Qwen/Qwen3-8B",
}


_reward_model = None
_policy_model = None


def load_model(model_name: str, use_flash: bool = False, device: str = "auto"):
    global _reward_model, _policy_model
    if model_name in REWARD_MODELS:
        model_name_hf = REWARD_MODELS[model_name]
        print(f"Loading reward model {model_name_hf}...")
        if _reward_model is None:
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": device,
                "num_labels": 1,
            }
            if use_flash:
                load_kwargs["attn_implementation"] = "flash_attention_2"

            _reward_model = AutoModelForSequenceClassification.from_pretrained(
                model_name_hf, **load_kwargs
            )
            model = _reward_model
        else:
            model = _reward_model

    elif model_name in POLICY_MODELS:
        model_name_hf = POLICY_MODELS[model_name]
        print(f"Loading policy model {model_name_hf}...")
        if _policy_model is None:
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": device,
            }
            if use_flash:
                load_kwargs["attn_implementation"] = "flash_attention_2"

            _policy_model = AutoModelForCausalLM.from_pretrained(
                model_name_hf, **load_kwargs
            )
            model = _policy_model
        else:
            model = _policy_model

    print("Model loaded. Set to eval mode and disabled gradients.")
    model.eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
    if tokenizer.pad_token is None:
        print("No pad token found, setting to eos token")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        print("No pad token id found, setting to eos token id")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token

    return model, tokenizer


def is_local_model(model_name: str) -> bool:
    if model_name in POLICY_MODELS or model_name in REWARD_MODELS:
        return True
    else:
        return False


def parse_json_response(
    resp: OpenaiResponse, log_json_error: bool = True
) -> Tuple[Any, str]:
    """
    Returns a tuple (json array, reasoning)
    """
    output, reasoning = None, "N/A"
    try:
        raw_text = resp.first_response
        if is_thinking_model(resp.model):
            reasoning = resp.reasoning_content
        try:
            output = json.loads(
                raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
            )
            if not is_thinking_model(resp.model):
                reasoning = raw_text.rsplit("```json", 1)[0].strip()

        except Exception as e:
            output = raw_text
            if not is_thinking_model(resp.model):
                reasoning = raw_text
            if log_json_error:
                logger.error(f"Response JSON parse error: {e}")
                logger.error(f"API response: {resp}")
    except Exception as e:
        logger.error(f"Response does not have text: {e}")
        logger.error(f"API response: {resp}")

    return output, reasoning


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def count_words(text):
    # Split on whitespace and common delimiters
    # This regex splits on spaces, newlines, and common code delimiters
    words = re.findall(r"\S+", text)
    return len(words)


async def time_operation(operation_name, coroutine):
    """Time an async operation and print the duration."""
    start_time = time.time()
    result = await coroutine
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"  {operation_name} completed in {duration:.2f} seconds")
    return result


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (assume not notebook)
    except NameError:
        return False  # Probably standard Python interpreter


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


def get_to_pass_reasoning(reasoning: int | str | None, max_tokens: int) -> dict | None:
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


# %%
# WARNING - These are no longer used and not guaranteed to work!


def custom_cache(cache_dir: str = ".cache"):
    """
    Decorator that caches function results to disk based on arguments hash.
    """

    def decorator(func):
        cache_path = Path(cache_dir) / func.__name__
        cache_path.mkdir(parents=True, exist_ok=True)

        def get_cache_key(args, kwargs):
            try:
                # Try to serialize for hashing
                cache_data = (args, sorted(kwargs.items()))
                return hashlib.md5(pickle.dumps(cache_data)).hexdigest()
            except (TypeError, pickle.PicklingError) as e:
                # Fall back to string representation if pickling fails
                logger.warning(f"Cache key generation using str fallback due to: {e}")
                cache_str = f"{func.__name__}_{args}_{sorted(kwargs.items())}"
                return hashlib.md5(cache_str.encode()).hexdigest()

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache_key = get_cache_key(args, kwargs)
                cache_file = cache_path / f"{cache_key}.pkl"

                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        logger.info(f"Loading cached result from {cache_file}")
                        return pickle.load(f)

                logger.info(f"Computing {func.__name__} for {cache_key}")
                result = await func(*args, **kwargs)
                # Write to temp file first to avoid corruption
                temp_file = cache_file.with_suffix(".tmp")
                with open(temp_file, "wb") as f:
                    pickle.dump(result, f)
                temp_file.rename(cache_file)

                return result

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache_key = get_cache_key(args, kwargs)
                cache_file = cache_path / f"{cache_key}.pkl"

                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        logger.info(f"Loading cached result from {cache_file}")
                        return pickle.load(f)

                logger.info(f"Computing {func.__name__} for {cache_key}")
                result = func(*args, **kwargs)
                # Write to temp file first to avoid corruption
                temp_file = cache_file.with_suffix(".tmp")
                with open(temp_file, "wb") as f:
                    pickle.dump(result, f)
                temp_file.rename(cache_file)

                return result

            return sync_wrapper

    return decorator


def setup_prompt_logger(log_path: str | None, to_stdout: bool = False):
    logger = logging.getLogger("prompt_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't bubble to root

    # Ensure a single console handler exists
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        stream = (
            __import__("sys").stdout if to_stdout else None
        )  # default stderr if None
        ch = logging.StreamHandler(stream)
        ch.setLevel(logging.INFO)

        class PromptConsoleFormatter(logging.Formatter):
            def format(self, record):
                # If the message is a list/tuple, print each item on its own line
                if isinstance(record.msg, (list, tuple)):
                    lines = [str(item) for item in record.msg]
                    return "[PROMPT] " + "\n[PROMPT] ".join(lines)
                return "[PROMPT] " + record.getMessage()

        ch.setFormatter(PromptConsoleFormatter())
        logger.addHandler(ch)

    # Optionally add/update a JSONL file handler for later analysis
    if log_path:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        except Exception:
            pass

        absolute_path = os.path.abspath(log_path)
        has_file_handler = False
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if os.path.abspath(h.baseFilename) == absolute_path:
                        has_file_handler = True
                        break
                except Exception:
                    continue

        if not has_file_handler:
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setLevel(logging.INFO)

            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    # Preserve lists so the JSON file has an array; otherwise store a string
                    prompts_value = (
                        [str(item) for item in record.msg]
                        if isinstance(record.msg, (list, tuple))
                        else record.getMessage()
                    )
                    payload = {
                        "prompts": prompts_value,
                        "meta": getattr(record, "meta", {}),
                    }
                    return json.dumps(payload, indent=4, ensure_ascii=False)

            fh.setFormatter(JsonFormatter())
            logger.addHandler(fh)

    return logger


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
                classifier_score = rating.raw_score
            elif rating.rater.rating_function_type in ["elo_lm", "absolute_lm"]:
                lm_judge_score = rating.raw_score

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
                classifier_score = rating.raw_score
            elif rating.rater.rating_function_type in ["elo_lm", "absolute_lm"]:
                lm_judge_score = rating.raw_score

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
