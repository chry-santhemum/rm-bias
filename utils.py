# %%
import json
import math
import logging
import datetime
import re
from slist import Slist
from IPython import get_ipython
from pathlib import Path
import hashlib
import pickle
import asyncio
import functools

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from llm_types import ChatHistory
from state import SeedState, Attack
from client import OpenaiResponse, get_universal_caller, sample_from_model_parallel
from standard_prompts import make_prompt_mix

logging.getLogger(__name__)

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

def load_model(model_name: str, use_flash: bool = False, device: str = "auto"):
    if model_name in REWARD_MODELS:
        model_name_hf = REWARD_MODELS[model_name]
        print(f"Loading reward model {model_name_hf}...")
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
            "num_labels": 1,
        }
        if use_flash:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_hf, **load_kwargs
        )

        print("Reward model loaded. Set to eval mode and disabled gradients.")

    elif model_name in POLICY_MODELS:
        model_name_hf = POLICY_MODELS[model_name]
        print(f"Loading policy model {model_name_hf}...")
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
        }
        if use_flash:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(model_name_hf, **load_kwargs)

        print("Policy model loaded. Set to eval mode and disabled gradients.")

    model.eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)

    # Set pad token
    if tokenizer.pad_token is None:
        print("No pad token found, setting to eos token")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        print("No pad token id found, setting to eos token id")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token

    return model, tokenizer


def is_thinking_model(model_name: str) -> bool:
    """
    Whether or not there is an explicit thinking mode for this model.
    """
    THINKING_MODELS = [
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "openai/gpt-5",
        "openai/gpt-5-nano",
        "openai/gpt-5-mini",
        "openai/o3",
        "deepseek/deepseek-r1",
    ]
    return model_name in THINKING_MODELS


def is_local_model(model_name: str) -> bool:
    if model_name in POLICY_MODELS or model_name in REWARD_MODELS:
        return True
    else:
        return False


def parse_json(result: tuple[str, str] | str):
    """
    Parse model responses that are formatted as json.
    """
    try:
        if isinstance(result, tuple):
            result = result[-1]
        json_str = result.split("```json")[1].split("```")[0]
        json_obj = json.loads(json_str)
    except json.JSONDecodeError:
        logging.error(f"Could not parse the following response: {result}")
        return None

    return json_obj


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def count_words(text):
    # Split on whitespace and common delimiters
    # This regex splits on spaces, newlines, and common code delimiters
    words = re.findall(r"\S+", text)
    return len(words)


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


# %%
def custom_cache(cache_dir: str=".cache"):
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
                return hashlib.md5(
                    pickle.dumps(cache_data)
                ).hexdigest()
            except (TypeError, pickle.PicklingError) as e:
                # Fall back to string representation if pickling fails
                logging.warning(f"Cache key generation using str fallback due to: {e}")
                cache_str = f"{func.__name__}_{args}_{sorted(kwargs.items())}"
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


@custom_cache(cache_dir=".cache")
async def per_prompt_stats(
    prompt: str,
    rater_name: str,
    policy_name: str,
    N: int = 16,  # sample 16 by default
    temperature: float = 0.8,
    policy_max_tokens: int = 512,
    policy_max_par: int = 64,  # max parallel sampling from policy
    rater_max_par: int = 32,  # batch size / max parallel for rater
    winsorize: float = 0.05,
    policy_system_prompt: str|None=None,
) -> dict:
    """
    Take N samples from the policy model and compute reward statistics.
    """
    caller = get_universal_caller()
    if policy_system_prompt is None:
        message = ChatHistory().add_user(prompt)
    else:
        message = ChatHistory.from_system(policy_system_prompt).add_user(prompt)

    policy_responses: Slist[OpenaiResponse] = await sample_from_model_parallel(
        prompts=[message for _ in range(N)],
        caller=caller,
        max_par=policy_max_par,
        full_logging=False,
        temperature=temperature,
        model=policy_name,
        max_tokens=policy_max_tokens,
    )
    # print("\n".join([resp.first_response for resp in policy_responses]))

    full_convos = policy_responses.map(
        lambda resp: message.add_assistant(resp.first_response)
    )

    all_outputs = {}

    if is_local_model(rater_name):
        reward_model = RewardModel(rater)
        rater_name = rater
        rewards = []
        for i in range(0, N, rater_max_par):
            batch = full_convos[i : i + rater_max_par]
            rewards.extend(reward_model(batch, normalized=False).tolist())

    else:
        rater_name = rater_name
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
        if winsorize > 0:
            lower = np.percentile(rewards_cleaned, 100 * winsorize)
            upper = np.percentile(rewards_cleaned, 100 * (1 - winsorize))
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
