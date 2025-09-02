"""
Standard prompts for estimating mean and stdev of rater models.

Datasets used:
- Alpaca GPT4
- Ultrafeedback_binarized
- Skywork Preference V0.2
- LMArena
"""

import random
import json

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers.trainer_utils import set_seed as hf_set_seed

def set_seed_all(seed: int):
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA RNG
    hf_set_seed(seed)

def make_prompt_mix(num_total: int = 2048, seed: int = 10086):
    """
    This should be a determinstic function.
    """
    num_per_dataset = num_total // 4
    set_seed_all(seed)

    alpaca_gpt4 = load_dataset("vicgalle/alpaca-gpt4", split="train")
    print("Alpaca GPT4 before filtering: ", len(alpaca_gpt4))
    alpaca_gpt4_prompts = alpaca_gpt4.filter(
        lambda item: isinstance(item["output"], str) and len(item["output"]) <= 2000,
        num_proc=16,  # type: ignore
    )
    print("Alpaca GPT4 after filtering: ", len(alpaca_gpt4_prompts))
    # Sort for deterministic ordering before sampling
    alpaca_gpt4_prompts = alpaca_gpt4_prompts.sort("instruction")
    alpaca_gpt4_prompts = alpaca_gpt4_prompts.select(
        random.sample(range(len(alpaca_gpt4_prompts)), num_per_dataset)  # type: ignore
    ).map(
        lambda item: {
            "prompt": item["instruction"]
            + ("\n\n" + item["input"] if item["input"] else ""),
        },
        num_proc=1,
        remove_columns=alpaca_gpt4.column_names,  # type: ignore
    )

    ultrafeedback = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized", split="train_prefs"
    )
    print("Ultrafeedback before filtering: ", len(ultrafeedback))
    ultrafeedback_prompts = ultrafeedback.filter(
        lambda item: len(item["chosen"][1]["content"]) <= 2000,
        num_proc=16,  # type: ignore
    )
    print("Ultrafeedback after filtering: ", len(ultrafeedback_prompts))
    # Sort for deterministic ordering before sampling
    ultrafeedback_prompts = ultrafeedback_prompts.sort("prompt")
    ultrafeedback_prompts = ultrafeedback_prompts.select(
        random.sample(range(len(ultrafeedback_prompts)), num_per_dataset)  # type: ignore
    ).map(
        lambda item: {
            "prompt": item["prompt"],
        },
        num_proc=1,
        remove_columns=ultrafeedback.column_names,  # type: ignore
    )

    skywork = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2", split="train")
    print("Skywork before filtering: ", len(skywork))
    skywork_prompts = skywork.filter(
        lambda item: len(item["chosen"]) == 2
        and len(item["chosen"][1]["content"]) <= 2000,
        num_proc=16,  # type: ignore
    )
    print("Skywork after filtering: ", len(skywork_prompts))
    # Sort by first message content for deterministic ordering
    skywork_prompts = skywork_prompts.map(
        lambda item: {"sort_key": item["chosen"][0]["content"]},
        num_proc=1,  # type: ignore
    ).sort("sort_key")
    skywork_prompts = skywork_prompts.select(
        random.sample(range(len(skywork_prompts)), num_per_dataset)  # type: ignore
    ).map(
        lambda item: {
            "prompt": item["chosen"][0]["content"],
        },
        num_proc=1,
        remove_columns=skywork.column_names,  # type: ignore
    )

    lmarena = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
    print("LMArena before filtering: ", len(lmarena))
    lmarena_prompts = lmarena.filter(
        lambda item: len(json.loads(item["prompt"])) == 1
        and len(str(json.loads(item["response_a"])[0])) <= 2000,
        num_proc=16,  # type: ignore
    )
    print("LMArena after filtering: ", len(lmarena_prompts))
    # Sort for deterministic ordering before sampling
    lmarena_prompts = lmarena_prompts.sort("prompt")
    lmarena_prompts = lmarena_prompts.select(
        random.sample(range(len(lmarena_prompts)), num_per_dataset)  # type: ignore
    ).map(
        lambda item: {
            "prompt": json.loads(item["prompt"])[0],
        },
        num_proc=1,
        remove_columns=lmarena.column_names,  # type: ignore
    )

    mix_dataset = concatenate_datasets(
        [
            alpaca_gpt4_prompts,
            skywork_prompts,
            lmarena_prompts,
            ultrafeedback_prompts,
        ]
    )

    # Reset seed right before shuffle for determinism
    set_seed_all(seed)
    mix_dataset = mix_dataset.shuffle()

    return mix_dataset
