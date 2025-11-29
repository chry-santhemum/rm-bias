"""
Standard mixture of user prompts.

Prompts are taken from the following datasets:
- vicgalle/alpaca-gpt4
- HuggingFaceH4/ultrafeedback_binarized
- Skywork/Skywork-Reward-Preference-80K-v0.2
- lmarena-ai/arena-human-preference-55k
"""

import random
import json
from datasets import load_dataset, concatenate_datasets, Dataset
from utils import set_seed_all


def load_alpaca_gpt4_prompts(num_samples: int) -> Dataset:
    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
    print("Alpaca before filtering: ", len(dataset))  # type: ignore
    
    dataset = dataset.filter(
        lambda item: isinstance(item["output"], str) and len(item["output"]) <= 2000,
        num_proc=8,  # type: ignore
    )
    print("Alpaca after filtering: ", len(dataset))
    
    dataset = dataset.sort("instruction")
    original_columns = dataset.column_names  # type: ignore
    dataset = dataset.select(  # type: ignore
        random.sample(range(len(dataset)), num_samples)
    ).map(
        lambda item: {
            "prompt": item["instruction"]
            + ("\n\n" + item["input"] if item["input"] else ""),
        },
        num_proc=8,
        remove_columns=original_columns,  # type: ignore
    )
    return dataset  # type: ignore


def load_ultrafeedback_prompts(num_samples: int) -> Dataset:
    dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized", split="train_prefs"
    )
    print("Ultrafeedback before filtering: ", len(dataset))  # type: ignore
    
    dataset = dataset.filter(
        lambda item: len(item["chosen"][1]["content"]) <= 2000,
        num_proc=8,  # type: ignore
    )
    print("Ultrafeedback after filtering: ", len(dataset))
    
    dataset = dataset.sort("prompt")
    original_columns = dataset.column_names  # type: ignore
    dataset = dataset.select(  # type: ignore
        random.sample(range(len(dataset)), num_samples)
    ).map(
        lambda item: {"prompt": item["prompt"]},
        num_proc=8,
        remove_columns=original_columns,  # type: ignore
    )
    return dataset  # type: ignore


def load_skywork_prompts(num_samples: int) -> Dataset:
    dataset = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2", split="train")
    print("Skywork before filtering: ", len(dataset))  # type: ignore
    
    dataset = dataset.filter(
        lambda item: len(item["chosen"]) == 2
        and len(item["chosen"][1]["content"]) <= 2000,
        num_proc=8,  # type: ignore
    )
    print("Skywork after filtering: ", len(dataset))
    
    dataset = dataset.map(
        lambda item: {"sort_key": item["chosen"][0]["content"]},
        num_proc=8,  # type: ignore
    ).sort("sort_key")
    original_columns = dataset.column_names  # type: ignore
    dataset = dataset.select(  # type: ignore
        random.sample(range(len(dataset)), num_samples)
    ).map(
        lambda item: {"prompt": item["chosen"][0]["content"]},
        num_proc=8,
        remove_columns=original_columns,  # type: ignore
    )
    return dataset  # type: ignore


def load_lmarena_prompts(num_samples: int) -> Dataset:
    dataset = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
    print("LMArena before filtering: ", len(dataset))  # type: ignore
    
    dataset = dataset.filter(
        lambda item: len(json.loads(item["prompt"])) == 1
        and len(str(json.loads(item["response_a"])[0])) <= 2000,
        num_proc=8,  # type: ignore
    )
    print("LMArena after filtering: ", len(dataset))
    
    dataset = dataset.sort("prompt")
    original_columns = dataset.column_names  # type: ignore
    dataset = dataset.select(  # type: ignore
        random.sample(range(len(dataset)), num_samples)
    ).map(
        lambda item: {"prompt": json.loads(item["prompt"])[0]},
        num_proc=8,
        remove_columns=original_columns,  # type: ignore
    )
    return dataset  # type: ignore


def make_prompt_mix(num_total: int = 2048, seed: int = 10086) -> list[str]:
    num_per_dataset = num_total // 4
    set_seed_all(seed)

    alpaca_gpt4_prompts = load_alpaca_gpt4_prompts(num_per_dataset)
    ultrafeedback_prompts = load_ultrafeedback_prompts(num_per_dataset)
    skywork_prompts = load_skywork_prompts(num_per_dataset)
    lmarena_prompts = load_lmarena_prompts(num_per_dataset)

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

    return list(mix_dataset["prompt"])
