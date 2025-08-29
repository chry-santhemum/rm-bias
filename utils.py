# %%
import json
import logging
from dotenv import load_dotenv

load_dotenv()

import datetime
import re
from dataclasses import dataclass
from IPython import get_ipython
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

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
