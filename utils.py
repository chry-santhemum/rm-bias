# %%
import re
import json
import time
import random
import logging
import datetime
from typing import Any, Tuple
from collections import defaultdict
from IPython.core.getipython import get_ipython

import numpy as np
import torch
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.trainer_utils import set_seed as hf_set_seed

from caller.client import (
    OpenaiResponse,
    is_thinking_model,
)

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


def set_seed_all(seed: int):
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA RNG
    hf_set_seed(seed)


def parse_json_response(
    resp: OpenaiResponse, log_json_error: bool = True
) -> Tuple[Any, str | None]:
    """
    Returns a tuple (parsed output, reasoning).

    If output is a valid json array, it is parsed and returned.
    Else, if output exists, it is returned as is.
    """
    output, reasoning = None, None
    try:
        raw_text = resp.first_response
        if is_thinking_model(resp.model):
            reasoning = resp.reasoning_content
        try:
            if "```json" in raw_text:
                output = json.loads(
                    raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
                )
                if not is_thinking_model(resp.model):
                    reasoning = raw_text.rsplit("```json", 1)[0].strip()
            else:
                output = json.loads(raw_text)

        except Exception as e:
            output = raw_text
            if not is_thinking_model(resp.model):
                reasoning = raw_text
            if log_json_error:
                logger.error(f"Response JSON parse error: {e}")
                logger.error(f"API response: {resp}")
                logger.error(f"Full traceback:", exc_info=True)
    except Exception as e:
        logger.error(f"Response does not have text: {e}")
        logger.error(f"API response: {resp}")
        logger.error(f"Full traceback:", exc_info=True)

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


def get_to_pass_reasoning(reasoning: int | str | None, max_tokens: int|None) -> dict | None:
    if isinstance(reasoning, str):
        if max_tokens is None:
            to_pass_reasoning = {"effort": reasoning}
        else:
            to_pass_reasoning = {
                "max_tokens": get_tokens_from_effort(reasoning, max_tokens),
                "effort": reasoning,
            }
    elif isinstance(reasoning, int):
        if max_tokens is None:
            to_pass_reasoning = {"max_tokens": reasoning}
        else:
            to_pass_reasoning = {
                "max_tokens": reasoning,
                "effort": get_effort_from_tokens(reasoning, max_tokens),
            }
    else:
        to_pass_reasoning = None
    return to_pass_reasoning


# %%

class ClusterModel:
    def __init__(
        self,
        embedding_model_name: str,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 8,
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.umap_model = UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            min_dist=0.0,
            metric="cosine",
            low_memory=False,
        )

    def embed(self, inputs: list[str]) -> np.ndarray:
        return self.embedding_model.encode(inputs)

    def reduce_embed(self, inputs: list[str]) -> np.ndarray:
        """Embed then do dimensionality reduction"""

        embeddings: np.ndarray = self.embed(inputs)
        return self.umap_model.fit_transform(embeddings)  # type: ignore

    def cluster(
        self, inputs: list[str], n_clusters: int
    ) -> Tuple[list[str], list[int]]:
        reduced_embeddings = self.reduce_embed(inputs)

        # # log the pairwise distance matrix
        # logger.info(
        #     f"Pairwise distance matrix:\n"
        #     f"{pairwise_distances(reduced_embeddings, metric='cosine')}"
        # )

        kmeans = KMeans(
            n_clusters=min(len(inputs), n_clusters), random_state=10086, n_init="auto"
        )
        kmeans.fit(reduced_embeddings)

        closest_point_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, reduced_embeddings
        )

        sorted_indices = sorted(closest_point_indices)
        selected = [inputs[i] for i in sorted_indices]

        return selected, sorted_indices

    def cluster_dbscan(
        self,
        inputs: list[str],
        dbscan_eps: float,
    ) -> Tuple[dict[int, list[str]], dict[int, list[int]]]:
        embeddings = self.embed(inputs)

        # log the pairwise distance matrix
        logger.info(
            f"Pairwise distance matrix:\n"
            f"{pairwise_distances(embeddings, metric='cosine')}"
        )

        dbscan = DBSCAN(
            eps=dbscan_eps, min_samples=2 * self.umap_n_components, metric="cosine"
        )
        dbscan.fit(embeddings)

        niches = defaultdict(list)
        indices = defaultdict(list)
        for i, label in enumerate(dbscan.labels_):
            niches[label].append(inputs[i])
            indices[label].append(i)

        logger.info(
            "Niches:\n"
            + "\n".join(
                [
                    f"Niche {label}:\n{"\n".join(members)}"
                    for label, members in niches.items()
                ]
            )
        )

        return niches, indices
