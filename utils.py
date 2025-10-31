# %%
import re
import json
import time
import random
import logging
import datetime
import asyncio
from pprint import pprint
from typing import Any, Tuple, Optional
from pathlib import Path
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

from caller import Response

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
_embedding_model = None


def load_model(model_name: str, use_flash: bool = False, device: str = "auto"):
    global _reward_model, _policy_model
    if model_name in REWARD_MODELS:
        model_name_hf = REWARD_MODELS[model_name]
        print(f"Loading reward model {model_name_hf}...")
        if _reward_model is None:
            load_kwargs = {
                "dtype": torch.bfloat16,
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
                "dtype": torch.bfloat16,
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


def logging_setup(filename: str, level: int = logging.WARNING):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        filename=filename,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(name)s - %(message)s",
    )


def remove_outliers(data: list[float], z_score: float = 3.0) -> list[float]:
    mean, std = np.mean(data), np.std(data)
    return [x for x in data if np.abs(x - mean) < z_score * std]


async def async_gather(tasks: list, max_parallel: Optional[int] = None):
    if max_parallel is None or max_parallel >= len(tasks):
        return await asyncio.gather(*tasks)
    else:
        semaphore = asyncio.Semaphore(max_parallel)

        async def sem_task(task):
            async with semaphore:
                return await task

        return await asyncio.gather(*(sem_task(task) for task in tasks))


def parse_json_response(
    resp: Response, log_json_error: bool = True
) -> Tuple[Any, str | None]:
    """
    Returns a tuple (parsed output, reasoning).

    If output contains a valid json array, it is parsed and returned.
    Else, if output exists, it is returned as is.
    """
    raw_text = resp.first_response
    if raw_text is None:
        return None, None
    
    output, reasoning = None, None
    if resp.reasoning_content is not None:
        reasoning = resp.reasoning_content
        print("Found reasoning content in response: ", reasoning)
    try:
        if "```json" in raw_text:
            output = json.loads(
                raw_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            )
            if reasoning is None:
                reasoning = raw_text.rsplit("```json", 1)[0].strip()
        else:
            output = json.loads(raw_text)

    except Exception as e:
        output = raw_text
        if reasoning is None:
            reasoning = raw_text
        if log_json_error:
            logger.error(f"Response JSON parse error: {e}")
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


# %%


class ClusterModel:
    def __init__(
        self,
        embedding_model_name: str,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 8,
    ):
        global _embedding_model
        if _embedding_model is None:
            _embedding_model = SentenceTransformer(embedding_model_name)
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
        return _embedding_model.encode(inputs)

    def reduce_embed(self, inputs: list[str]) -> np.ndarray:
        """Embed then do dimensionality reduction"""
        print("Embedding...")
        embeddings: np.ndarray = self.embed(inputs)
        print("Reducing dimensionality...")
        return self.umap_model.fit_transform(embeddings)  # type: ignore

    def cluster(
        self, inputs: list[str], n_clusters: int
    ) -> list[dict[str, Any]]:
        reduced_embeddings = self.reduce_embed(inputs)

        # # log the pairwise distance matrix
        # logger.info(
        #     f"Pairwise distance matrix:\n"
        #     f"{pairwise_distances(reduced_embeddings, metric='cosine')}"
        # )

        kmeans = KMeans(
            n_clusters=min(len(inputs), n_clusters), random_state=10086, n_init="auto"
        )
        print("Fitting KMeans...")
        kmeans.fit(reduced_embeddings)

        closest_point_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, reduced_embeddings
        )
        labels = [int(label) for label in kmeans.labels_.tolist()]

        results = []
        for cluster_idx, center_idx in enumerate(closest_point_indices):
            # print(f"Cluster {cluster_idx}")
            results.append(
                {
                    "cluster_idx": cluster_idx,
                    "center_idx": center_idx,
                    "center_input": inputs[center_idx],
                    "content_indices": []
                }
            )

        for input_idx, label in enumerate(labels):
            results[label]["content_indices"].append(input_idx)
        
        for result in results:
            assert result["center_idx"] in result["content_indices"]

        return results

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
