# %%
import re
import json
import time
import random
import logging
import datetime
import asyncio
from pprint import pprint
from typing import Any, Tuple, Optional, Iterable, Awaitable
from pathlib import Path
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio
from IPython.core.getipython import get_ipython

import numpy as np
import torch
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.trainer_utils import set_seed as hf_set_seed

from caller import Response
from state import Rollout

logger = logging.getLogger(__name__)


# %%
REWARD_MODELS = {
    "skywork": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    "skywork-v2": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "tulu3": "allenai/Llama-3.1-Tulu-3-8B-RM",
    "llamarb2": "allenai/Llama-3.1-8B-Instruct-RM-RB2",
    "skywork-v2-qwen-8b": "Skywork/Skywork-Reward-V2-Qwen3-8B",
    "sleeper": "saepark/sleeper-classicRM",
}

POLICY_MODELS = {
    "llama-3.1-8b-instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "tulu3-sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "qwen-3-8b-instruct": "Qwen/Qwen3-8B",
}

def load_model(model_name: str, device: str = "cuda:0", attn_implementation: str = "sdpa"):
    if model_name in REWARD_MODELS:
        model_name_hf = REWARD_MODELS[model_name]
        print(f"Loading reward model {model_name_hf}...")
        load_kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
            "num_labels": 1,
        }
        if device != "auto":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_hf, **load_kwargs
            )
            model.to(device)
        else:
            load_kwargs["device_map"] = "auto"
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_hf, **load_kwargs
            )

    elif model_name in POLICY_MODELS:
        model_name_hf = POLICY_MODELS[model_name]
        print(f"Loading policy model {model_name_hf}...")
        load_kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
        }
        if device != "auto":
            model = AutoModelForCausalLM.from_pretrained(
                model_name_hf, **load_kwargs
            )
            model.to(device)  # type: ignore
        else:
            load_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(
                model_name_hf, **load_kwargs
            )

    else:
        raise ValueError(f"Model {model_name} not found.")

    print("Model loaded. Set to eval mode and disabled gradients.")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

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


def logging_setup(filename: str|None, level: int = logging.WARNING, console: bool=False):
    """
    Set up logging to file / stdout.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(name)s - %(message)s"
    )

    logger = logging.getLogger()
    logger.setLevel(level)

    # Add file handler if needed
    if filename is not None:
        filepath = Path(filename).resolve()
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == filepath
            for h in logger.handlers
        )
        if not has_file_handler:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(filename, mode="w")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # Add console handler if needed
    if console:
        has_console_handler = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )
        if not has_console_handler:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    # Silence httpx INFO logs (LLM APIs)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def remove_outliers(data: list[float], z_score: float = 3.0) -> list[float]:
    mean, std = np.mean(data), np.std(data)
    return [x for x in data if np.abs(x - mean) < z_score * std]


async def async_gather(tasks: list[Awaitable[Any]], max_parallel: Optional[int] = None):
    if max_parallel is None or max_parallel >= len(tasks):
        return await tqdm_asyncio.gather(*tasks)
    else:
        semaphore = asyncio.Semaphore(max_parallel)

        async def sem_task(task):
            async with semaphore:
                return await task

        return await tqdm_asyncio.gather(*(sem_task(task) for task in tasks))

    
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
        logger.debug("Found reasoning content in response: ", reasoning)
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
            logger.exception(f"Response JSON parse error: {e}")
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


def json_to_rollouts(
    json_data: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, list[Rollout]]]:
    rollouts = {}
    for attribute, attribute_data in json_data.items():
        rollouts[attribute] = {}
        for user, user_data in attribute_data.items():
            rollouts[attribute][user] = [
                Rollout(response=rollout["response"], score=rollout["score"])
                for rollout in user_data
            ]
    return rollouts

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
        """Returns: [n_inputs, d_embed]"""
        return self.embedding_model.encode(inputs)  # type: ignore

    def reduce_embed(self, inputs: list[str]) -> np.ndarray:
        """Embed then do dimensionality reduction"""
        print("Embedding...")
        embeddings: np.ndarray = self.embed(inputs)
        print("Reducing dimensionality...")
        return self.umap_model.fit_transform(embeddings)  # type: ignore

    def cluster(self, inputs: list[str], n_clusters: int) -> list[dict[str, Any]]:
        """
        Returns a list of cluster information.
        """
        reduced_embeddings = self.reduce_embed(inputs)

        kmeans = KMeans(
            n_clusters=min(len(inputs), n_clusters), random_state=10086, n_init="auto"
        )
        print("Fitting KMeans...")
        kmeans.fit(reduced_embeddings)

        labels = [int(label) for label in kmeans.labels_.tolist()]  # type: ignore

        # Group points by cluster
        cluster_points = defaultdict(list)
        for input_idx, label in enumerate(labels):
            cluster_points[label].append(input_idx)

        results = []
        for cluster_idx in range(kmeans.n_clusters):
            content_indices = cluster_points[cluster_idx]

            if not content_indices:
                continue

            # Select representative sample: find the medoid (point that minimizes
            # sum of distances to all other points in the cluster)
            if len(content_indices) == 1:
                center_idx = content_indices[0]
            else:
                # Compute pairwise distances within cluster
                cluster_embeddings = reduced_embeddings[content_indices]
                pairwise_dists = pairwise_distances(cluster_embeddings, metric="cosine")

                # Find point with minimum sum of distances to all other points
                sum_dists = pairwise_dists.sum(axis=1)
                medoid_idx_in_cluster = np.argmin(sum_dists)
                center_idx = content_indices[medoid_idx_in_cluster]

            results.append(
                {
                    "cluster_idx": cluster_idx,
                    "center_idx": center_idx,
                    "center_input": inputs[center_idx],
                    "content_indices": content_indices,
                }
            )

        for result in results:
            assert result["center_idx"] in result["content_indices"]

        return results

    def cluster_dbscan(
        self,
        inputs: list[str],
        dbscan_eps: float,
    ) -> Tuple[dict[int, list[str]], dict[int, list[int]]]:
        embeddings = self.embed(inputs)

        # # log the pairwise distance matrix
        # logger.info(
        #     f"Pairwise distance matrix:\n"
        #     f"{pairwise_distances(embeddings, metric='cosine')}"
        # )

        dbscan = DBSCAN(
            eps=dbscan_eps, min_samples=2 * self.umap_n_components, metric="cosine"
        )
        dbscan.fit(embeddings)

        niches = defaultdict(list)
        indices = defaultdict(list)
        for i, label in enumerate(dbscan.labels_):
            niches[label].append(inputs[i])
            indices[label].append(i)

        logger.debug(
            "Niches:\n"
            + "\n".join(
                [
                    f"Niche {label}:\n{"\n".join(members)}"
                    for label, members in niches.items()
                ]
            )
        )

        return niches, indices
