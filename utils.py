import re
import gc
import json
import time
import random
import datetime
import asyncio
import functools
import inspect

from json_repair import repair_json
from loguru import logger
from typing import Any, Tuple, Optional, Awaitable, Literal, overload
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
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.trainer_utils import set_seed as hf_set_seed

from caller import Response
from state import Rollout, Score

Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

def load_model(
    model_name: str,
    model_type: Literal["reward", "generation"],
    device: str = "auto",
    attn_implementation: str = "sdpa",
) -> tuple[PreTrainedModel, Tokenizer]:

    if model_type == "reward":
        print(f"Loading reward model {model_name} with device {device}...")
        load_kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
            "num_labels": 1,
        }
        if device != "auto":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **load_kwargs
            )
            model.to(device)
        else:
            load_kwargs["device_map"] = "auto"
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **load_kwargs
            )

    elif model_type == "generation":
        print(f"Loading generation model {model_name} with device {device}...")
        load_kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
        }
        if device != "auto":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )
            model.to(device)  # type: ignore
        else:
            load_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )

    print("Model loaded. Set to eval mode and disabled gradients.")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("No pad token found, setting to eos token")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        print("No pad token id found, setting to eos token id")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token

    return model, tokenizer


def set_seed_all(seed: int):
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA RNG
    hf_set_seed(seed)


# # Deprecated since we're using loguru

# def logging_setup(filename: str|None, level: int = logging.WARNING, console: bool=False):
#     """
#     Set up logging to file / stdout.
#     """
#     formatter = logging.Formatter(
#         "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(name)s - %(message)s"
#     )

#     logger = logging.getLogger()
#     logger.setLevel(level)

#     # Add file handler if needed
#     if filename is not None:
#         filepath = Path(filename).resolve()
#         has_file_handler = any(
#             isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == filepath
#             for h in logger.handlers
#         )
#         if not has_file_handler:
#             filepath.parent.mkdir(parents=True, exist_ok=True)
#             file_handler = logging.FileHandler(filename, mode="w")
#             file_handler.setFormatter(formatter)
#             logger.addHandler(file_handler)

#     # Add console handler if needed
#     if console:
#         has_console_handler = any(
#             isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
#             for h in logger.handlers
#         )
#         if not has_console_handler:
#             console_handler = logging.StreamHandler()
#             console_handler.setFormatter(formatter)
#             logger.addHandler(console_handler)

#     # Silence httpx INFO logs (LLM APIs)
#     logging.getLogger("httpx").setLevel(logging.WARNING)


def remove_outliers(data: list[float], z_score: float|None = None, clip_percent: float|None = None) -> list[float]:
    """z_score precedes clip_percent"""

    data_np = np.array(data)
    if z_score is not None:
        mean, std = np.mean(data_np), np.std(data_np)
        if std == 0:
            # All values are the same
            return list(data_np)
        mask = np.abs(data_np - mean) < z_score * std
        return list(data_np[mask])
    elif clip_percent is not None:
        low = np.percentile(data_np, clip_percent * 100)
        high = np.percentile(data_np, (1 - clip_percent) * 100)
        mask = (data_np >= low) & (data_np <= high)
        return list(data_np[mask])
    else:
        raise ValueError("Either z_score or clip_percent must be specified")


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
    resp: Response, log_json_error: bool = True, marker: str = "json",
) -> Tuple[Any, str | None]:
    raw_text = resp.first_response
    if raw_text is None:
        logger.warning(f"Response is None: {resp}")
        return None, None

    output, reasoning = None, None
    if resp.reasoning_content is not None:
        reasoning = resp.reasoning_content

    try:
        if raw_text.strip().startswith(f"```{marker}"):
            json_str = raw_text.split(f"```{marker}", 1)[1].rsplit("```", 1)[0].strip()
            if reasoning is None:
                reasoning = raw_text.rsplit(f"```{marker}", 1)[0].strip()
        else:
            json_str = raw_text.strip()

        # Repair and parse
        output = json.loads(repair_json(json_str))

    except Exception as e:
        output = raw_text.strip()
        if reasoning is None:
            reasoning = raw_text.strip()
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
    model_name: str = "unknown",
) -> dict[str, dict[str, list[Rollout]]]:
    """
    Convert JSON rollout data to Rollout objects.
    Expects JSON with structure: attribute -> user_prompt -> list of rollout dicts.
    Each rollout dict should have 'response' and optionally 'student_score'/'score' fields.
    """
    rollouts = {}
    for attribute, attribute_data in json_data.items():
        rollouts[attribute] = {}
        for user, user_data in attribute_data.items():
            rollouts[attribute][user] = []
            for rollout in user_data:
                # Handle both old format (score) and new format (student_score)
                if "student_score" in rollout:
                    student_score = Score(
                        score=rollout["student_score"].get("score"),
                        raw_score=rollout["student_score"].get("raw_score"),
                        reasoning=rollout["student_score"].get("reasoning"),
                        model_name=rollout["student_score"].get("model_name", model_name),
                    )
                elif "score" in rollout:
                    # Legacy format
                    student_score = Score(
                        score=rollout["score"],
                        raw_score=rollout["score"],
                        reasoning=None,
                        model_name=model_name,
                    )
                else:
                    student_score = Score(score=None, raw_score=None, reasoning=None, model_name=model_name)

                teacher_score = None
                if "teacher_score" in rollout and rollout["teacher_score"] is not None:
                    teacher_score = Score(
                        score=rollout["teacher_score"].get("score"),
                        raw_score=rollout["teacher_score"].get("raw_score"),
                        reasoning=rollout["teacher_score"].get("reasoning"),
                        model_name=rollout["teacher_score"].get("model_name", model_name),
                    )

                rollouts[attribute][user].append(Rollout(
                    response=rollout["response"],
                    student_score=student_score,
                    teacher_score=teacher_score,
                ))
    return rollouts

# %% borrowed from https://github.com/GraySwanAI/nanoGCG/blob/d4a20a8b4a12c3ee814a77a61423b7bdcd151f82/nanogcg/utils.py
# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(starting_batch_size: int, function: callable = None):
    """
    A basic decorator that will try to execute `function`. 
    If it fails from exceptions related to out-of-memory or CUDNN, 
    the batch size is cut in half and passed to `function`.

    NOTE: `function` must take in a `batch_size` parameter as its first argument.

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator

# %%
class ClusterModel:
    def __init__(
        self,
        embedding_model_name: str,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 5,
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
