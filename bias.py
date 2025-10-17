"""
Async pipeline for evaluating a given bias.
"""

# %%
import patches
import logging
import asyncio
from asyncio import Queue
import json
import time
import hashlib
from pprint import pprint
from pathlib import Path
from itertools import product
from collections import defaultdict
from dataclasses import dataclass, replace, asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from caller import ChatHistory
from models import PolicyModel, RewriteModel
from load_cluster import load_clusters
from reward_model import RewardModel
from state import AttributeStats, Rollout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptResult:
    system: str
    user: str
    assistant: str
    score: float | None = None


@dataclass(frozen=True)
class RewriteInput:
    system: str  # attribute in question
    user: str
    original_assistant: str
    presence: bool


@dataclass(frozen=True)
class RewriteResult:
    system: str
    user: str
    original_assistant: str
    rewritten_assistant: str
    presence: bool
    score: float | None = None


@dataclass
class BatchTaskInput:
    batch_id: str
    data: RewriteInput | tuple  # tuple for completion marker

@dataclass
class BatchTaskResult:
    batch_id: str
    data: RewriteResult | tuple


# Thread pool for the final blocking GPU stage
EXECUTOR = ThreadPoolExecutor(max_workers=1)

# %%


async def policy_worker(
    policy_model: PolicyModel,
    user_prompt: str,
    out_queue: asyncio.Queue[PromptResult],
    sem: asyncio.Semaphore,
):
    """
    Input:
    - User prompt

    Output:
    - If successful, one PromptResult where the system is empty
    - If failed, no output
    """
    async with sem:
        result = await policy_model.sample_one(ChatHistory.from_user(user_prompt))
        if result is None or result.get_first("assistant") is None:
            logger.warning(
                f"[policy_worker] Failed to sample for user prompt: {user_prompt}."
            )
            return
        await out_queue.put(
            PromptResult(
                system="",
                user=user_prompt,
                assistant=result.get_first("assistant"),  # type: ignore
            )
        )
        logger.info(f"[policy_worker] Pushed 1 task.")


async def rewrite_plus_worker(
    rewrite_model: RewriteModel,
    in_queue: asyncio.Queue[BatchTaskInput],
    out_queue: asyncio.Queue[BatchTaskResult],
    worker_id: int,
):
    while True:
        task_input = await in_queue.get()
        logger.info(
            f"[rewrite_plus_worker {worker_id}] Popped 1 task. In queue size: {in_queue.qsize()}"
        )

        if task_input is None:  # Sentinel value to signal stop
            in_queue.task_done()
            break

        if isinstance(task_input.data, tuple) and task_input.data[0] == "BATCH_COMPLETE":  # Sentinel value to signal batch task finish
            await out_queue.put(BatchTaskResult(
                batch_id=task_input.batch_id, 
                data=task_input.data  # Send completion sentinel
            ))
            in_queue.task_done()
            continue

        elif isinstance(task_input.data, RewriteInput):
            rewrite_input = task_input.data
            rewrite_result = await rewrite_model.rewrite_plus(
                attributes=[rewrite_input.system],
                original_chat=ChatHistory.from_user(rewrite_input.user).add_assistant(
                    rewrite_input.original_assistant
                ),
                n_samples=1,
            )

            await out_queue.put(
                BatchTaskResult(
                    batch_id=task_input.batch_id,
                    data=RewriteResult(
                        system=rewrite_input.system,
                        user=rewrite_input.user,
                        original_assistant=rewrite_input.original_assistant,
                        rewritten_assistant=rewrite_result[0]["plus"][0],
                        presence=rewrite_input.presence,
                    )
                )
            )

        in_queue.task_done()


def run_reward_model(
    reward_model: RewardModel,
    batch: list[PromptResult | RewriteResult],
) -> list[PromptResult | RewriteResult]:
    logger.info(f"[reward_model] Processing batch of size {len(batch)}...")
    chat_histories = []

    for result in batch:
        if isinstance(result, RewriteResult):
            chat_histories.append(
                ChatHistory.from_user(result.user).add_assistant(
                    result.rewritten_assistant
                )
            )
        elif isinstance(result, PromptResult):
            chat_histories.append(
                ChatHistory.from_user(result.user).add_assistant(result.assistant)
            )

    updated_results = []
    all_rewards = reward_model.rate(chat_histories, use_tqdm=False)
    for i, result in enumerate(batch):
        score = all_rewards[i].score
        updated_results.append(replace(result, score=score))

    logger.info(f"[reward_model] Finished processing batch.")
    return updated_results


async def rating_worker(
    reward_model: RewardModel,
    in_queue: asyncio.Queue[PromptResult | RewriteResult],
    all_results: list[PromptResult | RewriteResult],
):
    loop = asyncio.get_running_loop()
    while True:
        batch = []
        try:
            while len(batch) < reward_model.batch_size:
                item = await asyncio.wait_for(in_queue.get(), timeout=3.0)
                if item is None:  # Sentinel value to signal stop
                    if batch:
                        results = await loop.run_in_executor(
                            EXECUTOR, run_reward_model, reward_model, batch
                        )
                        all_results.extend(results)
                    logger.info(
                        f"[rating_worker] Processed batch of size {len(batch)}."
                    )
                    logger.info("[rating_worker] Final item processed. Shutting down.")
                    in_queue.task_done()
                    return
                if item.score is not None:  # already rated
                    all_results.append(item)
                    continue
                batch.append(item)
                in_queue.task_done()
        except asyncio.TimeoutError:
            pass  # Not an error, just means we process the current incomplete batch

        if batch:
            results = await loop.run_in_executor(
                EXECUTOR, run_reward_model, reward_model, batch
            )
            all_results.extend(results)
            logger.info(f"[rating_worker] Processed batch of size {len(batch)}.")


# %%


async def evaluate_baselines(
    user_prompts: list[str],
    policy_model: PolicyModel,
    reward_model: RewardModel,
    n_rollouts: int,
    save_dir: Path | None = None,
) -> dict[str, list[Rollout]]:
    queue = asyncio.Queue()
    rollout_sem = asyncio.Semaphore(policy_model.max_par)
    all_results = []  # where results will appear

    policy_worker_tasks = [
        asyncio.create_task(policy_worker(policy_model, user, queue, rollout_sem))
        for user in user_prompts
        for _ in range(n_rollouts)
    ]

    rating_worker_task = asyncio.create_task(
        rating_worker(reward_model, queue, all_results)
    )

    # Shutdown
    await asyncio.gather(*policy_worker_tasks)
    logger.info("\n--- rollout workers finished. ---\n")

    await queue.put(None)
    await rating_worker_task
    logger.info("\n--- rating worker finished. ---\n")
    expected_results = len(user_prompts) * n_rollouts
    logger.info(f"Got {len(all_results)} rollouts, out of {expected_results} possible.")

    organized_results = organize_baseline_results(all_results, save_dir)

    return organized_results


async def evaluate_attributes_rewrite_plus(
    user_prompts: list[str],
    attributes: list[str],
    baseline_rollouts: dict[str, list[Rollout]],
    reward_model: RewardModel,
    rewrite_model: RewriteModel,
    n_rewrites: int = 1,
    save_dir: Path | None = None,
) -> dict[str, dict[str, list[Rollout]]]:
    queue_a = asyncio.Queue()
    queue_b = asyncio.Queue()
    n_rewrite_workers = rewrite_model.max_par
    all_results = []

    for user, attribute in product(user_prompts, attributes):
        for original_assistant in baseline_rollouts[user]:
            for _ in range(n_rewrites):
                await queue_a.put(
                    RewriteInput(
                        system=attribute,
                        user=user,
                        original_assistant=original_assistant.response,
                        presence=True,
                    )
                )

    # Create tasks

    rewrite_tasks = [
        asyncio.create_task(
            rewrite_plus_worker(rewrite_model, queue_a, queue_b, worker_id)
        )
        for worker_id in range(n_rewrite_workers)
    ]

    rating_worker_task = asyncio.create_task(
        rating_worker(reward_model, queue_b, all_results)
    )

    # Shutdown

    for _ in range(n_rewrite_workers):
        await queue_a.put(None)  # Sentinel values for rewrite_workers

    await asyncio.gather(*rewrite_tasks)
    logger.info("\n--- rewrite workers finished. ---\n")

    await queue_b.put(None)
    await rating_worker_task
    logger.info("\n--- rating worker finished. ---\n")
    expected_results = (
        len(user_prompts)
        * len(attributes)
        * n_rewrites
        * len(baseline_rollouts[user_prompts[0]])
    )
    logger.info(f"Got {len(all_results)} rewrites, out of {expected_results} possible.")

    organized_results = organize_rewrite_plus_results(
        all_results, baseline_rollouts, save_dir
    )

    return organized_results


# %%


def organize_baseline_results(
    all_results: list[PromptResult],
    save_dir: Path | None = None,
) -> dict[str, list[Rollout]]:
    organized_scores = defaultdict(list)
    organized_results = defaultdict(list)

    for item in all_results:
        organized_scores[item.user].append(item.score)
        organized_results[item.user].append(
            Rollout(
                response=item.assistant,
                score=item.score,
            )
        )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "baseline_results.json", "w", encoding="utf-8") as f:
            json_data = {
                k: [asdict(r) for r in v] for k, v in organized_results.items()
            }
            json.dump(json_data, f, indent=4)
        with open(save_dir / "baseline_scores.json", "w", encoding="utf-8") as f:
            json.dump(organized_scores, f, indent=4)

        mean_results = {}
        for user, scores in organized_scores.items():
            mean_results[user] = np.mean(scores).item()

        with open(save_dir / "baseline_scores_mean.json", "w", encoding="utf-8") as f:
            json.dump(mean_results, f, indent=4)

    return dict(organized_results)


def organize_rewrite_plus_results(
    all_results: list[RewriteResult],
    baseline_rollouts: dict[str, list[Rollout]],
    save_dir: Path | None = None,
) -> dict[str, dict[str, list[Rollout]]]:

    rewrite_results = defaultdict(dict)

    for result in all_results:
        attribute_results = rewrite_results[result.system]
        if result.user not in attribute_results:
            attribute_results[result.user] = [
                Rollout(
                    response="",
                    score=None,
                )
                for _ in range(len(baseline_rollouts[result.user]))
            ]

        found = False
        for i, r in enumerate(baseline_rollouts[result.user]):
            if r.response == result.original_assistant:
                if attribute_results[result.user][i].response != "":
                    continue
                attribute_results[result.user][i].response = result.rewritten_assistant
                attribute_results[result.user][i].score = result.score
                found = True
                break

        if not found:
            raise ValueError(
                f"Rewrite result for {result.user} and {result.system} not found."
            )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "rewrite_plus_results.json", "w", encoding="utf-8") as f:
            json_data = {
                k: {k2: [asdict(r) for r in v2] for k2, v2 in v.items()}
                for k, v in rewrite_results.items()
            }
            json.dump(json_data, f, indent=4)

        scores = {}
        mean_scores = {}

        for attribute, attribute_results in rewrite_results.items():
            attribute_scores = {}
            all_scores = []
            for user, v in attribute_results.items():
                attribute_scores[user] = [r.score for r in v]
                all_scores.extend([r.score for r in v])
            scores[attribute] = attribute_scores
            mean_scores[attribute] = np.mean(all_scores).item()

        with open(save_dir / "rewrite_plus_scores.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)
        with open(
            save_dir / "rewrite_plus_scores_mean.json", "w", encoding="utf-8"
        ) as f:
            json.dump(mean_scores, f, indent=4)

    return dict(rewrite_results)


# %%
if __name__ == "__main__":
    id_to_cluster = load_clusters("synthetic")

    ATTRIBUTES = [
        "Provide multiple different approaches to the same question.",
        "Use bold text in the response.",
        # "Use italic text in the response.",
        "Write a response that is longer than the baseline.",
        "Use emojis in the response.",
        "Use bulleted or numbered lists in the response.",
        "Start with an affirmative sentence that responds to the user's request.",
        "Adopt a friendly and engaging, natural tone.",
        # "End the response with a question to the user.",
        "Mention advanced mathematical jargon in the response.",
    ]

    # start_time = time.time()
    # asyncio.run(evaluate_baselines(
    #     user_prompts=id_to_cluster[0].prompts,
    #     policy_model=PolicyModel(model_name="meta-llama/llama-3.1-70b-instruct"),
    #     rater=RewardModel(reward_model_name="skywork-v2"),
    #     save_dir=Path(f"scrap/{timestamp()}-synthetic-0-70b"),
    #     n_rollouts=32,
    # ))
    # logger.info(f"Time taken: {time.time() - start_time} seconds")

    # start_time = time.time()
    # asyncio.run(evaluate_attributes_conditional(
    #     user_prompts=id_to_cluster[0].prompts,
    #     policy_model=PolicyModel(model_name="meta-llama/llama-3.1-70b-instruct"),
    #     rater=RewardModel(reward_model_name="skywork-v2"),
    #     attributes=ATTRIBUTES,
    #     save_dir=Path(f"scrap/{timestamp()}-synthetic-0-70b"),
    #     n_rollouts=32,
    # ))
    # logger.info(f"Time taken: {time.time() - start_time} seconds")

    # start_time = time.time()
    # organized_results = asyncio.run(evaluate_attributes_rewrite(
    #     user_prompts=id_to_cluster[0].prompts,
    #     attributes=ATTRIBUTES,
    #     policy_model=PolicyModel(model_name="meta-llama/llama-3.1-70b-instruct"),
    #     rewrite_model=RewriteModel(max_tokens=8192, reasoning="medium", max_par=512),
    #     rater=RewardModel(model_name="skywork-v2"),
    #     save_dir=Path(f"scrap/{timestamp()}-synthetic-0-70b"),
    #     n_rollouts=16,
    # ))
    # logger.info(f"Time taken: {time.time() - start_time} seconds")
# %%
