"""
Async pipeline for evaluating bias through rewrite. 
"""

# %%
import patches
import logging
import asyncio
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, replace, asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from caller import ChatHistory
from models import RewriteModel
from reward_model import RewardModel, RatingResult
from state import Rollout


logger = logging.getLogger(__name__)


# %%


@dataclass(frozen=True)
class BatchSentinel:
    batch_id: str
    expected_items: int


@dataclass(frozen=True)
class RewriteInput:
    system: str  # attribute in question
    user: str
    original_assistant: str
    presence: bool
    batch_id: str


@dataclass(frozen=True)
class RewriteResult:
    system: str
    user: str
    original_assistant: str
    rewritten_assistant: str
    presence: bool
    batch_id: str
    score: float | None = None


# Thread pool is supplied by the runner (instance-level), not module-level

# %%


async def rewrite_worker(
    rewrite_model: RewriteModel,
    in_queue: asyncio.Queue,
    out_queue: asyncio.Queue,
    worker_id: int,
):
    while True:
        task_input = await in_queue.get()
        logger.info(
            f"[rewrite_worker {worker_id}] Popped 1 task. In queue size: {in_queue.qsize()}"
        )

        if task_input is None:  # Sentinel value to signal stop
            in_queue.task_done()
            break

        if isinstance(task_input, BatchSentinel):
            await out_queue.put(task_input)
            in_queue.task_done()
            continue

        rewrite_result = await rewrite_model.rewrite(
            attribute=task_input.system,
            original_chat=ChatHistory.from_user(task_input.user).add_assistant(
                task_input.original_assistant
            ),
            presence=task_input.presence,
            n_samples=1,
        )

        if rewrite_result[0] is None:
            logger.warning(
                f"[rewrite_worker {worker_id}] Failed to rewrite:\n<begin_user_prompt>\n{task_input.user}\n<end_user_prompt>"
            )
            # Put a placeholder result
            await out_queue.put(RewriteResult(
                system=task_input.system,
                user=task_input.user,
                original_assistant=task_input.original_assistant,
                rewritten_assistant="",
                presence=task_input.presence,
                batch_id=task_input.batch_id,
            ))
        else:
            await out_queue.put(RewriteResult(
                system=task_input.system,
                user=task_input.user,
                original_assistant=task_input.original_assistant,
                rewritten_assistant=rewrite_result[0],
                presence=task_input.presence,
                batch_id=task_input.batch_id,
            ))
        logger.info(f"[rewrite_worker {worker_id}] Pushed 1 task.")
        in_queue.task_done()


async def rewrite_rating_worker(
    reward_model: RewardModel,
    in_queue: asyncio.Queue[RewriteResult],
    all_results: dict[str, list[RewriteResult]],
    all_futures: dict[str, asyncio.Future],
    executor: ThreadPoolExecutor,
):
    loop = asyncio.get_running_loop()
    done = False
    batches_progress_left = defaultdict(int)

    while True:
        batch = []
        try:
            while len(batch) < reward_model.batch_size:
                item = await asyncio.wait_for(in_queue.get(), timeout=3.0)
                logger.info(
                    f"[rewrite_rating_worker] Popped 1 task. In queue size: {in_queue.qsize()}"
                )

                if item is None:  # Sentinel value to signal stop
                    logger.info(
                        f"[rewrite_rating_worker] Sentinal value received. Shutting down."
                    )
                    in_queue.task_done()
                    done = True
                    break

                if isinstance(item, BatchSentinel):
                    logger.info(
                        f"[rewrite_rating_worker] Batch sentinel received for batch id: {item.batch_id}."
                    )
                    in_queue.task_done()
                    batches_progress_left[item.batch_id] += item.expected_items
                    continue

                if item.score is not None:  # already rated
                    logger.info(
                        f"[rewrite_rating_worker] Item already rated. Adding to results."
                    )
                    batches_progress_left[item.batch_id] -= 1
                    all_results[item.batch_id].append(item)
                    in_queue.task_done()
                    continue

                if item.rewritten_assistant != "":
                    batch.append(item)
                batches_progress_left[item.batch_id] -= 1
                logger.info(
                    f"[rewrite_rating_worker] Received valid item. New batch size: {len(batch)}."
                )
                in_queue.task_done()

        except asyncio.TimeoutError:
            pass  # Not an error, just means we process the current incomplete batch

        if len(batch) == 0 and not done:
            # nothing ready yet; continue collecting
            continue

        logger.info(f"[rewrite_rating_worker] Processing batch of size {len(batch)}...")
        chat_histories: list[ChatHistory|None] = [
            ChatHistory.from_user(rewrite_result.user).add_assistant(
                rewrite_result.rewritten_assistant
            )
            for rewrite_result in batch
        ]

        if len(chat_histories) > 0:
            reward_scores: list[RatingResult] = await loop.run_in_executor(
                executor, reward_model.rate, chat_histories
            )
            for rewrite_result, reward in zip(batch, reward_scores):
                all_results[rewrite_result.batch_id].append(
                    replace(rewrite_result, score=reward.score)
                )
        logger.info(
            f"[rewrite_rating_worker] Finished processing batch of size {len(batch)}."
        )

        completed_batch_ids = []
        for batch_id, progress_left in list(batches_progress_left.items()):
            logger.info(
                f"[rewrite_rating_worker] Batch {batch_id} has {progress_left} items left to process."
            )
            if progress_left == 0:
                all_futures[batch_id].set_result(all_results[batch_id])
                completed_batch_ids.append(batch_id)

        for batch_id in completed_batch_ids:
            del all_results[batch_id]
            del batches_progress_left[batch_id]

        if done:
            logger.info("[rewrite_rating_worker] Final item processed. Shutting down.")
            break


def organize_rewrite_results(
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

    # clear any blanks
    for attribute, attribute_results in rewrite_results.items():
        for user, user_results in attribute_results.items():
            user_results = [r for r in user_results if r.response != "" and r.score is not None]

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
                all_scores.extend([r.score for r in v if r.score is not None])
            scores[attribute] = attribute_scores
            mean_scores[attribute] = np.mean(all_scores).item()

        with open(save_dir / "rewrite_plus_scores.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)
        with open(
            save_dir / "rewrite_plus_scores_mean.json", "w", encoding="utf-8"
        ) as f:
            json.dump(mean_scores, f, indent=4)

    return dict(rewrite_results)
