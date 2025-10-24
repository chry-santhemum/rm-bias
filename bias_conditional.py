"""
Async pipeline for evaluating bias through system prompting. 
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
from models import PolicyModel
from reward_model import RewardModel
from state import Rollout
from bias_baseline import PromptResult, baseline_rating_worker
from bias_rewrite import BatchSentinel


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionalInput:
    system: str  # attribute in question
    user: str
    batch_id: str



async def conditional_policy_worker(
    policy_model: PolicyModel,
    in_queue: asyncio.Queue,
    out_queue: asyncio.Queue,
    worker_id: int,
):
    """
    Output:
    - If successful, puts a PromptResult
    - If failed, no output
    """
    while True:
        task_input = await in_queue.get()
        logger.info(
            f"[conditional_policy_worker {worker_id}] Popped 1 task. In queue size: {in_queue.qsize()}"
        )

        if task_input is None:  # Stop sentinel
            in_queue.task_done()
            break

        if isinstance(task_input, BatchSentinel):
            await out_queue.put(task_input)
            in_queue.task_done()
            continue

        result = await policy_model.sample([ChatHistory.from_system(task_input.system).add_user(task_input.user)])

        if result[0] is None or result[0].get_first("assistant") is None:
            logger.warning(
                f"[conditional_policy_worker {worker_id}] Failed to sample for user prompt:\n<user_prompt>\n{task_input}\n</user_prompt>"
            )
            in_queue.task_done()
            continue

        await out_queue.put(
            PromptResult(
                system=task_input.system,
                user=task_input.user,
                assistant=result[0].get_first("assistant"),  # type: ignore
            )
        )
        in_queue.task_done()
        logger.info(f"[conditional_policy_worker {worker_id}] Pushed 1 task.")



async def conditional_rating_worker(
    reward_model: RewardModel,
    in_queue: asyncio.Queue[PromptResult],
    all_results: list[PromptResult],
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

                if item is None:  # Sentinel value to signal stop
                    in_queue.task_done()
                    done = True
                    break

                if isinstance(item, BatchSentinel):
                    logger.info(
                        f"[conditional_rating_worker] Batch sentinel received for batch id: {item.batch_id}."
                    )
                    in_queue.task_done()
                    batches_progress_left[item.batch_id] += item.expected_items
                    continue

                if item.score is not None:  # already rated
                    all_results.append(item)
                    batches_progress_left[item.batch_id] -= 1
                    in_queue.task_done()
                    continue

                batch.append(item)
                batches_progress_left[item.batch_id] -= 1
                in_queue.task_done()
                
        except asyncio.TimeoutError:
            pass  # Not an error, just means we process the current incomplete batch

        if len(batch) == 0 and not done:
            # nothing to process yet; continue collecting
            continue

        logger.info(
            f"[conditional_rating_worker] Processing batch of size {len(batch)}..."
        )
        chat_histories = [
            ChatHistory.from_user(result.user).add_assistant(result.assistant)
            for result in batch
        ]

        if len(chat_histories) > 0:
            reward_scores = await loop.run_in_executor(
                executor, reward_model.rate, chat_histories
            )
            for result, reward_score in zip(batch, reward_scores):
                all_results.append(replace(result, score=reward_score.score))

        logger.info(
            f"[conditional_rating_worker] Fnished processing batch of size {len(batch)}."
        )

        completed_batch_ids = []
        for batch_id, progress_left in list(batches_progress_left.items()):
            logger.info(
                f"[conditional_rating_worker] Batch {batch_id} has {progress_left} items left to process."
            )
            if progress_left == 0:
                all_futures[batch_id].set_result(all_results[batch_id])
                completed_batch_ids.append(batch_id)

        for batch_id in completed_batch_ids:
            del all_results[batch_id]
            del batches_progress_left[batch_id]

        if done:
            logger.info("[conditional_rating_worker] Final item processed. Shutting down.")
            break



def organize_conditional_results(
    all_results: list[PromptResult],
    save_dir: Path | None = None,
) -> dict[str, dict[str, list[Rollout]]]:

    conditional_results = defaultdict(dict)

    for result in all_results:
        attribute_results = conditional_results[result.system]
        if result.user not in attribute_results:
            attribute_results[result.user] = []

        attribute_results[result.user].append(
            Rollout(
                response = result.assistant,
                score = result.score
            )
        )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "conditional_results.json", "w", encoding="utf-8") as f:
            json_data = {
                k: {k2: [asdict(r) for r in v2] for k2, v2 in v.items()}
                for k, v in conditional_results.items()
            }
            json.dump(json_data, f, indent=4)

        scores = {}
        mean_scores = {}

        for attribute, attribute_results in conditional_results.items():
            attribute_scores = {}
            all_scores = []
            for user, v in attribute_results.items():
                attribute_scores[user] = [r.score for r in v]
                all_scores.extend([r.score for r in v])
            scores[attribute] = attribute_scores
            mean_scores[attribute] = np.mean(all_scores).item()

        with open(save_dir / "conditional_scores.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)
        with open(
            save_dir / "conditional_scores_mean.json", "w", encoding="utf-8"
        ) as f:
            json.dump(mean_scores, f, indent=4)

    return dict(conditional_results)