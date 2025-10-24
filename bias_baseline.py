"""
Async pipeline for generating baseline rollouts and rewards.
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

logger = logging.getLogger(__name__)

# Thread pool is now managed locally within evaluate_baselines()


@dataclass(frozen=True)
class PromptResult:
    system: str
    user: str
    assistant: str
    score: float | None = None
    batch_id: str | None = None


async def baseline_policy_worker(
    policy_model: PolicyModel,
    in_queue: asyncio.Queue[str],
    out_queue: asyncio.Queue[PromptResult],
    worker_id: int,
):
    """
    Output:
    - If successful, puts a PromptResult where the system is empty
    - If failed, no output
    """
    while True:
        task_input = await in_queue.get()
        logger.info(
            f"[baseline_policy_worker {worker_id}] Popped 1 task. In queue size: {in_queue.qsize()}"
        )

        if task_input is None:  # Stop sentinel
            in_queue.task_done()
            break

        result = await policy_model.sample([ChatHistory.from_user(task_input)])

        if result[0] is None or result[0].get_first("assistant") is None:
            logger.warning(
                f"[baseline_policy_worker {worker_id}] Failed to sample for user prompt:\n<user_prompt>\n{task_input}\n</user_prompt>"
            )
            in_queue.task_done()
            continue

        await out_queue.put(
            PromptResult(
                system="",
                user=task_input,
                assistant=result[0].get_first("assistant"),  # type: ignore
            )
        )
        in_queue.task_done()
        logger.info(f"[baseline_policy_worker {worker_id}] Pushed 1 task.")


async def baseline_rating_worker(
    reward_model: RewardModel,
    in_queue: asyncio.Queue[PromptResult],
    all_results: list[PromptResult],
    executor: ThreadPoolExecutor,
):
    loop = asyncio.get_running_loop()
    done = False
    while True:
        batch = []
        try:
            while len(batch) < reward_model.batch_size:
                item = await asyncio.wait_for(in_queue.get(), timeout=3.0)

                if item is None:  # Sentinel value to signal stop
                    in_queue.task_done()
                    done = True
                    break

                if item.score is not None:  # already rated
                    all_results.append(item)
                    in_queue.task_done()
                    continue

                batch.append(item)
                in_queue.task_done()
        except asyncio.TimeoutError:
            pass  # Not an error, just means we process the current incomplete batch

        if len(batch) == 0 and not done:
            # nothing to process yet; continue collecting
            continue

        logger.info(
            f"[baseline_rating_worker] Processing batch of size {len(batch)}..."
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
            f"[baseline_rating_worker] Fnished processing batch of size {len(batch)}."
        )

        if done:
            logger.info("[baseline_rating_worker] Final item processed. Shutting down.")
            break


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


async def evaluate_baselines(
    user_prompts: list[str],
    policy_model: PolicyModel,
    reward_model: RewardModel,
    n_rollouts: int,
    save_dir: Path | None = None,
) -> dict[str, list[Rollout]]:
    queue_a = asyncio.Queue()
    queue_b = asyncio.Queue()
    n_policy_workers = policy_model.max_par
    all_results = []  # where results will appear

    for user in user_prompts:
        for _ in range(n_rollouts):
            await queue_a.put(user)

    policy_worker_tasks = [
        asyncio.create_task(
            baseline_policy_worker(policy_model, queue_a, queue_b, worker_id)
        )
        for worker_id in range(n_policy_workers)
    ]

    # Use a local executor for rating to avoid lingering threads
    with ThreadPoolExecutor(max_workers=1) as executor:
        rating_worker_task = asyncio.create_task(
            baseline_rating_worker(reward_model, queue_b, all_results, executor)
        )

        # send one stop sentinel per policy worker
        for _ in range(n_policy_workers):
            await queue_a.put(None)

        await asyncio.gather(*policy_worker_tasks)
        logger.info("\n--- baseline policy workers finished. ---\n")

        await queue_b.put(None)
        await rating_worker_task
        logger.info("\n--- baseline rating worker finished. ---\n")

    organized_results = organize_baseline_results(all_results, save_dir)

    return organized_results
