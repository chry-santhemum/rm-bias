"""
Async pipeline for generating baseline rollouts and rewards.
"""

# %%

import logging
import asyncio
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, replace, asdict
from typing import Mapping, Sequence

import numpy as np
from caller import ChatHistory
from models import PolicyModel, RewriteModel
from reward_model import RewardModel
from state import Rollout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptInput:
    system: str | None
    user: str
    batch_id: str


@dataclass(frozen=True)
class PromptResult:
    system: str | None
    user: str
    assistant: str | None
    batch_id: str
    score: float | None = None


@dataclass(frozen=True)
class RewriteInput:
    system: str
    user: str
    original_assistant: str
    presence: bool
    batch_id: str


@dataclass(frozen=True)
class RewriteResult:
    system: str
    user: str
    original_assistant: str
    rewritten_assistant: str | None
    presence: bool
    batch_id: str
    score: float | None = None


@dataclass(frozen=True)
class BatchSentinel:
    batch_id: str
    expected_items: int


async def policy_worker(
    policy_model: PolicyModel,
    in_queue: asyncio.Queue[PromptInput],
    out_queue: asyncio.Queue[PromptResult],
    worker_id: int,
):
    while True:
        task_input = await in_queue.get()
        if in_queue.qsize() % 100 == 0:
            logger.info(
                f"[policy_worker {worker_id}] Popped 1 task. In queue size: {in_queue.qsize()}"
            )

        if task_input is None:  # Stop sentinel
            in_queue.task_done()
            break

        if task_input.system is not None:
            result = await policy_model.sample(
                [ChatHistory.from_system(task_input.system).add_user(task_input.user)]
            )
        else:
            result = await policy_model.sample([ChatHistory.from_user(task_input.user)])
    
        if (
            result[0] is None or
            (not result[0].has_response) or
            (result[0].finish_reason != "stop")
        ):
            logger.warning(
                f"[policy_worker {worker_id}] Failed to sample for user prompt:\n<user_prompt>\n{task_input}\n</user_prompt>"
            )
            await out_queue.put(
                PromptResult(
                    system=task_input.system,
                    user=task_input.user,
                    assistant=None,
                    batch_id=task_input.batch_id,
                )
            )
            in_queue.task_done()
            continue

        await out_queue.put(
            PromptResult(
                system=task_input.system,
                user=task_input.user,
                assistant=result[0].first_response,  # type: ignore
                batch_id=task_input.batch_id,
            )
        )
        in_queue.task_done()
        logger.debug(f"[policy_worker {worker_id}] Pushed 1 task.")


async def rewrite_worker(
    rewrite_model: RewriteModel,
    batch_size: int,
    in_queue: asyncio.Queue[RewriteInput],
    out_queue: asyncio.Queue[RewriteResult],
    worker_id: int,
):
    async def rewrite_batch(batch: list[RewriteInput]):
        if not batch:
            return

        rewritten_responses = await rewrite_model.rewrite(
            attributes=[input.system for input in batch],
            original_chats=[ChatHistory.from_user(input.user).add_assistant(
                input.original_assistant
            ) for input in batch],
        )

        rewrite_results = []
        for i, response in enumerate(rewritten_responses):
            input = batch[i]
            if response is None:
                logger.warning(
                    f"[rewrite_worker {worker_id}] Failed to rewrite:\n<begin_user_prompt>\n{input.user}\n<end_user_prompt>"
                )
            rewrite_results.append(RewriteResult(
                system=input.system,
                user=input.user,
                original_assistant=input.original_assistant,
                rewritten_assistant=response,
                presence=input.presence,
                batch_id=input.batch_id,
            ))

        for result in rewrite_results:
            await out_queue.put(result)
            logger.debug(f"[rewrite_worker {worker_id}] Pushed 1 task.")

        for _ in batch:
            in_queue.task_done()
        batch.clear() 


    current_batch: list[RewriteInput] = []

    while True:
        task_input = await in_queue.get()
        if in_queue.qsize() % 100 == 0:
            logger.info(
                f"[rewrite_worker {worker_id}] Popped 1 task. In queue size: {in_queue.qsize()}"
            )

        if task_input is None:  # Sentinel value to signal stop
            in_queue.task_done()
            await rewrite_batch(current_batch)
            return

        current_batch.append(task_input)
        if len(current_batch) >= batch_size:
            await rewrite_batch(current_batch)


async def rating_worker(
    reward_model: RewardModel,
    in_queue: asyncio.Queue[PromptResult | RewriteResult | BatchSentinel],
    all_results: Mapping[str, Sequence[PromptResult | RewriteResult]],
    all_futures: Mapping[str, asyncio.Future],
):
    done = False
    total_items_processed = 0
    processed_counts = defaultdict(int)
    expected_counts = defaultdict(int)

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
                        f"Batch {item.batch_id} sentinel received."
                    )
                    in_queue.task_done()
                    expected_counts[item.batch_id] += item.expected_items
                    continue

                if isinstance(item, PromptResult):
                    if item.assistant is None:
                        logger.debug(
                            f"[rating_worker] Received prompt result with no assistant response."
                        )
                    else:
                        logger.debug(
                            f"[rating_worker] Received valid prompt item. New batch size: {len(batch)}."
                        )
                        batch.append(item)
                elif isinstance(item, RewriteResult):
                    if item.rewritten_assistant is None:
                        logger.debug(
                            f"[rating_worker] Received rewritten result with no assistant response."
                        )
                    else:
                        logger.debug(
                            f"[rating_worker] Received valid rewritten item. New batch size: {len(batch)}."
                        )
                        batch.append(item)

                processed_counts[item.batch_id] += 1
                in_queue.task_done()

        except asyncio.TimeoutError:
            pass  # Not an error, just means we process the current incomplete batch

        if len(batch) == 0 and not done:
            # nothing to process yet; continue collecting
            continue

        logger.debug(f"[rating_worker] Processing batch of size {len(batch)}...")
        chat_histories = []
        for result in batch:
            if isinstance(result, PromptResult):
                chat_histories.append(ChatHistory.from_user(result.user).add_assistant(result.assistant))  # type: ignore
            elif isinstance(result, RewriteResult):
                chat_histories.append(ChatHistory.from_user(result.user).add_assistant(result.rewritten_assistant))  # type: ignore

        if len(chat_histories) > 0:
            reward_scores = await reward_model.async_rate(chat_histories, use_tqdm=False)
            for result, reward_score in zip(batch, reward_scores):
                all_results[result.batch_id].append(replace(result, score=reward_score.score))  # type: ignore
        
        total_items_processed += len(batch)
        logger.info(f"[rating_worker] Finished processing minibatch of size {len(batch)}. Total items processed: {total_items_processed}.")

        completed_batch_ids = []
        for batch_id, expected in list(expected_counts.items()):
            processed = processed_counts.get(batch_id, 0)
            if processed >= expected:
                all_futures[batch_id].set_result(all_results[batch_id])
                completed_batch_ids.append(batch_id)
                logger.info(f"[rating_worker] Batch {batch_id} completed: {processed}/{expected}.")
            else:
                logger.debug(
                    f"[rating_worker] Batch {batch_id} progress: {processed}/{expected}."
                )

        for batch_id in completed_batch_ids:
            del all_results[batch_id]  # type: ignore
            if batch_id in processed_counts:
                del processed_counts[batch_id]
            del expected_counts[batch_id]

        if done:
            logger.info("[rating_worker] Final item processed. Shutting down.")
            break


# %%
# Organizing results


def organize_baseline_results(
    all_results: list[PromptResult],
    save_dir: Path | None = None,
) -> dict[str, list[Rollout]]:
    organized_scores = defaultdict(list)
    organized_results = defaultdict(list)

    for item in all_results:
        if item.score is None:
            continue
        assert item.assistant is not None
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


def organize_conditional_results(
    all_results: list[PromptResult],
    save_dir: Path | None = None,
) -> dict[str, dict[str, list[Rollout]]]:

    conditional_results = defaultdict(dict)

    for result in all_results:
        if result.score is None:
            continue
        assert result.assistant is not None

        attribute_results = conditional_results[result.system]
        if result.user not in attribute_results:
            attribute_results[result.user] = []

        attribute_results[result.user].append(
            Rollout(response=result.assistant, score=result.score)
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


def organize_rewrite_results(
    all_results: list[RewriteResult],
    baseline_rollouts: dict[str, list[Rollout]],
    n_rollouts: int | None = None,
    save_dir: Path | None = None,
) -> dict[str, dict[str, list[Rollout | None]]]:

    rewrite_results = defaultdict(dict)

    for result in all_results:
        attribute_results = rewrite_results[result.system]
        if result.user not in attribute_results:
            attribute_results[result.user] = [
                None
                for _ in range(
                    n_rollouts
                    if n_rollouts is not None
                    else len(baseline_rollouts[result.user])
                )
            ]

        found = False
        for i, r in enumerate(baseline_rollouts[result.user]):
            if n_rollouts is not None and i >= n_rollouts:
                break
            if r.response == result.original_assistant:
                if attribute_results[result.user][i] is not None:
                    continue
                attribute_results[result.user][i] = Rollout(
                    response=result.rewritten_assistant,  # type: ignore
                    score=result.score,  # type: ignore
                )
                found = True
                break

        if not found:
            raise ValueError(
                f"Rewrite result for {result.user} and {result.system} not found."
            )

    # # clear any blanks
    # for attribute, attribute_results in rewrite_results.items():
    #     for user, user_results in attribute_results.items():
    #         attribute_results[user] = [r for r in user_results if r is not None]

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


# %%
async def evaluate_baselines(
    user_prompts: list[str],
    policy_model: PolicyModel,
    reward_model: RewardModel,
    n_rollouts: int,
    save_dir: Path | None = None,
) -> dict[str, list[Rollout]]:
    queue_a = asyncio.Queue()
    queue_b = asyncio.Queue()
    batch_id = "0"
    n_policy_workers = policy_model.max_par
    all_results = {batch_id: []}
    all_futures = {batch_id: asyncio.get_running_loop().create_future()}

    for user in user_prompts:
        for _ in range(n_rollouts):
            await queue_a.put(PromptInput(system=None, user=user, batch_id=batch_id))
    await queue_a.put(
        BatchSentinel(batch_id=batch_id, expected_items=len(user_prompts) * n_rollouts)
    )

    policy_worker_tasks = [
        asyncio.create_task(policy_worker(policy_model, queue_a, queue_b, worker_id))
        for worker_id in range(n_policy_workers)
    ]

    rating_worker_task = asyncio.create_task(
        rating_worker(reward_model, queue_b, all_results, all_futures)
    )

    all_results = await all_futures[batch_id]
    logger.info(
        f"Expected {len(user_prompts) * n_rollouts} results, got {len(all_results)}"
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
