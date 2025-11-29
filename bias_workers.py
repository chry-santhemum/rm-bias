# %%
import logging
import asyncio
import time
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, replace, asdict
from typing import Mapping, Sequence

import numpy as np
from caller import ChatHistory
from models import PolicyModel, RewriteModel
from reward_models import RewardModel
from state import Rollout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptInput:
    system: str | None
    user: str
    batch_id: str


@dataclass(frozen=True)
class PromptOutput:
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
    reference_user: str | None = None
    reference_response_A: str | None = None
    reference_response_B: str | None = None


@dataclass(frozen=True)
class RewriteOutput:
    system: str
    user: str
    original_assistant: str
    rewritten_assistant: str | None
    presence: bool
    batch_id: str
    score: float | None = None


@dataclass(frozen=True)
class BatchStartMarker:
    batch_id: str
    expected_items: int


async def policy_worker(
    policy_model: PolicyModel,
    batch_size: int,
    in_queue: asyncio.Queue[PromptInput],
    out_queue: asyncio.Queue[PromptOutput],
    worker_id: int,
):
    async def sample_batch(batch: list[PromptInput]):
        if not batch:
            return
        
        chat_histories = []
        for input in batch:
            if input.system is not None:
                chat_histories.append(ChatHistory.from_system(input.system).add_user(input.user))
            else:
                chat_histories.append(ChatHistory.from_user(input.user))
                
        responses = await policy_model.sample(chat_histories)
        sample_results = []

        for i, response in enumerate(responses):
            input = batch[i]
            if (
                response is None or
                (not response.has_response) or
                (response.finish_reason != "stop")
            ):
                logger.warning(
                    f"[policy_worker {worker_id}] Failed to sample for user prompt:\n<user_prompt>\n{input.user}\n</user_prompt>"
                )
                sample_results.append(PromptOutput(
                    system=input.system,
                    user=input.user,
                    assistant=None,
                    batch_id=input.batch_id,
                ))
            else:
                sample_results.append(PromptOutput(
                    system=input.system,
                    user=input.user,
                    assistant=response.first_response,  # type: ignore
                    batch_id=input.batch_id,
                ))

        for result in sample_results:
            await out_queue.put(result)
            in_queue.task_done()

        batch.clear()
        
    current_batch: list[PromptInput] = []
        
    while True:
        task_input = await in_queue.get()
        if in_queue.qsize() % 100 == 0:
            logger.info(
                f"[policy_worker {worker_id}] Popped 1 task. In queue size: {in_queue.qsize()}"
            )

        if task_input is None:  # Stop sentinel
            in_queue.task_done()
            await sample_batch(current_batch)
            break
        
        current_batch.append(task_input)
        if len(current_batch) >= batch_size:
            await sample_batch(current_batch)


async def rewrite_worker(
    rewrite_model: RewriteModel,
    batch_size: int,
    in_queue: asyncio.Queue[RewriteInput | None],
    out_queue: asyncio.Queue[RewriteOutput],
    worker_id: int,
):
    async def rewrite_batch(batch: list[RewriteInput]):
        start_time = time.time()
        if not batch:
            return
        
        reference_chats = []
        for input in batch:
            if input.reference_user is None:
                reference_chats.append(None)
            else:
                reference_chats.append({
                    "user": input.reference_user,
                    "response_A": input.reference_response_A,
                    "response_B": input.reference_response_B,
                })

        responses = await rewrite_model.rewrite(
            attributes=[input.system for input in batch],
            original_chats=[ChatHistory.from_user(input.user).add_assistant(
                input.original_assistant
            ) for input in batch],
            reference_chats=reference_chats,
        )

        rewrite_results = []
        for i, response in enumerate(responses):
            input = batch[i]
            if response is None:
                logger.warning(
                    f"[rewrite_worker {worker_id}] Failed to rewrite:\n<begin_user_prompt>\n{input.user}\n<end_user_prompt>"
                )
            rewrite_results.append(RewriteOutput(
                system=input.system,
                user=input.user,
                original_assistant=input.original_assistant,
                rewritten_assistant=response,
                presence=input.presence,
                batch_id=input.batch_id,
            ))

        for result in rewrite_results:
            await out_queue.put(result)
            in_queue.task_done()
            logger.debug(f"[rewrite_worker {worker_id}] Pushed 1 task.")

        if worker_id < 3:
            logger.info(f"[rewrite_worker {worker_id}] Finished rewriting minibatch of size {len(batch)} in {(time.time() - start_time):.2f} seconds.")
        batch.clear() 

    current_batch: list[RewriteInput] = []

    while True:
        try:
            task_input = await asyncio.wait_for(in_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            # Flush any accumulated items in current_batch
            if current_batch:
                await rewrite_batch(current_batch)
            continue

        if in_queue.qsize() % 100 == 0:
            logger.info(
                f"[rewrite_worker {worker_id}] Popped 1 task. In queue size: {in_queue.qsize()}"
            )

        if task_input is None:  # Stop sentinel
            in_queue.task_done()
            await rewrite_batch(current_batch)
            return

        current_batch.append(task_input)
        if len(current_batch) >= batch_size:
            await rewrite_batch(current_batch)


async def rating_worker(
    reward_model: RewardModel,
    in_queue: asyncio.Queue[PromptOutput | RewriteOutput | BatchStartMarker],
    all_results: Mapping[str, Sequence[PromptOutput | RewriteOutput]],
    all_futures: Mapping[str, asyncio.Future],
):
    done = False
    # Keep track of progress by batch id
    processed_counts = defaultdict(int)
    expected_counts = defaultdict(int)

    async def rate_batch(batch: list[PromptOutput | RewriteOutput]):
        start_time = time.time()
        chat_histories = []
        for result in batch:
            if isinstance(result, PromptOutput):
                chat_histories.append(ChatHistory.from_user(result.user).add_assistant(result.assistant))  # type: ignore
            elif isinstance(result, RewriteOutput):
                chat_histories.append(ChatHistory.from_user(result.user).add_assistant(result.rewritten_assistant))  # type: ignore

        reward_scores = await reward_model.async_rate(chat_histories, use_tqdm=False)
        for result, reward_score in zip(batch, reward_scores):
            all_results[result.batch_id].append(replace(result, score=reward_score.score))  # type: ignore
        
        logger.info(f"[rating_worker] Finished processing minibatch of size {len(batch)} in {(time.time() - start_time):.2f} seconds. Queue size: {in_queue.qsize()}")


    while True:
        batch = []
        last_flush_time = time.time()

        try:
            while len(batch) < reward_model.batch_size:
                item = await asyncio.wait_for(in_queue.get(), timeout=3.0)

                if item is None:  # Sentinel value to signal stop
                    logger.info("Stop sentinel received.")
                    in_queue.task_done()
                    done = True
                    break

                if isinstance(item, BatchStartMarker):
                    logger.info(f"Batch {item.batch_id} sentinel received.")
                    in_queue.task_done()
                    expected_counts[item.batch_id] += item.expected_items
                    continue

                if isinstance(item, PromptOutput):
                    if item.assistant is not None:
                        batch.append(item)

                elif isinstance(item, RewriteOutput):
                    if item.rewritten_assistant is not None:
                        batch.append(item)

                processed_counts[item.batch_id] += 1
                in_queue.task_done()
                
                if time.time() - last_flush_time > 3.0:
                    break

        except asyncio.TimeoutError:
            pass  # Not an error, just means we process the current incomplete batch
        
        # Only rate when there is something to rate; but always progress-check below
        # to avoid the edge case of rewrite worker failing to produce valid outputs,
        # such that the rating worker doesn't know that it's done
        if len(batch) > 0:
            try:
                await rate_batch(batch)
            except Exception:
                logger.exception("rating_worker: rating failed; marking %d items with score=None", len(batch))
                # Ensure progress even if rating crashes; keep result alignment
                for result in batch:
                    all_results[result.batch_id].append(replace(result, score=None))  # type: ignore
            last_flush_time = time.time()

        completed_batch_ids = []
        for batch_id, expected in list(expected_counts.items()):
            processed = processed_counts.get(batch_id, 0)
            if processed == expected:
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
            # Break outer loop
            break


# %%
# Organizing results

def organize_samples(
    all_results: list[PromptOutput],
    save_dir: Path | None = None,
):
    organized_scores = defaultdict(dict)
    organized_rollouts = defaultdict(dict)

    for item in all_results:
        if item.score is None:
            continue
        assert item.assistant is not None

        attribute_rollouts = organized_rollouts[item.system]
        if item.user not in attribute_rollouts:
            attribute_rollouts[item.user] = []
        
        attribute_scores = organized_scores[item.system]
        if item.user not in attribute_scores:
            attribute_scores[item.user] = []

        attribute_rollouts[item.user].append(Rollout(response=item.assistant, score=item.score))
        attribute_scores[item.user].append(item.score)

    mean_scores = {}
    for attribute, attribute_scores in organized_scores.items():
        all_scores = []
        for user, scores in attribute_scores.items():
            all_scores.extend(scores)
        mean_scores[attribute] = np.mean(all_scores).item()
    
    keys = list(organized_rollouts.keys())
    if len(keys) == 1 and keys[0] is None:
        organized_rollouts = organized_rollouts[None]
        organized_scores = organized_scores[None]
        mean_scores = mean_scores[None]


    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "sample_rollouts.json", "w", encoding="utf-8") as f:
            if len(keys) == 1 and keys[0] is None:
                json_data = {k: [asdict(r) for r in v] for k, v in organized_rollouts.items()}
            else:
                json_data = {
                    k: {k2: [asdict(r) for r in v2] for k2, v2 in v.items()}
                    for k, v in organized_rollouts.items()
                }
            json.dump(json_data, f, indent=4)

        with open(save_dir / "sample_scores.json", "w", encoding="utf-8") as f:
            json.dump(organized_scores, f, indent=4)

        with open(save_dir / "sample_scores_mean.json", "w", encoding="utf-8") as f:
            json.dump(mean_scores, f, indent=4)

    return dict(organized_rollouts)


def organize_rewrites(
    all_results: list[RewriteOutput],
    baseline_rollouts: dict[str, list[Rollout]],
    n_rollouts: int | None = None,
    save_dir: Path | None = None,
) -> dict[str, dict[str, list[Rollout | None]]]:

    organized_rollouts = defaultdict(dict)

    for result in all_results:
        attribute_rollouts = organized_rollouts[result.system]
        if result.user not in attribute_rollouts:
            attribute_rollouts[result.user] = [
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
                if attribute_rollouts[result.user][i] is not None:
                    continue
                attribute_rollouts[result.user][i] = Rollout(
                    response=result.rewritten_assistant,  # type: ignore
                    score=result.score,  # type: ignore
                )
                found = True
                break

        if not found:
            raise ValueError(
                f"Rewrite result for {result.user} and {result.system} not matched."
            )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "rewrite_rollouts.json", "w", encoding="utf-8") as f:
            json_data = {
                k: {k2: [asdict(r) for r in v2] for k2, v2 in v.items()}
                for k, v in organized_rollouts.items()
            }
            json.dump(json_data, f, indent=4)

        scores = {}
        mean_scores = {}

        for attribute, attribute_rollouts in organized_rollouts.items():
            attribute_scores = {}
            all_scores = []
            for user, v in attribute_rollouts.items():
                attribute_scores[user] = [r.score for r in v]
                all_scores.extend([r.score for r in v if r.score is not None])
            scores[attribute] = attribute_scores
            mean_scores[attribute] = np.mean(all_scores).item()

        with open(save_dir / "rewrite_scores.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)
        with open(
            save_dir / "rewrite_scores_mean.json", "w", encoding="utf-8"
        ) as f:
            json.dump(mean_scores, f, indent=4)

    return dict(organized_rollouts)


# %%
async def evaluate_baselines(
    user_prompts: list[str],
    policy_model: PolicyModel,
    reward_model: RewardModel,
    n_rollouts: int,
    save_dir: Path | None = None,
) -> dict[str, list[Rollout]]:
    """
    Saves results under the directory save_dir.
    """
    queue_a = asyncio.Queue(maxsize = 2 * policy_model.max_par)
    queue_b = asyncio.Queue()
    batch_id = "0"
    n_policy_workers = 128
    batch_size = max(1, policy_model.max_par // n_policy_workers)
    all_results = {batch_id: []}
    all_futures = {batch_id: asyncio.get_running_loop().create_future()}
    start_time = time.time()

    # Start workers
    policy_worker_tasks = [
        asyncio.create_task(policy_worker(policy_model, batch_size, queue_a, queue_b, worker_id))
        for worker_id in range(n_policy_workers)
    ]
    rating_worker_task = asyncio.create_task(
        rating_worker(reward_model, queue_b, all_results, all_futures)
    )

    # Send inputs
    logger.info(f"Batch {batch_id} expects {len(user_prompts) * n_rollouts} results...")
    await queue_b.put(
        BatchStartMarker(batch_id=batch_id, expected_items=len(user_prompts) * n_rollouts)
    )

    for user in user_prompts:
        for _ in range(n_rollouts):
            await queue_a.put(PromptInput(system=None, user=user, batch_id=batch_id))

    # send one stop sentinel per policy worker
    for _ in range(n_policy_workers):
        await queue_a.put(None)

    # Wait for policy workers to finish first, so we know no more items will be produced
    await asyncio.gather(*policy_worker_tasks)
    logger.info("\n--- baseline policy workers finished. ---\n")

    # Now send stop sentinel to rating worker so it can finalize incomplete batches
    await queue_b.put(None)
    await rating_worker_task
    logger.info("\n--- baseline rating worker finished. ---\n")

    # Get results from future (should be set by now, either when complete or when finalized)
    all_results = await all_futures[batch_id]
    logger.info(
        f"Expected {len(user_prompts) * n_rollouts} results, got {len(all_results)}"
    )

    organized_results = organize_samples(all_results, save_dir)
    logger.info(f"Evaluated baselines in {(time.time() - start_time):.2f} seconds")

    return organized_results  # type: ignore
