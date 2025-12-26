import asyncio
import time
import json
from loguru import logger
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, replace, asdict
from typing import Mapping, Sequence

import numpy as np
from caller import ChatHistory
from api_models import GenerationModel, RewriteModel
from reward_models import RewardModel
from state import Rollout, RewriteScore


@dataclass(frozen=True, kw_only=True, slots=True)
class PromptInput:
    system: str | None
    user: str
    batch_id: str


@dataclass(kw_only=True, slots=True)
class PromptOutput:
    system: str | None
    user: str
    assistant: str | None
    batch_id: str
    raw_score: float | None = None
    model: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class RewriteInput:
    system: str
    user: str
    original_assistant: str
    presence: bool
    batch_id: str

@dataclass(kw_only=True, slots=True)
class RewriteOutput:
    system: str
    user: str
    original_assistant: str
    rewritten_assistant: str | None
    rewriter_reasoning: str | None
    presence: bool
    batch_id: str
    raw_score: float | None = None


@dataclass(frozen=True)
class BatchStartMarker:
    batch_id: str
    expected_items: int


async def policy_worker(
    policy_model: GenerationModel,
    batch_size: int,
    in_queue: asyncio.Queue[PromptInput|None],
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
                    f"[policy_worker {worker_id}] Failed to sample for user prompt:\nuser prompt:\n{input.user}\n"
                    f"model: {response.model if response is not None else 'None'}\n"
                    f"finish reason: {response.finish_reason if response is not None else 'None'}\n"
                    f"response:\n{response.first_response if response is not None else 'None'}"
                )
                sample_results.append(PromptOutput(
                    system=input.system,
                    user=input.user,
                    assistant=None,
                    batch_id=input.batch_id,
                    model=response.model if response is not None else None,
                ))
            else:
                sample_results.append(PromptOutput(
                    system=input.system,
                    user=input.user,
                    assistant=response.first_response.strip(),  # type: ignore
                    batch_id=input.batch_id,
                    model=response.model,
                ))

        for result in sample_results:
            await out_queue.put(result)
            in_queue.task_done()

        batch.clear()
        
    current_batch: list[PromptInput] = []
        
    while True:
        try:
            task_input = await asyncio.wait_for(in_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            # Flush any accumulated items in current_batch
            if current_batch:
                await sample_batch(current_batch)
            continue
        
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
    in_queue: asyncio.Queue[RewriteInput|None],
    out_queue: asyncio.Queue[RewriteOutput],
    worker_id: int,
):
    async def rewrite_batch(batch: list[RewriteInput]):
        start_time = time.time()
        if not batch:
            return

        responses = await rewrite_model.rewrite(
            attributes=[input.system for input in batch],
            original_chats=[ChatHistory.from_user(input.user).add_assistant(
                input.original_assistant
            ) for input in batch],
            presence=[input.presence for input in batch]
        )

        rewrite_results = []
        for i, response in enumerate(responses):
            input = batch[i]
            if response.text is None:
                logger.warning(
                    f"rewrite_worker {worker_id} - Failed to rewrite:\nUser prompt:\n{input.user}"  # this doesn't happen much
                )
            rewrite_results.append(RewriteOutput(
                system=input.system,
                user=input.user,
                original_assistant=input.original_assistant,
                rewritten_assistant=response.text,
                rewriter_reasoning=response.reasoning,
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
        existing_indices = []
        for i, result in enumerate(batch):
            if isinstance(result, PromptOutput):
                if result.assistant is None:
                    continue
                chat_histories.append(ChatHistory.from_user(result.user).add_assistant(result.assistant))  # type: ignore
                existing_indices.append(i)
            elif isinstance(result, RewriteOutput):
                if result.rewritten_assistant is None:
                    continue
                chat_histories.append(ChatHistory.from_user(result.user).add_assistant(result.rewritten_assistant))  # type: ignore
                existing_indices.append(i)

        reward_scores = await reward_model.async_rate(chat_histories, use_tqdm=False)
        for i, reward_score in zip(existing_indices, reward_scores):
            batch[i].raw_score = reward_score.score
        
        all_results[batch[0].batch_id].extend(batch)  # type: ignore      
        logger.debug(f"[rating_worker] Finished processing minibatch of size {len(existing_indices)}/{len(batch)} in {(time.time() - start_time):.2f} seconds. Queue size: {in_queue.qsize()}")
        batch.clear()

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
                    logger.info(f"Batch {item.batch_id} sentinel received: {item.expected_items} items.")
                    in_queue.task_done()
                    expected_counts[item.batch_id] += item.expected_items
                    continue

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
            if in_queue.qsize() >= 256 or in_queue.qsize() <= 32:
                logger.info(f"[rating_worker] Flushing, waited for {(time.time() - last_flush_time):.2f} seconds. Queue size: {in_queue.qsize()}")
            last_flush_time = time.time()
            try:
                await rate_batch(batch)
            except Exception:
                logger.exception(f"rating_worker: rating failed; marking {len(batch)} items with score=None")
                all_results[batch[0].batch_id].extend(batch)  # type: ignore

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

def organize_baselines(
    all_results: list[PromptOutput],
    student_model_name: str,
    save_dir: Path | None = None,
) -> dict[str, list[Rollout]]:
    organized_rollouts = defaultdict(list)
    organized_scores = defaultdict(list)

    for item in all_results:
        assert item.system is None
        if item.raw_score is None:  # only possible when reward model bugged out
            logger.warning(f"Baseline result for {item.user} is None. Skipping.")
            continue
        assert item.assistant is not None
        student_score = RewriteScore(
            score=None,
            raw_score=item.raw_score,
            reasoning=None,
            model_name=student_model_name,
        )
        organized_rollouts[item.user].append(Rollout(
            response=item.assistant,
            student_score=student_score,
            teacher_score=None,
            presence=None,  # None means not set
            model=item.model
        ))
        organized_scores[item.user].append(item.raw_score)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "rollouts.json", "w") as f:
            json_data = {k: [
                {
                    "response": r.response,
                    "model": r.model,
                    "student_score": r.student_score.raw_score,
                } 
                for r in v
            ] for k, v in organized_rollouts.items()}
            json.dump(json_data, f, indent=4, sort_keys=True)

        with open(save_dir / "scores.json", "w") as f:
            json.dump(organized_scores, f, indent=4, sort_keys=True)

    return dict(organized_rollouts)


def organize_rewrites(
    all_results: list[RewriteOutput],
    baseline_rollouts: dict[str, list[Rollout]],
    student_model_name: str,
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
        if result.rewritten_assistant is None:
            continue

        found = False
        for i, baseline in enumerate(baseline_rollouts[result.user]):
            if n_rollouts is not None and i >= n_rollouts:
                break
            if baseline.response == result.original_assistant:
                if attribute_rollouts[result.user][i] is not None:
                    continue

                # Compute reward diff
                # When presence=True: rewritten has attr, baseline doesn't
                #   score_diff = rewritten - baseline (positive = bias toward attr)
                # When presence=False: rewritten doesn't have attr, baseline does
                #   score_diff = baseline - rewritten (flip so positive = bias toward attr)
                # Special case: if rewrite is unchanged (double failure), score_diff = 0
                if result.rewritten_assistant == result.original_assistant:
                    # Unchanged rewrite (double failure) - no bias signal
                    score_diff = 0.0
                    rewritten_raw = baseline.student_score.raw_score  # Use baseline score
                else:
                    baseline_raw = baseline.student_score.raw_score
                    rewritten_raw = result.raw_score
                    if baseline_raw is not None and rewritten_raw is not None:
                        if result.presence:
                            score_diff = rewritten_raw - baseline_raw
                        else:
                            score_diff = baseline_raw - rewritten_raw
                    else:
                        score_diff = None

                student_score = RewriteScore(
                    score=score_diff,
                    raw_score=rewritten_raw,
                    reasoning=None,
                    model_name=student_model_name,
                )
                attribute_rollouts[result.user][i] = Rollout(
                    response=result.rewritten_assistant,
                    student_score=student_score,
                    teacher_score=None,
                    presence=result.presence,
                )
                found = True
                break

        if not found:
            raise ValueError(
                f"Rewrite result for {result.user} and {result.system} not matched."
            )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "rollouts.json", "w") as f:
            json_data = {
                k: {k2: [{
                "response": r.response,
                "student_score": r.student_score.raw_score,
                "student_diff": r.student_score.score,
                "presence": r.presence,
                } if r is not None else None for r in v2
                ] for k2, v2 in v.items()
                } for k, v in organized_rollouts.items()
            }
            json.dump(json_data, f, indent=4, sort_keys=True)

        rewrite_diffs = {}
        for attribute, attribute_rollouts in organized_rollouts.items():
            attribute_diffs = {}
            for user, v in attribute_rollouts.items():
                attribute_diffs[user] = [
                    r.student_score.score for r in v
                    if r is not None and r.student_score is not None and r.student_score.score is not None
                ]
            rewrite_diffs[attribute] = attribute_diffs

        with open(save_dir / "student_diffs.json", "w") as f:
            json.dump(rewrite_diffs, f, indent=4, sort_keys=True)

    return dict(organized_rollouts)


# %%
async def evaluate_baselines(
    user_prompts: list[str],
    policy_model: GenerationModel,
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
    n_policy_workers = 64
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
    logger.info("--- baseline policy workers finished. ---")

    # Now send stop sentinel to rating worker so it can finalize incomplete batches
    await queue_b.put(None)
    await rating_worker_task
    logger.info("--- baseline rating worker finished. ---")

    # Get results from future (should be set by now, either when complete or when finalized)
    all_results = await all_futures[batch_id]
    logger.info(
        f"Expected {len(user_prompts) * n_rollouts} results, got {len(all_results)}"
    )

    organized_results = organize_baselines(all_results, reward_model.model_name, save_dir)
    logger.info(f"Evaluated baselines in {(time.time() - start_time):.2f} seconds")

    return organized_results
