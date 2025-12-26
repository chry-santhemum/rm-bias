import time
import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any
from loguru import logger

from api_models import RewriteModel
from reward_models import RewardModel
from state import Rollout
from bias_workers import (
    BatchStartMarker,
    RewriteInput,
    RewriteOutput,
    rewrite_worker,
    rating_worker,
    organize_rewrites,
)


class BiasEvaluator:
    def __init__(
        self,
        rewrite_model: RewriteModel,
        reward_model: RewardModel,
        n_rewrite_workers: int,
    ):
        self.rewrite_model = rewrite_model
        self.reward_model = reward_model
        self.n_rewrite_workers = n_rewrite_workers

        self._workers_started = False

        # Defer queues and worker startup until inside a running event loop
        self._batch_id: int
        self._batch_id_lock: asyncio.Lock | None
        self.queue_input: asyncio.Queue | None
        self.queue_rewrite: asyncio.Queue | None
        self.batch_results: dict[str, list[RewriteOutput]]
        self.batch_futures: dict[str, asyncio.Future]
        self.rewrite_workers: list[asyncio.Task] = []
        self.rating_worker: asyncio.Task | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "rewrite_model": self.rewrite_model.to_dict(),
            "reward_model": self.reward_model.to_dict(),
            "n_rewrite_workers": self.n_rewrite_workers
        }

    async def _ensure_workers_started(self):
        if self._workers_started:
            return
        self._batch_id = 0
        self._batch_id_lock = asyncio.Lock()
        self.queue_input = asyncio.Queue(maxsize=2 * self.rewrite_model.max_par)
        self.queue_rewrite = asyncio.Queue()
        self.batch_results = {}
        self.batch_futures = {}

        self.rewrite_workers = [
            asyncio.create_task(
                rewrite_worker(
                    rewrite_model=self.rewrite_model,
                    batch_size=max(
                        1, self.rewrite_model.max_par // self.n_rewrite_workers
                    ),
                    in_queue=self.queue_input,
                    out_queue=self.queue_rewrite,
                    worker_id=worker_id,
                )
            )
            for worker_id in range(self.n_rewrite_workers)
        ]

        self.rating_worker = asyncio.create_task(
            rating_worker(
                self.reward_model,
                self.queue_rewrite,
                self.batch_results,
                self.batch_futures,
            )
        )
        # Surface crashes of the rating worker early
        def _rating_done_cb(task: asyncio.Task):
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                logger.info("[rating_worker] cancelled, exiting.")
                return
            if exc is not None:
                logger.exception("[rating_worker] exited with exception")
            else:
                logger.info("[rating_worker] exited without exception.")
        self.rating_worker.add_done_callback(_rating_done_cb)
        
        self._workers_started = True

    async def _run_rewrite_batch(
        self,
        rewrite_inputs: list[RewriteInput],
    ) -> list[RewriteOutput]:
        """Run a batch of rewrite inputs through the worker pipeline."""
        assert self._batch_id_lock is not None
        async with self._batch_id_lock:
            batch_id = str(self._batch_id)
            self.batch_results[batch_id] = []
            loop = asyncio.get_running_loop()
            self.batch_futures[batch_id] = loop.create_future()
            self._batch_id += 1

        # Send batch start marker
        logger.info(f"Batch {batch_id} expects {len(rewrite_inputs)} results...")
        await self.queue_rewrite.put(  # type: ignore
            BatchStartMarker(batch_id=batch_id, expected_items=len(rewrite_inputs))
        )

        # Put tasks (update batch_id since RewriteInput is frozen)
        for rewrite_input in rewrite_inputs:
            input_with_batch_id = replace(rewrite_input, batch_id=batch_id)
            await self.queue_input.put(input_with_batch_id)  # type: ignore

        # Wait for results
        batch_results = await self.batch_futures[batch_id]
        del self.batch_futures[batch_id]

        return batch_results

    async def evaluate_attributes(
        self,
        user_prompts: list[str],
        attributes: list[str],
        baselines: dict[str, list[Rollout]],
        n_rollouts: int | None = None,  # max number of baseline responses to rewrite
        save_dir: Path | None = None,
    ):
        """Sends roughly len(user_prompts) * len(attributes) * n_rollouts rewrite requests.

        When presence=True rewrite returns unchanged (baseline already has attribute),
        automatically retries with presence=False to remove the attribute instead.
        """
        await self._ensure_workers_started()
        start_time = time.time()

        # Statistics tracking per attribute
        from collections import defaultdict
        rewrite_stats: dict[str, dict[str, int]] = defaultdict(lambda: {
            "total_attempted": 0,
            "positive_unchanged": 0,  # First pass failures (presence=True returned unchanged)
            "retry_unchanged": 0,     # Double failures (presence=False also returned unchanged)
        })

        # Build all rewrite inputs for first pass (presence=True)
        first_pass_inputs: list[RewriteInput] = []
        for user in user_prompts:
            for j, attribute in enumerate(attributes):
                for i, original_assistant in enumerate(baselines[user]):
                    if n_rollouts is not None and i >= n_rollouts:
                        break
                    rewrite_stats[attribute]["total_attempted"] += 1
                    first_pass_inputs.append(
                        RewriteInput(
                            system=attribute,
                            user=user,
                            original_assistant=original_assistant.response,
                            presence=True,
                            batch_id="",  # Will be set in _run_rewrite_batch
                        )
                    )

        # Run first pass
        first_pass_results = await self._run_rewrite_batch(first_pass_inputs)
        logger.info(f"First pass complete: {len(first_pass_results)} results")

        # Identify items where rewrite returned unchanged (model said "already has attribute")
        # These need retry with presence=False
        changed_results: list[RewriteOutput] = []
        retry_inputs: list[RewriteInput] = []

        for result in first_pass_results:
            # Check if rewrite returned the original (unchanged)
            # This happens when model outputs "None" - rewritten_assistant is set to original
            if (
                result.rewritten_assistant is not None
                and result.rewritten_assistant == result.original_assistant
                and result.presence  # Only retry if this was a presence=True attempt
            ):
                rewrite_stats[result.system]["positive_unchanged"] += 1
                retry_inputs.append(
                    RewriteInput(
                        system=result.system,
                        user=result.user,
                        original_assistant=result.original_assistant,
                        presence=False,
                        batch_id="",  # Will be set in _run_rewrite_batch
                    )
                )
            else:
                changed_results.append(result)

        # Run second pass if there are items to retry
        if retry_inputs:
            logger.info(f"Retrying {len(retry_inputs)} items with presence=False")
            retry_results = await self._run_rewrite_batch(retry_inputs)
            logger.info(f"Retry pass complete: {len(retry_results)} results")

            # Check for double failures (retry also returned unchanged)
            for result in retry_results:
                if (
                    result.rewritten_assistant is not None
                    and result.rewritten_assistant == result.original_assistant
                ):
                    rewrite_stats[result.system]["retry_unchanged"] += 1
                    logger.debug(
                        f"Double failure:\nuser prompt:\n{result.user}\n"
                        f"attribute:\n{result.system}\n"
                        f"original assistant:\n{result.original_assistant}\n"
                        f"rewriter reasoning:\n{result.rewriter_reasoning}"
                    )

            # Combine results: use retry results (including double failures)
            all_results = changed_results + retry_results
        else:
            all_results = changed_results

        # Log statistics
        total_positive_unchanged = sum(s["positive_unchanged"] for s in rewrite_stats.values())
        total_retry_unchanged = sum(s["retry_unchanged"] for s in rewrite_stats.values())
        total_attempted = sum(s["total_attempted"] for s in rewrite_stats.values())

        if total_positive_unchanged > 0:
            logger.info(
                f"Rewrite stats: {total_positive_unchanged}/{total_attempted} positive rewrites unchanged, "
                f"{total_retry_unchanged}/{total_positive_unchanged} retries also unchanged (double failures)"
            )
        else:
            logger.info(f"Rewrite stats: all {total_attempted} positive rewrites succeeded")

        # Log per-attribute stats
        for attr, stats in rewrite_stats.items():
            logger.info(
                f"  {attr[:50]}...: {stats['positive_unchanged']}/{stats['total_attempted']} "
                f"positive unchanged, {stats['retry_unchanged']} double failures"
            )

        organized_results = organize_rewrites(
            all_results, baselines, self.reward_model.model_name, n_rollouts, save_dir
        )

        logger.info(f"Evaluation complete: {len(all_results)} results in {(time.time() - start_time):.2f} seconds")
        return organized_results

    async def shutdown(self):
        if not self._workers_started:
            return

        assert self.queue_input is not None and self.queue_rewrite is not None
        for _ in range(self.n_rewrite_workers):
            await self.queue_input.put(None)  # Sentinel values for rewrite_workers

        await asyncio.gather(*self.rewrite_workers)
        logger.info("--- rewrite workers finished. ---")

        # Ensure all produced rewrite results are fully consumed and marked done
        # before signalling the rating worker to exit. This prevents leaving
        # unfinished tasks on the queue and avoids shutdown hangs.
        await self.queue_rewrite.join()
        await self.queue_rewrite.put(None)  # Sentinel value for rating_worker
        if self.rating_worker is not None:
            await self.rating_worker
        logger.info("--- rewrite rating worker finished. ---")

        self._batch_id = 0
        self._batch_id_lock = None
        self._workers_started = False
        self.queue_input = None
        self.queue_rewrite = None
        self.batch_results = {}
        self.batch_futures = {}
        self.rewrite_workers = []
        self.rating_worker = None


    # Make this into an async context manager to ensure proper shutdown
    async def __aenter__(self):
        await self._ensure_workers_started()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.shutdown()
        return False
