# %%
import json
import time
import asyncio
from itertools import product
from pathlib import Path
from loguru import logger
import plotly.graph_objects as go

from utils import timestamp, remove_outliers
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

    async def evaluate_attributes(
        self,
        user_prompts: list[str],
        attributes: list[str],
        baselines: dict[str, list[Rollout]],
        references: list[dict[str, str] | None] | None = None,
        presence: bool=True,
        n_rollouts: int | None = None,  # max number of baseline responses to rewrite
        save_dir: Path | None = None,
    ):
        """Sends roughly len(user_prompts) * len(attributes) * n_rollouts rewrite requests."""
        await self._ensure_workers_started()
        start_time = time.time()

        # Generate batch ID and future
        assert self._batch_id_lock is not None
        async with self._batch_id_lock:  
            batch_id = str(self._batch_id)
            self.batch_results[batch_id] = []
            loop = asyncio.get_running_loop()
            self.batch_futures[batch_id] = loop.create_future()
            self._batch_id += 1

        # Pre-compute expected number of results
        expected_result_count = 0
        for user in user_prompts:
            n_user_rollouts = (
                min(n_rollouts, len(baselines[user]))
                if n_rollouts is not None
                else len(baselines[user])
            )
            expected_result_count += n_user_rollouts * len(attributes)

        # Send batch start marker
        logger.info(f"Batch {batch_id} expects {expected_result_count} results...")
        await self.queue_rewrite.put(  # type: ignore
            BatchStartMarker(batch_id=batch_id, expected_items=expected_result_count)
        )

        # Put tasks
        for user in user_prompts:
            for j, attribute in enumerate(attributes):
                ref_triple = references[j] if references is not None else None
                for i, original_assistant in enumerate(baselines[user]):
                    if n_rollouts is not None and i >= n_rollouts:
                        break
                    if ref_triple is not None:
                        await self.queue_input.put(  # type: ignore
                            RewriteInput(
                                system=attribute,
                                user=user,
                                original_assistant=original_assistant.response,
                                presence=presence,
                                batch_id=batch_id,
                                reference_user=ref_triple["user_prompt"],
                                reference_response_A=ref_triple["response_A"],
                                reference_response_B=ref_triple["response_B"],
                            )
                        )
                    else:
                        await self.queue_input.put(  # type: ignore
                            RewriteInput(
                                system=attribute,
                                user=user,
                                original_assistant=original_assistant.response,
                                presence=presence,
                                batch_id=batch_id,
                            )
                        )

        # Wait for results for this batch
        batch_results = await self.batch_futures[batch_id]

        organized_results = organize_rewrites(
            batch_results, baselines, self.reward_model.model_name, n_rollouts, save_dir
        )
        del self.batch_futures[batch_id]  # type: ignore

        logger.info(f"Batch {batch_id}: {expected_result_count} results evaluated in {(time.time() - start_time):.2f} seconds")
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
