# %%
import json
import time
import asyncio
from itertools import product
from pathlib import Path
import logging
import plotly.graph_objects as go

from utils import timestamp, remove_outliers
from models import RewriteModel
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

logger = logging.getLogger(__name__)




class BiasEvaluator:
    def __init__(
        self,
        rewrite_model: RewriteModel,
        reward_model: RewardModel,
        n_rewrite_workers: int,
    ):
        self.rewrite_model = rewrite_model
        self.reward_model = reward_model
        self._workers_started = False
        self.n_rewrite_workers = n_rewrite_workers

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
                return
            if exc is not None:
                logger.exception("rating_worker exited with exception")
            else:
                logger.warning("rating_worker exited unexpectedly without exception.")
        self.rating_worker.add_done_callback(_rating_done_cb)
        
        self._workers_started = True

    async def evaluate_attributes(
        self,
        user_prompts: list[str],
        attributes: list[str],
        baselines: dict[str, list[Rollout]],
        n_rollouts: int | None = None,  # number of baseline responses to rewrite
        save_dir: Path | None = None,
    ):
        await self._ensure_workers_started()
        start_time = time.time()

        # Generate batch ID and asyncio.Future (protected by lock)
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
        for user, attribute in product(user_prompts, attributes):
            for i, original_assistant in enumerate(baselines[user]):
                if n_rollouts is not None and i >= n_rollouts:
                    break
                await self.queue_input.put(  # type: ignore
                    RewriteInput(
                        system=attribute,
                        user=user,
                        original_assistant=original_assistant.response,
                        presence=True,
                        batch_id=batch_id,
                    )
                )

        # Wait for results for this batch
        batch_results = await self.batch_futures[batch_id]
        logger.info(
            f"Batch {batch_id}: Expected {expected_result_count} results, got {len(batch_results)}"
        )
        organized_results = organize_rewrites(
            batch_results, baselines, n_rollouts, save_dir
        )
        del self.batch_futures[batch_id]  # type: ignore

        logger.info(f"Batch {batch_id}: evaluated in {(time.time() - start_time):.2f} seconds")
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



def plot_seed_validation_data(
    results_dir: Path,
    seed_index: int,
    model_slugs: (
        list[str] | None
    ) = None,  # e.g., ["gemini-2-5-flash-lite-preview-09-2025", "grok-4-fast"]
):
    # Load baseline results
    if not (results_dir / "baseline_results.json").exists():
        with open(
            results_dir / "val_baselines" / "baseline_results.json",
            "r",
            encoding="utf-8",
        ) as f:
            baseline_results = json.load(f)
    else:
        with open(results_dir / "baseline_results.json", "r", encoding="utf-8") as f:
            baseline_results = json.load(f)

    # Load cluster info
    with open(
        results_dir / f"seed_{seed_index}_cluster.json", "r", encoding="utf-8"
    ) as f:
        cluster_info = json.load(f)

    # Define model directories
    model_dirs = [results_dir / f"validate/seed_{seed_index}_validate"]
    model_names = ["gpt-5-nano"]

    if model_slugs:
        for slug in model_slugs:
            model_dirs.append(
                results_dir / f"validate/seed_{seed_index}_validate_{slug}"
            )
            model_names.append(slug)

    # Define consistent colors for each model
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

    # Load data for all models
    all_model_data = []
    for model_dir, model_name in zip(model_dirs, model_names):
        with open(model_dir / "rewrite_plus_scores.json", "r", encoding="utf-8") as f:
            validate_results = json.load(f)

        with open(
            model_dir.parent / f"seed_{seed_index}_judge.json", "r", encoding="utf-8"
        ) as f:
            judge_results = json.load(f)

        all_model_data.append(
            {"name": model_name, "validate": validate_results, "judge": judge_results}
        )

    # Helper function to wrap text
    def wrap_text(text, width):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return "<br>".join(lines)

    # Get list of attributes (from first model)
    attributes = list(all_model_data[0]["validate"].keys())

    # Create figure
    fig = go.Figure()

    # For each attribute, add violin plots for each model
    for attr_idx, attribute in enumerate(attributes):
        for model_idx, model_data in enumerate(all_model_data):
            validate_results = model_data["validate"]
            judge_results = model_data["judge"]

            # Compute differences from baseline
            attribute_results = validate_results[attribute]
            attribute_diffs = []
            winrates = []

            for prompt, prompt_rewards in attribute_results.items():
                baseline_rewards = [r["score"] for r in baseline_results[prompt]]

                # Compute element-wise differences
                for attr_score, base_score in zip(prompt_rewards, baseline_rewards):
                    if attr_score is None or base_score is None:
                        continue
                    attribute_diffs.append(attr_score - base_score)

            for prompt, prompt_judge_winrates in judge_results[attribute].items():
                winrates_clean = [wr for wr in prompt_judge_winrates if wr is not None]
                winrates.extend(winrates_clean)

            # Remove outliers
            attribute_diffs = remove_outliers(attribute_diffs)

            fig.add_trace(
                go.Violin(
                    y=attribute_diffs,
                    name=model_data["name"],
                    x=[attr_idx] * len(attribute_diffs),  # Group by attribute index
                    legendgroup=model_data["name"],
                    scalegroup=model_data["name"],
                    showlegend=(attr_idx == 0),  # Only show legend for first attribute
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    marker=dict(color=colors[model_idx]),
                    line=dict(color=colors[model_idx]),
                    offsetgroup=model_idx,  # Offset side-by-side
                )
            )

    # Update layout
    fig.update_layout(
        title=f"Seed {seed_index}: {cluster_info['summary']}",
        xaxis_title="Attribute",
        yaxis_title="Reward Difference (Attribute - Baseline)",
        height=1000,
        width=1400,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(attributes))),
            ticktext=[wrap_text(attr, width=60) for attr in attributes],
            tickangle=45,
        ),
        violinmode="group",  # Group violins side-by-side
        legend=dict(
            title="Model",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # Add reference line at 0
    fig.add_hline(
        y=0, line_dash="dash", line_color="black", line_width=1.5, opacity=0.6
    )

    return fig


# %%
if __name__ == "__main__":
    # asyncio.run(main())
    save_dir = Path(f"data/scrap/{timestamp()}_rewrite_different_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    for seed_index in [1, 8, 9, 14]:
        fig = plot_seed_validation_data(
            results_dir=Path("data/evo/20251101-070815-synthetic_2"),
            seed_index=seed_index,
            model_slugs=["gemini-2.5-flash-lite-preview-09-2025", "grok-4-fast"],
        )
        fig.write_html(save_dir / f"evo-synthetic_2-seed_{seed_index}.html")
