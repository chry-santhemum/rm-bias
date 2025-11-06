# %%
import json
import time
import asyncio
import uuid
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
import numpy as np
import plotly.graph_objects as go

from utils import logging_setup, timestamp, remove_outliers
from models import RewriteModel
from reward_model import RewardModel
from state import Rollout
from bias_workers_baseline import evaluate_baselines
from bias_workers_rewrite import (
    BatchSentinel,
    RewriteInput,
    RewriteResult,
    rewrite_worker,
    rewrite_rating_worker,
    organize_rewrite_results,
)

logger = logging.getLogger(__name__)


class BiasEvaluator:

    def __init__(
        self, 
        rewrite_model: RewriteModel,
        reward_model: RewardModel,
    ):
        self.rewrite_model = rewrite_model
        self.reward_model = reward_model

        # defer queues and worker startup until inside a running event loop
        self._workers_started = False
        self.queue_input: asyncio.Queue | None = None
        self.queue_rewrite: asyncio.Queue | None = None
        self.batch_results: dict[str, list[RewriteResult]] = {}
        self.batch_futures: dict[str, asyncio.Future] = {}
        self.rewrite_workers: list[asyncio.Task] = []
        self.rating_worker: asyncio.Task | None = None
        self.rating_executor: ThreadPoolExecutor | None = None


    async def _ensure_workers_started(self):
        if self._workers_started:
            return

        # must be called from within a running event loop
        self.queue_input = asyncio.Queue()
        self.queue_rewrite = asyncio.Queue()

        self.rewrite_workers = [
            asyncio.create_task(
                rewrite_worker(
                    self.rewrite_model, self.queue_input, self.queue_rewrite, worker_id
                )
            )
            for worker_id in range(self.rewrite_model.max_par)
        ]

        self.rating_executor = ThreadPoolExecutor(max_workers=1)
        self.rating_worker = asyncio.create_task(
            rewrite_rating_worker(
                self.reward_model,
                self.queue_rewrite,
                self.batch_results,
                self.batch_futures,
                self.rating_executor,
            )
        )
        self._workers_started = True


    async def evaluate_attributes(
        self,
        user_prompts: list[str],
        attributes: list[str],
        baseline_rollouts: dict[str, list[Rollout]],
        save_dir: Path | None = None,
    ):
        start_time = time.time()
        await self._ensure_workers_started()

        # Generate batch ID and asyncio.Future
        batch_id = str(uuid.uuid4())
        self.batch_results[batch_id] = []
        loop = asyncio.get_running_loop()
        self.batch_futures[batch_id] = loop.create_future()
        expected_result_count = 0

        for user, attribute in product(user_prompts, attributes):
            for original_assistant in baseline_rollouts[user]:
                assert self.queue_input is not None
                await self.queue_input.put(
                    RewriteInput(
                        system=attribute,
                        user=user,
                        original_assistant=original_assistant.response,
                        presence=True,
                        batch_id=batch_id,
                    )
                )
                expected_result_count += 1

        # Send batch task completion sentinel
        logger.info(f"Batch {batch_id} expects {expected_result_count} results...")
        assert self.queue_input is not None
        await self.queue_input.put(
            BatchSentinel(batch_id=batch_id, expected_items=expected_result_count)
        )

        # Wait for results for this batch
        batch_results = await self.batch_futures[batch_id]
        logger.info(f"Expected {expected_result_count} results, got {len(batch_results)}")
        organized_results = organize_rewrite_results(
            batch_results, baseline_rollouts, save_dir
        )
        del self.batch_futures[batch_id]

        logger.info(f"Attributes evaluated in {(time.time() - start_time):.2f} seconds")
        return organized_results

    async def shutdown(self):
        if not self._workers_started:
            return

        assert self.queue_input is not None and self.queue_rewrite is not None
        for _ in range(self.rewrite_model.max_par):
            await self.queue_input.put(None)  # Sentinel values for rewrite_workers

        await asyncio.gather(*self.rewrite_workers)
        logger.info("\n--- rewrite workers finished. ---\n")

        await self.queue_rewrite.put(None)
        if self.rating_worker is not None:
            await self.rating_worker
        logger.info("\n--- rewrite rating worker finished. ---\n")

        if self.rating_executor is not None:
            self.rating_executor.shutdown(wait=True)
            self.rating_executor = None
        self._workers_started = False


    # Make this into an async context manager to ensure proper shutdown
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.shutdown()
        return False



async def main():
    logging_setup(filename=f"logs/scrap/rewrite_different_models_{timestamp()}.log", level=logging.INFO, console=False)
    run_name = "20251101-070815-synthetic_2"
    reward_model = RewardModel(model_name="skywork-v2", batch_size=32)
    model1 = RewriteModel(
        # model_name="google/gemini-2.5-flash-preview-09-2025", 
        model_name="google/gemini-2.5-flash-lite-preview-09-2025",
        max_par=512,
        reasoning=4096,
    )
    model2 = RewriteModel(
        model_name="x-ai/grok-4-fast",
        max_par=512,
        reasoning="medium",
    )

    with open(
        f"data/evo/{run_name}/val_baselines/baseline_results.json", "r"
    ) as f:
        baselines_json = json.load(f)

    val_baselines = dict()
    for user, rollouts in baselines_json.items():
        val_baselines[user] = [
            Rollout(response=rollout["response"], score=rollout["score"])
            for rollout in rollouts
        ]

    seed_ids = [1, 8, 9, 14]

    for rewrite_model in [model1, model2]:
        async with BiasEvaluator(rewrite_model, reward_model) as evaluator:
            tasks = []
            for seed_id in seed_ids:
                with open(
                    f"data/evo/{run_name}/validate/seed_{seed_id}_validate/rewrite_plus_results.json", "r"
                ) as f:
                    past_validation_json = json.load(f)

                attributes = list(past_validation_json.keys())
                user_prompts = list(past_validation_json[attributes[0]].keys())

                tasks.append(asyncio.create_task(evaluator.evaluate_attributes(
                    user_prompts=user_prompts,
                    attributes=attributes,
                    baseline_rollouts=val_baselines,
                    save_dir=Path(f"data/evo/{run_name}/validate/seed_{seed_id}_validate_{rewrite_model.model_slug}"),
                )))

            await asyncio.gather(*tasks)


def plot_seed_validation_data(
    results_dir: Path,
    seed_index: int,
    model_slugs: list[str] | None = None,  # e.g., ["gemini-2-5-flash-lite-preview-09-2025", "grok-4-fast"]
):
    # Load baseline results
    if not (results_dir / "baseline_results.json").exists():
        with open(results_dir / "val_baselines" / "baseline_results.json", "r", encoding="utf-8") as f:
            baseline_results = json.load(f)
    else:
        with open(results_dir / "baseline_results.json", "r", encoding="utf-8") as f:
            baseline_results = json.load(f)

    # Load cluster info
    with open(results_dir / f"seed_{seed_index}_cluster.json", "r", encoding="utf-8") as f:
        cluster_info = json.load(f)

    # Define model directories
    model_dirs = [results_dir / f"validate/seed_{seed_index}_validate"]
    model_names = ["gpt-5-nano"]
    
    if model_slugs:
        for slug in model_slugs:
            model_dirs.append(results_dir / f"validate/seed_{seed_index}_validate_{slug}")
            model_names.append(slug)

    # Define consistent colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Load data for all models
    all_model_data = []
    for model_dir, model_name in zip(model_dirs, model_names):
        with open(model_dir / "rewrite_plus_scores.json", "r", encoding="utf-8") as f:
            validate_results = json.load(f)
        
        with open(model_dir.parent / f"seed_{seed_index}_judge.json", "r", encoding="utf-8") as f:
            judge_results = json.load(f)
        
        all_model_data.append({
            'name': model_name,
            'validate': validate_results,
            'judge': judge_results
        })

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
    attributes = list(all_model_data[0]['validate'].keys())

    # Create figure
    fig = go.Figure()

    # For each attribute, add violin plots for each model
    for attr_idx, attribute in enumerate(attributes):
        wrapped_attr = wrap_text(attribute, width=60)
        
        for model_idx, model_data in enumerate(all_model_data):
            validate_results = model_data['validate']
            judge_results = model_data['judge']
            
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
            
            # Create x-axis label with winrate
            if model_idx == 0:
                # Only show attribute name on first model's violin
                x_label = f"{wrapped_attr}<br>{model_data['name']}<br>(WR: {np.mean(winrates):.2f})"
            else:
                x_label = f"{model_data['name']}<br>(WR: {np.mean(winrates):.2f})"

            fig.add_trace(
                go.Violin(
                    y=attribute_diffs,
                    name=model_data['name'],
                    x=[attr_idx] * len(attribute_diffs),  # Group by attribute index
                    legendgroup=model_data['name'],
                    scalegroup=model_data['name'],
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
            tickmode='array',
            tickvals=list(range(len(attributes))),
            ticktext=[wrap_text(attr, width=60) for attr in attributes],
            tickangle=45
        ),
        violinmode='group',  # Group violins side-by-side
        legend=dict(
            title="Model",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
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
            model_slugs=["gemini-2.5-flash-lite-preview-09-2025", "grok-4-fast"]
        )
        fig.write_html(save_dir / f"evo-synthetic_2-seed_{seed_index}.html")

