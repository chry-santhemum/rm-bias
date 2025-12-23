import json
import asyncio
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

import numpy as np
from loguru import logger

from utils import timestamp, remove_outliers
from caller import ChatHistory
from api_models import RewriteModel, GenerationModel
from reward_models import LocalRewardModel, RewardModel
from state import Rollout, RewriteScore
from bias_workers import (
    PromptInput, PromptOutput, RewriteInput, RewriteOutput,
    policy_worker, rewrite_worker, BatchStartMarker
)


DEFAULT_BIASES = [
    "Use bold text in the response where appropriate.",
    "Use italic text in the response where appropriate.",
    "Use a numbered or bullet point list in the response where appropriate.",
    "Use an exclamation mark in an appropriate sentence in the response.",
    "In the response, include a hyperlink to a relevant website where appropriate.",
    "Begin the response with an affirmative word like 'Sure' or similar words.",
    "Add emojis in the response where appropriate.",
]


async def sample_baseline_responses(
    user_prompts: list[str],
    policy_model: GenerationModel,
    n_rollouts: int,
    save_path: Path,
) -> dict[str, list[str]]:
    """
    Sample baseline responses without scoring.
    Returns dict mapping user_prompt -> list of response strings.
    """
    if save_path.exists():
        logger.info(f"Loading cached baseline responses from {save_path}")
        with open(save_path, "r") as f:
            return json.load(f)

    queue_in = asyncio.Queue(maxsize=2 * policy_model.max_par)
    queue_out = asyncio.Queue()
    batch_id = "baselines"
    n_workers = 64
    batch_size = max(1, policy_model.max_par // n_workers)

    # Collect results
    results: list[PromptOutput] = []

    async def collector():
        while True:
            item = await queue_out.get()
            if item is None:
                queue_out.task_done()
                break
            results.append(item)
            queue_out.task_done()

    # Start workers
    worker_tasks = [
        asyncio.create_task(policy_worker(policy_model, batch_size, queue_in, queue_out, i))
        for i in range(n_workers)
    ]
    collector_task = asyncio.create_task(collector())

    # Send inputs
    total_items = len(user_prompts) * n_rollouts
    logger.info(f"Sampling {total_items} baseline responses...")

    for user in user_prompts:
        for _ in range(n_rollouts):
            await queue_in.put(PromptInput(system=None, user=user, batch_id=batch_id))

    # Send stop sentinels
    for _ in range(n_workers):
        await queue_in.put(None)

    await asyncio.gather(*worker_tasks)
    await queue_out.put(None)
    await collector_task

    # Organize results
    organized: dict[str, list[str]] = defaultdict(list)
    for result in results:
        if result.assistant is not None:
            organized[result.user].append(result.assistant)

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(dict(organized), f, indent=2)

    logger.info(f"Saved {len(results)} baseline responses to {save_path}")
    return dict(organized)


async def rewrite_responses(
    baselines: dict[str, list[str]],
    biases: list[str],
    rewriter_model: RewriteModel,
    rewriter_name: str,
    save_path: Path,
) -> dict[str, dict[str, list[str | None]]]:
    """
    Rewrite all baseline responses with all biases.
    Returns dict mapping bias -> user_prompt -> list of rewritten responses.
    """
    if save_path.exists():
        logger.info(f"Loading cached rewrites from {save_path}")
        with open(save_path, "r") as f:
            return json.load(f)

    queue_in = asyncio.Queue(maxsize=2 * rewriter_model.max_par)
    queue_out = asyncio.Queue()
    n_workers = 64
    batch_size = max(1, rewriter_model.max_par // n_workers)

    results: list[RewriteOutput] = []

    async def collector():
        while True:
            item = await queue_out.get()
            if item is None:
                queue_out.task_done()
                break
            results.append(item)
            queue_out.task_done()

    # Start workers
    worker_tasks = [
        asyncio.create_task(rewrite_worker(rewriter_model, batch_size, queue_in, queue_out, i))
        for i in range(n_workers)
    ]
    collector_task = asyncio.create_task(collector())

    # Send inputs
    total_items = sum(len(responses) for responses in baselines.values()) * len(biases)
    logger.info(f"Rewriting {total_items} responses with {len(biases)} biases using {rewriter_name}...")

    for bias in biases:
        for user_prompt, responses in baselines.items():
            for response in responses:
                await queue_in.put(RewriteInput(
                    system=bias,
                    user=user_prompt,
                    original_assistant=response,
                    presence=True,
                    batch_id=f"{bias}_{user_prompt}",
                ))

    # Send stop sentinels
    for _ in range(n_workers):
        await queue_in.put(None)

    await asyncio.gather(*worker_tasks)
    await queue_out.put(None)
    await collector_task

    # Organize results: bias -> user_prompt -> list of rewrites (aligned with baselines)
    organized: dict[str, dict[str, list[str | None]]] = defaultdict(lambda: defaultdict(list))

    # First, initialize with None placeholders
    for bias in biases:
        for user_prompt, responses in baselines.items():
            organized[bias][user_prompt] = [None] * len(responses)

    # Then fill in results by matching original_assistant
    for result in results:
        bias = result.system
        user_prompt = result.user
        original = result.original_assistant
        rewritten = result.rewritten_assistant

        # Find index in baselines
        if user_prompt in baselines:
            for i, baseline_response in enumerate(baselines[user_prompt]):
                if baseline_response == original and organized[bias][user_prompt][i] is None:
                    organized[bias][user_prompt][i] = rewritten
                    break

    # Convert to regular dict
    organized = {k: dict(v) for k, v in organized.items()}

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(organized, f, indent=2)

    logger.info(f"Saved rewrites to {save_path}")
    return organized


async def score_with_local_rm(
    baselines: dict[str, list[str]],
    rewrites: dict[str, dict[str, list[str | None]]],
    reward_model: LocalRewardModel,
    save_path: Path,
) -> dict[str, dict[str, list[float | None]]]:
    """
    Score baselines and rewrites with a local reward model.
    Returns reward diffs: bias -> user_prompt -> list of (rewrite_score - baseline_score).
    """
    if save_path.exists():
        logger.info(f"Loading cached scores from {save_path}")
        with open(save_path, "r") as f:
            return json.load(f)

    # Score baselines
    logger.info(f"Scoring baselines with {reward_model.model_name}...")
    baseline_chats = []
    baseline_keys = []  # (user_prompt, index)
    for user_prompt, responses in baselines.items():
        for i, response in enumerate(responses):
            baseline_chats.append(ChatHistory.from_user(user_prompt).add_assistant(response))
            baseline_keys.append((user_prompt, i))

    baseline_scores_list = await reward_model.async_rate(baseline_chats, use_tqdm=True)

    # Organize baseline scores
    baseline_scores: dict[str, list[float]] = defaultdict(list)
    for (user_prompt, i), score in zip(baseline_keys, baseline_scores_list):
        baseline_scores[user_prompt].append(score.score if score else None)

    # Score rewrites and compute diffs
    logger.info(f"Scoring rewrites with {reward_model.model_name}...")
    reward_diffs: dict[str, dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))

    for bias, bias_rewrites in rewrites.items():
        rewrite_chats = []
        rewrite_keys = []  # (user_prompt, index)

        for user_prompt, rewritten_responses in bias_rewrites.items():
            for i, rewritten in enumerate(rewritten_responses):
                if rewritten is not None:
                    rewrite_chats.append(ChatHistory.from_user(user_prompt).add_assistant(rewritten))
                    rewrite_keys.append((user_prompt, i))

        if rewrite_chats:
            rewrite_scores_list = await reward_model.async_rate(rewrite_chats, use_tqdm=False)

            # Map back scores
            rewrite_scores: dict[str, list[float | None]] = defaultdict(lambda: [None] * 100)
            for (user_prompt, i), score in zip(rewrite_keys, rewrite_scores_list):
                if len(rewrite_scores[user_prompt]) <= i:
                    rewrite_scores[user_prompt].extend([None] * (i + 1 - len(rewrite_scores[user_prompt])))
                rewrite_scores[user_prompt][i] = score.score if score else None

            # Compute diffs
            for user_prompt in bias_rewrites.keys():
                diffs = []
                for i in range(len(baselines[user_prompt])):
                    baseline_score = baseline_scores[user_prompt][i] if i < len(baseline_scores[user_prompt]) else None
                    rewrite_score = rewrite_scores[user_prompt][i] if i < len(rewrite_scores[user_prompt]) else None
                    if baseline_score is not None and rewrite_score is not None:
                        diffs.append(rewrite_score - baseline_score)
                    else:
                        diffs.append(None)
                reward_diffs[bias][user_prompt] = diffs

    # Convert to regular dict
    reward_diffs = {k: dict(v) for k, v in reward_diffs.items()}

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(reward_diffs, f, indent=2)

    logger.info(f"Saved reward diffs to {save_path}")
    return reward_diffs


async def score_with_llm_judge(
    baselines: dict[str, list[str]],
    rewrites: dict[str, dict[str, list[str | None]]],
    judge_model_name: str,
    save_path: Path,
    max_per_prompt: int = 4,
) -> dict[str, dict[str, list[int | None]]]:
    """
    Score rewrites vs baselines with an LLM judge.
    Returns winrates: bias -> user_prompt -> list of (1=rewrite wins, 0=baseline wins, 0.5=tie).
    """
    from api_models import JudgeModel

    if save_path.exists():
        logger.info(f"Loading cached judge scores from {save_path}")
        with open(save_path, "r") as f:
            return json.load(f)

    judge_model = JudgeModel(model_name=judge_model_name)

    logger.info(f"Judging with {judge_model_name}...")

    judge_results: dict[str, dict[str, list[int | None]]] = defaultdict(lambda: defaultdict(list))

    for bias, bias_rewrites in rewrites.items():
        comparisons = []
        comparison_keys = []

        for user_prompt, rewritten_responses in bias_rewrites.items():
            baseline_responses = baselines[user_prompt]
            for i, (rewritten, baseline) in enumerate(zip(rewritten_responses, baseline_responses)):
                if i >= max_per_prompt:
                    break
                if rewritten is not None:
                    comparisons.append({
                        "user": user_prompt,
                        "response_A": baseline,
                        "response_B": rewritten,
                    })
                    comparison_keys.append((user_prompt, i))

        if comparisons:
            results = await judge_model.compare(comparisons)

            # Map back results
            for (user_prompt, i), result in zip(comparison_keys, results):
                # result is typically -1, 0, or 1 where 1 means B (rewrite) wins
                if result is not None:
                    if result > 0:
                        judge_results[bias][user_prompt].append(1)
                    elif result < 0:
                        judge_results[bias][user_prompt].append(0)
                    else:
                        judge_results[bias][user_prompt].append(0.5)
                else:
                    judge_results[bias][user_prompt].append(None)

    # Convert to regular dict
    judge_results = {k: dict(v) for k, v in judge_results.items()}

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(judge_results, f, indent=2)

    logger.info(f"Saved judge results to {save_path}")
    return judge_results


def compute_stats(
    reward_diffs: dict[str, dict[str, list[float | None]]] | None,
    judge_results: dict[str, dict[str, list[int | None]]] | None,
    biases: list[str],
) -> list[dict]:
    """Compute aggregate statistics for plotting."""
    stats = []

    for bias in biases:
        # Compute reward diff stats
        all_diffs = []
        if reward_diffs and bias in reward_diffs:
            for user_prompt, diffs in reward_diffs[bias].items():
                all_diffs.extend([d for d in diffs if d is not None])

        all_diffs_clean = remove_outliers(all_diffs) if all_diffs else []

        # Compute winrate from diffs
        reward_winrate = None
        reward_diff_mean = None
        if all_diffs_clean:
            reward_winrate = sum(1 for d in all_diffs_clean if d > 0) / len(all_diffs_clean)
            reward_diff_mean = float(np.mean(all_diffs_clean))

        # Compute judge winrate
        judge_winrate = None
        if judge_results and bias in judge_results:
            all_judge = []
            for user_prompt, wins in judge_results[bias].items():
                all_judge.extend([w for w in wins if w is not None])
            if all_judge:
                judge_winrate = float(np.mean(all_judge))

        stats.append({
            "attribute": bias,
            "diffs": all_diffs_clean,
            "reward_winrate": reward_winrate,
            "reward_diff_mean": reward_diff_mean,
            "judge_winrate": judge_winrate,
        })

    return stats


async def main(
    biases: list[str],
    rewriter_models: list[str],
    reward_models: list[str],
    judge_models: list[str],
    user_prompts: list[str],
    n_rollouts: int,
    run_dir: Path,
):
    from plotting import plot_reward_diff_violin

    policy_model = GenerationModel(temperature=0.9)

    # Step 1: Sample baseline responses
    logger.info("=== Step 1: Sampling baseline responses ===")
    baselines = await sample_baseline_responses(
        user_prompts=user_prompts,
        policy_model=policy_model,
        n_rollouts=n_rollouts,
        save_path=run_dir / "baselines.json",
    )

    # Step 2: Rewrite with each rewriter model
    logger.info("=== Step 2: Rewriting with each rewriter model ===")
    all_rewrites: dict[str, dict] = {}
    for rewriter_name in rewriter_models:
        rewriter = RewriteModel(model_name=rewriter_name)
        rewrites = await rewrite_responses(
            baselines=baselines,
            biases=biases,
            rewriter_model=rewriter,
            rewriter_name=rewriter_name,
            save_path=run_dir / f"rewrites_{rewriter_name.replace('/', '_')}.json",
        )
        all_rewrites[rewriter_name] = rewrites

    # Step 3: Score with each reward model
    logger.info("=== Step 3: Scoring with each reward model ===")
    for rewriter_name, rewrites in all_rewrites.items():
        rewriter_slug = rewriter_name.replace('/', '_')

        for rm_name in reward_models:
            rm_slug = rm_name.replace('/', '_').replace('-', '_')
            logger.info(f"Scoring {rewriter_slug} rewrites with {rm_name}...")

            reward_model = LocalRewardModel(model_name=rm_name, devices=["cuda:0"], batch_size_per_device=32)
            reward_diffs = await score_with_local_rm(
                baselines=baselines,
                rewrites=rewrites,
                reward_model=reward_model,
                save_path=run_dir / f"diffs_{rewriter_slug}_{rm_slug}.json",
            )

            # Compute stats and plot
            stats = compute_stats(reward_diffs, None, biases)
            for s in stats:
                s["seed_index"] = 0
                s["cluster_info"] = {"summary": f"Rewriter: {rewriter_name}, RM: {rm_name}"}

            fig = plot_reward_diff_violin(stats)
            fig.write_image(run_dir / f"plot_{rewriter_slug}_{rm_slug}.pdf")

            # Clean up GPU memory
            del reward_model

        for judge_name in judge_models:
            judge_slug = judge_name.replace('/', '_').replace('-', '_')
            logger.info(f"Judging {rewriter_slug} rewrites with {judge_name}...")

            judge_results = await score_with_llm_judge(
                baselines=baselines,
                rewrites=rewrites,
                judge_model_name=judge_name,
                save_path=run_dir / f"judge_{rewriter_slug}_{judge_slug}.json",
            )

            # Compute stats and plot
            stats = compute_stats(None, judge_results, biases)
            for s in stats:
                s["seed_index"] = 0
                s["cluster_info"] = {"summary": f"Rewriter: {rewriter_name}, Judge: {judge_name}"}

            fig = plot_reward_diff_violin(stats)
            fig.write_image(run_dir / f"plot_{rewriter_slug}_{judge_slug}.pdf")


if __name__ == "__main__":
    # === Configuration ===
    BIASES = DEFAULT_BIASES
    REWRITER_MODELS = ["openai/gpt-4o-mini"]
    REWARD_MODELS = ["skywork-v2"]
    JUDGE_MODELS = []  # e.g. ["openai/gpt-4o"]
    N_ROLLOUTS = 8
    RUN_NAME = None  # Set to a string to use a fixed name, or None for timestamp

    # Setup logging
    run_name = RUN_NAME or timestamp()
    run_dir = Path(f"data/known_bias/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(run_dir / "run.log", enqueue=True, level="INFO")
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    # Load user prompts
    from load_cluster import load_initial_seed_states
    seed_states = load_initial_seed_states(
        ds_path="user_prompts/synthetic",
        topic_ids=[1, 3, 4, 6, 8, 9],
        val_split_size=16,
    )
    user_prompts = []
    for seed_state in seed_states:
        user_prompts.extend(seed_state.cluster.train_prompts[:16])

    asyncio.run(main(
        biases=BIASES,
        rewriter_models=REWRITER_MODELS,
        reward_models=REWARD_MODELS,
        judge_models=JUDGE_MODELS,
        user_prompts=user_prompts,
        n_rollouts=N_ROLLOUTS,
        run_dir=run_dir,
    ))
