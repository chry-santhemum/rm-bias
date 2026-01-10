# %%
"""
Experiment to test the "Triple space between words" bias.

Compares baseline responses vs the same responses with spaces replaced by triple spaces.
No LLM rewriting - just programmatic space replacement.

Supports two prompt sources:
- cluster: Load prompts from user_prompts/handpick/cluster_*.json
- prompt_mix: Use make_prompt_mix() from standard_prompts.py
"""

import json
import asyncio
import argparse
import re
from pathlib import Path
from loguru import logger

import numpy as np

from utils import timestamp, remove_outliers, set_seed_all
from api_models import GenerationModel
from reward_models import LocalRewardModel
from caller import ChatHistory


def triple_space(text: str) -> str:
    """Replace single spaces with triple spaces."""
    # Replace single spaces (but not multiple spaces) with triple spaces
    return re.sub(r' ', '   ', text)


async def main(
    dataset_name: str,
    user_prompts: list[str],
    run_dir: Path,
):
    import torch

    # Setup CUDA
    all_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    logger.info(f"Using CUDA devices: {all_cuda_devices}")

    policy_model_names = [
        "meta-llama/llama-3.2-1b-instruct",
        "mistralai/ministral-3b",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "google/gemma-2-9b-it",
        "qwen/qwen-2.5-72b-instruct",
    ]

    policy_model = GenerationModel(
        model_name=policy_model_names,
        max_par=512,
        max_tokens=1024,
        temperature=1.0,
        enable_cache=False,
    )

    reward_model = LocalRewardModel(
        model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        devices=all_cuda_devices,
        batch_size_per_device=64,
    )

    # ========== Step 1: Generate baseline responses ==========
    logger.info("=" * 60)
    logger.info("Step 1: Generating baseline responses...")
    logger.info("=" * 60)

    chat_histories = [ChatHistory.from_user(prompt) for prompt in user_prompts]
    responses = await policy_model.sample(chat_histories)

    # Collect successful responses
    baselines: list[tuple[str, str]] = []  # (user_prompt, response)
    for prompt, response in zip(user_prompts, responses):
        if response is None or not response.has_response or response.finish_reason != "stop":
            logger.warning(f"Failed to generate response for prompt: {prompt[:50]}...")
            continue
        baselines.append((prompt, response.first_response.strip()))

    logger.success(f"Generated {len(baselines)} baseline responses")

    # ========== Step 2: Create triple-spaced versions ==========
    logger.info("=" * 60)
    logger.info("Step 2: Creating triple-spaced versions...")
    logger.info("=" * 60)

    # Prepare chat histories for scoring
    baseline_chats = []
    triple_spaced_chats = []

    for prompt, response in baselines:
        baseline_chats.append(
            ChatHistory.from_user(prompt).add_assistant(response)
        )
        triple_spaced_chats.append(
            ChatHistory.from_user(prompt).add_assistant(triple_space(response))
        )

    logger.success(f"Created {len(triple_spaced_chats)} triple-spaced versions")

    # ========== Step 3: Score both versions ==========
    logger.info("=" * 60)
    logger.info("Step 3: Scoring baseline and triple-spaced responses...")
    logger.info("=" * 60)

    baseline_scores = await reward_model.async_rate(baseline_chats, use_tqdm=True)
    triple_scores = await reward_model.async_rate(triple_spaced_chats, use_tqdm=True)

    logger.success("Scoring complete")

    # ========== Step 4: Compute reward diffs ==========
    logger.info("=" * 60)
    logger.info("Step 4: Computing reward diffs...")
    logger.info("=" * 60)

    reward_diffs = []
    results_data = []

    for i, ((prompt, response), baseline_score, triple_score) in enumerate(
        zip(baselines, baseline_scores, triple_scores)
    ):
        if baseline_score.score is None or triple_score.score is None:
            continue

        diff = triple_score.score - baseline_score.score
        reward_diffs.append(diff)

        results_data.append({
            "user_prompt": prompt,
            "baseline_response": response,
            "triple_spaced_response": triple_space(response),
            "baseline_score": baseline_score.score,
            "triple_score": triple_score.score,
            "diff": diff,
        })

    # Compute stats
    cleaned_diffs = remove_outliers(reward_diffs) if reward_diffs else []
    stats = {
        "mean": float(np.mean(cleaned_diffs)) if cleaned_diffs else None,
        "stderr": float(np.std(cleaned_diffs) / np.sqrt(len(cleaned_diffs))) if len(cleaned_diffs) > 1 else None,
        "n_samples": len(reward_diffs),
    }

    if reward_diffs:
        winrates = [1 if d > 0 else 0 if d < 0 else 0.5 for d in reward_diffs]
        stats["winrate"] = float(np.mean(winrates))
        stats["winrate_stderr"] = float(np.std(winrates) / np.sqrt(len(winrates))) if len(winrates) > 1 else None
    else:
        stats["winrate"] = None
        stats["winrate_stderr"] = None

    # ========== Step 5: Save results ==========
    logger.info("=" * 60)
    logger.info("Step 5: Saving results...")
    logger.info("=" * 60)

    with open(run_dir / "stats_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(run_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # ========== Print summary ==========
    logger.success("=" * 60)
    logger.success(f"RESULTS FOR {dataset_name.upper()}")
    logger.success("=" * 60)

    baseline_mean = np.mean([r["baseline_score"] for r in results_data])
    triple_mean = np.mean([r["triple_score"] for r in results_data])

    logger.success(f"Baseline scores:      mean={baseline_mean:.4f}")
    logger.success(f"Triple-spaced scores: mean={triple_mean:.4f}")
    logger.success(f"Reward diff (triple - baseline):")
    logger.success(f"  mean={stats['mean']:+.4f}, stderr={stats['stderr']:.4f}")
    logger.success(f"  winrate={stats['winrate']:.3f} (triple beats baseline)")
    logger.success(f"  n={stats['n_samples']}")

    return {
        "diffs": reward_diffs,
        "stats": stats,
    }


if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser(description="Triple space bias experiment")
    parser.add_argument("--prompt_source", type=str, default="cluster",
                        choices=["cluster", "prompt_mix"],
                        help="Source of user prompts (default: cluster)")
    parser.add_argument("--num_prompts", type=int, default=2048,
                        help="Number of prompts when using prompt_mix (default: 2048)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for output directory (default: timestamp)")
    args = parser.parse_args()

    SEED = args.seed

    # Setup run directory and logging
    run_name = args.run_name or timestamp()
    base_run_dir = Path(f"data/exp_triple_space_bias/{run_name}")
    base_run_dir.mkdir(parents=True, exist_ok=True)

    Path("logs/exp_triple_space_bias").mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        f"logs/exp_triple_space_bias/{run_name}.log",
        enqueue=True, level="INFO",
        retention="7 days"
    )
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    if args.prompt_source == "prompt_mix":
        # === prompt_mix mode: single run with prompts from make_prompt_mix ===
        from standard_prompts import make_prompt_mix

        print(f"Testing: single space → triple space")
        print(f"Using {args.num_prompts} prompts from make_prompt_mix")
        print("No API cost (programmatic rewrite)")
        time.sleep(3)

        user_prompts = make_prompt_mix(num_total=args.num_prompts, seed=SEED)
        dataset_name = "prompt_mix"

        logger.info("=" * 80)
        logger.info(f"PROMPT_MIX: {len(user_prompts)} prompts")
        logger.info("=" * 80)

        set_seed_all(SEED)

        run_dir = base_run_dir / dataset_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "dataset": dataset_name,
            "prompt_source": "prompt_mix",
            "n_prompts": len(user_prompts),
            "transformation": "single_space -> triple_space",
            "seed": SEED,
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Loaded {len(user_prompts)} prompts from make_prompt_mix")

        results = asyncio.run(main(
            dataset_name=dataset_name,
            user_prompts=user_prompts,
            run_dir=run_dir,
        ))

        # Save summary for single run
        summary_data = {
            "run_name": run_name,
            "prompt_source": "prompt_mix",
            "overall": {
                "mean": results["stats"]["mean"],
                "stderr": results["stats"]["stderr"],
                "winrate": results["stats"]["winrate"],
                "n_samples": results["stats"]["n_samples"],
            }
        }
        with open(base_run_dir / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

    else:
        # === cluster mode: run for each cluster file ===
        HANDPICK_DIR = Path("user_prompts/handpick")

        # Find all cluster files
        cluster_files = sorted(HANDPICK_DIR.glob("cluster_*.json"), key=lambda p: int(p.stem.split("_")[1]))

        print(f"Testing: single space → triple space")
        print(f"On {len(cluster_files)} clusters from {HANDPICK_DIR}")
        print("No API cost (programmatic rewrite)")
        time.sleep(3)

        # Run experiment for each cluster
        all_results = {}
        for cluster_file in cluster_files:
            cluster_id = int(cluster_file.stem.split("_")[1])
            dataset_name = f"cluster_{cluster_id}"

            logger.info("=" * 80)
            logger.info(f"CLUSTER {cluster_id}: {cluster_file.name}")
            logger.info("=" * 80)

            # Load prompts from JSON file
            with open(cluster_file, "r") as f:
                data = json.load(f)
            user_prompts = data["prompts"]
            cluster_summary = data.get("summary", "")

            set_seed_all(SEED)

            run_dir = base_run_dir / dataset_name
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save config
            config = {
                "dataset": dataset_name,
                "cluster_id": cluster_id,
                "cluster_summary": cluster_summary,
                "prompts_file": str(cluster_file),
                "n_prompts": len(user_prompts),
                "transformation": "single_space -> triple_space",
                "user_prompts": user_prompts,
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Summary: {cluster_summary}")
            logger.info(f"Loaded {len(user_prompts)} prompts")

            results = asyncio.run(main(
                dataset_name=dataset_name,
                user_prompts=user_prompts,
                run_dir=run_dir,
            ))
            results["cluster_id"] = cluster_id
            results["cluster_summary"] = cluster_summary
            all_results[dataset_name] = results

        # Print combined summary
        logger.success("=" * 80)
        logger.success("COMBINED SUMMARY ACROSS ALL CLUSTERS")
        logger.success("=" * 80)

        # Sort by mean diff to see which clusters have positive vs negative bias
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1]["stats"]["mean"] if x[1]["stats"]["mean"] is not None else 0,
            reverse=True
        )

        for dataset_name, results in sorted_results:
            stats = results["stats"]
            cluster_id = results["cluster_id"]
            summary = results["cluster_summary"][:50] + "..." if len(results["cluster_summary"]) > 50 else results["cluster_summary"]
            if stats["mean"] is not None:
                sign = "+" if stats["mean"] > 0 else ""
                logger.success(
                    f"cluster_{cluster_id:2d}: mean={sign}{stats['mean']:.4f}, "
                    f"stderr={stats['stderr']:.4f}, "
                    f"winrate={stats['winrate']:.3f}, "
                    f"n={stats['n_samples']} | {summary}"
                )

        # Aggregate across all clusters
        all_diffs = []
        for results in all_results.values():
            all_diffs.extend(results["diffs"])

        if all_diffs:
            cleaned_all = remove_outliers(all_diffs)
            overall_mean = np.mean(cleaned_all)
            overall_stderr = np.std(cleaned_all) / np.sqrt(len(cleaned_all))
            overall_winrate = np.mean([1 if d > 0 else 0 if d < 0 else 0.5 for d in all_diffs])

            logger.success(f"\n{'OVERALL':10s}: "
                f"mean={overall_mean:+.4f}, "
                f"stderr={overall_stderr:.4f}, "
                f"winrate={overall_winrate:.3f}, "
                f"n={len(all_diffs)}"
            )

        # Save combined summary
        summary_data = {
            "run_name": run_name,
            "prompt_source": "cluster",
            "clusters": [
                {
                    "cluster_id": results["cluster_id"],
                    "cluster_summary": results["cluster_summary"],
                    "mean": results["stats"]["mean"],
                    "stderr": results["stats"]["stderr"],
                    "winrate": results["stats"]["winrate"],
                    "n_samples": results["stats"]["n_samples"],
                }
                for _, results in sorted_results
            ],
            "overall": {
                "mean": float(overall_mean) if all_diffs else None,
                "stderr": float(overall_stderr) if all_diffs else None,
                "winrate": float(overall_winrate) if all_diffs else None,
                "n_samples": len(all_diffs),
            }
        }
        with open(base_run_dir / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)
