# %%
"""
Experiment to validate discovered attributes from local RM evo runs on test sets.

Takes an evo run path with local RM teacher, loads passing attributes from validation,
and runs them on the test set using the same local reward models.

Output format matches data/evo/.../validate/ directories.
"""

import json
import asyncio
import argparse
from pathlib import Path
from dataclasses import asdict
from loguru import logger

import numpy as np
import matplotlib.pyplot as plt

from utils import timestamp
from state import BaselineRollout, AttributeStats, asdict_no_none
from api_models import GenerationModel, RewriteModel, concat_as_bullet
from reward_models import LocalRewardModel
from bias_evaluator import BiasEvaluator
from baselines import evaluate_baselines_from_prompts
from caller import ChatHistory
from filtering import (
    get_seed_indices,
    get_teacher_type,
    aggregate_across_rewriters,
    compute_partial_conjunction_stats,
    passes_partial_conjunction_criteria,
    PARTIAL_CONJUNCTION_P_THRESHOLD,
)


# Attributes to hold constant during rewrites
SAME_ATTRS = concat_as_bullet([
    "Whether or not the answer refuses or complies with the request",
    "The approximate length of the response",
    "The style and tone of the response",
])


def print_test_set_stats(run_path: Path) -> None:
    """Print partial conjunction stats for test set results (like compare_bonferroni)."""
    teacher_type = get_teacher_type(run_path)
    teacher_threshold = 0.0 if teacher_type == "local" else 0.5
    seed_indices = get_seed_indices(run_path)

    print("\n" + "=" * 100)
    print("TEST SET PARTIAL CONJUNCTION RESULTS")
    print(f"Teacher type: {teacher_type}, threshold: {teacher_threshold}")
    print(f"P-value threshold: {PARTIAL_CONJUNCTION_P_THRESHOLD}")
    print("=" * 100)

    all_passing = []
    all_failing = []

    for seed_idx in seed_indices:
        stats = compute_partial_conjunction_stats(run_path, seed_idx, teacher_type=teacher_type)
        if not stats:
            continue

        print(f"\n--- Topic {seed_idx} ({len(stats)} attributes) ---")

        for attr, data in stats.items():
            passes = passes_partial_conjunction_criteria(
                data["student_p_pc_bonferroni"],
                data["teacher_p_pc_bonferroni"],
                data["student_mean"],
                data["teacher_mean"],
                teacher_threshold=teacher_threshold,
            )

            status = "PASS" if passes else "FAIL"
            if passes:
                all_passing.append((seed_idx, attr))
            else:
                all_failing.append((seed_idx, attr))

            print(f"\n[{status}] {attr}")
            print(f"  Student: mean={data['student_mean']:+.3f} ± {data['student_ci']:.3f}, p_pc_bonf={data['student_p_pc_bonferroni']:.4f}")
            print(f"  Teacher: mean={data['teacher_mean']:+.3f} [{data['teacher_ci_lower']:.3f}, {data['teacher_ci_upper']:.3f}], p_pc_bonf={data['teacher_p_pc_bonferroni']:.4f}")

            # Per-rewriter breakdown
            for rw_name, rw_data in data["per_rewriter"].items():
                rw_short = rw_name.replace("openai_", "").replace("anthropic_", "").replace("x-ai_", "")[:12]
                print(f"    {rw_short}: student={rw_data['student_mean']:+.3f}, teacher={rw_data['teacher_mean']:+.3f}, n={rw_data['n_samples']}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Passing: {len(all_passing)} attributes")
    print(f"Failing: {len(all_failing)} attributes")

    if all_passing:
        print("\nPassing attributes:")
        for seed_idx, attr in all_passing:
            print(f"  Topic {seed_idx}: {attr[:70]}{'...' if len(attr) > 70 else ''}")


def load_models_from_config(
    config: dict,
    devices: list[str],
) -> tuple[LocalRewardModel, LocalRewardModel]:
    """Load student and teacher LocalRewardModels from an evo run config."""
    student_cfg = config["student_model"]
    teacher_cfg = config["teacher_model"]

    student_model = LocalRewardModel(
        model_name=student_cfg["model_name"],
        devices=devices,
        batch_size_per_device=student_cfg.get("batch_size", 32) // len(devices),
        attn_implementation=student_cfg.get("attn_implementation", "sdpa"),
    )

    teacher_model = LocalRewardModel(
        model_name=teacher_cfg["model_name"],
        devices=devices,
        batch_size_per_device=teacher_cfg.get("batch_size", 32) // len(devices),
        attn_implementation=teacher_cfg.get("attn_implementation", "sdpa"),
    )

    return student_model, teacher_model


def load_passing_attributes(
    evo_run_path: Path,
) -> dict[int, list[str]]:
    """
    Load attributes that pass loose (tolerant) criteria from an evo run's validation data.

    Uses aggregate_across_rewriters with strict=False, which:
    - Pools scores across all 3 rewriters
    - Requires student_mean > 0, teacher_mean < threshold (0 for local, 0.5 for API)
    - Requires Bonferroni-corrected p < 0.05 for both

    Args:
        evo_run_path: Path to the evo run directory

    Returns:
        Dict mapping seed_index -> list of passing attribute strings
    """
    teacher_type = get_teacher_type(evo_run_path)
    seed_indices = get_seed_indices(evo_run_path)
    topic_attributes: dict[int, list[str]] = {}

    for seed_idx in seed_indices:
        # aggregate_across_rewriters with strict=False returns only passing attributes
        aggregated = aggregate_across_rewriters(
            evo_run_path, seed_idx, strict=False, teacher_type=teacher_type
        )
        if aggregated:
            topic_attributes[seed_idx] = list(aggregated.keys())

    return topic_attributes


async def validate_topic(
    topic_id: int,
    attributes: list[str],
    user_prompts: list[str],
    policy_model: GenerationModel,
    rewriters: list[RewriteModel],
    student_model: LocalRewardModel,
    teacher_model: LocalRewardModel,
    n_baseline_rollouts: int,
    n_rewrite_rollouts: int,
    run_dir: Path,
):
    """Validate attributes for a single topic using local RMs.

    Returns dict of rewriter_name -> dict of attribute -> AttributeStats
    """
    topic_run_dir = run_dir / f"seed_{topic_id}_validate"
    topic_run_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate baselines and score with student model
    logger.info(f"Topic {topic_id}: Generating {n_baseline_rollouts} baseline rollouts for {len(user_prompts)} prompts...")
    baselines: dict[str, list[BaselineRollout]] = await evaluate_baselines_from_prompts(
        user_prompts=user_prompts,
        policy_model=policy_model,
        reward_model=student_model,
        n_rollouts=n_baseline_rollouts,
        save_dir=topic_run_dir / "baselines",
    )
    logger.success(f"Topic {topic_id}: Generated baselines for {len(baselines)} prompts")

    # Step 2: Score baselines with teacher model (required for local RM teacher)
    # This adds teacher scores to the existing baseline objects
    logger.info(f"Topic {topic_id}: Scoring baselines with teacher model...")
    for user_prompt, rollouts in baselines.items():
        chats = [
            ChatHistory.from_user(user_prompt).add_assistant(r.response)
            for r in rollouts
        ]
        teacher_scores = await teacher_model.async_rate(chats, use_tqdm=False)
        for r, score in zip(rollouts, teacher_scores):
            r.scores[teacher_model.model_name] = score.score

    # Save updated baselines with teacher scores
    baselines_path = topic_run_dir / "baselines" / "baselines.json"
    with open(baselines_path, "w") as f:
        json.dump(
            {user: [asdict(r) for r in rollouts] for user, rollouts in baselines.items()},
            f, indent=4, sort_keys=True
        )
    logger.success(f"Topic {topic_id}: Scored baselines with teacher model")

    # Step 3: Rewrite with all attributes using BiasEvaluator
    logger.info(f"Topic {topic_id}: Rewriting responses for {len(attributes)} attributes...")

    async with BiasEvaluator(rewrite_models=rewriters, reward_model=student_model) as evaluator:
        # Returns: dict[rewriter_model, dict[attribute, dict[user_prompt, list[Rollout|None]]]]
        evaluate_results = await evaluator.evaluate_attributes(
            user_prompts=user_prompts,
            attributes=attributes,
            same_attrs=[SAME_ATTRS] * len(attributes),
            baselines=baselines,
            n_rollouts=n_rewrite_rollouts,
            save_dir=topic_run_dir / "rewrites",
        )

    logger.success(f"Topic {topic_id}: Completed rewrites for {len(attributes)} attributes")

    # Step 4: Build validation_results structure for teacher judging
    # Structure: dict[rewriter_name, dict[seed_idx, dict[attribute, AttributeStats]]]
    # We only have one "seed" (the topic_id), so seed_idx = topic_id
    validation_results: dict[str, dict[int, dict[str, AttributeStats]]] = {}

    for rewriter_name, rewriter_stats in evaluate_results.items():
        validation_results[rewriter_name] = {
            topic_id: {
                attr: AttributeStats(attribute=attr, rollouts=rollouts)
                for attr, rollouts in rewriter_stats.items()
            }
        }

    # Step 5: Populate teacher scores (local RM)
    logger.info(f"Topic {topic_id}: Running teacher model evaluation...")

    # Need baselines in the format expected by judge_rollouts
    baselines_for_judge: dict[int, dict[str, list[BaselineRollout]]] = {topic_id: baselines}

    for rewriter_name, rewriter_stats in validation_results.items():
        await teacher_model.judge_rollouts(
            evaluate_results=rewriter_stats,
            baselines=baselines_for_judge,
            first_n_rollouts=n_rewrite_rollouts,
            first_n_user_prompts=len(user_prompts),
        )

    logger.success(f"Topic {topic_id}: Teacher evaluation complete")

    # Step 6: Save outputs per rewriter
    for rewriter_name, rewriter_stats in validation_results.items():
        seed_stats = rewriter_stats[topic_id]
        rewriter_dir = topic_run_dir / rewriter_name.replace("/", "_")
        rewriter_dir.mkdir(parents=True, exist_ok=True)

        # Save complete rollouts with teacher scores
        with open(rewriter_dir / "rollouts.json", "w") as f:
            json_data = {
                attr: {
                    user: [asdict_no_none(r) if r else None for r in rollouts]
                    for user, rollouts in attr_stats.rollouts.items()
                }
                for attr, attr_stats in seed_stats.items()
            }
            json.dump(json_data, f, indent=4, sort_keys=True)

        # Save candidate stats
        candidate_stats = []
        for attribute, attribute_stats in seed_stats.items():
            candidate_stats.append({
                "attribute": attribute,
                "student_winrate": attribute_stats.winrate("student"),
                "teacher_winrate": attribute_stats.winrate("teacher"),
            })

        with open(rewriter_dir / "candidate_stats.json", "w") as f:
            json.dump(candidate_stats, f, indent=4, sort_keys=True)

        # Create scatter plot
        valid_points = [
            (s["student_winrate"], s["teacher_winrate"])
            for s in candidate_stats
            if s["student_winrate"] is not None and s["teacher_winrate"] is not None
        ]

        if valid_points:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(
                [p[0] for p in valid_points],
                [p[1] for p in valid_points],
                c='blue', alpha=0.7, marker='o'
            )

            ax.set_xlabel('Student Winrate')
            ax.set_ylabel('Teacher Winrate')
            ax.set_title(f'Validation: {rewriter_name} - Topic {topic_id}')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(rewriter_dir / "validation_scatter.pdf")
            plt.close()

        logger.info(f"Topic {topic_id}: Saved results for {rewriter_name}")

    return validation_results


async def main(
    topic_attributes: dict[int, list[str]],
    evo_config: dict,
    run_dir: Path,
    n_baseline_rollouts: int,
    n_rewrite_rollouts: int,
    max_prompts: int | None = None,
):
    import torch

    # Setup CUDA
    all_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    logger.info(f"Using CUDA devices: {all_cuda_devices}")

    # Policy model setup (same as exp_attribute_validation.py)
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

    # Rewriters setup (same as train.py val_rewriters)
    rewriters = [
        RewriteModel(
            model_name="openai/gpt-5-mini",
            max_tokens=4096,
            reasoning="low",
            force_caller="openrouter",
        ),
        RewriteModel(
            model_name="anthropic/claude-haiku-4.5",
            max_par=128,
            max_tokens=8192,
            reasoning=6000,
            force_caller="openrouter",
        ),
        RewriteModel(
            model_name="x-ai/grok-4.1-fast",
            max_tokens=8192,
            reasoning="medium",
        ),
    ]

    # Load student and teacher models from evo config
    logger.info("Loading reward models from evo run config...")
    student_model, teacher_model = load_models_from_config(evo_config, all_cuda_devices)
    logger.success(f"Loaded student: {student_model.model_name}, teacher: {teacher_model.model_name}")

    # Process each topic sequentially
    all_results = {}
    for topic_id, attributes in topic_attributes.items():
        logger.info("=" * 80)
        logger.info(f"Processing topic {topic_id} with {len(attributes)} attributes")
        logger.info("=" * 80)

        # Load prompts from test set
        test_file = Path(f"user_prompts/handpick_test/cluster_{topic_id}.json")
        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            continue

        with open(test_file, "r") as f:
            data = json.load(f)
        user_prompts = data["prompts"]
        if max_prompts is not None:
            user_prompts = user_prompts[:max_prompts]
        topic_summary = data.get("summary", "")

        logger.info(f"Topic {topic_id} summary: {topic_summary}")
        logger.info(f"Using {len(user_prompts)} test prompts" + (f" (limited from {len(data['prompts'])})" if max_prompts else ""))

        results = await validate_topic(
            topic_id=topic_id,
            attributes=attributes,
            user_prompts=user_prompts,
            policy_model=policy_model,
            rewriters=rewriters,
            student_model=student_model,
            teacher_model=teacher_model,
            n_baseline_rollouts=n_baseline_rollouts,
            n_rewrite_rollouts=n_rewrite_rollouts,
            run_dir=run_dir,
        )
        all_results[topic_id] = results

    # Print summary
    logger.success("=" * 80)
    logger.success("VALIDATION COMPLETE")
    logger.success("=" * 80)

    for topic_id, topic_results in all_results.items():
        logger.success(f"\nTopic {topic_id}:")
        for rewriter_name, rewriter_stats in topic_results.items():
            seed_stats = rewriter_stats[topic_id]
            logger.success(f"  {rewriter_name}:")
            for attr, attr_stats in seed_stats.items():
                student_wr = attr_stats.winrate("student")
                teacher_wr = attr_stats.winrate("teacher")
                student_str = f"{student_wr:+.3f}" if student_wr is not None else "None"
                teacher_str = f"{teacher_wr:+.3f}" if teacher_wr is not None else "None"
                attr_short = attr[:50] + "..." if len(attr) > 50 else attr
                logger.success(f"    {attr_short}: student={student_str}, teacher={teacher_str}")

    # Print partial conjunction stats for test results
    print_test_set_stats(run_dir)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate attributes from local RM evo runs on test sets")
    parser.add_argument("--evo_run_path", type=Path, required=True,
                        help="Path to the evo run directory with local RM teacher")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for output directory (default: timestamp)")
    parser.add_argument("--n_baseline_rollouts", type=int, default=4,
                        help="Number of baseline rollouts per prompt")
    parser.add_argument("--n_rewrite_rollouts", type=int, default=4,
                        help="Number of rewrite rollouts per attribute")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Max prompts per topic for test runs (default: use all)")
    args = parser.parse_args()

    # Load evo run config
    config_path = args.evo_run_path / "config.json"
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        exit(1)

    with open(config_path, "r") as f:
        evo_config = json.load(f)

    # Verify this is a local RM teacher run
    teacher_type = evo_config.get("teacher_model", {}).get("type", "api")
    if teacher_type != "local":
        print(f"ERROR: Expected local teacher model, got type='{teacher_type}'")
        print("This script is for validating attributes from evo runs with local RM teachers.")
        print("Use exp_attribute_validation.py for API teacher runs.")
        exit(1)

    # Load passing attributes from validation data (using loose criteria: p < 0.05)
    print(f"Loading passing attributes from {args.evo_run_path}...")
    topic_attributes = load_passing_attributes(args.evo_run_path)

    if not topic_attributes:
        print("ERROR: No passing attributes found in the evo run.")
        print("Using loose criteria: pooled across rewriters, Bonferroni p < 0.05")
        exit(1)

    n_topics = len(topic_attributes)
    n_total_attrs = sum(len(attrs) for attrs in topic_attributes.values())

    print(f"\nFound passing attributes:")
    for topic_id, attrs in sorted(topic_attributes.items()):
        print(f"  Topic {topic_id}: {len(attrs)} attributes")
        for attr in attrs:
            print(f"    - {attr[:70]}{'...' if len(attr) > 70 else ''}")

    # Get number of prompts per topic from test files
    n_prompts_per_topic = 64  # Default
    first_topic_id = next(iter(topic_attributes.keys()))
    test_file = Path(f"user_prompts/handpick_test/cluster_{first_topic_id}.json")
    if test_file.exists():
        with open(test_file, "r") as f:
            n_prompts_per_topic = len(json.load(f)["prompts"])
    if args.max_prompts is not None:
        n_prompts_per_topic = min(n_prompts_per_topic, args.max_prompts)

    print(f"\nValidation configuration:")
    print(f"  Source evo run: {args.evo_run_path}")
    print(f"  Topics: {n_topics}")
    print(f"  Total attributes: {n_total_attrs}")
    print(f"  Prompts per topic: {n_prompts_per_topic}")
    print(f"  Baseline rollouts: {args.n_baseline_rollouts}")
    print(f"  Rewrite rollouts: {args.n_rewrite_rollouts}")
    print(f"  Student model: {evo_config['student_model']['model_name']}")
    print(f"  Teacher model: {evo_config['teacher_model']['model_name']}")
    print("\n(No API cost estimate - using local reward models)")

    # Setup run directory and logging
    run_name = args.run_name or timestamp()
    run_dir = Path(f"data/exp_attribute_validation_local/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    Path("logs/exp_attribute_validation_local").mkdir(parents=True, exist_ok=True)

    # Enable caller library logging
    logger.enable("caller")
    logger.remove()

    # Main log file (excludes caller library verbose logs)
    logger.add(
        f"logs/exp_attribute_validation_local/{run_name}.log",
        enqueue=True, level="INFO",
        filter=lambda record: not (record["name"] or "").startswith("caller"),
        retention="7 days"
    )

    # Caller library logs (WARNING and above only)
    logger.add(
        f"logs/exp_attribute_validation_local/{run_name}.log",
        enqueue=True, level="WARNING",
        filter="caller",
        retention="7 days"
    )

    # Warnings log with full backtrace for debugging
    logger.add(
        f"logs/exp_attribute_validation_local/{run_name}_warnings.log",
        enqueue=True, level="WARNING",
        backtrace=True, diagnose=True,
        retention="7 days"
    )

    # Console output
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    # Save config
    config = {
        "run_name": run_name,
        "evo_run_path": str(args.evo_run_path),
        "selection_criteria": "loose (pooled, Bonferroni p < 0.05)",
        "topic_attributes": topic_attributes,
        "n_baseline_rollouts": args.n_baseline_rollouts,
        "n_rewrite_rollouts": args.n_rewrite_rollouts,
        "max_prompts": args.max_prompts,
        "same_attrs": SAME_ATTRS,
        "student_model": evo_config["student_model"],
        "teacher_model": evo_config["teacher_model"],
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Starting validation run: {run_name}")
    logger.info(f"Results will be saved to: {run_dir}")

    asyncio.run(main(
        topic_attributes=topic_attributes,
        evo_config=evo_config,
        run_dir=run_dir,
        n_baseline_rollouts=args.n_baseline_rollouts,
        n_rewrite_rollouts=args.n_rewrite_rollouts,
        max_prompts=args.max_prompts,
    ))
