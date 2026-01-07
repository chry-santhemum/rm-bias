"""
Temporary script to re-run teacher model judging on saved validation rollouts.
This is useful when the original run failed partway through teacher judging.

Usage:
    python rerun_teacher_judge.py --run_path data/evo/20260107-075251-list_reverse-handpick-plus
    python rerun_teacher_judge.py --run_path data/evo/20260107-075251-list_reverse-handpick-plus --test_n 3
"""

import json
import argparse
import asyncio
from pathlib import Path
from dataclasses import asdict

from state import AttributeStats, Rollout, RewriteScore, BaselineRollout, asdict_no_none
from reward_models import APIRewardModel
from plotting import plot_validation_data


def load_baselines(baselines_dir: Path, seed_indices: list[int]) -> dict[int, dict[str, list[BaselineRollout]]]:
    """Load baseline rollouts for each seed from the baselines directory."""
    baselines = {}
    for seed_idx in seed_indices:
        baseline_path = baselines_dir / f"cluster_{seed_idx}.json"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)

        baselines[seed_idx] = {
            user_prompt: [BaselineRollout(**r) for r in rollouts]
            for user_prompt, rollouts in baseline_data.items()
        }

    return baselines


def load_rollouts_as_attribute_stats(
    rollouts_path: Path,
    first_n_user_prompts: int = 0,
) -> dict[str, AttributeStats]:
    """
    Load rollouts.json and convert to AttributeStats format for judge_rollouts().

    Handles two formats:
    1. Old format: student_diff (float), student_score (float)
    2. New format: student_score (dict with score, raw_score, reasoning, model_name)

    Args:
        rollouts_path: Path to rollouts.json file
        first_n_user_prompts: If > 0, only load first N user prompts per attribute (for testing)

    Returns:
        dict mapping attribute -> AttributeStats
    """
    with open(rollouts_path, "r") as f:
        rollouts_data = json.load(f)

    results = {}
    for attribute, user_prompts_data in rollouts_data.items():
        rollouts_dict: dict[str, list[Rollout | None]] = {}

        user_prompt_count = 0
        for user_prompt, rollouts_list in user_prompts_data.items():
            if first_n_user_prompts > 0 and user_prompt_count >= first_n_user_prompts:
                break
            user_prompt_count += 1

            converted_rollouts = []
            for r in rollouts_list:
                if r is None:
                    converted_rollouts.append(None)
                    continue

                # Handle student_score - can be dict (new format) or float (old format)
                student_score_data = r.get("student_score")
                if isinstance(student_score_data, dict):
                    # New format: nested dict with score, raw_score, etc.
                    student_score = RewriteScore(
                        score=student_score_data.get("score"),
                        raw_score=student_score_data.get("raw_score"),
                        reasoning=student_score_data.get("reasoning"),
                        model_name=student_score_data.get("model_name", "unknown"),
                    )
                else:
                    # Old format: student_score is raw score (float), student_diff is the diff
                    student_score = RewriteScore(
                        score=r.get("student_diff"),  # diff is what we care about
                        raw_score=student_score_data,  # raw score
                        reasoning=None,
                        model_name="unknown",
                    )

                # Convert teacher_score if it exists (may be None)
                teacher_score = None
                teacher_score_data = r.get("teacher_score")
                if teacher_score_data is not None:
                    if isinstance(teacher_score_data, dict):
                        teacher_score = RewriteScore(
                            score=teacher_score_data.get("score"),
                            raw_score=teacher_score_data.get("raw_score"),
                            reasoning=teacher_score_data.get("reasoning"),
                            model_name=teacher_score_data.get("model_name", "unknown"),
                        )
                    else:
                        # Old format: just a float
                        teacher_score = RewriteScore(
                            score=teacher_score_data,
                            raw_score=None,
                            reasoning=None,
                            model_name="unknown",
                        )

                rollout = Rollout(
                    rewritten_response=r["rewritten_response"],
                    baseline_response=r["baseline_response"],
                    student_score=student_score,
                    teacher_score=teacher_score,
                    policy_model=r.get("policy_model"),
                )
                converted_rollouts.append(rollout)

            rollouts_dict[user_prompt] = converted_rollouts

        results[attribute] = AttributeStats(
            attribute=attribute,
            rollouts=rollouts_dict,
        )

    return results


def save_rollouts_and_stats(
    rewriter_dir: Path,
    seed_stats: dict[str, AttributeStats],
):
    """Save rollouts.json and candidate_stats.json for a rewriter/seed."""
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


async def main():
    parser = argparse.ArgumentParser(description="Re-run teacher model judging on saved validation rollouts")
    parser.add_argument("--run_path", type=str, required=True, help="Path to the run directory")
    parser.add_argument("--test_n", type=int, default=0, help="Only judge first N user prompts per attribute (0=all)")
    parser.add_argument("--skip_plot", action="store_true", help="Skip plotting after judging")
    parser.add_argument("--judge_first_n_rollouts", type=int, default=1, help="Number of rollouts to judge per user prompt")
    parser.add_argument("--judge_first_n_user_prompts", type=int, default=64, help="Number of user prompts to judge per attribute")
    args = parser.parse_args()

    run_path = Path(args.run_path)
    if not run_path.exists():
        raise FileNotFoundError(f"Run path not found: {run_path}")

    # Load config to get dataset name and seed indices
    config_path = run_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset = config["dataset"]
    seed_indices = config["topic_ids"]

    print(f"Run: {run_path.name}")
    print(f"Dataset: {dataset}")
    print(f"Seeds: {seed_indices}")
    if args.test_n > 0:
        print(f"TEST MODE: Only judging first {args.test_n} user prompts per attribute")

    # Load baselines
    baselines_dir = Path(f"data/baselines/{dataset}/val")
    print(f"\nLoading baselines from {baselines_dir}...")
    baselines = load_baselines(baselines_dir, seed_indices)

    # Initialize teacher model
    print("\nInitializing teacher model (claude-sonnet-4.5)...")
    teacher_model = APIRewardModel(
        model_name="anthropic/claude-sonnet-4.5",
        max_par=512,
        force_caller="openrouter",
        max_tokens=1050,
        reasoning=1024,
    )

    # Find all rewriters
    validate_dir = run_path / "validate"
    sample_seed_dir = validate_dir / f"seed_{seed_indices[0]}_validate"
    rewriter_names = [d.name for d in sample_seed_dir.iterdir() if d.is_dir()]
    print(f"Rewriters: {rewriter_names}")

    # Process each rewriter
    for rewriter_name in rewriter_names:
        print(f"\n{'='*60}")
        print(f"Processing rewriter: {rewriter_name}")
        print(f"{'='*60}")

        # Build evaluate_results structure for all seeds
        evaluate_results: dict[int, dict[str, AttributeStats]] = {}

        for seed_idx in seed_indices:
            rollouts_path = validate_dir / f"seed_{seed_idx}_validate" / rewriter_name / "rollouts.json"
            if not rollouts_path.exists():
                print(f"  WARNING: No rollouts found for seed {seed_idx}, skipping")
                continue

            # Use test_n to limit user prompts if specified
            load_limit = args.test_n if args.test_n > 0 else 0
            seed_stats = load_rollouts_as_attribute_stats(rollouts_path, first_n_user_prompts=load_limit)
            evaluate_results[seed_idx] = seed_stats
            print(f"  Loaded seed {seed_idx}: {len(seed_stats)} attributes")

        if not evaluate_results:
            print(f"  No data found for rewriter {rewriter_name}, skipping")
            continue

        # Run teacher judging
        judge_n_prompts = args.test_n if args.test_n > 0 else args.judge_first_n_user_prompts
        print(f"\n  Running teacher judge (first_n_rollouts={args.judge_first_n_rollouts}, first_n_user_prompts={judge_n_prompts})...")

        await teacher_model.judge_rollouts(
            evaluate_results=evaluate_results,
            baselines=baselines,
            first_n_rollouts=args.judge_first_n_rollouts,
            first_n_user_prompts=judge_n_prompts,
        )

        # Save results for each seed
        for seed_idx, seed_stats in evaluate_results.items():
            rewriter_dir = validate_dir / f"seed_{seed_idx}_validate" / rewriter_name
            print(f"  Saving results for seed {seed_idx} to {rewriter_dir}")
            save_rollouts_and_stats(rewriter_dir, seed_stats)

    # Run plotting
    if not args.skip_plot:
        print(f"\n{'='*60}")
        print("Generating plots...")
        print(f"{'='*60}")
        plot_validation_data(run_path=run_path, write_path=run_path)
        print("Plots saved!")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
