# ABOUTME: Validates rewrite fidelity for exp_attribute_validation runs.
# ABOUTME: Checks that baselines don't contain attributes and rewrites do.

import asyncio
import json
from pathlib import Path
from collections import defaultdict

from api_models import JudgeModel


async def run_validation(run_path: Path) -> dict:
    """
    Validate that:
    1. Baseline responses do NOT contain the attribute
    2. Rewritten responses DO contain the attribute

    Returns a dict with results per attribute.
    """
    # Load config
    with open(run_path / "config.json") as f:
        config = json.load(f)

    # Initialize judge model
    judge_model = JudgeModel(
        model_name="openai/gpt-4.1-mini",
        max_tokens=2048,
        max_par=512,
    )

    # Get all seeds and rewriters
    seed_dirs = sorted([d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("seed_")])

    # Collect all pairs to judge
    baseline_pairs = []  # (attribute, response_text)
    baseline_indices = []  # (seed_idx, attribute, prompt, rollout_idx)

    rewrite_pairs = []
    rewrite_indices = []  # (seed_idx, attribute, rewriter, prompt, rollout_idx)

    for seed_dir in seed_dirs:
        seed_idx = seed_dir.name.split("_")[1]
        attributes = config["topic_attributes"].get(seed_idx, [])

        if not attributes:
            continue

        # Load baselines
        baselines_path = seed_dir / "baselines" / "baselines.json"
        with open(baselines_path) as f:
            baselines = json.load(f)

        # Collect baseline pairs (for each attribute, check all baselines)
        for attr in attributes:
            for prompt, rollouts in baselines.items():
                for rollout_idx, rollout in enumerate(rollouts):
                    if rollout and rollout.get("response"):
                        # Format as conversation for the judge
                        conversation = f"User: {prompt}\n\nAssistant: {rollout['response']}"
                        baseline_pairs.append((attr, conversation))
                        baseline_indices.append((seed_idx, attr, prompt, rollout_idx))

        # Load rewrites for each rewriter
        rewrites_dir = seed_dir / "rewrites"
        rewriters = sorted([d.name for d in rewrites_dir.iterdir() if d.is_dir()])

        for rewriter in rewriters:
            rollouts_path = rewrites_dir / rewriter / "rollouts.json"
            if not rollouts_path.exists():
                continue

            with open(rollouts_path) as f:
                rollouts = json.load(f)

            for attr, prompts in rollouts.items():
                for prompt, rollout_list in prompts.items():
                    for rollout_idx, rollout in enumerate(rollout_list):
                        if rollout and rollout.get("rewritten_response"):
                            conversation = f"User: {prompt}\n\nAssistant: {rollout['rewritten_response']}"
                            rewrite_pairs.append((attr, conversation))
                            rewrite_indices.append((seed_idx, attr, rewriter, prompt, rollout_idx))

    print(f"Baseline pairs to judge: {len(baseline_pairs)}")
    print(f"Rewrite pairs to judge: {len(rewrite_pairs)}")
    print(f"Total judge calls: {len(baseline_pairs) + len(rewrite_pairs)}")

    # Judge all pairs
    print("\nJudging baseline presence...")
    baseline_results = await judge_model.judge_presence(baseline_pairs)

    print("Judging rewrite presence...")
    rewrite_results = await judge_model.judge_presence(rewrite_pairs)

    # Aggregate results by attribute
    baseline_stats = defaultdict(lambda: {"present": 0, "absent": 0, "failed": 0})
    rewrite_stats = defaultdict(lambda: {"present": 0, "absent": 0, "failed": 0})

    for idx, result in zip(baseline_indices, baseline_results):
        seed_idx, attr, prompt, rollout_idx = idx
        if result is None:
            baseline_stats[attr]["failed"] += 1
        elif result:
            baseline_stats[attr]["present"] += 1
        else:
            baseline_stats[attr]["absent"] += 1

    for idx, result in zip(rewrite_indices, rewrite_results):
        seed_idx, attr, rewriter, prompt, rollout_idx = idx
        if result is None:
            rewrite_stats[attr]["failed"] += 1
        elif result:
            rewrite_stats[attr]["present"] += 1
        else:
            rewrite_stats[attr]["absent"] += 1

    # Build final results table
    results = {}
    all_attributes = set(baseline_stats.keys()) | set(rewrite_stats.keys())

    for attr in sorted(all_attributes):
        b_stats = baseline_stats[attr]
        r_stats = rewrite_stats[attr]

        b_total = b_stats["present"] + b_stats["absent"]
        r_total = r_stats["present"] + r_stats["absent"]

        results[attr] = {
            "baseline_present_pct": round(100 * b_stats["present"] / b_total, 1) if b_total > 0 else None,
            "baseline_present": b_stats["present"],
            "baseline_total": b_total,
            "baseline_failed": b_stats["failed"],
            "rewrite_present_pct": round(100 * r_stats["present"] / r_total, 1) if r_total > 0 else None,
            "rewrite_present": r_stats["present"],
            "rewrite_total": r_total,
            "rewrite_failed": r_stats["failed"],
        }

    return results


async def main():
    run_path = Path("data/exp_attribute_validation/20260112-162826")

    results = await run_validation(run_path)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    output = json.dumps(results, indent=2)
    print(output)

    # Save results
    output_path = run_path / "fidelity_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
