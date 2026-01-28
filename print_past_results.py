"""
Script to collect and output past run results from data/evo/.

For each run (excluding chatgpt runs), outputs results to
data/print_past_results/{run_name}.json with attributes organized by seed
for both strict and tolerant filtering modes.
"""

import json
from pathlib import Path

import filtering
from filtering import (
    aggregate_across_rewriters,
    get_seed_indices,
    get_teacher_type,
)

# Set p-value threshold for filtering
filtering.P_VALUE_THRESHOLD = 0.05


def _get_cluster_summary(run_path: Path, seed_index: int) -> str | None:
    """Get the cluster summary for a seed from step stats files."""
    # Try to find the cluster summary in any step_*_stats directory
    for step_dir in sorted(run_path.glob("step_*_stats")):
        seed_file = step_dir / f"seed_{seed_index}.json"
        if seed_file.exists():
            with open(seed_file, "r") as f:
                data = json.load(f)
            if data and len(data) > 0:
                meta = data[0].get("meta", {})
                return meta.get("cluster_summary")
    return None


def _format_attribute(attr_name: str, data: dict, cluster_summary: str | None) -> dict:
    """Format an attribute's data for output."""
    return {
        "cluster_summary": cluster_summary,
        "attribute": attr_name,
        "reward_diff_mean": data["student_mean"],
        "reward_diff_ci": data["student_ci"],
        "reward_diff_p": data["student_p"],
        "reward_diff_p_bonferroni": data["student_p_bonferroni"],
        "judge_winrate_mean": data["teacher_mean"],
        "judge_winrate_ci": data["teacher_ci"],
        "judge_winrate_p": data["teacher_p"],
        "judge_winrate_p_bonferroni": data["teacher_p_bonferroni"],
        "n_hypotheses": data["n_hypotheses"],
    }


def collect_run_results(run_path: Path) -> dict | None:
    """
    Collect valid attributes and their metrics for a single run.

    Returns dict with:
        - run_name: str
        - config: dict
        - strict: dict mapping seed_index -> list of attributes
        - tolerant: dict mapping seed_index -> list of attributes

    Returns None if no validation data exists.
    """
    # Load config (exclude 'prompts' field)
    config_path = run_path / "config.json"
    if not config_path.exists():
        return None

    with open(config_path, "r") as f:
        config = json.load(f)

    # Remove prompts field
    config.pop("prompts", None)

    # Get seed indices
    seed_indices = get_seed_indices(run_path)
    if not seed_indices:
        return None

    # Collect attributes for both strict and tolerant modes
    strict_by_seed: dict[int, list[dict]] = {}
    tolerant_by_seed: dict[int, list[dict]] = {}

    # Auto-detect teacher type from config
    teacher_type = get_teacher_type(run_path)

    for seed_index in seed_indices:
        cluster_summary = _get_cluster_summary(run_path, seed_index)

        # Strict mode
        strict_aggregated = aggregate_across_rewriters(
            run_path, seed_index, strict=True, teacher_type=teacher_type
        )
        strict_by_seed[seed_index] = [
            _format_attribute(attr_name, data, cluster_summary)
            for attr_name, data in strict_aggregated.items()
        ]

        # Tolerant mode
        tolerant_aggregated = aggregate_across_rewriters(
            run_path, seed_index, strict=False, teacher_type=teacher_type
        )
        tolerant_by_seed[seed_index] = [
            _format_attribute(attr_name, data, cluster_summary)
            for attr_name, data in tolerant_aggregated.items()
        ]

    # Check if any data exists
    has_strict = any(attrs for attrs in strict_by_seed.values())
    has_tolerant = any(attrs for attrs in tolerant_by_seed.values())
    if not has_strict and not has_tolerant:
        return None

    return {
        "run_name": run_path.name,
        "config": config,
        "strict": strict_by_seed,
        "tolerant": tolerant_by_seed,
    }


def main():
    evo_dir = Path("data/evo")
    output_dir = Path("data/print_past_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all run directories, excluding chatgpt runs
    run_dirs = sorted(evo_dir.iterdir())
    for run_path in run_dirs:
        if not run_path.is_dir():
            continue

        # Skip chatgpt runs
        if "chatgpt" in run_path.name.lower():
            print(f"Skipping {run_path.name} (chatgpt run)")
            continue

        print(f"Processing {run_path.name}...")
        results = collect_run_results(run_path)

        if results is None:
            print(f"  Skipped (no valid data)")
            continue

        # Write output file
        output_path = output_dir / f"{run_path.name}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Count results
        n_strict = sum(len(attrs) for attrs in results["strict"].values())
        n_tolerant = sum(len(attrs) for attrs in results["tolerant"].values())
        n_seeds = len(results["strict"])
        print(f"  {n_seeds} seeds: {n_strict} strict, {n_tolerant} tolerant (p<{filtering.P_VALUE_THRESHOLD})")


if __name__ == "__main__":
    main()
