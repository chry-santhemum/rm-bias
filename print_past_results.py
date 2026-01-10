"""
Script to collect and output past run results from data/evo/.

Filters for attributes that pass strict mode criteria across all rewriters,
and outputs results to data/print_past_results/{run_name}.json.
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


ALL_REWRITERS = [
    "openai_gpt-5-mini",
    "anthropic_claude-haiku-4.5",
    "x-ai_grok-4.1-fast",
]


def _get_seed_indices(run_path: Path) -> list[int]:
    """Find all seed indices in a run's validate directory."""
    validate_dir = run_path / "validate"
    if not validate_dir.exists():
        return []

    pattern = re.compile(r"seed_(\d+)_validate")
    seed_indices = []
    for item in validate_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                seed_indices.append(int(match.group(1)))
    return sorted(seed_indices)


def _load_rollouts_data(
    run_path: Path,
    seed_index: int,
    rewriter_name: str,
) -> dict[str, list[dict]]:
    """
    Load per-sample scores from rollouts.json.

    Returns dict mapping attribute -> list of sample dicts, each with:
        - student_score: float (reward diff for this sample)
        - teacher_score: float (normalized to [0, 1] from [-1, 1])
    """
    rollouts_path = (
        run_path
        / "validate"
        / f"seed_{seed_index}_validate"
        / rewriter_name
        / "rollouts.json"
    )

    if not rollouts_path.exists():
        return {}

    with open(rollouts_path, "r") as f:
        rollouts = json.load(f)

    result: dict[str, list[dict]] = {}
    for attribute, prompts in rollouts.items():
        samples = []
        for prompt, rollout_list in prompts.items():
            for rollout in rollout_list:
                if rollout is None:
                    continue
                student_data = rollout.get("student_score")
                teacher_data = rollout.get("teacher_score")
                if student_data is None or teacher_data is None:
                    continue
                student = student_data.get("score")
                teacher_raw = teacher_data.get("score")
                if student is not None and teacher_raw is not None:
                    # Convert teacher from [-1, 1] to [0, 1]
                    teacher = (teacher_raw + 1) / 2
                    samples.append({
                        "student_score": student,
                        "teacher_score": teacher,
                    })
        result[attribute] = samples

    return result


def _aggregate_across_rewriters(
    run_path: Path,
    seed_index: int,
    strict: bool,
) -> dict[str, dict]:
    """
    Aggregate per-sample scores across all rewriters for a seed.

    Args:
        run_path: Path to the run directory
        seed_index: Which seed to load
        strict: If True, only include attributes where ALL rewriters have
            mean student_score >= 0 AND mean teacher_score <= 0.5.
            If False, aggregate all samples first, then filter on aggregated mean.

    Returns:
        Dict mapping attribute -> {
            "student_scores": list[float],  # all samples
            "teacher_scores": list[float],  # all samples
            "student_mean": float,
            "teacher_mean": float,
            "student_ci": float,  # 95% CI half-width
            "teacher_ci": float,  # 95% CI half-width
        }
    """
    # Load rollouts from all rewriters
    rewriter_data: dict[str, dict[str, list[dict]]] = {}
    for rewriter in ALL_REWRITERS:
        rewriter_data[rewriter] = _load_rollouts_data(run_path, seed_index, rewriter)

    # Find all attributes across rewriters
    all_attributes = set()
    for data in rewriter_data.values():
        all_attributes.update(data.keys())

    if strict:
        # Strict mode: filter to attributes where ALL rewriters pass threshold
        valid_attributes = set()
        for attribute in all_attributes:
            passes_all = True
            for rewriter in ALL_REWRITERS:
                samples = rewriter_data[rewriter].get(attribute, [])
                if not samples:
                    passes_all = False
                    break
                student_mean = np.mean([s["student_score"] for s in samples])
                teacher_mean = np.mean([s["teacher_score"] for s in samples])
                if student_mean < 0 or teacher_mean > 0.5:
                    passes_all = False
                    break
            if passes_all:
                valid_attributes.add(attribute)
    else:
        # Non-strict: will filter after aggregation
        valid_attributes = all_attributes

    # Aggregate samples across rewriters
    result: dict[str, dict] = {}
    for attribute in valid_attributes:
        all_student = []
        all_teacher = []
        for rewriter in ALL_REWRITERS:
            samples = rewriter_data[rewriter].get(attribute, [])
            all_student.extend([s["student_score"] for s in samples])
            all_teacher.extend([s["teacher_score"] for s in samples])

        if not all_student:
            continue

        student_mean = np.mean(all_student)
        teacher_mean = np.mean(all_teacher)

        # Non-strict filtering: check aggregated means
        if not strict:
            if student_mean < 0 or teacher_mean > 0.5:
                continue

        # Compute 95% CI (t-distribution)
        n = len(all_student)
        if n > 1:
            student_sem = scipy_stats.sem(all_student)
            teacher_sem = scipy_stats.sem(all_teacher)
            t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
            student_ci = t_crit * student_sem
            teacher_ci = t_crit * teacher_sem
        else:
            student_ci = 0.0
            teacher_ci = 0.0

        result[attribute] = {
            "student_scores": all_student,
            "teacher_scores": all_teacher,
            "student_mean": float(student_mean),
            "teacher_mean": float(teacher_mean),
            "student_ci": float(student_ci),
            "teacher_ci": float(teacher_ci),
        }

    return result


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


def collect_run_results(run_path: Path) -> dict | None:
    """
    Collect valid attributes and their metrics for a single run.

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
    seed_indices = _get_seed_indices(run_path)
    if not seed_indices:
        return None

    # Collect attributes across all seeds
    attributes = []
    for seed_index in seed_indices:
        aggregated = _aggregate_across_rewriters(run_path, seed_index, strict=True)
        cluster_summary = _get_cluster_summary(run_path, seed_index)

        for attr_name, data in aggregated.items():
            attributes.append({
                "seed": seed_index,
                "cluster_summary": cluster_summary,
                "attribute": attr_name,
                "reward_diff_mean": data["student_mean"],
                "reward_diff_ci": data["student_ci"],
                "judge_winrate_mean": data["teacher_mean"],
                "judge_winrate_ci": data["teacher_ci"],
            })

    if not attributes:
        return None

    return {
        "run_name": run_path.name,
        "config": config,
        "attributes": attributes,
    }


def main():
    evo_dir = Path("data/evo")
    output_dir = Path("data/print_past_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all run directories
    run_dirs = sorted(evo_dir.iterdir())
    for run_path in run_dirs:
        if not run_path.is_dir():
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

        n_attrs = len(results["attributes"])
        print(f"  Found {n_attrs} valid attributes")


if __name__ == "__main__":
    main()
