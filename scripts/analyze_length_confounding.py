import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def word_count(text: str) -> int:
    """Crude word count using whitespace splitting."""
    return len(text.split())


def extract_from_rollout(resp: dict) -> tuple[float, int] | None:
    """Extract student_diff and length_diff from a rollout entry.

    Returns (student_diff, length_diff) or None if data is missing.
    """
    if resp is None:
        return None

    baseline_len = word_count(resp["baseline_response"])
    rewritten_len = word_count(resp["rewritten_response"])

    # Handle both old format (student_diff) and new format (student_score.score)
    if "student_diff" in resp:
        student_diff = resp["student_diff"]
    elif "student_score" in resp:
        student_diff = resp["student_score"]["score"]
    else:
        return None

    if student_diff is None:
        return None

    return student_diff, rewritten_len - baseline_len


def collect_from_step_file(step_file: Path) -> list[tuple[float, int]]:
    """Collect (student_diff, length_diff) pairs from a step stats file.

    Step files are lists of attribute stats, each with rollouts.
    """
    with open(step_file) as f:
        data = json.load(f)

    results = []
    for attr_stats in data:
        rollouts = attr_stats.get("rollouts", {})
        for prompt, responses in rollouts.items():
            for resp in responses:
                extracted = extract_from_rollout(resp)
                if extracted:
                    results.append(extracted)

    return results


def collect_from_validate_dir(validate_dir: Path) -> list[tuple[float, int]]:
    """Collect (student_diff, length_diff) pairs from all rewriters in a validate dir.

    Validation rollouts are stored as {attribute: {prompt: [rollouts]}}.
    """
    results = []

    for rewriter_dir in validate_dir.iterdir():
        if not rewriter_dir.is_dir():
            continue

        rollouts_path = rewriter_dir / "rollouts.json"
        if not rollouts_path.exists():
            continue

        with open(rollouts_path) as f:
            rollouts = json.load(f)

        for attribute, prompts_data in rollouts.items():
            for prompt, responses in prompts_data.items():
                for resp in responses:
                    extracted = extract_from_rollout(resp)
                    if extracted:
                        results.append(extracted)

    return results


def collect_all_for_seed(run_path: Path, seed_index: int) -> list[tuple[float, int]]:
    """Collect all (student_diff, length_diff) pairs for a seed.

    Gathers data from:
    - All step_*_stats/seed_{seed_index}.json files
    - validate/seed_{seed_index}_validate/*/rollouts.json
    """
    results = []

    # Collect from step files
    for step_dir in run_path.glob("step_*_stats"):
        step_file = step_dir / f"seed_{seed_index}.json"
        if step_file.exists():
            step_results = collect_from_step_file(step_file)
            results.extend(step_results)

    # Collect from validation (all rewriters)
    validate_dir = run_path / "validate" / f"seed_{seed_index}_validate"
    if validate_dir.exists():
        val_results = collect_from_validate_dir(validate_dir)
        results.extend(val_results)

    return results


def plot_length_confounding_simple(
    data: list[tuple[float, int]],
    seed_index: int,
    output_path: Path,
) -> None:
    """Create a simple scatterplot of reward diff vs length diff with linear regression."""
    if not data:
        print(f"No data for seed {seed_index}")
        return

    x = np.array([d[1] for d in data])  # length_diff
    y = np.array([d[0] for d in data])  # student_diff

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(x, y, alpha=0.3, s=20, c="steelblue", edgecolors="none")

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r-", linewidth=2, label=f"y = {slope:.4f}x + {intercept:.2f}")

    # Reference lines
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    # Labels with large font
    ax.set_xlabel("Length Difference (words)", fontsize=14)
    ax.set_ylabel("Reward Difference", fontsize=14)
    ax.set_title(f"Seed {seed_index}: r = {r_value:.3f}, n = {len(data)}", fontsize=16)

    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def analyze_seed(
    run_path: Path | str,
    seed_index: int,
    output_dir: Path | None = None,
) -> list[tuple[float, int]]:
    """Analyze length confounding for a seed, aggregating all steps and rewriters."""
    if isinstance(run_path, str):
        run_path = Path(run_path)

    data = collect_all_for_seed(run_path, seed_index)
    print(f"Seed {seed_index}: {len(data)} response pairs collected")

    if output_dir is None:
        output_dir = run_path

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    output_json = output_dir / f"seed_{seed_index}_length_analysis.json"
    with open(output_json, "w") as f:
        json.dump([{"student_diff": d[0], "length_diff": d[1]} for d in data], f, indent=2)

    # Create plot
    output_plot = output_dir / f"seed_{seed_index}_length_confounding.pdf"
    plot_length_confounding_simple(data, seed_index, output_plot)

    return data


def analyze_run(run_path: Path | str) -> None:
    """Analyze length confounding for all seeds in a run."""
    if isinstance(run_path, str):
        run_path = Path(run_path)

    # Find all seeds by looking at step_0_stats
    step0_dir = run_path / "step_0_stats"
    if not step0_dir.exists():
        print(f"step_0_stats not found: {step0_dir}")
        return

    seed_indices = []
    for f in step0_dir.glob("seed_*.json"):
        match = re.match(r"seed_(\d+)\.json", f.name)
        if match:
            seed_indices.append(int(match.group(1)))

    seed_indices.sort()

    if not seed_indices:
        print("No seed files found")
        return

    for seed_index in seed_indices:
        analyze_seed(run_path, seed_index)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_length_confounding.py <run_path> [seed_index]")
        print("Example: python analyze_length_confounding.py data/evo/20260103-171901-list_reverse-handpick-plus 4")
        sys.exit(1)

    run_path = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        seed_index = int(sys.argv[2])
        analyze_seed(run_path, seed_index)
    else:
        analyze_run(run_path)
