# ABOUTME: Analyzes whether biases found by the pipeline are confounded with response length.
# ABOUTME: Creates scatterplots of student diff vs response length diff for each attribute.

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def word_count(text: str) -> int:
    """Crude word count using whitespace splitting."""
    return len(text.split())


def load_baseline_lengths(baseline_path: Path) -> dict[str, float]:
    """Load baseline rollouts and compute average word count per prompt."""
    with open(baseline_path) as f:
        baselines = json.load(f)

    prompt_lengths = {}
    for prompt, responses in baselines.items():
        if responses:
            avg_length = np.mean([word_count(r["response"]) for r in responses])
            prompt_lengths[prompt] = avg_length
    return prompt_lengths


def analyze_seed(
    validate_rollouts_path: Path,
    baseline_lengths: dict[str, float],
) -> list[dict]:
    """Analyze a single seed's validation rollouts.

    Returns a list of dicts, one per individual rewrite pair:
        - attribute: the attribute text
        - prompt: the prompt text (truncated for storage)
        - student_diff: student diff for this response
        - length_diff: (rewritten - baseline) word count
    """
    with open(validate_rollouts_path) as f:
        rollouts = json.load(f)

    results = []
    for attribute, prompts_data in rollouts.items():
        for prompt, responses in prompts_data.items():
            if prompt not in baseline_lengths:
                continue

            baseline_len = baseline_lengths[prompt]
            for resp in responses:
                rewritten_len = word_count(resp["response"])
                results.append({
                    "attribute": attribute[:100],  # Truncate for storage
                    "prompt": prompt[:100],  # Truncate for storage
                    "student_diff": resp["student_diff"],
                    "length_diff": rewritten_len - baseline_len,
                })

    return results


def plot_length_confounding(
    results: list[dict],
    seed_index: int,
    output_path: Path,
) -> None:
    """Create a scatterplot of student diff vs length diff, colored by attribute."""
    if not results:
        print(f"No results for seed {seed_index}")
        return

    # Group by attribute for coloring
    attributes = list(set(r["attribute"] for r in results))
    attr_to_idx = {attr: i for i, attr in enumerate(attributes)}

    # Use a colormap with enough distinct colors
    cmap = plt.colormaps["tab10" if len(attributes) <= 10 else "tab20"]
    colors = [cmap(attr_to_idx[r["attribute"]] / len(attributes)) for r in results]

    x = [r["length_diff"] for r in results]
    y = [r["student_diff"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each attribute separately for legend
    for i, attr in enumerate(attributes):
        mask = [r["attribute"] == attr for r in results]
        x_attr = [x[j] for j, m in enumerate(mask) if m]
        y_attr = [y[j] for j, m in enumerate(mask) if m]
        # Truncate attribute for legend
        label = attr[:50] + "..." if len(attr) > 50 else attr
        ax.scatter(x_attr, y_attr, alpha=0.6, s=30, label=label, color=cmap(i / len(attributes)))

    # Add trend line
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label=f"trend (slope={z[0]:.4f})")

        # Calculate correlation
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"Seed {seed_index}: Student Diff vs Length Diff (r={corr:.3f}, n={len(x)})")
    else:
        ax.set_title(f"Seed {seed_index}: Student Diff vs Length Diff")

    ax.set_xlabel("Length Diff (words: rewritten - baseline)")
    ax.set_ylabel("Student Diff")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Place legend outside the plot
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def analyze_run(run_path: Path | str) -> None:
    """Analyze length confounding for a complete run."""
    if isinstance(run_path, str):
        run_path = Path(run_path)

    # Load baseline lengths
    baseline_path = run_path / "val_baselines" / "rollouts.json"
    if not baseline_path.exists():
        print(f"Baseline rollouts not found: {baseline_path}")
        return

    baseline_lengths = load_baseline_lengths(baseline_path)
    print(f"Loaded {len(baseline_lengths)} baseline prompts")

    # Find all seed validation directories
    validate_dir = run_path / "validate"
    if not validate_dir.exists():
        print(f"Validate directory not found: {validate_dir}")
        return

    pattern = re.compile(r"seed_(\d+)_validate")
    seed_dirs = []
    for item in validate_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                seed_dirs.append((int(match.group(1)), item))
    seed_dirs.sort(key=lambda x: x[0])

    if not seed_dirs:
        print("No seed directories found")
        return

    # Analyze each seed
    for seed_index, seed_dir in seed_dirs:
        rollouts_path = seed_dir / "rollouts.json"
        if not rollouts_path.exists():
            print(f"Rollouts not found: {rollouts_path}")
            continue

        results = analyze_seed(rollouts_path, baseline_lengths)
        print(f"Seed {seed_index}: {len(results)} response pairs analyzed")

        # Save results
        output_json = seed_dir / "length_analysis.json"
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

        # Create plot
        output_plot = seed_dir / "length_confounding.pdf"
        plot_length_confounding(results, seed_index, output_plot)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_length_confounding.py <run_path>")
        print("Example: python analyze_length_confounding.py data/evo/20251217-160232-list_reverse-clio-plus")
        sys.exit(1)

    run_path = Path(sys.argv[1])
    analyze_run(run_path)
