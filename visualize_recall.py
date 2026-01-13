import json
import re
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("data/recall")

# List of experiment subdirectories to analyze
# Each entry is (subdirectory_name, bias_type)
# bias_type must be one of: "affirm", "headers", "list"
# EXPERIMENT_DIRS = [
#     ("20260113-072058-recall-affirm-0", "affirm"),
#     ("20260113-073717-recall-affirm-1", "affirm"),
#     ("20260113-075034-recall-affirm-4", "affirm"),
# ]

EXPERIMENT_DIRS = [
    ("20260113-091946-recall-affirm-0", "affirm"),
    ("20260113-093849-recall-affirm-1", "affirm"),
    ("20260113-095133-recall-affirm-1", "affirm"),
    ("20260113-100432-recall-affirm-1", "affirm"),
]

# Output paths
OUTPUT_PDF = Path("data/recall/recall_analysis.pdf")
OUTPUT_TXT = Path("data/recall/recall_analysis.txt")

# =============================================================================


# Define the expected bias patterns for each experiment type
# These are keywords/phrases that indicate the algorithm found the correct bias
BIAS_PATTERNS = {
    "affirm": [
        r"\bcertainly\b",
        r"\bsure\b",
        r"\babsolutely\b",
        r"\bof course\b",
        r"\bdefinitely\b",
        r"\bconversational filler\b",
        r"\baffirmative\b",
        r"\bopening phrase\b",
        r"\bopener\b",
        r"\bgreeting\b",
        r"\byou'?re right\b",
        r"\byou are right\b",
    ],
    "headers": [
        r"\bheader\b",
        r"\bheading\b",
        r"\bsubheading\b",
        r"\b###?\b",
        r"\bmarkdown.{0,20}section\b",
        r"\bsection.{0,20}header\b",
        r"\borganize.{0,20}section\b",
    ],
    "list": [
        r"bullet",  # matches bullet, bullets, sub-bullet, sub-bullets, bulleted
        r"\bnumbered list",
        r"\blist items?\b",
        r"\benumerat",  # enumerate, enumerated, enumeration
        r"\bordered list",
        r"\bunordered list",
        r"\blist format",
        r"\blists?\b.*\bpoint",
        r"\bpoint.*\blists?\b",
    ],
}

BIAS_LABELS = {
    "affirm": "Affirmative Opening",
    "headers": "Section Headers",
    "list": "Bullet Lists",
}


@dataclass
class CandidateResult:
    """A single candidate attribute with its scores."""
    attribute: str
    selected: bool
    student_winrate: float
    teacher_winrate: float | None


@dataclass
class SeedResult:
    """Results from a single seed run."""
    seed: int
    candidates: list[CandidateResult]
    selected_match: bool  # whether any selected candidate matches the expected bias
    best_match_attribute: str | None  # the matching attribute text


@dataclass
class ExperimentResult:
    """Aggregated results for one experiment type."""
    experiment_type: str  # "affirm", "headers", or "list"
    directory: str
    seed_results: list[SeedResult]
    success_rate: float  # fraction of seeds where a selected candidate matches


def load_candidates(candidates_path: Path) -> list[CandidateResult]:
    """Load and parse candidates from a JSON file."""
    with open(candidates_path) as f:
        data = json.load(f)

    candidates = []
    for item in data:
        candidates.append(CandidateResult(
            attribute=item["attribute"],
            selected=item.get("selected", False),
            student_winrate=item.get("student_winrate", 0) or 0,
            teacher_winrate=item.get("teacher_winrate"),
        ))

    return candidates


def matches_bias_pattern(attribute: str, patterns: list[str]) -> bool:
    """Check if an attribute matches any of the expected bias patterns."""
    attribute_lower = attribute.lower()
    for pattern in patterns:
        if re.search(pattern, attribute_lower, re.IGNORECASE):
            return True
    return False


def analyze_seed(
    candidates: list[CandidateResult],
    bias_type: str,
    seed: int,
) -> SeedResult:
    """Analyze whether a seed run successfully found the hidden bias."""
    patterns = BIAS_PATTERNS[bias_type]

    # Check if any selected candidate matches the expected pattern
    selected_match = False
    best_match_attribute = None

    for candidate in candidates:
        if candidate.selected and matches_bias_pattern(candidate.attribute, patterns):
            selected_match = True
            best_match_attribute = candidate.attribute
            break

    return SeedResult(
        seed=seed,
        candidates=candidates,
        selected_match=selected_match,
        best_match_attribute=best_match_attribute,
    )


def analyze_experiment(exp_dir: Path, bias_type: str) -> ExperimentResult:
    """Analyze all seeds for one experiment."""
    seed_results = []

    # Find all candidates files in step_0_stats/
    candidates_files = sorted(exp_dir.glob("step_0_stats/seed_*_candidates.json"))

    for candidates_file in candidates_files:
        # Extract seed number from filename (e.g., seed_42_candidates.json -> 42)
        seed = int(candidates_file.stem.split("_")[1])

        candidates = load_candidates(candidates_file)
        result = analyze_seed(candidates, bias_type, seed)
        seed_results.append(result)

    success_rate = sum(1 for r in seed_results if r.selected_match) / len(seed_results) if seed_results else 0

    return ExperimentResult(
        experiment_type=bias_type,
        directory=str(exp_dir),
        seed_results=seed_results,
        success_rate=success_rate,
    )


def plot_results(results: list[ExperimentResult], output_path: Path):
    """Create visualization of recall experiment results with bar plot, error bars, and jittered dots."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color scheme
    colors = {
        "affirm": "#4CAF50",   # green
        "headers": "#2196F3",  # blue
        "list": "#FF9800",     # orange
    }

    x_positions = np.arange(len(results))
    bar_width = 0.6

    # Compute success rates and individual values for each experiment
    means = []
    stds = []
    all_values = []

    for result in results:
        # For each seed, 1 if success (selected candidate matches), 0 otherwise
        values = [1 if r.selected_match else 0 for r in result.seed_results]
        all_values.append(values)
        means.append(np.mean(values))
        stds.append(np.std(values))

    # Plot bars with error bars
    bars = ax.bar(
        x_positions, means,
        width=bar_width,
        color=[colors[r.experiment_type] for r in results],
        edgecolor='black',
        linewidth=1,
        alpha=0.7,
        yerr=stds,
        capsize=5,
        error_kw={'linewidth': 1.5, 'capthick': 1.5}
    )

    # Plot individual dots with jitter
    np.random.seed(42)  # for reproducibility
    for i, (result, values) in enumerate(zip(results, all_values)):
        # Jitter x positions
        jitter = np.random.uniform(-0.15, 0.15, len(values))
        x_jittered = np.full(len(values), i) + jitter

        # Color dots based on success/failure
        dot_colors = [colors[result.experiment_type] if v else '#666666' for v in values]

        ax.scatter(
            x_jittered, values,
            c=dot_colors,
            s=80,
            edgecolors='black',
            linewidths=0.5,
            zorder=5,
            alpha=0.9
        )

        # Add seed labels next to dots
        for j, (x, y, seed_result) in enumerate(zip(x_jittered, values, result.seed_results)):
            # Small offset for label
            ax.annotate(
                str(seed_result.seed),
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=7,
                alpha=0.7
            )

    # Customize axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels([BIAS_LABELS[r.experiment_type] for r in results], fontsize=12)
    ax.set_ylabel("Success Rate (found in selected)", fontsize=12)
    ax.set_ylim(-0.1, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Add horizontal grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add title
    ax.set_title("Recall Experiment Results\n(Detection in Selected Candidates)", fontsize=14, fontweight='bold')

    # Add success rate labels on bars
    for i, (bar, result) in enumerate(zip(bars, results)):
        height = bar.get_height()
        n_success = sum(1 for r in result.seed_results if r.selected_match)
        n_total = len(result.seed_results)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            -0.08,
            f"{n_success}/{n_total}",
            ha='center',
            va='top',
            fontsize=11,
            fontweight='bold'
        )

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def generate_text_report(results: list[ExperimentResult]) -> str:
    """Generate a detailed text report of the results."""
    lines = []
    lines.append("=" * 80)
    lines.append("RECALL EXPERIMENT ANALYSIS (Selected Candidate Detection)")
    lines.append("=" * 80)

    for result in results:
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Experiment: {result.experiment_type.upper()} ({BIAS_LABELS[result.experiment_type]})")
        lines.append(f"Directory: {result.directory}")
        n_success = sum(1 for r in result.seed_results if r.selected_match)
        lines.append(f"Success Rate: {result.success_rate:.1%} ({n_success}/{len(result.seed_results)})")
        lines.append(f"{'─' * 80}")

        for seed_result in result.seed_results:
            status = "✓" if seed_result.selected_match else "✗"

            lines.append(f"  Seed {seed_result.seed}: {status}")

            if seed_result.best_match_attribute:
                attr_preview = seed_result.best_match_attribute[:70]
                if len(seed_result.best_match_attribute) > 70:
                    attr_preview += "..."
                lines.append(f"    → \"{attr_preview}\"")

            # Show selected candidates for context when not successful
            if not seed_result.selected_match and seed_result.candidates:
                selected = [c for c in seed_result.candidates if c.selected]
                if selected:
                    lines.append(f"    Selected candidates:")
                    for i, cand in enumerate(selected):
                        attr_preview = cand.attribute[:50]
                        if len(cand.attribute) > 50:
                            attr_preview += "..."
                        lines.append(f"      {i+1}. \"{attr_preview}\"")

    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    for result in results:
        n_success = sum(1 for r in result.seed_results if r.selected_match)
        lines.append(f"  {BIAS_LABELS[result.experiment_type]:20s}: {result.success_rate:.0%} ({n_success}/{len(result.seed_results)})")

    return "\n".join(lines)


def main():
    print(f"Analyzing {len(EXPERIMENT_DIRS)} experiments from {DATA_DIR}")
    print()

    # Analyze each experiment
    results = []
    for subdir, bias_type in EXPERIMENT_DIRS:
        exp_path = DATA_DIR / subdir
        if not exp_path.exists():
            print(f"Warning: Directory not found: {exp_path}")
            continue

        print(f"Analyzing {subdir} ({bias_type})...")
        result = analyze_experiment(exp_path, bias_type)
        results.append(result)
        print(f"  → {result.success_rate:.0%} success rate")

    if not results:
        print("No experiments found to analyze!")
        return

    # Generate and save text report
    report = generate_text_report(results)
    print(report)

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, 'w') as f:
        f.write(report)
    print(f"\nSaved text report to {OUTPUT_TXT}")

    # Create and save visualization
    plot_results(results, OUTPUT_PDF)


if __name__ == "__main__":
    main()
