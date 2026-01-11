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
EXPERIMENT_DIRS = [
    ("20260110-171712-recall-affirm-0", "affirm"),
    ("20260110-174744-recall-headers-3", "headers"),
    ("20260110-182231-recall-list-10", "list"),
]

# How many top candidates to check for bias match
TOP_K = 5

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
    student_winrate: float
    teacher_winrate: float | None
    disagreement_score: float  # student - teacher (or just student if teacher is None)


@dataclass
class SeedResult:
    """Results from a single seed run."""
    seed: int
    candidates: list[CandidateResult]
    top_k_match: bool  # whether top-k contains a matching attribute
    best_match_rank: int | None  # rank of best matching attribute (1-indexed), None if no match
    best_match_attribute: str | None  # the matching attribute text


@dataclass
class ExperimentResult:
    """Aggregated results for one experiment type."""
    experiment_type: str  # "affirm", "headers", or "list"
    directory: str
    seed_results: list[SeedResult]
    success_rate: float  # fraction of seeds where top-k contains a match


def load_candidates(candidates_path: Path) -> list[CandidateResult]:
    """Load and parse candidates from a JSON file."""
    with open(candidates_path) as f:
        data = json.load(f)

    candidates = []
    for item in data:
        student = item.get("student_winrate", 0) or 0
        teacher = item.get("teacher_winrate")

        # Compute disagreement score: how much more does student like this than teacher?
        # Higher = student prefers, teacher dislikes (the signal we're looking for)
        if teacher is not None:
            disagreement = student - teacher
        else:
            # If no teacher score, just use student score
            # (but these are less reliable as we can't confirm teacher dislikes it)
            disagreement = student

        candidates.append(CandidateResult(
            attribute=item["attribute"],
            student_winrate=student,
            teacher_winrate=teacher,
            disagreement_score=disagreement,
        ))

    # Sort by disagreement score (highest first = strongest bias signal)
    candidates.sort(key=lambda x: x.disagreement_score, reverse=True)
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
    top_k: int = 5,
) -> SeedResult:
    """Analyze whether a seed run successfully found the hidden bias."""
    patterns = BIAS_PATTERNS[bias_type]

    # Check if any of the top-k candidates match the expected pattern
    top_k_match = False
    best_match_rank = None
    best_match_attribute = None

    for i, candidate in enumerate(candidates[:top_k]):
        if matches_bias_pattern(candidate.attribute, patterns):
            top_k_match = True
            if best_match_rank is None:
                best_match_rank = i + 1
                best_match_attribute = candidate.attribute

    # Also find best match in full list (for reporting)
    if best_match_rank is None:
        for i, candidate in enumerate(candidates):
            if matches_bias_pattern(candidate.attribute, patterns):
                best_match_rank = i + 1
                best_match_attribute = candidate.attribute
                break

    return SeedResult(
        seed=seed,
        candidates=candidates,
        top_k_match=top_k_match,
        best_match_rank=best_match_rank,
        best_match_attribute=best_match_attribute,
    )


def analyze_experiment(exp_dir: Path, bias_type: str, top_k: int = 5) -> ExperimentResult:
    """Analyze all seeds for one experiment."""
    seed_results = []

    # Find all seed directories
    seed_dirs = sorted(exp_dir.glob("random_seed_*"))

    for seed_dir in seed_dirs:
        # Extract seed number from directory name
        seed = int(seed_dir.name.split("_")[-1])

        # Find the candidates file (seed_{topic_id}_candidates.json)
        candidates_files = list(seed_dir.glob("step_0_stats/seed_*_candidates.json"))
        if not candidates_files:
            print(f"Warning: No candidates file found in {seed_dir}")
            continue

        candidates = load_candidates(candidates_files[0])
        result = analyze_seed(candidates, bias_type, seed, top_k)
        seed_results.append(result)

    success_rate = sum(1 for r in seed_results if r.top_k_match) / len(seed_results) if seed_results else 0

    return ExperimentResult(
        experiment_type=bias_type,
        directory=str(exp_dir),
        seed_results=seed_results,
        success_rate=success_rate,
    )


def plot_results(results: list[ExperimentResult], top_k: int, output_path: Path):
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
        # For each seed, 1 if success (found in top-k), 0 otherwise
        values = [1 if r.top_k_match else 0 for r in result.seed_results]
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
    ax.set_ylabel(f"Success Rate (found in top-{top_k})", fontsize=12)
    ax.set_ylim(-0.1, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Add horizontal grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add title
    ax.set_title(f"Recall Experiment Results\n(Detection in Top-{top_k} Candidates)", fontsize=14, fontweight='bold')

    # Add success rate labels on bars
    for i, (bar, result) in enumerate(zip(bars, results)):
        height = bar.get_height()
        n_success = sum(1 for r in result.seed_results if r.top_k_match)
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


def generate_text_report(results: list[ExperimentResult], top_k: int) -> str:
    """Generate a detailed text report of the results."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"RECALL EXPERIMENT ANALYSIS (Top-{top_k} Detection)")
    lines.append("=" * 80)

    for result in results:
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Experiment: {result.experiment_type.upper()} ({BIAS_LABELS[result.experiment_type]})")
        lines.append(f"Directory: {result.directory}")
        n_success = sum(1 for r in result.seed_results if r.top_k_match)
        lines.append(f"Success Rate: {result.success_rate:.1%} ({n_success}/{len(result.seed_results)})")
        lines.append(f"{'─' * 80}")

        for seed_result in result.seed_results:
            status = "✓" if seed_result.top_k_match else "✗"
            rank_info = f"rank #{seed_result.best_match_rank}" if seed_result.best_match_rank else "no match"

            lines.append(f"  Seed {seed_result.seed}: {status} ({rank_info})")

            if seed_result.best_match_attribute:
                attr_preview = seed_result.best_match_attribute[:70]
                if len(seed_result.best_match_attribute) > 70:
                    attr_preview += "..."
                lines.append(f"    → \"{attr_preview}\"")

            # Show top-3 candidates for context when not successful
            if not seed_result.top_k_match and seed_result.candidates:
                lines.append(f"    Top-3 found:")
                for i, cand in enumerate(seed_result.candidates[:3]):
                    score = f"Δ={cand.disagreement_score:.2f}"
                    attr_preview = cand.attribute[:50]
                    if len(cand.attribute) > 50:
                        attr_preview += "..."
                    lines.append(f"      {i+1}. [{score}] \"{attr_preview}\"")

    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    for result in results:
        n_success = sum(1 for r in result.seed_results if r.top_k_match)
        lines.append(f"  {BIAS_LABELS[result.experiment_type]:20s}: {result.success_rate:.0%} ({n_success}/{len(result.seed_results)})")

    return "\n".join(lines)


def main():
    print(f"Analyzing {len(EXPERIMENT_DIRS)} experiments from {DATA_DIR}")
    print(f"Top-k threshold: {TOP_K}")
    print()

    # Analyze each experiment
    results = []
    for subdir, bias_type in EXPERIMENT_DIRS:
        exp_path = DATA_DIR / subdir
        if not exp_path.exists():
            print(f"Warning: Directory not found: {exp_path}")
            continue

        print(f"Analyzing {subdir} ({bias_type})...")
        result = analyze_experiment(exp_path, bias_type, TOP_K)
        results.append(result)
        print(f"  → {result.success_rate:.0%} success rate")

    if not results:
        print("No experiments found to analyze!")
        return

    # Generate and save text report
    report = generate_text_report(results, TOP_K)
    print(report)

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, 'w') as f:
        f.write(report)
    print(f"\nSaved text report to {OUTPUT_TXT}")

    # Create and save visualization
    plot_results(results, TOP_K, OUTPUT_PDF)


if __name__ == "__main__":
    main()
