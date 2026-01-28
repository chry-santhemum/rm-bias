import json
import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit


DATA_DIR = Path("data/recall")

# SNR sweep experiments: constant signal (3.0), varying noise
# All use topic_id=0, affirm bias
SNR_EXPERIMENT_DIRS = [
    ("20260113-102137-recall-affirm-0", "affirm"),  # noise=3
    ("20260113-105016-recall-affirm-0", "affirm"),  # noise=5
    ("20260113-110553-recall-affirm-0", "affirm"),  # noise=7
    ("20260113-112551-recall-affirm-0", "affirm"),  # noise=10
    ("20260113-114340-recall-affirm-0", "affirm"),  # noise=17
    ("20260114-043854-recall-affirm-0", "affirm"),  # noise=30
]

# Baseline comparison experiments: different bias types and topic_ids
# All use signal=3.0, noise=3.0
BASELINE_EXPERIMENT_DIRS = [
    ("20260113-102137-recall-affirm-0", "affirm"),
    ("20260113-121232-recall-affirm-5", "affirm"),
    ("20260113-124505-recall-affirm-9", "affirm"),
    ("20260113-132117-recall-headers-0", "headers"),
    ("20260113-135032-recall-headers-3", "headers"),
    ("20260113-141941-recall-headers-5", "headers"),
    ("20260113-175807-recall-list-0", "list"),
    ("20260113-182905-recall-list-1", "list"),
    ("20260113-145154-recall-list-2", "list"),
    ("20260113-151642-recall-list-3", "list"),
    ("20260113-154528-recall-list-6", "list"),
    ("20260113-161700-recall-list-8", "list"),
]

# Output paths
OUTPUT_SNR_PDF = Path("data/recall/recall_snr.pdf")
OUTPUT_SNR_NOFIT_PDF = Path("data/recall/recall_snr_nofit.pdf")
OUTPUT_SNR_NOCI_PDF = Path("data/recall/recall_snr_noci.pdf")
OUTPUT_SNR_NOFIT_NOCI_PDF = Path("data/recall/recall_snr_nofit_noci.pdf")
OUTPUT_BASELINE_PDF = Path("data/recall/recall_baseline.pdf")
OUTPUT_BASELINE_NOFIT_PDF = Path("data/recall/recall_baseline_nofit.pdf")
OUTPUT_BASELINE_NOCI_PDF = Path("data/recall/recall_baseline_noci.pdf")
OUTPUT_BASELINE_NOFIT_NOCI_PDF = Path("data/recall/recall_baseline_nofit_noci.pdf")
OUTPUT_TXT = Path("data/recall/recall_analysis.txt")

# Baseline attribute percentages from analyze_baselines.py
# Maps (bias_type, topic_id) -> percentage of baseline responses with that attribute
BASELINE_PERCENTAGES = {
    ("affirm", 0): 28.2, ("affirm", 1): 6.6, ("affirm", 2): 2.3,
    ("affirm", 3): 1.6, ("affirm", 4): 0.8, ("affirm", 5): 10.6,
    ("affirm", 6): 2.5, ("affirm", 7): 1.1, ("affirm", 8): 4.2,
    ("affirm", 9): 20.5, ("affirm", 10): 4.9, ("affirm", 11): 10.3,
    ("affirm", 12): 1.6,
    ("headers", 0): 85.9, ("headers", 1): 1.2, ("headers", 2): 1.3,
    ("headers", 3): 12.8, ("headers", 4): 16.0, ("headers", 5): 24.9,
    ("headers", 6): 1.1, ("headers", 7): 2.8, ("headers", 8): 14.2,
    ("headers", 9): 27.1, ("headers", 10): 15.7, ("headers", 11): 7.0,
    ("headers", 12): 2.9,
    ("list", 0): 59.0, ("list", 1): 47.4, ("list", 2): 35.2,
    ("list", 3): 96.4, ("list", 4): 82.9, ("list", 5): 97.8,
    ("list", 6): 1.6, ("list", 7): 67.7, ("list", 8): 71.4,
    ("list", 9): 91.7, ("list", 10): 93.6, ("list", 11): 87.4,
    ("list", 12): 99.6,
}

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

BIAS_COLORS = {
    "affirm": "#1b9e77",   # teal (Dark2)
    "headers": "#d95f02",  # orange (Dark2)
    "list": "#7570b3",     # purple (Dark2)
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
    topic_id: int | None = None
    noise_strength: float | None = None
    bias_strength: float | None = None


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


def load_experiment_config(exp_dir: Path) -> dict:
    """Load config from first seed's config file."""
    config_files = sorted(exp_dir.glob("config_seed_*.json"))
    if config_files:
        with open(config_files[0]) as f:
            return json.load(f)
    return {}


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

    # Load config to get experiment parameters
    config = load_experiment_config(exp_dir)

    return ExperimentResult(
        experiment_type=bias_type,
        directory=str(exp_dir),
        seed_results=seed_results,
        success_rate=success_rate,
        topic_id=config.get("topic_id"),
        noise_strength=config.get("noise_strength"),
        bias_strength=config.get("bias_strength"),
    )


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0, center - margin), min(1, center + margin))


def logistic(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    """Logistic function: y = 100 / (1 + exp(-k*(x-x0)))."""
    return 100 / (1 + np.exp(-k * (x - x0)))


def plot_snr_results(results: list[ExperimentResult], output_path: Path, include_fit: bool = True, include_ci: bool = True):
    """Create scatter plot of success rate vs signal/noise ratio with optional logistic fit."""
    # Compute SNR, success rate, and confidence intervals
    data_points = []
    for result in results:
        if result.bias_strength and result.noise_strength:
            snr = result.bias_strength / result.noise_strength
            n_seeds = len(result.seed_results)
            n_success = sum(1 for r in result.seed_results if r.selected_match)
            ci_low, ci_high = wilson_ci(n_success, n_seeds)
            data_points.append((snr, result.success_rate * 100, ci_low * 100, ci_high * 100))

    # Sort by SNR
    data_points.sort(key=lambda x: x[0])
    snrs = np.array([d[0] for d in data_points])
    success_rates = np.array([d[1] for d in data_points])
    ci_lows = np.array([d[2] for d in data_points])
    ci_highs = np.array([d[3] for d in data_points])

    fig = go.Figure()

    # Scatter points with optional error bars
    scatter_kwargs = dict(
        x=snrs,
        y=success_rates,
        mode='markers',
        marker=dict(
            size=12,
            color=BIAS_COLORS["affirm"],
            line=dict(color='black', width=1),
        ),
        name='Observed',
        showlegend=False,
    )
    if include_ci:
        scatter_kwargs['error_y'] = dict(
            type='data',
            array=ci_highs - success_rates,
            arrayminus=success_rates - ci_lows,
            color='rgba(0,0,0,0.5)',
            thickness=1.5,
            width=4,
        )
    fig.add_trace(go.Scatter(**scatter_kwargs))

    # Fit and plot logistic curve if requested
    if include_fit and len(snrs) >= 3:
        try:
            # Initial guess: k=5, x0=0.5 (midpoint of SNR range)
            popt, _ = curve_fit(logistic, snrs, success_rates, p0=[5, 0.5], maxfev=5000)
            k_fit, x0_fit = popt

            # Generate smooth curve
            x_smooth = np.linspace(0, max(snrs) * 1.1, 100)
            y_smooth = logistic(x_smooth, k_fit, x0_fit)

            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                line=dict(color=BIAS_COLORS["affirm"], width=2, dash='solid'),
                name='Logistic fit',
                showlegend=False,
            ))
        except RuntimeError:
            pass  # Skip fit if convergence fails

    # Layout styling
    fig.update_layout(
        xaxis=dict(
            title=dict(text='Signal / Noise Ratio', font=dict(size=16)),
            tickfont=dict(size=14),
            range=[0, max(snrs) * 1.1],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text='Success Rate (%)', font=dict(size=16)),
            tickfont=dict(size=14),
            range=[0, 105],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=1,
            zeroline=False,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=20, b=60),
        width=600,
        height=400,
    )

    fig.write_image(output_path)
    print(f"Saved SNR plot to {output_path}")


def plot_baseline_results(results: list[ExperimentResult], output_path: Path, include_fit: bool = True, include_ci: bool = True):
    """Create scatter plot of success rate vs baseline attribute frequency with optional quadratic fit."""
    fig = go.Figure()

    # Collect all data points for fitting
    all_x = []
    all_y = []

    # Group by bias type for coloring
    for bias_type in ["affirm", "headers", "list"]:
        x_vals = []
        y_vals = []
        ci_upper = []
        ci_lower = []
        for result in results:
            if result.experiment_type == bias_type and result.topic_id is not None:
                baseline_pct = BASELINE_PERCENTAGES.get((bias_type, result.topic_id))
                if baseline_pct is not None:
                    x_vals.append(baseline_pct)
                    y_vals.append(result.success_rate * 100)
                    all_x.append(baseline_pct)
                    all_y.append(result.success_rate * 100)
                    # Compute Wilson CI
                    n_seeds = len(result.seed_results)
                    n_success = sum(1 for r in result.seed_results if r.selected_match)
                    ci_low, ci_high = wilson_ci(n_success, n_seeds)
                    ci_upper.append(ci_high * 100 - result.success_rate * 100)
                    ci_lower.append(result.success_rate * 100 - ci_low * 100)

        if x_vals:
            scatter_kwargs = dict(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    size=12,
                    color=BIAS_COLORS[bias_type],
                    line=dict(color='black', width=1),
                ),
                cliponaxis=False,
                name=BIAS_LABELS[bias_type],
            )
            if include_ci:
                scatter_kwargs['error_y'] = dict(
                    type='data',
                    array=ci_upper,
                    arrayminus=ci_lower,
                    color='rgba(0,0,0,0.5)',
                    thickness=1.5,
                    width=4,
                )
            fig.add_trace(go.Scatter(**scatter_kwargs))

    # Fit and plot quadratic curve if requested
    if include_fit and len(all_x) >= 3:
        all_x = np.array(all_x)
        all_y = np.array(all_y)

        # Fit quadratic: y = a*x^2 + b*x + c
        coeffs = np.polyfit(all_x, all_y, 2)

        # Generate smooth curve
        x_smooth = np.linspace(0, 100, 100)
        y_smooth = np.polyval(coeffs, x_smooth)
        # Clip to valid range
        y_smooth = np.clip(y_smooth, 0, 100)

        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            line=dict(color='rgba(100,100,100,0.6)', width=2, dash='solid'),
            name='Quadratic fit',
            showlegend=False,
        ))

    # Layout styling
    fig.update_layout(
        xaxis=dict(
            title=dict(text='Baseline Attribute Frequency (%)', font=dict(size=16)),
            tickfont=dict(size=14),
            range=[0, 100],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text='Success Rate (%)', font=dict(size=16)),
            tickfont=dict(size=14),
            range=[0, 105],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=1,
            zeroline=False,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=20, b=60),
        width=600,
        height=400,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
        ),
    )

    fig.write_image(output_path)
    print(f"Saved baseline plot to {output_path}")


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
    # Analyze SNR experiments
    print(f"Analyzing {len(SNR_EXPERIMENT_DIRS)} SNR experiments from {DATA_DIR}")
    print()

    snr_results = []
    for subdir, bias_type in SNR_EXPERIMENT_DIRS:
        exp_path = DATA_DIR / subdir
        if not exp_path.exists():
            print(f"Warning: Directory not found: {exp_path}")
            continue

        print(f"Analyzing {subdir} ({bias_type})...")
        result = analyze_experiment(exp_path, bias_type)
        snr_results.append(result)
        print(f"  → {result.success_rate:.0%} success rate (SNR={result.bias_strength}/{result.noise_strength})")

    # Analyze baseline experiments
    print(f"\nAnalyzing {len(BASELINE_EXPERIMENT_DIRS)} baseline experiments from {DATA_DIR}")
    print()

    baseline_results = []
    for subdir, bias_type in BASELINE_EXPERIMENT_DIRS:
        exp_path = DATA_DIR / subdir
        if not exp_path.exists():
            print(f"Warning: Directory not found: {exp_path}")
            continue

        print(f"Analyzing {subdir} ({bias_type})...")
        result = analyze_experiment(exp_path, bias_type)
        baseline_results.append(result)
        baseline_pct = BASELINE_PERCENTAGES.get((bias_type, result.topic_id), "?")
        print(f"  → {result.success_rate:.0%} success rate (baseline={baseline_pct}%)")

    # Generate text report for baseline experiments
    if baseline_results:
        report = generate_text_report(baseline_results)
        print(report)

        OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_TXT, 'w') as f:
            f.write(report)
        print(f"\nSaved text report to {OUTPUT_TXT}")

    # Create plots (all combinations of fit and CI)
    if snr_results:
        plot_snr_results(snr_results, OUTPUT_SNR_PDF, include_fit=True, include_ci=True)
        plot_snr_results(snr_results, OUTPUT_SNR_NOFIT_PDF, include_fit=False, include_ci=True)
        plot_snr_results(snr_results, OUTPUT_SNR_NOCI_PDF, include_fit=True, include_ci=False)
        plot_snr_results(snr_results, OUTPUT_SNR_NOFIT_NOCI_PDF, include_fit=False, include_ci=False)

    if baseline_results:
        plot_baseline_results(baseline_results, OUTPUT_BASELINE_PDF, include_fit=True, include_ci=True)
        plot_baseline_results(baseline_results, OUTPUT_BASELINE_NOFIT_PDF, include_fit=False, include_ci=True)
        plot_baseline_results(baseline_results, OUTPUT_BASELINE_NOCI_PDF, include_fit=True, include_ci=False)
        plot_baseline_results(baseline_results, OUTPUT_BASELINE_NOFIT_NOCI_PDF, include_fit=False, include_ci=False)


if __name__ == "__main__":
    main()
