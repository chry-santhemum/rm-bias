"""
Analyze rewriter invariance for exp_attribute_validation data.

Creates:
1. Correlation plots for all rewriter pairs (scatter with r, p, n annotations)
2. QQ plots per attribute comparing rewriter pair distributions
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Dark2 color scheme
DARK2_COLORS = [
    "#1B9E77",  # teal
    "#D95F02",  # orange
    "#7570B3",  # purple
    "#E7298A",  # pink
    "#66A61E",  # green
    "#E6AB02",  # gold
    "#A6761D",  # brown
    "#666666",  # gray
]


def wrap_text(text: str, width: int = 60) -> str:
    """Wrap text at specified width, inserting <br> for line breaks."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return "<br>".join(lines)


def load_rewriter_data(data_dir: Path) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    """
    Load all rewriter data from exp_attribute_validation format.

    Args:
        data_dir: Path to the run directory (e.g., data/exp_attribute_validation/20260112-162826)

    Returns:
        {seed_index: {rewriter_name: {attribute: {prompt: [student_diffs]}}}}
    """
    result: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}

    # Find all seed directories
    seed_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("seed_") and "validate" in d.name])

    for seed_dir in seed_dirs:
        # Extract seed index from directory name (e.g., "seed_0_validate" -> "0")
        seed_index = seed_dir.name.split("_")[1]
        result[seed_index] = {}

        rewrites_dir = seed_dir / "rewrites"
        if not rewrites_dir.exists():
            continue

        # Find all rewriter directories
        for rewriter_dir in rewrites_dir.iterdir():
            if not rewriter_dir.is_dir():
                continue

            rewriter_name = rewriter_dir.name
            rollouts_path = rewriter_dir / "rollouts.json"

            if not rollouts_path.exists():
                continue

            with open(rollouts_path) as f:
                rollouts_data = json.load(f)

            # Extract student_diffs from rollouts
            result[seed_index][rewriter_name] = {}
            for attribute, prompts_data in rollouts_data.items():
                result[seed_index][rewriter_name][attribute] = {}
                for prompt, rollouts in prompts_data.items():
                    diffs = [
                        r["student_diff"]
                        for r in rollouts
                        if r is not None and r.get("student_diff") is not None
                    ]
                    result[seed_index][rewriter_name][attribute][prompt] = diffs

    return result


def plot_pairwise_correlation(
    data: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
    output_dir: Path,
):
    """
    For each rewriter pair, plot correlation of reward diffs
    for the same (attribute, prompt, rollout_index).

    Aggregates across all attributes and seeds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all rewriter names
    all_rewriters = set()
    for seed_data in data.values():
        all_rewriters.update(seed_data.keys())
    rewriter_names = sorted(all_rewriters)

    # For each pair of rewriters
    for rewriter_a, rewriter_b in combinations(rewriter_names, 2):
        # Collect paired diffs
        diffs_a = []
        diffs_b = []

        for seed_index, seed_data in data.items():
            if rewriter_a not in seed_data or rewriter_b not in seed_data:
                continue

            data_a = seed_data[rewriter_a]
            data_b = seed_data[rewriter_b]

            # Find common attributes
            common_attrs = set(data_a.keys()) & set(data_b.keys())

            for attr in common_attrs:
                # Find common prompts
                common_prompts = set(data_a[attr].keys()) & set(data_b[attr].keys())

                for prompt in common_prompts:
                    rollouts_a = data_a[attr][prompt]
                    rollouts_b = data_b[attr][prompt]

                    # Pair by rollout index
                    for i in range(min(len(rollouts_a), len(rollouts_b))):
                        diffs_a.append(rollouts_a[i])
                        diffs_b.append(rollouts_b[i])

        if len(diffs_a) < 2:
            print(f"Skipping {rewriter_a} vs {rewriter_b}: insufficient paired data")
            continue

        # Create scatter plot
        fig = go.Figure()

        # Scatter points with transparency
        fig.add_trace(go.Scatter(
            x=diffs_a,
            y=diffs_b,
            mode="markers",
            marker=dict(size=5, color=DARK2_COLORS[0], opacity=0.3),
            hovertemplate=f"{rewriter_a}: %{{x:.2f}}<br>{rewriter_b}: %{{y:.2f}}<extra></extra>",
            showlegend=False,
        ))

        # Compute correlation stats (no fit line drawn)
        _, _, r_value, p_value, _ = stats.linregress(diffs_a, diffs_b)

        # y=x reference line
        all_vals = diffs_a + diffs_b
        min_val, max_val = min(all_vals), max(all_vals)
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="gray", dash="dot", width=1),
            name="y=x",
        ))

        fig.update_layout(
            title=f"{rewriter_a} vs {rewriter_b}<br>r={r_value:.3f}, p={p_value:.1e}, n={len(diffs_a)}",
            xaxis_title=f"{rewriter_a} reward diff",
            yaxis_title=f"{rewriter_b} reward diff",
            width=700,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
        )

        save_path = output_dir / f"{rewriter_a}_vs_{rewriter_b}.pdf"
        fig.write_image(save_path)
        print(f"Saved correlation plot: {save_path}")


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    # Replace problematic characters
    result = text.lower()
    for char in [" ", "'", '"', "/", "\\", ":", ";", ",", ".", "(", ")", "[", "]", "{", "}"]:
        result = result.replace(char, "_")
    # Collapse multiple underscores
    while "__" in result:
        result = result.replace("__", "_")
    # Trim length and strip underscores
    return result[:60].strip("_")


def plot_qq_per_attribute(
    data: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
    output_dir: Path,
):
    """
    For each attribute, create QQ plots comparing each rewriter pair's
    reward diff distributions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all rewriter names
    all_rewriters = set()
    for seed_data in data.values():
        all_rewriters.update(seed_data.keys())
    rewriter_names = sorted(all_rewriters)

    # Collect all attributes across all seeds
    all_attributes = set()
    for seed_data in data.values():
        for rewriter_data in seed_data.values():
            all_attributes.update(rewriter_data.keys())

    for attribute in sorted(all_attributes):
        attr_slug = slugify(attribute)
        attr_output_dir = output_dir / attr_slug
        attr_output_dir.mkdir(parents=True, exist_ok=True)

        # For each pair of rewriters
        for rewriter_a, rewriter_b in combinations(rewriter_names, 2):
            # Collect all diffs for each rewriter for this attribute
            diffs_a = []
            diffs_b = []

            for seed_index, seed_data in data.items():
                if rewriter_a not in seed_data or rewriter_b not in seed_data:
                    continue

                data_a = seed_data.get(rewriter_a, {})
                data_b = seed_data.get(rewriter_b, {})

                if attribute not in data_a or attribute not in data_b:
                    continue

                # Collect all diffs from each rewriter
                for prompt, rollouts in data_a[attribute].items():
                    diffs_a.extend(rollouts)
                for prompt, rollouts in data_b[attribute].items():
                    diffs_b.extend(rollouts)

            if len(diffs_a) < 2 or len(diffs_b) < 2:
                continue

            # Compute quantiles at common probability points for QQ plot
            n_points = min(len(diffs_a), len(diffs_b), 100)
            probs = np.linspace(0.01, 0.99, n_points)  # Avoid extreme tails
            quantiles_a = np.quantile(diffs_a, probs)
            quantiles_b = np.quantile(diffs_b, probs)

            # Create QQ plot
            fig = go.Figure()

            # QQ points
            fig.add_trace(go.Scatter(
                x=quantiles_a,
                y=quantiles_b,
                mode="markers",
                marker=dict(size=6, color=DARK2_COLORS[0]),
                hovertemplate=f"{rewriter_a}: %{{x:.2f}}<br>{rewriter_b}: %{{y:.2f}}<extra></extra>",
                showlegend=False,
            ))

            # y=x reference line
            all_vals = list(quantiles_a) + list(quantiles_b)
            min_val, max_val = min(all_vals), max(all_vals)
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="gray", dash="dash", width=2),
                name="y=x",
            ))

            # Compute KS test
            ks_stat, ks_pvalue = stats.ks_2samp(diffs_a, diffs_b)

            # Wrap long attribute text
            attr_display = wrap_text(attribute, width=60)

            fig.update_layout(
                title=dict(
                    text=f"{rewriter_a} vs {rewriter_b}<br>{attr_display}<br>KS stat={ks_stat:.3f}, p={ks_pvalue:.1e}",
                    y=0.95,
                    yanchor="top",
                ),
                xaxis_title=f"{rewriter_a} quantiles",
                yaxis_title=f"{rewriter_b} quantiles",
                width=600,
                height=470,
                margin=dict(t=120),
            )

            save_path = attr_output_dir / f"{rewriter_a}_vs_{rewriter_b}.pdf"
            fig.write_image(save_path)

        print(f"Saved QQ plots for attribute: {attr_slug}")


if __name__ == "__main__":
    data_dir = Path("data/exp_attribute_validation/20260112-162826")
    output_dir = Path("data/exp_attribute_validation/20260112-162826/invariance")

    print("Loading rewriter data...")
    data = load_rewriter_data(data_dir)

    # Print summary
    print(f"Loaded {len(data)} seeds")
    for seed_idx, seed_data in data.items():
        print(f"  Seed {seed_idx}: {list(seed_data.keys())}")

    print("\nCreating correlation plots...")
    plot_pairwise_correlation(data, output_dir / "correlation")

    print("\nCreating QQ plots per attribute...")
    plot_qq_per_attribute(data, output_dir / "qq")

    print("\nDone!")
