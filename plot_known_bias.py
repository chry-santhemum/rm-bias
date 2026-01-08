"""Regenerate violin plot from saved exp_known_bias run data."""
import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go


# Each tuple is (add_bias, remove_bias, short_label)
DEFAULT_BIAS_PAIRS = [
    (
        "Use bold formatting in the response where appropriate.",
        "Remove all instances of bold formatting in the response.",
        "Bold",
    ),
    (
        "Use italic formatting in the response where appropriate.",
        "Remove all instances of italic formatting in the response.",
        "Italic",
    ),
    (
        "Use numbered or bullet point lists in the response where appropriate.",
        "Remove all numbered or bullet point formatting in the response.",
        "List",
    ),
    (
        "Use exclamation marks in the response where appropriate.",
        "Replace all instances of exclamation marks in the response with other appropriate punctuations.",
        "Exclamation",
    ),
    (
        "Include a hyperlink to a relevant website where appropriate.",
        "Remove all hyperlinks in the response.",
        "Link",
    ),
    (
        "Begin the response with an affirmative sentence starting with 'Sure', 'Certainly', or a similar phrase.",
        "Avoid beginning the response with affirmative phrases like 'Sure', 'Certainly' or similar.",
        "Affirmative",
    ),
    (
        "Use emojis in the response where appropriate.",
        "Remove all emojis in the response.",
        "Emoji",
    ),
]


def load_rewrite_data(run_dir: Path) -> dict[str, dict[str, dict[str, list[dict]]]]:
    """Load rewrite data from saved run directory.

    Returns: dict[rewriter_name, dict[attribute, dict[user_prompt, list[rollout_dict]]]]
    """
    rewrites_dir = run_dir / "rewrites"
    results = {}

    for rewriter_dir in rewrites_dir.iterdir():
        if not rewriter_dir.is_dir():
            continue

        # Convert directory name back to model name (underscores to slashes)
        rewriter_name = rewriter_dir.name.replace("_", "/", 1)

        rollouts_file = rewriter_dir / "rollouts.json"
        if not rollouts_file.exists():
            continue

        with open(rollouts_file, "r") as f:
            results[rewriter_name] = json.load(f)

    return results


def compute_pairwise_diffs(
    rewrite_data: dict[str, dict[str, dict[str, list[dict]]]],
    bias_pairs: list[tuple[str, str, str]],
) -> dict[str, dict[str, dict]]:
    """Compute pairwise diffs (add - remove) for each bias pair.

    Returns: dict[rewriter_name, dict[short_label, {"diffs": list, "stats": dict}]]
    """
    plot_data_by_rewriter = {}

    for rewriter_name, rewriter_results in rewrite_data.items():
        plot_data_by_rewriter[rewriter_name] = {}

        for add_attr, remove_attr, short_label in bias_pairs:
            add_rollouts = rewriter_results.get(add_attr, {})
            remove_rollouts = rewriter_results.get(remove_attr, {})

            pairwise_diffs = []

            for user_prompt in add_rollouts:
                if user_prompt not in remove_rollouts:
                    continue
                add_user_rollouts = add_rollouts[user_prompt]
                remove_user_rollouts = remove_rollouts[user_prompt]

                for add_r, remove_r in zip(add_user_rollouts, remove_user_rollouts):
                    if add_r is None or remove_r is None:
                        continue
                    # student_diff is already rewritten - baseline
                    add_score = add_r.get("student_diff")
                    remove_score = remove_r.get("student_diff")
                    if add_score is not None and remove_score is not None:
                        pairwise_diffs.append(add_score - remove_score)

            # Compute stats
            stats = {
                "diff_mean": float(np.mean(pairwise_diffs)) if pairwise_diffs else None,
                "diff_stderr": float(np.std(pairwise_diffs) / np.sqrt(len(pairwise_diffs))) if len(pairwise_diffs) > 1 else None,
                "n_samples": len(pairwise_diffs),
            }

            plot_data_by_rewriter[rewriter_name][short_label] = {
                "diffs": pairwise_diffs,
                "stats": stats,
            }

    return plot_data_by_rewriter


def create_violin_plot(
    plot_data_by_rewriter: dict[str, dict[str, dict]],
    bias_pairs: list[tuple[str, str, str]],
    output_path: Path,
):
    """Create violin plot from pairwise diff data."""

    def hex_to_rgba(hex_color, alpha):
        h = hex_color.lstrip('#')
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"

    def separate_outliers(data, k=1.5):
        """Separate data into inliers and outliers using IQR method."""
        if len(data) < 4:
            return data, []
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        inliers = [x for x in data if lower <= x <= upper]
        outliers = [x for x in data if x < lower or x > upper]
        return inliers, outliers

    # Get short labels in order
    short_labels = [bp[2] for bp in bias_pairs]

    # Compute aggregate stats for each attribute (pooled across all rewriters)
    aggregate_stats = {}
    for short_label in short_labels:
        pooled_diffs = []
        for rewriter_data in plot_data_by_rewriter.values():
            if short_label in rewriter_data:
                pooled_diffs.extend(rewriter_data[short_label]["diffs"])

        if pooled_diffs:
            # Remove outliers for stats computation
            inliers, _ = separate_outliers(pooled_diffs)
            if inliers:
                mean = np.mean(inliers)
                stderr = np.std(inliers) / np.sqrt(len(inliers))
                ci_95 = 1.96 * stderr
                aggregate_stats[short_label] = {"mean": mean, "ci": ci_95}
            else:
                aggregate_stats[short_label] = {"mean": None, "ci": None}
        else:
            aggregate_stats[short_label] = {"mean": None, "ci": None}

    # Build tick labels with stats
    tick_labels = []
    for short_label in short_labels:
        stats = aggregate_stats[short_label]
        if stats["mean"] is not None:
            mean_str = f"{stats['mean']:+.2f}"
            ci_str = f"\u00b1{stats['ci']:.2f}"
            tick_labels.append(f"{short_label}<br>{mean_str}<br>{ci_str}")
        else:
            tick_labels.append(short_label)

    # Dark2 color palette
    dark2_colors = ["#1b9e77", "#d95f02", "#7570b3"]
    rewriter_names = sorted(plot_data_by_rewriter.keys())
    offsets = [-0.25, 0, 0.25]  # x-offsets for each rewriter

    # Truncation range for y-axis
    y_min, y_max = -15, 15

    fig = go.Figure()

    # Track overflow counts
    overflow_annotations = []

    for rewriter_idx, rewriter_name in enumerate(rewriter_names):
        rewriter_data = plot_data_by_rewriter[rewriter_name]
        color_hex = dark2_colors[rewriter_idx % len(dark2_colors)]
        color_fill = hex_to_rgba(color_hex, 0.6)
        short_rewriter_name = rewriter_name.split("/")[-1]
        offset = offsets[rewriter_idx]

        for label_idx, short_label in enumerate(short_labels):
            if short_label not in rewriter_data:
                continue
            diffs = rewriter_data[short_label]["diffs"]
            if not diffs:
                continue

            inliers, outliers = separate_outliers(diffs)
            x_pos = label_idx + offset

            # Count points exceeding the truncation range
            count_above = sum(1 for d in diffs if d > y_max)
            count_below = sum(1 for d in diffs if d < y_min)

            if count_above > 0:
                overflow_annotations.append((x_pos, y_max, f"{count_above}\u2191", color_hex))
            if count_below > 0:
                overflow_annotations.append((x_pos, y_min, f"{count_below}\u2193", color_hex))

            # Clip outliers to be within the visible range for display
            clipped_outliers = [max(y_min, min(y_max, o)) for o in outliers]

            # Violin from inliers only
            fig.add_trace(go.Violin(
                y=inliers,
                x0=x_pos,
                name=short_rewriter_name,
                legendgroup=rewriter_name,
                showlegend=(label_idx == 0),
                line_color=color_hex,
                fillcolor=color_fill,
                box_visible=True,
                box=dict(
                    visible=True,
                    width=0.12,
                    line=dict(color="black", width=1.5),
                    fillcolor=color_fill,
                ),
                meanline_visible=True,
                meanline=dict(color="#e41a1c", width=2),
                scalemode="width",
                width=0.22,
                side="both",
                points=False,
            ))

            # Add outliers as separate scatter with transparency (clipped to range)
            if clipped_outliers:
                fig.add_trace(go.Scatter(
                    x=[x_pos] * len(clipped_outliers),
                    y=clipped_outliers,
                    mode="markers",
                    marker=dict(
                        color=hex_to_rgba(color_hex, 0.5),
                        size=6,
                        line=dict(color=color_hex, width=1),
                    ),
                    legendgroup=rewriter_name,
                    showlegend=False,
                    hoverinfo="y",
                ))

    # Add overflow annotations
    for x_pos, y_pos, text, color in overflow_annotations:
        y_offset = 0.8 if "\u2191" in text else -0.8
        fig.add_annotation(
            x=x_pos,
            y=y_pos + y_offset,
            text=text,
            showarrow=False,
            font=dict(size=9, color=color),
            xanchor="center",
            yanchor="middle",
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        yaxis_title="Reward diff (present \u2212 absent)",
        yaxis=dict(
            range=[y_min - 2.5, y_max + 2.5],
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(short_labels))),
            ticktext=tick_labels,
            title="",
        ),
        violinmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=60, b=100),
    )

    # Save to same location as input, or specified output
    fig.write_image(str(output_path.with_suffix(".pdf")))
    fig.write_html(str(output_path.with_suffix(".html")))
    print(f"Saved violin plot to {output_path.with_suffix('.pdf')}")


def main(run_dir: Path, output_path: Path | None = None):
    """Regenerate violin plot from saved run data."""
    print(f"Loading data from {run_dir}")

    rewrite_data = load_rewrite_data(run_dir)
    print(f"Loaded data for {len(rewrite_data)} rewriters: {list(rewrite_data.keys())}")

    plot_data = compute_pairwise_diffs(rewrite_data, DEFAULT_BIAS_PAIRS)

    if output_path is None:
        output_path = run_dir / "pairwise_diff_violin"

    create_violin_plot(plot_data, DEFAULT_BIAS_PAIRS, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate violin plot from saved run data")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory")
    parser.add_argument("--output", "-o", type=Path, help="Output path (without extension)")

    args = parser.parse_args()
    main(args.run_dir, args.output)
