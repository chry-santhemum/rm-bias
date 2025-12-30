import json
import html
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from loguru import logger

from bias_evaluator import BiasEvaluator
from api_models import RewriteModel, SAME_ATTRS
from reward_models import LocalRewardModel
from state import Rollout, RewriteScore
from utils import remove_outliers


async def multiple_rewriters(
    reward_model: LocalRewardModel,
    rewriter_models: list[RewriteModel],
    user_prompts: list[str],
    attributes: list[str],
    baselines: dict[str, list[Rollout]],
    n_rollouts: int,
    save_dir: Path,
    n_rewrite_workers: int = 128,
):
    """
    Evaluate whether experimental results stay the same across different rewriter pipelines.

    For each rewriter model:
    1. Evaluate all attributes using that rewriter
    2. Save results to separate files

    Then create a violin plot showing reward distributions for each (attribute, rewriter) pair.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Store results from each rewriter
    all_rewriter_results = []

    for rewriter_idx, rewriter_model in enumerate(rewriter_models):
        rewriter_name = rewriter_model.model_name.split("/")[-1]  # Extract model name
        print(f"Evaluating with rewriter {rewriter_idx + 1}/{len(rewriter_models)}: {rewriter_name}")

        # Create evaluator for this rewriter
        evaluator = BiasEvaluator(
            rewrite_model=rewriter_model,
            reward_model=reward_model,
            n_rewrite_workers=n_rewrite_workers,
        )

        # Evaluate attributes
        async with evaluator:  # type: ignore
            results = await evaluator.evaluate_attributes(
                user_prompts=user_prompts,
                attributes=attributes,
                same_attrs=[SAME_ATTRS] * len(attributes),
                baselines=baselines,
                n_rollouts=n_rollouts,
            )

        # Save full rollouts
        rollouts_data = {}
        for attribute, rollouts_by_prompt in results.items():
            rollouts_data[attribute] = {}
            for user_prompt, rollouts in rollouts_by_prompt.items():
                rollouts_data[attribute][user_prompt] = [
                    {
                        "response": r.response,
                        "student_score": r.student_score.score,
                    } if r is not None else None
                    for r in rollouts
                ]

        with open(save_dir / f"{rewriter_name}_rollouts.json", "w") as f:
            json.dump(rollouts_data, f, indent=4, sort_keys=True)

        # Save student diffs (similar to validation format)
        student_diffs = {}
        for attribute, rollouts_by_prompt in results.items():
            student_diffs[attribute] = {}
            for user_prompt, rollouts in rollouts_by_prompt.items():
                diffs = [
                    r.student_score.score if r is not None and r.student_score.score is not None else None
                    for r in rollouts
                ]
                student_diffs[attribute][user_prompt] = diffs

        with open(save_dir / f"{rewriter_name}_student_diffs.json", "w") as f:
            json.dump(student_diffs, f, indent=4, sort_keys=True)

        # Collect data for plotting
        all_rewriter_results.append({
            "rewriter_name": rewriter_name,
            "results": student_diffs,
        })

        logger.success(f"Evaluated {rewriter_name}.")

    return all_rewriter_results

def compute_winrate_from_diffs(diffs: list[float | None]) -> tuple[float | None, float | None]:
    """
    Compute win rate from diffs: 1 if > 0, 0 if < 0, 0.5 if == 0.
    Returns (mean, stderr).
    """
    winrates = []
    for d in diffs:
        if d is None:
            continue
        elif d > 0:
            winrates.append(1)
        elif d < 0:
            winrates.append(0)
        else:
            winrates.append(0.5)

    if not winrates:
        return None, None

    mean = np.mean(winrates)
    stderr = np.std(winrates) / np.sqrt(len(winrates)) if len(winrates) > 1 else None
    return float(mean), float(stderr) if stderr is not None else None


def compute_attribute_stats_from_diffs(
    student_diffs: dict[str, dict[str, list[float | None]]],
    teacher_diffs: dict[str, dict[str, list[float | None]]] | None = None,
) -> dict[str, dict]:
    """
    Compute win rates and mean diffs for each attribute from raw diffs.

    Returns dict mapping attribute -> {
        student_winrate, student_winrate_stderr,
        student_diff_mean, student_diff_stderr,
        teacher_winrate, teacher_winrate_stderr
    }
    """
    stats = {}

    for attribute, user_prompt_diffs in student_diffs.items():
        all_diffs = []
        for diffs in user_prompt_diffs.values():
            all_diffs.extend(diffs)

        # Filter None values for diff calculation
        valid_diffs = [d for d in all_diffs if d is not None]
        cleaned_diffs = remove_outliers(valid_diffs) if valid_diffs else []

        # Compute student win rate
        stu_wr, stu_wr_err = compute_winrate_from_diffs(all_diffs)

        # Compute student diff mean
        stu_diff_mean = float(np.mean(cleaned_diffs)) if cleaned_diffs else None
        stu_diff_err = float(np.std(cleaned_diffs) / np.sqrt(len(cleaned_diffs))) if len(cleaned_diffs) > 1 else None

        # Compute teacher win rate if available
        tch_wr, tch_wr_err = None, None
        if teacher_diffs and attribute in teacher_diffs:
            teacher_all_diffs = []
            for diffs in teacher_diffs[attribute].values():
                teacher_all_diffs.extend(diffs)
            tch_wr, tch_wr_err = compute_winrate_from_diffs(teacher_all_diffs)

        stats[attribute] = {
            "student_winrate": stu_wr,
            "student_winrate_stderr": stu_wr_err,
            "student_diff_mean": stu_diff_mean,
            "student_diff_stderr": stu_diff_err,
            "teacher_winrate": tch_wr,
            "teacher_winrate_stderr": tch_wr_err,
        }

    return stats


def format_attribute_stats(item: dict) -> str:
    """Format student WR, student diff, and teacher WR stats for display."""
    parts = []

    # Student win rate (0-1)
    stu_wr = item.get("student_winrate")
    if stu_wr is not None:
        stu_wr_err = item.get("student_winrate_stderr")
        if stu_wr_err is not None:
            parts.append(f"Student WR: {stu_wr:.2f}±{stu_wr_err:.2f}")
        else:
            parts.append(f"Student WR: {stu_wr:.2f}")

    # Student diff mean
    stu_diff = item.get("student_diff_mean")
    if stu_diff is not None:
        stu_diff_err = item.get("student_diff_stderr")
        if stu_diff_err is not None:
            parts.append(f"Student Δ: {stu_diff:.2f}±{stu_diff_err:.2f}")
        else:
            parts.append(f"Student Δ: {stu_diff:.2f}")

    # Teacher win rate (0-1)
    tch_wr = item.get("teacher_winrate")
    if tch_wr is not None:
        tch_wr_err = item.get("teacher_winrate_stderr")
        if tch_wr_err is not None:
            parts.append(f"Teacher WR: {tch_wr:.2f}±{tch_wr_err:.2f}")
        else:
            parts.append(f"Teacher WR: {tch_wr:.2f}")

    if parts:
        return f"<i>({', '.join(parts)})</i>"
    return ""


def plot_multi_rewriter_boxplot(
    all_rewriter_results: list[dict],
    attributes: list[str],
    save_path: Path,
    title: str|None,
    x_range: tuple[float, float] = (-15, 15),
    attribute_stats: dict[str, dict] | None = None,
):
    """
    Create horizontal boxplots showing reward distributions for each attribute across different rewriters.

    Each attribute gets a y-axis position with multiple horizontal boxplots (one per rewriter).
    X-axis is clipped to x_range with overflow annotations.
    """
    # Helper function to wrap text at specified width
    def wrap_text(text, width=60):
        text = html.escape(text)
        text = text.replace("$", "&#36;")
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

    x_min, x_max = x_range

    # Define colors for each rewriter
    colors = [
        'rgb(31, 119, 180)',   # Blue
        'rgb(255, 127, 14)',   # Orange
        'rgb(44, 160, 44)',    # Green
        'rgb(214, 39, 40)',    # Red
        'rgb(148, 103, 189)',  # Purple
        'rgb(140, 86, 75)',    # Brown
    ]

    fig = go.Figure()

    # Sort alphabetically (reverse so A appears at top of plot)
    sorted_attributes = sorted(attributes, reverse=True)

    # Track overflow annotations: (x, y, text, xanchor)
    overflow_annotations: list[tuple[float, float, str, str]] = []

    # For each attribute, create boxplots for each rewriter
    for attr_idx, attribute in enumerate(sorted_attributes):
        # Collect data from all rewriters for this attribute
        for rewriter_idx, rewriter_result in enumerate(all_rewriter_results):
            rewriter_name = rewriter_result["rewriter_name"]
            student_diffs = rewriter_result["results"]

            # Collect all diffs for this attribute
            all_diffs = []
            if attribute in student_diffs:
                for user_prompt, diffs in student_diffs[attribute].items():
                    all_diffs.extend([d for d in diffs if d is not None])

            if len(all_diffs) == 0:
                continue

            # Calculate mean and stderr with outliers removed
            cleaned_diffs = remove_outliers(all_diffs)
            if len(cleaned_diffs) == 0:
                cleaned_diffs = all_diffs  # Fall back if all removed
            mean_diff = np.mean(cleaned_diffs)
            stderr_diff = np.std(cleaned_diffs) / np.sqrt(len(cleaned_diffs)) if len(cleaned_diffs) > 1 else 0

            # Create y-position with offset for multiple boxes per attribute
            n_rewriters = len(all_rewriter_results)
            offset_range = 0.8  # Total range for offsets
            offset = (rewriter_idx - (n_rewriters - 1) / 2) * (offset_range / n_rewriters)
            y_position = attr_idx + offset

            # Track overflow for this rewriter's data
            data_min = min(all_diffs)
            data_max = max(all_diffs)
            if data_max > x_max:
                overflow_annotations.append((x_max, y_position, f"→{data_max:.1f}", "left"))
            if data_min < x_min:
                overflow_annotations.append((x_min, y_position, f"{data_min:.1f}←", "right"))

            # Create display name with stats
            annotation = f"{rewriter_name}: {mean_diff:.2f}±{stderr_diff:.2f}"

            # Create box trace
            fig.add_trace(
                go.Box(
                    x=all_diffs,
                    y0=y_position,
                    name=rewriter_name,
                    orientation='h',
                    marker_color=colors[rewriter_idx % len(colors)],
                    line=dict(color='black', width=1.5),  # Black outline for visible median/mean
                    fillcolor=colors[rewriter_idx % len(colors)],
                    opacity=0.6,
                    boxpoints="outliers",
                    boxmean=True,  # Show mean as dashed line
                    jitter=0.3,
                    pointpos=0,
                    legendgroup=rewriter_name,
                    showlegend=(attr_idx == 0),  # Only show legend for first attribute
                    hovertemplate=f"{annotation}<br>Value: %{{x:.3f}}<extra></extra>",
                )
            )

    # Add overflow annotations at correct y-positions
    for x, y, text, xanchor in overflow_annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=8, color="gray"),
            xanchor=xanchor,
        )

    # Create y-axis tick labels (one per attribute) with stats
    y_tick_vals = list(range(len(attributes)))
    y_tick_labels = []
    for attr in sorted_attributes:
        label = wrap_text(attr, width=60)
        if attribute_stats and attr in attribute_stats:
            stats_str = format_attribute_stats(attribute_stats[attr])
            if stats_str:
                label += f"<br>{stats_str}"
            # Red flag: student WR > 0.5 AND teacher WR < 0.5
            stu_wr = attribute_stats[attr].get("student_winrate")
            tch_wr = attribute_stats[attr].get("teacher_winrate")
            if stu_wr is not None and tch_wr is not None and stu_wr > 0.5 and tch_wr < 0.5:
                label = f"<span style='color:red'>{label}</span>"
        y_tick_labels.append(label)

    # Calculate plot height
    n_attributes = len(attributes)
    plot_height = max(500, min(1600, 100 + n_attributes * 120))

    fig.update_layout(
        title=dict(text=title or "Reward Diffs across Rewriters", font=dict(size=18)),
        xaxis_title="Student reward diff",
        yaxis_title=None,
        height=plot_height,
        width=1000,
        xaxis=dict(
            range=[x_min - 1, x_max + 1],  # Slight padding for overflow annotations
            tickfont=dict(size=11),
            title=dict(font=dict(size=12)),
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            tickmode='array',
            tickvals=y_tick_vals,
            ticktext=y_tick_labels,
            automargin=True,
            ticklabelstandoff=25,  # Add space between labels and plot
        ),
        font=dict(size=11),
        boxgap=0.1,
        boxgroupgap=0.1,
        margin=dict(l=10, r=40, t=80, b=60),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.15,
        ),
    )

    # Add vertical reference line at 0
    fig.add_vline(
        x=0, line_dash="dash", line_color="black", line_width=1.5, opacity=0.6
    )

    fig.write_image(save_path)
    print(f"Saved multi-rewriter boxplot to {save_path}")


def mean_diff_correlation(
    all_rewriter_results: list[dict],
    original_rewriter_name: str,
    save_path: Path,
):
    """
    For each attribute in each seed state,
    compute the mean reward uplift for each rewriter.
    Then, draw a scatter plot of the mean reward uplift of each rewriter
    vs the original rewriter, and fit a linear regression line.
    Save the plots to the given save path, one plot for each rewriter.

    Also, plot a similar scatter plot without aggregating per attribute first.
    In other words, each point should be a single rewrite attempt.
    """
    from scipy import stats

    # Find original rewriter results
    original_results = None
    for r in all_rewriter_results:
        if r["rewriter_name"] == original_rewriter_name:
            original_results = r["results"]
            break
    if original_results is None:
        raise ValueError(f"Original rewriter '{original_rewriter_name}' not found")

    save_path.mkdir(parents=True, exist_ok=True)

    for rewriter_result in all_rewriter_results:
        rewriter_name = rewriter_result["rewriter_name"]
        if rewriter_name == original_rewriter_name:
            continue

        new_results = rewriter_result["results"]

        # Build aggregated data (per-attribute means)
        orig_means = []
        new_means = []
        attribute_labels = []

        # Build paired individual data
        orig_diffs_individual = []
        new_diffs_individual = []

        for attribute in original_results:
            # Compute mean for original rewriter
            orig_all_diffs = []
            for diffs in original_results[attribute].values():
                orig_all_diffs.extend([d for d in diffs if d is not None])

            # Compute mean for new rewriter
            new_all_diffs = []
            for diffs in new_results[attribute].values():
                new_all_diffs.extend([d for d in diffs if d is not None])

            if orig_all_diffs and new_all_diffs:
                orig_cleaned = remove_outliers(orig_all_diffs)
                new_cleaned = remove_outliers(new_all_diffs)
                if orig_cleaned and new_cleaned:
                    orig_means.append(np.mean(orig_cleaned))
                    new_means.append(np.mean(new_cleaned))
                    attribute_labels.append(attribute)

            # Paired individual diffs
            for user_prompt in original_results[attribute]:
                orig_prompt_diffs = original_results[attribute][user_prompt]
                new_prompt_diffs = new_results[attribute][user_prompt]

                for i in range(min(len(orig_prompt_diffs), len(new_prompt_diffs))):
                    orig_d = orig_prompt_diffs[i]
                    new_d = new_prompt_diffs[i]
                    if orig_d is not None and new_d is not None:
                        orig_diffs_individual.append(orig_d)
                        new_diffs_individual.append(new_d)

        # Plot 1: Aggregated by attribute
        if len(orig_means) >= 2:
            fig = go.Figure()

            # Scatter points
            fig.add_trace(go.Scatter(
                x=orig_means,
                y=new_means,
                mode='markers',
                marker=dict(size=8, color='rgb(31, 119, 180)'),
                text=attribute_labels,
                hovertemplate='%{text}<br>Original: %{x:.2f}<br>New: %{y:.2f}<extra></extra>',
            ))

            # Regression line
            slope, intercept, r_value, p_value, _ = stats.linregress(orig_means, new_means)
            x_line = np.array([min(orig_means), max(orig_means)])
            y_line = slope * x_line + intercept

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'r={r_value:.3f}, p={p_value:.3e}',
            ))

            # y=x reference line
            all_vals = orig_means + new_means
            min_val, max_val = min(all_vals), max(all_vals)
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', dash='dot'),
                name='y=x',
            ))

            fig.update_layout(
                title=f'{rewriter_name} vs {original_rewriter_name} (per-attribute means)<br>r={r_value:.3f}, p={p_value:.3e}',
                xaxis_title=f'{original_rewriter_name} mean diff',
                yaxis_title=f'{rewriter_name} mean diff',
                width=700,
                height=600,
            )

            fig.write_image(save_path / f'{rewriter_name}_vs_original_aggregated.pdf')

        # Plot 2: Individual paired diffs
        if len(orig_diffs_individual) >= 2:
            fig = go.Figure()

            # Scatter points with transparency
            fig.add_trace(go.Scatter(
                x=orig_diffs_individual,
                y=new_diffs_individual,
                mode='markers',
                marker=dict(size=4, color='rgb(31, 119, 180)', opacity=0.3),
                hovertemplate='Original: %{x:.2f}<br>New: %{y:.2f}<extra></extra>',
            ))

            # Regression line
            slope, intercept, r_value, p_value, _ = stats.linregress(orig_diffs_individual, new_diffs_individual)
            x_line = np.array([min(orig_diffs_individual), max(orig_diffs_individual)])
            y_line = slope * x_line + intercept

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'r={r_value:.3f}, p={p_value:.3e}',
            ))

            # y=x reference line
            all_vals = orig_diffs_individual + new_diffs_individual
            min_val, max_val = min(all_vals), max(all_vals)
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', dash='dot'),
                name='y=x',
            ))

            fig.update_layout(
                title=f'{rewriter_name} vs {original_rewriter_name} (individual rewrites, n={len(orig_diffs_individual)})<br>r={r_value:.3f}, p={p_value:.3e}',
                xaxis_title=f'{original_rewriter_name} diff',
                yaxis_title=f'{rewriter_name} diff',
                width=700,
                height=600,
            )

            fig.write_image(save_path / f'{rewriter_name}_vs_original_individual.pdf')

    print(f"Saved correlation plots to {save_path}")


def mean_diff_correlation_all_seeds(
    all_seeds_results: list[tuple[str, list[dict]]],  # list of (seed_index, all_rewriter_results)
    original_rewriter_name: str,
    save_path: Path,
):
    """
    Like mean_diff_correlation but aggregates data across all seeds.
    Each point in the aggregated plot is one (seed, attribute) pair.
    """
    from scipy import stats

    save_path.mkdir(parents=True, exist_ok=True)

    # Collect all rewriter names (excluding original)
    all_rewriter_names = set()
    for _, all_rewriter_results in all_seeds_results:
        for r in all_rewriter_results:
            if r["rewriter_name"] != original_rewriter_name:
                all_rewriter_names.add(r["rewriter_name"])

    for rewriter_name in all_rewriter_names:
        # Accumulate across seeds
        orig_means = []
        new_means = []
        labels = []
        orig_diffs_individual = []
        new_diffs_individual = []

        for seed_index, all_rewriter_results in all_seeds_results:
            # Find original and new rewriter results for this seed
            original_results = None
            new_results = None
            for r in all_rewriter_results:
                if r["rewriter_name"] == original_rewriter_name:
                    original_results = r["results"]
                elif r["rewriter_name"] == rewriter_name:
                    new_results = r["results"]

            if original_results is None or new_results is None:
                continue

            for attribute in original_results:
                if attribute not in new_results:
                    continue

                # Compute means for this (seed, attribute)
                orig_all_diffs = []
                for diffs in original_results[attribute].values():
                    orig_all_diffs.extend([d for d in diffs if d is not None])

                new_all_diffs = []
                for diffs in new_results[attribute].values():
                    new_all_diffs.extend([d for d in diffs if d is not None])

                if orig_all_diffs and new_all_diffs:
                    orig_cleaned = remove_outliers(orig_all_diffs)
                    new_cleaned = remove_outliers(new_all_diffs)
                    if orig_cleaned and new_cleaned:
                        orig_means.append(np.mean(orig_cleaned))
                        new_means.append(np.mean(new_cleaned))
                        labels.append(f"seed{seed_index}: {attribute[:40]}...")

                # Paired individual diffs
                for user_prompt in original_results[attribute]:
                    if user_prompt not in new_results[attribute]:
                        continue
                    orig_prompt_diffs = original_results[attribute][user_prompt]
                    new_prompt_diffs = new_results[attribute][user_prompt]

                    for i in range(min(len(orig_prompt_diffs), len(new_prompt_diffs))):
                        orig_d = orig_prompt_diffs[i]
                        new_d = new_prompt_diffs[i]
                        if orig_d is not None and new_d is not None:
                            orig_diffs_individual.append(orig_d)
                            new_diffs_individual.append(new_d)

        # Plot 1: Aggregated by (seed, attribute)
        if len(orig_means) >= 2:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=orig_means,
                y=new_means,
                mode='markers',
                marker=dict(size=8, color='rgb(31, 119, 180)'),
                text=labels,
                hovertemplate='%{text}<br>Original: %{x:.2f}<br>New: %{y:.2f}<extra></extra>',
            ))

            slope, intercept, r_value, p_value, _ = stats.linregress(orig_means, new_means)
            x_line = np.array([min(orig_means), max(orig_means)])
            y_line = slope * x_line + intercept

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'r={r_value:.3f}, p={p_value:.3e}',
            ))

            all_vals = orig_means + new_means
            min_val, max_val = min(all_vals), max(all_vals)
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', dash='dot'),
                name='y=x',
            ))

            fig.update_layout(
                title=f'{rewriter_name} vs {original_rewriter_name} (all seeds, per-attribute)<br>r={r_value:.3f}, p={p_value:.3e}, n={len(orig_means)}',
                xaxis_title=f'{original_rewriter_name} mean diff',
                yaxis_title=f'{rewriter_name} mean diff',
                width=700,
                height=600,
            )

            fig.write_image(save_path / f'{rewriter_name}_vs_original_aggregated.pdf')

        # Plot 2: Individual paired diffs across all seeds
        if len(orig_diffs_individual) >= 2:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=orig_diffs_individual,
                y=new_diffs_individual,
                mode='markers',
                marker=dict(size=4, color='rgb(31, 119, 180)', opacity=0.2),
                hovertemplate='Original: %{x:.2f}<br>New: %{y:.2f}<extra></extra>',
            ))

            slope, intercept, r_value, p_value, _ = stats.linregress(orig_diffs_individual, new_diffs_individual)
            x_line = np.array([min(orig_diffs_individual), max(orig_diffs_individual)])
            y_line = slope * x_line + intercept

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'r={r_value:.3f}, p={p_value:.3e}',
            ))

            all_vals = orig_diffs_individual + new_diffs_individual
            min_val, max_val = min(all_vals), max(all_vals)
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', dash='dot'),
                name='y=x',
            ))

            fig.update_layout(
                title=f'{rewriter_name} vs {original_rewriter_name} (all seeds, individual)<br>r={r_value:.3f}, p={p_value:.3e}, n={len(orig_diffs_individual)}',
                xaxis_title=f'{original_rewriter_name} diff',
                yaxis_title=f'{rewriter_name} diff',
                width=700,
                height=600,
            )

            fig.write_image(save_path / f'{rewriter_name}_vs_original_individual.pdf')

    print(f"Saved cross-seed correlation plots to {save_path}")


def compute_paired_rewriter_diffs_by_attribute(
    all_rewriter_results: list[dict],
    original_rewriter_name: str,
) -> dict[str, dict[str, dict]]:
    """
    Compute paired reward diffs per attribute between each new rewriter and the original.

    Returns:
        Dict mapping attribute -> rewriter_name -> {
            "diffs": list of paired diffs,
            "mean": mean diff,
            "p10": 10th percentile,
            "p90": 90th percentile,
            "n_samples": count,
        }
    """
    # Find original rewriter results
    original_results = None
    for r in all_rewriter_results:
        if r["rewriter_name"] == original_rewriter_name:
            original_results = r["results"]
            break

    if original_results is None:
        raise ValueError(f"Original rewriter '{original_rewriter_name}' not found")

    # {attribute: {rewriter_name: {stats}}}
    per_attr_stats: dict[str, dict[str, dict]] = {}

    for rewriter_result in all_rewriter_results:
        rewriter_name = rewriter_result["rewriter_name"]
        if rewriter_name == original_rewriter_name:
            continue

        new_results = rewriter_result["results"]

        for attribute in original_results:
            if attribute not in new_results:
                continue

            diffs = []
            for user_prompt in original_results[attribute]:
                if user_prompt not in new_results[attribute]:
                    continue
                orig_scores = original_results[attribute][user_prompt]
                new_scores = new_results[attribute][user_prompt]

                for idx in range(min(len(orig_scores), len(new_scores))):
                    orig_score = orig_scores[idx]
                    new_score = new_scores[idx]
                    if orig_score is not None and new_score is not None:
                        diffs.append(new_score - orig_score)

            if diffs:
                # Remove outliers for mean calculation
                cleaned_diffs = remove_outliers(diffs)
                if len(cleaned_diffs) == 0:
                    cleaned_diffs = diffs  # Fall back if all removed
                if attribute not in per_attr_stats:
                    per_attr_stats[attribute] = {}
                per_attr_stats[attribute][rewriter_name] = {
                    "diffs": diffs,
                    "mean": float(np.mean(cleaned_diffs)),
                    "p10": float(np.percentile(diffs, 10)),
                    "p90": float(np.percentile(diffs, 90)),
                    "n_samples": len(diffs),
                }

    return per_attr_stats


def plot_rewriter_variation_by_attribute(
    per_attr_stats: dict[str, dict[str, dict]],
    attributes: list[str],
    original_effect_sizes: dict[str, float],
    title: str|None,
    save_path: Path,
    x_range: tuple[float, float] = (-10, 10),
    attribute_stats: dict[str, dict] | None = None,
):
    """
    Create horizontal error bar plot showing rewriter variation for each attribute.

    Similar layout to plot_multi_rewriter_boxplot but with error bars instead of boxes.
    Annotates each attribute with the original rewrite effect size for context.
    Error bars are truncated to x_range with overflow annotations.
    """
    import html

    def wrap_text(text, width=60):
        text = html.escape(text)
        text = text.replace("$", "&#36;")
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

    x_min, x_max = x_range

    # Define colors for each rewriter
    colors = [
        'rgb(31, 119, 180)',   # Blue
        'rgb(255, 127, 14)',   # Orange
        'rgb(44, 160, 44)',    # Green
        'rgb(214, 39, 40)',    # Red
        'rgb(148, 103, 189)',  # Purple
        'rgb(140, 86, 75)',    # Brown
    ]

    # Get all rewriter names
    all_rewriters = set()
    for attr_stats in per_attr_stats.values():
        all_rewriters.update(attr_stats.keys())
    rewriter_names = sorted(all_rewriters)

    fig = go.Figure()

    # Sort alphabetically (reverse so A appears at top of plot)
    sorted_attributes = sorted(attributes, reverse=True)

    # Track overflow annotations: (x, y, text, xanchor)
    overflow_annotations: list[tuple[float, float, str, str]] = []

    # For each attribute, create error bars for each rewriter
    for attr_idx, attribute in enumerate(sorted_attributes):
        if attribute not in per_attr_stats:
            continue

        for rewriter_idx, rewriter_name in enumerate(rewriter_names):
            if rewriter_name not in per_attr_stats[attribute]:
                continue

            stats = per_attr_stats[attribute][rewriter_name]
            mean = stats["mean"]
            p10 = stats["p10"]
            p90 = stats["p90"]

            # Clip values for display
            display_mean = np.clip(mean, x_min, x_max)
            display_p10 = max(p10, x_min)
            display_p90 = min(p90, x_max)

            # Calculate clipped error bar extents
            error_plus = max(0, display_p90 - display_mean)
            error_minus = max(0, display_mean - display_p10)

            # Create y-position with offset for multiple rewriters per attribute
            n_rewriters = len(rewriter_names)
            offset_range = 0.6
            offset = (rewriter_idx - (n_rewriters - 1) / 2) * (offset_range / n_rewriters)
            y_position = attr_idx + offset

            # Track overflow for this specific rewriter bar
            if p90 > x_max:
                overflow_annotations.append((x_max, y_position, f"→{p90:.1f}", "left"))
            if p10 < x_min:
                overflow_annotations.append((x_min, y_position, f"{p10:.1f}←", "right"))

            # Error bar: from p10 to p90, centered on mean (clipped)
            fig.add_trace(
                go.Scatter(
                    x=[display_mean],
                    y=[y_position],
                    mode='markers',
                    marker=dict(
                        color=colors[rewriter_idx % len(colors)],
                        size=8,
                        symbol='diamond',
                    ),
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=[error_plus],
                        arrayminus=[error_minus],
                        color=colors[rewriter_idx % len(colors)],
                        thickness=2,
                        width=6,
                    ),
                    name=rewriter_name,
                    legendgroup=rewriter_name,
                    showlegend=(attr_idx == len(attributes) - 1),  # Show legend for first plotted attr
                    hovertemplate=f"{rewriter_name}<br>Mean: {mean:.3f}<br>10th: {p10:.3f}<br>90th: {p90:.3f}<br>n={stats['n_samples']}<extra></extra>",
                )
            )

    # Add overflow annotations at correct y-positions
    for x, y, text, xanchor in overflow_annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=8, color="gray"),
            xanchor=xanchor,
        )

    # Create y-axis tick labels with original effect size and stats
    y_tick_vals = list(range(len(attributes)))
    y_tick_labels = []
    for attr in sorted_attributes:
        effect_size = original_effect_sizes.get(attr, 0.0)
        label = wrap_text(attr, width=70)
        # Append stats if available
        if attribute_stats and attr in attribute_stats:
            stats_str = format_attribute_stats(attribute_stats[attr])
            if stats_str:
                label += f"<br>{stats_str}"
            # Red flag: student WR > 0.5 AND teacher WR < 0.5
            stu_wr = attribute_stats[attr].get("student_winrate")
            tch_wr = attribute_stats[attr].get("teacher_winrate")
            if stu_wr is not None and tch_wr is not None and stu_wr > 0.5 and tch_wr < 0.5:
                label = f"<span style='color:red'>{label}</span>"
        y_tick_labels.append(label)

    # Calculate plot height
    n_attributes = len(attributes)
    plot_height = max(500, min(1600, 100 + n_attributes * 80))

    fig.update_layout(
        title=dict(text=title or "Rewriter Variation by Attribute (Mean ± 10/90 percentiles)", font=dict(size=18)),
        xaxis_title="Reward Diff (New − Original)",
        yaxis_title=None,
        height=plot_height,
        width=1000,
        xaxis=dict(
            range=[x_min - 1, x_max + 1],  # Slight padding for overflow annotations
            tickfont=dict(size=11),
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            tickmode='array',
            tickvals=y_tick_vals,
            ticktext=y_tick_labels,
            automargin=True,
            ticklabelstandoff=15,  # Add space between labels and plot
        ),
        font=dict(size=11),
        margin=dict(l=10, r=40, t=80, b=60),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.15,
        ),
    )

    # Add vertical reference line at 0
    fig.add_vline(
        x=0, line_dash="dash", line_color="black", line_width=1.5, opacity=0.6
    )

    fig.write_image(save_path)
    print(f"Saved per-attribute rewriter variation plot to {save_path}")


if __name__ == "__main__":
    import asyncio
    import torch

    rewriter_models = [
        RewriteModel(
            model_name="openai/gpt-5-nano",
            max_tokens=12000,
            reasoning="high",
        ),
        RewriteModel(
            model_name="anthropic/claude-haiku-4.5",
            max_par=128,
            max_tokens=10000,
            reasoning=8000,
        ),
        RewriteModel(
            model_name="x-ai/grok-4.1-fast",
            max_tokens=10000,
            reasoning="high",
        ),
        # RewriteModel(
        #     model_name="google/gemini-2.5-flash-lite-preview-09-2025",
        #     max_tokens=4096,
        #     reasoning=2000,
        # ),
        RewriteModel(
            model_name="google/gemini-3-flash-preview",
            max_tokens=10000,
            reasoning=8000,
        ),
    ]

    async def main(run_path: Path):
        run_name = run_path.name

        # Logging
        Path("logs/rewrite_val_invariance").mkdir(parents=True, exist_ok=True)
        logger.enable("caller")
        logger.remove()
        logger.add(
            f"logs/rewrite_val_invariance/{run_name}.log", 
            enqueue=True, level="INFO",
            filter=lambda record: not (record["name"] or "").startswith("caller"),
            retention="7 days"
        )
        logger.add(
            f"logs/rewrite_val_invariance/{run_name}.log",
            enqueue=True, level="WARNING",
            filter="caller",
            retention="7 days"
        )

        # Set up student model from run config
        with open(run_path / "config.json", "r") as f:
            run_config = json.load(f)

        student_model_config = run_config["bias_evaluator"]["reward_model"]
        # student_model = LocalRewardModel(
        #     model_name=student_model_config["model_name"],
        #     devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())],
        #     batch_size_per_device=student_model_config["batch_size"],
        #     attn_implementation=student_model_config["attn_implementation"],
        # )
        original_rewriter = run_config["bias_evaluator"]["rewrite_model"]["model_names"][0]

        # load baselines
        with open(run_path / "val_baselines/rollouts.json", "r") as f:
            baselines_json = json.load(f)

        baselines: dict[str, list[Rollout]] = {}
        for user, rollouts in baselines_json.items():
            baselines[user] = [
                Rollout(
                    response=rollout["response"],
                    student_score=RewriteScore(
                        score=None,
                        raw_score=rollout["student_score"],
                        reasoning=None,
                        model_name=student_model_config["model_name"],
                    ),
                    teacher_score=None,
                    model=rollout.get("model", None),
                ) for rollout in rollouts
            ]
        
        # Accumulate results across seeds for cross-seed correlation plots
        all_seeds_results: list[tuple[str, list[dict]]] = []

        for seed_index, prompts in run_config["prompts"].items():
            print(f"Evaluating seed {seed_index}...")
            val_prompts = prompts["val_prompts"]

            with open(run_path / f"validate/seed_{seed_index}_validate/candidate_stats.json", "r") as f:
                attribute_stats_list = json.load(f)
            attributes = [c["attribute"] for c in attribute_stats_list]

            # load original rewriter results (student diffs)
            with open(run_path / f"validate/seed_{seed_index}_validate/student_diffs.json", "r") as f:
                original_rewriter_results = json.load(f)

            # load teacher diffs if available
            teacher_diffs_path = run_path / f"validate/seed_{seed_index}_validate/teacher_diffs.json"
            teacher_diffs = None
            if teacher_diffs_path.exists():
                with open(teacher_diffs_path, "r") as f:
                    teacher_diffs = json.load(f)

            # Compute proper stats (winrates and diffs) from raw diffs
            attribute_stats_dict = compute_attribute_stats_from_diffs(
                student_diffs=original_rewriter_results,
                teacher_diffs=teacher_diffs,
            )
                
            # all_rewriter_results = await multiple_rewriters(
            #     reward_model=student_model,
            #     rewriter_models=rewriter_models,
            #     user_prompts=val_prompts,
            #     attributes=attributes,
            #     baselines=baselines,
            #     n_rollouts=8,
            #     save_dir=Path(f"data/rewrite_val_invariance/{run_name}/seed_{seed_index}"),
            # )
            
            # load rewriter results
            all_rewriter_results = []
            for rewriter_model in rewriter_models:
                rewriter_name = rewriter_model.model_name.split("/")[-1]
                with open(f"data/rewrite_val_invariance/{run_name}/seed_{seed_index}/{rewriter_name}_student_diffs.json", "r") as f:
                    rewriter_results = json.load(f)
                all_rewriter_results.append({
                    "rewriter_name": rewriter_name,
                    "results": rewriter_results
                })

            all_rewriter_results.append({
                "rewriter_name": original_rewriter.split("/")[-1],
                "results": original_rewriter_results
            })

            # # Create multi-rewriter boxplot
            # plot_multi_rewriter_boxplot(
            #     all_rewriter_results=all_rewriter_results,
            #     attributes=attributes,
            #     save_path=Path(f"data/rewrite_val_invariance/{run_name}/seed_{seed_index}/multi_rewriter_boxplot.pdf"),
            #     title=f"Seed {seed_index} ({prompts['summary']})<br>Reward Diffs across rewriters",
            #     attribute_stats=attribute_stats_dict,
            # )

            # # Compute original effect sizes per attribute (mean reward diff vs baseline)
            # original_effect_sizes = {}
            # for attr, user_prompt_diffs in original_rewriter_results.items():
            #     all_diffs = []
            #     for diffs in user_prompt_diffs.values():
            #         all_diffs.extend([d for d in diffs if d is not None])
            #     if all_diffs:
            #         cleaned_diffs = remove_outliers(all_diffs)
            #         if len(cleaned_diffs) == 0:
            #             cleaned_diffs = all_diffs
            #         original_effect_sizes[attr] = float(np.mean(cleaned_diffs))

            # # Compute and plot per-attribute rewriter variation
            # per_attr_stats = compute_paired_rewriter_diffs_by_attribute(
            #     all_rewriter_results=all_rewriter_results,
            #     original_rewriter_name=original_rewriter.split("/")[-1],
            # )
            # plot_rewriter_variation_by_attribute(
            #     per_attr_stats=per_attr_stats,
            #     attributes=attributes,
            #     original_effect_sizes=original_effect_sizes,
            #     save_path=Path(f"data/rewrite_val_invariance/{run_name}/seed_{seed_index}/rewriter_variation_by_attr.pdf"),
            #     title=f"Seed {seed_index} ({prompts['summary']})<br>New rewrite score - Original rewrite score (Mean ± 10/90 percentiles)",
            #     attribute_stats=attribute_stats_dict,
            # )

            # Correlation plots
            mean_diff_correlation(
                all_rewriter_results=all_rewriter_results,
                original_rewriter_name=original_rewriter.split("/")[-1],
                save_path=Path(f"data/rewrite_val_invariance/{run_name}/seed_{seed_index}/correlation"),
            )

            # Accumulate for cross-seed plots
            all_seeds_results.append((seed_index, all_rewriter_results))

        # Cross-seed correlation plots
        mean_diff_correlation_all_seeds(
            all_seeds_results=all_seeds_results,
            original_rewriter_name=original_rewriter.split("/")[-1],
            save_path=Path(f"data/rewrite_val_invariance/{run_name}/correlation_all_seeds"),
        )


    asyncio.run(main(run_path=Path("data/evo/20251228-165744-list_reverse-handpick-plus")))