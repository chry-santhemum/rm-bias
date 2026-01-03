"""ABOUTME: Plotting functions for reward model bias evaluation results.
ABOUTME: Creates violin plots showing reward diffs across attributes and rewriters."""

import html
import re
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from utils import remove_outliers


def wrap_text(text: str, width: int = 60) -> str:
    """Wrap text to specified width using <br> for line breaks."""
    text = text.replace("&nbsp;", " ")
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


def compute_stats_from_diffs(diffs: list[float | None]) -> dict:
    """Compute winrate, mean diff, and stderr from a list of diffs."""
    winrates = []
    valid_diffs = []

    for d in diffs:
        if d is None:
            continue
        valid_diffs.append(d)
        if d > 0:
            winrates.append(1)
        elif d < 0:
            winrates.append(0)
        else:
            winrates.append(0.5)

    if not valid_diffs:
        return {
            "winrate": None,
            "winrate_stderr": None,
            "diff_mean": None,
            "diff_stderr": None,
            "n_samples": 0,
        }

    cleaned_diffs = remove_outliers(valid_diffs) if valid_diffs else []
    if not cleaned_diffs:
        cleaned_diffs = valid_diffs

    return {
        "winrate": float(np.mean(winrates)) if winrates else None,
        "winrate_stderr": float(np.std(winrates) / np.sqrt(len(winrates))) if len(winrates) > 1 else None,
        "diff_mean": float(np.mean(cleaned_diffs)) if cleaned_diffs else None,
        "diff_stderr": float(np.std(cleaned_diffs) / np.sqrt(len(cleaned_diffs))) if len(cleaned_diffs) > 1 else None,
        "n_samples": len(valid_diffs),
    }


def process_rewriter_rollouts(rollouts_path: Path) -> dict[str, dict]:
    """
    Process a single rewriter's rollouts.json file.

    Returns dict mapping attribute -> {
        "diffs": list of student score diffs,
        "teacher_diffs": list of teacher score diffs (may be None),
        "stats": computed statistics
    }
    """
    with open(rollouts_path, "r") as f:
        rollouts_data = json.load(f)

    results = {}
    for attribute, user_prompts in rollouts_data.items():
        all_student_diffs = []
        all_teacher_diffs = []

        for user_prompt, rollouts in user_prompts.items():
            for rollout in rollouts:
                if rollout is None:
                    continue

                # Extract student score diff
                student_score = rollout.get("student_score", {})
                if isinstance(student_score, dict):
                    diff = student_score.get("score")
                else:
                    diff = None
                all_student_diffs.append(diff)

                # Extract teacher score diff if available
                teacher_score = rollout.get("teacher_score")
                if teacher_score is not None and isinstance(teacher_score, dict):
                    teacher_diff = teacher_score.get("score")
                else:
                    teacher_diff = None
                all_teacher_diffs.append(teacher_diff)

        results[attribute] = {
            "diffs": all_student_diffs,
            "teacher_diffs": all_teacher_diffs,
            "stats": compute_stats_from_diffs(all_student_diffs),
            "teacher_stats": compute_stats_from_diffs(all_teacher_diffs),
        }

    return results


def process_run_data_multi_rewriter(run_path: Path | str, seed_index: int) -> dict[str, dict[str, dict]]:
    """
    Process validation data for all rewriters for a given seed.

    Returns dict mapping rewriter_name -> attribute -> {diffs, stats, ...}
    """
    if isinstance(run_path, str):
        run_path = Path(run_path)

    validate_dir = run_path / "validate" / f"seed_{seed_index}_validate"
    if not validate_dir.exists():
        raise FileNotFoundError(f"Validate directory not found: {validate_dir}")

    all_rewriter_data = {}

    # Find all rewriter subdirectories
    for item in validate_dir.iterdir():
        if not item.is_dir():
            continue

        rollouts_path = item / "rollouts.json"
        if not rollouts_path.exists():
            continue

        rewriter_name = item.name
        all_rewriter_data[rewriter_name] = process_rewriter_rollouts(rollouts_path)

    return all_rewriter_data


def format_rewriter_stats_table(
    attribute: str,
    rewriter_data: dict[str, dict[str, dict]],
    rewriter_names: list[str],
    colors: list[str],
) -> str:
    """
    Format a mini stats table for an attribute showing each rewriter's stats.
    """
    lines = [wrap_text(attribute, width=55)]

    # Reverse order so label order matches violin order (top to bottom)
    n = len(rewriter_names)
    for i in range(n - 1, -1, -1):
        rewriter_name = rewriter_names[i]
        if rewriter_name not in rewriter_data:
            continue
        attr_data = rewriter_data[rewriter_name].get(attribute, {})
        stats = attr_data.get("stats", {})
        teacher_stats = attr_data.get("teacher_stats", {})

        # Short rewriter name (last part after /)
        short_name = rewriter_name.split("_")[-1] if "_" in rewriter_name else rewriter_name
        short_name = short_name[:12]  # Truncate if too long

        # Format student stats
        parts = []
        if stats.get("winrate") is not None:
            wr = stats["winrate"]
            wr_err = stats.get("winrate_stderr")
            if wr_err:
                parts.append(f"S:{wr:.2f}±{wr_err:.2f}")
            else:
                parts.append(f"S:{wr:.2f}")

        if stats.get("diff_mean") is not None:
            dm = stats["diff_mean"]
            dm_err = stats.get("diff_stderr")
            if dm_err:
                parts.append(f"SΔ:{dm:.1f}±{dm_err:.1f}")
            else:
                parts.append(f"SΔ:{dm:.1f}")

        # Add teacher winrate if available
        if teacher_stats.get("winrate") is not None:
            twr = teacher_stats["winrate"]
            parts.append(f"T:{twr:.2f}")

        color_hex = colors[i % len(colors)].replace("rgb(", "").replace(")", "")
        if color_hex.startswith("rgb"):
            color_style = colors[i % len(colors)]
        else:
            # Convert rgb to hex for HTML
            try:
                r, g, b = [int(x.strip()) for x in color_hex.split(",")]
                color_style = f"#{r:02x}{g:02x}{b:02x}"
            except:
                color_style = "black"

        stats_str = ", ".join(parts) if parts else "no data"
        lines.append(f"<span style='color:{color_style}'><b>{short_name}</b>: {stats_str}</span>")

    return "<br>".join(lines)


def plot_multi_rewriter_violin(
    rewriter_data: dict[str, dict[str, dict]],
    seed_index: int,
    cluster_summary: str | None = None,
    x_range: tuple[float, float] | None = None,
) -> go.Figure:
    """
    Create horizontal violin plots showing reward distributions for each attribute
    across different rewriters.

    Args:
        rewriter_data: dict mapping rewriter_name -> attribute -> {diffs, stats, ...}
        seed_index: seed index for title
        cluster_summary: optional cluster summary for title
        x_range: optional (min, max) range for x-axis display
    """
    # Define colors for each rewriter
    colors = [
        'rgb(31, 119, 180)',   # Blue
        'rgb(255, 127, 14)',   # Orange
        'rgb(44, 160, 44)',    # Green
        'rgb(214, 39, 40)',    # Red
        'rgb(148, 103, 189)',  # Purple
        'rgb(140, 86, 75)',    # Brown
        'rgb(227, 119, 194)',  # Pink
        'rgb(127, 127, 127)',  # Gray
    ]

    # Get all attributes across all rewriters
    all_attributes = set()
    for attr_data in rewriter_data.values():
        all_attributes.update(attr_data.keys())

    # Sort attributes alphabetically (reverse so A appears at top)
    sorted_attributes = sorted(all_attributes, reverse=True)
    rewriter_names = sorted(rewriter_data.keys())
    n_rewriters = len(rewriter_names)

    # Calculate x-axis range FIRST from all data (5th-95th percentile)
    all_diffs_flat = []
    for attr_data in rewriter_data.values():
        for attr_results in attr_data.values():
            all_diffs_flat.extend([d for d in attr_results.get("diffs", []) if d is not None])

    if x_range:
        x_min, x_max = x_range
    elif all_diffs_flat:
        x_min = np.percentile(all_diffs_flat, 2.5)
        x_max = np.percentile(all_diffs_flat, 97.5)
        # Ensure 0 is visible and add some padding
        x_min = min(x_min, -1)
        x_max = max(x_max, 1)
    else:
        x_min, x_max = -10, 10

    fig = go.Figure()

    # Track overflow annotations: (x, y, text, xanchor)
    overflow_annotations: list[tuple[float, float, str, str]] = []

    # For each attribute, create violin plots for each rewriter
    for attr_idx, attribute in enumerate(sorted_attributes):
        for rewriter_idx, rewriter_name in enumerate(rewriter_names):
            if rewriter_name not in rewriter_data:
                continue
            attr_data = rewriter_data[rewriter_name].get(attribute, {})
            diffs = attr_data.get("diffs", [])

            # Filter None values (keep all data, don't clip)
            valid_diffs = [d for d in diffs if d is not None]
            if not valid_diffs:
                continue

            # Calculate y-position with offset for multiple violins per attribute
            offset_range = 0.8
            offset = (rewriter_idx - (n_rewriters - 1) / 2) * (offset_range / n_rewriters)
            y_position = attr_idx * 1.2 + offset

            # Track overflow for this rewriter's data
            data_min = min(valid_diffs)
            data_max = max(valid_diffs)
            if data_max > x_max:
                overflow_annotations.append((x_max, y_position, f"→{data_max:.1f}", "left"))
            if data_min < x_min:
                overflow_annotations.append((x_min, y_position, f"{data_min:.1f}←", "right"))

            # Short rewriter name for legend
            short_name = rewriter_name.split("_")[-1] if "_" in rewriter_name else rewriter_name

            fig.add_trace(
                go.Violin(
                    x=valid_diffs,
                    y0=y_position,
                    name=short_name,
                    orientation='h',
                    side='positive',
                    line_color=colors[rewriter_idx % len(colors)],
                    fillcolor=colors[rewriter_idx % len(colors)],
                    opacity=0.4,  # More transparent to make box visible
                    box_visible=True,
                    box=dict(
                        fillcolor='white',
                        line=dict(color='black', width=2),
                        width=1.0,
                    ),
                    meanline_visible=True,
                    meanline=dict(color='darkred', width=3),
                    points="outliers",  # Show outlier points
                    marker=dict(
                        color=colors[rewriter_idx % len(colors)],
                        size=4,
                        opacity=0.7,
                    ),
                    width=1.0 / n_rewriters,  # Wider violins
                    legendgroup=rewriter_name,
                    showlegend=(attr_idx == 0),
                    hovertemplate=f"{short_name}<br>{attribute[:40]}...<br>Value: %{{x:.2f}}<extra></extra>",
                )
            )

    # Add overflow annotations
    for x, y, text, xanchor in overflow_annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=8, color="gray"),
            xanchor=xanchor,
        )

    # Create y-axis tick labels with stats table
    y_tick_vals = [i * 1.2 for i in range(len(sorted_attributes))]
    y_tick_labels = []
    for attr in sorted_attributes:
        label = format_rewriter_stats_table(attr, rewriter_data, rewriter_names, colors)
        y_tick_labels.append(label)

    # Build title
    title = f"Validation Reward Diffs - Seed {seed_index}"
    if cluster_summary:
        # Extract short labels from summary
        parts = []
        for line in cluster_summary.split("\n"):
            if line.startswith("Category:") or line.startswith("Intent:"):
                label = line.split(":", 1)[1].split("(")[0].strip()
                parts.append(label)
        short_summary = ", ".join(parts) if parts else cluster_summary[:80]
        title += f"<br>{short_summary}"

    # Calculate height based on number of attributes
    n_attributes = len(sorted_attributes)
    # More height per attribute to accommodate stats table
    plot_height = max(600, min(2000, 150 + n_attributes * 120))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),  # Larger title
        xaxis_title="Student reward diff",
        yaxis_title=None,
        height=plot_height,
        width=950,  # Narrower plot
        xaxis=dict(
            range=[x_min - 0.5, x_max + 0.5],  # Tighter padding
            tickfont=dict(size=11),
            title=dict(font=dict(size=12)),
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            tickfont=dict(size=11),  # Larger annotation font
            tickmode='array',
            tickvals=y_tick_vals,
            ticktext=y_tick_labels,
            automargin=True,
            ticklabelstandoff=20,  # Horizontal separation from plot
        ),
        font=dict(size=11),
        violingap=0.05,
        violingroupgap=0.05,
        margin=dict(l=10, r=150, t=80, b=60),  # Right margin for legend
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,  # Right of the plot
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=10),
        ),
    )

    # Add vertical reference line at 0
    fig.add_vline(
        x=0, line_dash="dash", line_color="black", line_width=1.5, opacity=0.6
    )

    return fig


def plot_validation_data(run_path: Path | str, write_path: Path | str):
    """
    Plot validation data for all seeds, supporting multiple rewriters.

    Discovers rewriter subdirectories under each seed's validate folder and
    creates multi-rewriter violin plots.
    """
    if isinstance(run_path, str):
        run_path = Path(run_path)
    if not run_path.exists():
        raise FileNotFoundError(f"run_path does not exist: {run_path}")
    if isinstance(write_path, str):
        write_path = Path(write_path)
    write_path.mkdir(parents=True, exist_ok=True)

    validate_dir = run_path / "validate"
    if not validate_dir.exists():
        print(f"No validate directory found at {validate_dir}")
        return

    # Gather all seed indices
    seed_indices = []
    pattern = re.compile(r'seed_(\d+)_validate')
    for item in validate_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                seed_indices.append(int(match.group(1)))
    seed_indices.sort()

    # Try to load cluster info for titles
    config_path = run_path / "config.json"
    cluster_summaries = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            prompts = config.get("prompts", {})
            for seed_idx, seed_data in prompts.items():
                if isinstance(seed_data, dict) and "summary" in seed_data:
                    cluster_summaries[int(seed_idx)] = seed_data["summary"]
        except Exception as e:
            print(f"Could not load config for cluster summaries: {e}")

    for seed_index in seed_indices:
        try:
            rewriter_data = process_run_data_multi_rewriter(run_path, seed_index)

            if not rewriter_data:
                print(f"No rewriter data found for seed {seed_index}")
                continue

            cluster_summary = cluster_summaries.get(seed_index)

            fig = plot_multi_rewriter_violin(
                rewriter_data=rewriter_data,
                seed_index=seed_index,
                cluster_summary=cluster_summary,
            )

            output_path = write_path / f"seed_{seed_index}.pdf"
            fig.write_image(output_path)
            print(f"Saved plot for seed {seed_index} to {output_path}")

        except Exception as e:
            print(f"Error processing seed {seed_index}: {e}")
            raise


# Legacy single-rewriter functions for backward compatibility

def process_run_data(run_path: Path | str, seed_index: int) -> list[dict]:
    """
    Legacy function for processing single-rewriter validation data.
    Reads from old student_diffs.json/teacher_diffs.json format.
    """
    if isinstance(run_path, str):
        run_path = Path(run_path)

    # Try new multi-rewriter format first
    validate_dir = run_path / "validate" / f"seed_{seed_index}_validate"
    rewriter_dirs = [d for d in validate_dir.iterdir() if d.is_dir()] if validate_dir.exists() else []

    if rewriter_dirs:
        # Use first rewriter for backward compatibility
        rewriter_data = process_run_data_multi_rewriter(run_path, seed_index)
        first_rewriter = sorted(rewriter_data.keys())[0]
        attr_data = rewriter_data[first_rewriter]

        plot_data = []
        for attribute, data in attr_data.items():
            stats = data.get("stats", {})
            teacher_stats = data.get("teacher_stats", {})
            plot_data.append({
                "attribute": attribute,
                "diffs": [d for d in data.get("diffs", []) if d is not None],
                "judge_winrate": teacher_stats.get("winrate"),
                "judge_stderr": teacher_stats.get("winrate_stderr"),
                "reward_winrate": stats.get("winrate"),
                "reward_stderr": stats.get("winrate_stderr"),
                "reward_diff_mean": stats.get("diff_mean"),
                "reward_diff_stderr": stats.get("diff_stderr"),
                "seed_index": seed_index,
                "cluster_info": None,
            })
        return plot_data

    # Fall back to old format
    student_diffs_path = run_path / f"validate/seed_{seed_index}_validate/student_diffs.json"
    if not student_diffs_path.exists():
        raise FileNotFoundError(f"No validation data found at {student_diffs_path}")

    with open(student_diffs_path, "r") as f:
        student_diffs = json.load(f)

    teacher_diffs_path = run_path / f"validate/seed_{seed_index}_validate/teacher_diffs.json"
    try:
        with open(teacher_diffs_path, "r") as f:
            teacher_diffs = json.load(f)
    except FileNotFoundError:
        teacher_diffs = None
        print(f"No teacher diffs found in {run_path.name} for seed {seed_index}")

    plot_data = []

    for attribute, attribute_results in student_diffs.items():
        attribute_diffs = []
        teacher_winrates = []

        for _, user_prompt_diffs in attribute_results.items():
            attribute_diffs.extend(user_prompt_diffs)

        if teacher_diffs is not None:
            for _, user_prompt_diffs in teacher_diffs[attribute].items():
                for wr in user_prompt_diffs:
                    if wr is None:
                        continue
                    elif wr > 0:
                        teacher_winrates.append(1)
                    elif wr < 0:
                        teacher_winrates.append(0)
                    else:
                        teacher_winrates.append(0.5)

        student_winrates = []
        for wr in attribute_diffs:
            if wr is None:
                continue
            elif wr > 0:
                student_winrates.append(1)
            elif wr < 0:
                student_winrates.append(0)
            else:
                student_winrates.append(0.5)

        attribute_diffs = [d for d in attribute_diffs if d is not None]
        attribute_diffs_clean = remove_outliers(attribute_diffs)

        student_mean = np.mean(student_winrates).item() if student_winrates else None
        student_stderr = (np.std(student_winrates) / np.sqrt(len(student_winrates))).item() if len(student_winrates) > 1 else None
        teacher_mean = np.mean(teacher_winrates).item() if teacher_winrates else None
        teacher_stderr = (np.std(teacher_winrates) / np.sqrt(len(teacher_winrates))).item() if len(teacher_winrates) > 1 else None

        reward_diff_mean = np.mean(attribute_diffs_clean).item() if attribute_diffs_clean else None
        reward_diff_stderr = (np.std(attribute_diffs_clean) / np.sqrt(len(attribute_diffs_clean))).item() if len(attribute_diffs_clean) > 1 else None

        plot_data.append({
            "attribute": attribute,
            "diffs": attribute_diffs,
            "judge_winrate": teacher_mean,
            "judge_stderr": teacher_stderr,
            "reward_winrate": student_mean,
            "reward_stderr": student_stderr,
            "reward_diff_mean": reward_diff_mean,
            "reward_diff_stderr": reward_diff_stderr,
            "seed_index": seed_index,
            "cluster_info": None,
        })

    return plot_data


def plot_reward_diff_violin(plot_data: list[dict]):
    """
    Legacy single-rewriter violin plot function.
    """
    fig = go.Figure()
    display_names = []

    for i, item in enumerate(reversed(plot_data)):
        base_name = wrap_text(item["attribute"], width=60)

        student_parts = []
        teacher_parts = []
        if item.get("reward_winrate") is not None:
            if item.get("reward_stderr") is not None:
                student_parts.append(f"Student WR: {item['reward_winrate']:.2f}±{item['reward_stderr']:.2f}")
            else:
                student_parts.append(f"Student WR: {item['reward_winrate']:.2f}")
        if item.get("reward_diff_mean") is not None:
            if item.get("reward_diff_stderr") is not None:
                student_parts.append(f"Student Diff: {item['reward_diff_mean']:.2f}±{item['reward_diff_stderr']:.2f}")
            else:
                student_parts.append(f"Student Diff: {item['reward_diff_mean']:.2f}")
        if item.get("judge_winrate") is not None:
            if item.get("judge_stderr") is not None:
                teacher_parts.append(f"Teacher WR: {item['judge_winrate']:.2f}±{item['judge_stderr']:.2f}")
            else:
                teacher_parts.append(f"Teacher WR: {item['judge_winrate']:.2f}")

        stats_lines = []
        if student_parts:
            stats_lines.append(', '.join(student_parts))
        if teacher_parts:
            stats_lines.append(', '.join(teacher_parts))

        if stats_lines:
            display_name = f"{base_name}<br>({stats_lines[0]})"
            if len(stats_lines) > 1:
                display_name += f"<br>({stats_lines[1]})"
        else:
            display_name = base_name

        display_names.append(display_name)

        fig.add_trace(
            go.Violin(
                x=item["diffs"],
                y0=display_name,
                name=display_name,
                orientation='h',
                side='positive',
                box_visible=True,
                box=dict(
                    fillcolor='white',
                    line=dict(color='black', width=2),
                ),
                meanline_visible=True,
                meanline=dict(color='red', width=3),
                points="suspectedoutliers",
                pointpos=-0.2,
                jitter=0.2,
                width=1.0,
            )
        )

    title = "Reward Diffs Violin Plot"
    if plot_data[0].get("cluster_info") is not None:
        cluster_info = plot_data[0]["cluster_info"]
        summary = cluster_info['summary']
        parts = []
        for line in summary.split("\n"):
            if line.startswith("Category:") or line.startswith("Intent:"):
                label = line.split(":", 1)[1].split("(")[0].strip()
                parts.append(label)
        short_summary = ", ".join(parts) if parts else summary[:100]
        title += f"<br>Seed {plot_data[0]['seed_index']}: {short_summary}"

    n_attributes = len(plot_data)
    plot_height = max(500, min(1600, 100 + n_attributes * 90))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Student reward diff",
        yaxis_title=None,
        height=plot_height,
        width=1200,
        showlegend=False,
        xaxis=dict(
            tickfont=dict(size=11),
            title=dict(font=dict(size=12)),
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            tickfont=dict(size=10),
            automargin=True,
            categoryorder='array',
            categoryarray=display_names,
        ),
        font=dict(size=11),
        violingap=0.05,
        violingroupgap=0.05,
        margin=dict(l=10, r=40, t=80, b=60),
    )

    fig.add_vline(
        x=0, line_dash="dash", line_color="black", line_width=1.5, opacity=0.6
    )
    return fig


if __name__ == "__main__":
    for run_name in [
        "20260103-121320-list_reverse-handpick-plus"
    ]:
        run_path = Path(f"data/evo/{run_name}")
        plot_validation_data(run_path=run_path, write_path=run_path)
