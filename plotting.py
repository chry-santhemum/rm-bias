"""All the plotting functions"""

import re
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from utils import remove_outliers


def process_run_data(run_path: Path|str, seed_index: int) -> list[dict]:
    if isinstance(run_path, str):
        run_path = Path(run_path)

    with open(run_path / f"validate/seed_{seed_index}_validate/student_diffs.json", "r", encoding="utf-8") as f:
        student_diffs = json.load(f)

    try:
        with open(run_path / f"validate/seed_{seed_index}_validate/teacher_diffs.json", "r", encoding="utf-8") as f:
            teacher_diffs = json.load(f)
    except FileNotFoundError:
        teacher_diffs = None
        print(f"No teacher diffs found in {run_path.name} for seed {seed_index}")

    # Collect difference data for each attribute
    plot_data = []

    # For each attribute, compute differences from baseline
    for attribute, attribute_results in student_diffs.items():
        attribute_diffs = []
        teacher_winrates = []

        for _, user_prompt_diffs in attribute_results.items():
            attribute_diffs.extend(user_prompt_diffs)

        if teacher_diffs is not None:
            for _, user_prompt_diffs in teacher_diffs[attribute].items():
                for wr in user_prompt_diffs:
                    if wr > 0:
                        teacher_winrates.append(1)
                    elif wr < 0:
                        teacher_winrates.append(0)
                    elif wr is None:
                        continue
                    else:
                        teacher_winrates.append(0.5)
            # teacher_winrates = remove_outliers(teacher_winrates, clip_percent = 0.05)

        # remove far outliers
        student_winrates = []
        for wr in attribute_diffs:
            if wr > 0:
                student_winrates.append(1)
            elif wr < 0:
                student_winrates.append(0)
            elif wr is None:
                continue
            else:
                student_winrates.append(0.5)

        attribute_diffs = remove_outliers(attribute_diffs, clip_percent = 0.05)
        ds_name = run_path.name.split("-")[-2]
        with open(
            f"user_prompts/{ds_name}/n_sub_0/cluster_{seed_index}.json", "r", encoding="utf-8"
        ) as f:
            cluster_info = json.load(f)

        # Calculate standard error for winrates
        student_mean = np.mean(student_winrates).item() if student_winrates else None
        student_stderr = (np.std(student_winrates) / np.sqrt(len(student_winrates))).item() if len(student_winrates) > 1 else None
        teacher_mean = np.mean(teacher_winrates).item() if teacher_winrates else None
        teacher_stderr = (np.std(teacher_winrates) / np.sqrt(len(teacher_winrates))).item() if len(teacher_winrates) > 1 else None

        plot_data.append(
            {
                "attribute": attribute,
                "diffs": attribute_diffs,
                "judge_winrate": teacher_mean,
                "judge_stderr": teacher_stderr,
                "reward_winrate": student_mean,
                "reward_stderr": student_stderr,
                "seed_index": seed_index,
                "cluster_info": cluster_info,
            }
        )
    
    return plot_data


def plot_reward_diff_violin(plot_data: list[dict]):
    """
    Each item in the input needs to have (at least): 
    - attribute
    - diffs: list[float]
    
    Creates horizontal violin plots to fit attribute labels compactly.
    Designed to handle up to 16 attributes.
    """
    # Helper function to wrap text at specified width
    def wrap_text(text, width=60):
        """Wrap text to specified width using <br> for line breaks"""
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

    # Create figure
    fig = go.Figure()

    # Store display names for y-axis
    display_names = []

    # Add violin plot for each attribute (in reverse order so first appears at top)
    for i, item in enumerate(reversed(plot_data)):
        # Create display name with wrapped text
        base_name = wrap_text(item["attribute"], width=60)
        
        # Build winrate suffix with error intervals
        winrate_parts = []
        if item.get("reward_winrate") is not None:
            if item.get("reward_stderr") is not None:
                winrate_parts.append(f"Student: {item['reward_winrate']:.2f}±{item['reward_stderr']:.2f}")
            else:
                winrate_parts.append(f"Student: {item['reward_winrate']:.2f}")
        if item.get("judge_winrate") is not None:
            if item.get("judge_stderr") is not None:
                winrate_parts.append(f"Teacher: {item['judge_winrate']:.2f}±{item['judge_stderr']:.2f}")
            else:
                winrate_parts.append(f"Teacher: {item['judge_winrate']:.2f}")
        
        if winrate_parts:
            display_name = f"{base_name}<br>({', '.join(winrate_parts)})"
        else:
            display_name = base_name

        # Check if this is a "red flag" attribute:
        # student winrate - err > 0.5 AND teacher winrate + err < 0.5
        is_red_flag = False
        student_wr = item.get("reward_winrate")
        student_err = item.get("reward_stderr") or 0
        teacher_wr = item.get("judge_winrate")
        teacher_err = item.get("judge_stderr") or 0
        
        if student_wr is not None and teacher_wr is not None:
            if (student_wr - student_err > 0.5) and (teacher_wr + teacher_err < 0.5):
                is_red_flag = True

        # Color the text red for red flag attributes
        if is_red_flag:
            display_name = f"<span style='color:red'>{display_name}</span>"

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
                meanline=dict(color='darkblue', width=2),
                points="all",
                pointpos=-0.1,
                jitter=0.3,
                width=1.0,
            )
        )

    title = "Reward Diffs Violin Plot"
    if plot_data[0].get("cluster_info", None) is not None:
        cluster_info = plot_data[0]["cluster_info"]
        title += f"<br>Seed {plot_data[0]['seed_index']}: {wrap_text(cluster_info['summary'], width=100)}"

    # Calculate height based on number of attributes (min 400, ~90px per attribute for taller rows)
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

    # Add vertical reference line at 0
    fig.add_vline(
        x=0, line_dash="dash", line_color="black", line_width=1.5, opacity=0.6
    )
    return fig


def plot_validation_data(run_path: Path|str, write_path: Path|str):
    if isinstance(run_path, str):
        run_path = Path(run_path)
    if not run_path.exists():
        raise FileNotFoundError(f"run_path does not exist: {run_path}")
    if isinstance(write_path, str):
        write_path = Path(write_path)
    write_path.mkdir(parents=True, exist_ok=True)

    validate_dir = run_path / "validate"
    # Gather all seed indices (folders matching "seed_{i}_validate")
    seed_indices = []
    if validate_dir.exists():
        pattern = re.compile(r'seed_(\d+)_validate')
        for item in validate_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    seed_indices.append(int(match.group(1)))
    seed_indices.sort()

    for seed_index in seed_indices:
        plot_data = process_run_data(run_path=run_path, seed_index=seed_index)
        fig = plot_reward_diff_violin(plot_data=plot_data)
        # fig.show()
        fig.write_image(write_path / f"seed_{seed_index}.pdf")
        print(f"Saved plot for seed {seed_index}")

# %%
if __name__ == "__main__":
    for run_name in [
        "20251211-081017-pair-synthetic-plus",
        "20251211-112052-list_reverse-synthetic-plus",
        "20251211-142409-pair-synthetic-plus",
    ]:
        run_path = Path(f"data/evo/{run_name}")
        write_path = Path(f"plots/{run_name}")
        plot_validation_data(run_path=run_path, write_path=write_path)
