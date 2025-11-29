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

    with open(run_path / "val_baselines" / "sample_rollouts.json", "r", encoding="utf-8") as f:
        val_baselines = json.load(f)

    with open(run_path / f"validate/seed_{seed_index}_validate/rewrite_scores.json", "r", encoding="utf-8") as f:
        val_results = json.load(f)

    try:
        with open(run_path / f"validate/seed_{seed_index}_judge.json", "r", encoding="utf-8") as f:
            judge_results = json.load(f)
    except FileNotFoundError:
        judge_results = None
        print(f"No judge results found in {run_path.name} for seed {seed_index}")

    # Collect difference data for each attribute
    plot_data = []

    # For each attribute, compute differences from baseline
    for attribute in val_results:
        attribute_results = val_results[attribute]
        attribute_diffs = []
        winrates = []

        for prompt, prompt_rewards in attribute_results.items():
            baseline_rewards = [r["score"] for r in val_baselines[prompt]]

            # Compute element-wise differences
            for attr_score, base_score in zip(prompt_rewards, baseline_rewards):
                if attr_score is None or base_score is None:
                    continue
                attribute_diffs.append(attr_score - base_score)

        if judge_results is not None:
            for prompt, prompt_judge_winrates in judge_results[attribute].items():
                winrates_clean = [wr for wr in prompt_judge_winrates if wr is not None]
                winrates.extend(winrates_clean)

        # remove far outliers
        attribute_diffs = remove_outliers(attribute_diffs)
        ds_name = run_path.name.split("-")[-2]
        with open(
            f"/workspace/rm-bias/data/{ds_name}/{seed_index}.json", "r", encoding="utf-8"
        ) as f:
            cluster_info = json.load(f)

        plot_data.append(
            {
                "attribute": attribute,
                "diffs": attribute_diffs,
                "winrate": np.mean(winrates).item() if winrates else None,
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
    """
    # Helper function to wrap text
    def wrap_text(text, width):
        """Wrap text to specified width"""
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

    # Create box plot
    fig = go.Figure()

    # Store positions and winrates for annotation
    display_names = []

    # Add violin plot for each attribute
    for i, item in enumerate(plot_data):
        display_name = wrap_text(item["attribute"], width=60)
        if item.get("winrate", None) is not None:
            display_name += f"<br>(Winrate: {item['winrate']:.2f})"
        else:
            display_name += "<br>(Winrate: N/A)"
        display_names.append(display_name)

        fig.add_trace(
            go.Violin(
                y=item["diffs"],
                name=display_name,
                box_visible=True,
                meanline_visible=True,
                points="all",
            )
        )

    title = f"Reward diffs violin plot"
    if plot_data[0].get("cluster_info", None) is not None:
        cluster_info = plot_data[0]["cluster_info"]
        title += f"<br><sub>Seed {plot_data[0]['seed_index']}: {cluster_info['summary']}</sub>"

    fig.update_layout(
        title=title,
        xaxis_title="Attributes",
        yaxis_title="Reward Difference (Rewrite - Baseline)",
        height=1000,
        width=1400,
        showlegend=False,
        xaxis=dict(tickangle=45),
    )

    # Add reference line at 0
    fig.add_hline(
        y=0, line_dash="dash", line_color="black", line_width=1.5, opacity=0.6
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
        fig.write_html(write_path / f"seed_{seed_index}.html")
        print(f"Saved plot for seed {seed_index}")


# %%
if __name__ == "__main__":
    run_path = Path("data/one_turn/20251128-091719-pair-synthetic_0-plus")
    timestamp = "-".join(run_path.name.split('-')[:2])
    write_path = Path(f"plots/{timestamp}")

    plot_validation_data(run_path=run_path, write_path=write_path)
