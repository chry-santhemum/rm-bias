"""All the plotting functions"""


import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from utils import remove_outliers


def process_run_data(run_path: Path|str, seed_index: int) -> list[dict]:
    if isinstance(run_path, str):
        run_path = Path(run_path)

    with open(
        run_path / "val_baselines" / "val_baselines.json",
        "r",
        encoding="utf-8",
    ) as f:
        val_baselines = json.load(f)

    with open(
        run_path / f"validate/seed_{seed_index}_validate/rewrite_plus_scores.json",
        "r",
        encoding="utf-8",
    ) as f:
        val_results = json.load(f)

    with open(
        run_path / f"validate/seed_{seed_index}_judge.json", "r", encoding="utf-8"
    ) as f:
        judge_results = json.load(f)

    # Collect difference data for each attribute
    plot_data = []

    # For each attribute, compute differences from baseline
    for attribute in judge_results:
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

        for prompt, prompt_judge_winrates in judge_results[attribute].items():
            winrates_clean = [wr for wr in prompt_judge_winrates if wr is not None]
            winrates.extend(winrates_clean)

        # remove far outliers
        attribute_diffs = remove_outliers(attribute_diffs)

        plot_data.append(
            {
                "attribute": attribute,
                "diffs": attribute_diffs,
                "winrate": np.mean(winrates).item(),
            }
        )
    
    return plot_data


def plot_seed_validation_data(
    run_path: Path|str,
    seed_index: int,
):
    if isinstance(run_path, str):
        run_path = Path(run_path)
        
    plot_data = process_run_data(run_path, seed_index)

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
        display_name = (
            wrap_text(item["attribute"], width=60)
            + f"<br>(Winrate: {item['winrate']:.2f})"
        )
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

    with open(
        run_path / f"seed_{seed_index}_cluster.json", "r", encoding="utf-8"
    ) as f:
        cluster_info = json.load(f)

    fig.update_layout(
        title=f"Seed {seed_index}: {cluster_info['summary']}",
        xaxis_title="Attribute",
        yaxis_title="Reward Difference (Attribute - Baseline)",
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



# %%

if __name__ == "__main__":
    run_path = Path("data/evo/20251107-084230-naive-synthetic_2")
    write_path = Path("data/scrap/20251107-084230")
    write_path.mkdir(parents=True, exist_ok=True)

    for seed_index in [1, 3, 4, 6, 8, 9, 12, 14, 16]:
        fig = plot_seed_validation_data(run_path=run_path, seed_index=seed_index)
        fig.show()
        fig.write_html(write_path / f"seed_{seed_index}.html")