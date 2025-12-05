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
        judge_winrates = []

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
                judge_winrates.extend(winrates_clean)

        # remove far outliers
        attribute_diffs = remove_outliers(attribute_diffs, clip_percent = 0.05)
        ds_name = run_path.name.split("-")[-2]
        with open(
            f"user_prompts/{ds_name}/n_sub_0/cluster_{seed_index}.json", "r", encoding="utf-8"
        ) as f:
            cluster_info = json.load(f)

        # Compute reward winrate: percentage of diffs > 0
        reward_winrate = None
        if attribute_diffs:
            reward_winrate = sum(1 for d in attribute_diffs if d > 0) / len(attribute_diffs)

        plot_data.append(
            {
                "attribute": attribute,
                "diffs": attribute_diffs,
                "judge_winrate": np.mean(judge_winrates).item() if judge_winrates else None,
                "reward_winrate": reward_winrate,
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

        # Add reward winrate (from reward diffs > 0)
        if item.get("reward_winrate") is not None:
            display_name += f"<br>(Reward WR: {item['reward_winrate']:.2f})"
        else:
            display_name += "<br>(Reward WR: N/A)"

        # Add judge winrate
        if item.get("judge_winrate") is not None:
            display_name += f"<br>(Judge WR: {item['judge_winrate']:.2f})"
        else:
            display_name += "<br>(Judge WR: N/A)"

        display_names.append(display_name)

        fig.add_trace(
            go.Violin(
                y=item["diffs"],
                name=display_name,
                box_visible=True,
                meanline_visible=True,
                points="all",
                pointpos=0,
                jitter=0.3,
            )
        )

    title = f"Reward diffs violin plot"
    if plot_data[0].get("cluster_info", None) is not None:
        cluster_info = plot_data[0]["cluster_info"]
        title += f"<br>Seed {plot_data[0]['seed_index']}: {cluster_info['summary']}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Attributes",
        yaxis_title="Reward Difference (Rewrite - Baseline)",
        height=1000,
        width=1400,
        showlegend=False,
        xaxis=dict(tickangle=45, tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=13), title=dict(font=dict(size=14))),
        font=dict(size=13),
        bargap=0.05,
        violingap=0.1,
        violingroupgap=0.05,
        margin=dict(l=60, r=40, t=80, b=120),
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
        fig.write_image(write_path / f"seed_{seed_index}.pdf")
        print(f"Saved plot for seed {seed_index}")


# %%
if __name__ == "__main__":
    run_path = Path("data/evo/20251205-053029-list-synthetic-plus")
    write_path = Path(f"plots/{run_path.name}")

    plot_validation_data(run_path=run_path, write_path=write_path)
