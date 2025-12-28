import json
import html
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from bias_evaluator import BiasEvaluator
from api_models import RewriteModel, SAME_ATTRS
from reward_models import LocalRewardModel
from state import Rollout

rewriter_models = [
    RewriteModel(
        model_name="openai/gpt-5-mini",
        max_par=1024,
        max_tokens=4096,
        reasoning="low",
        enable_cache=False,
        force_caller="openrouter",
    ),
    RewriteModel(
        model_name="anthropic/claude-haiku-4.5",
        max_par=1024,
        max_tokens=4096,
        reasoning="low",
        enable_cache=False,
        force_caller="openrouter",
    ),
]


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
        async with evaluator:
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

    # Create multi-rewriter violin plot
    plot_multi_rewriter_violin(
        all_rewriter_results=all_rewriter_results,
        attributes=attributes,
        save_path=save_dir / "multi_rewriter_violin.pdf",
    )


def plot_multi_rewriter_violin(
    all_rewriter_results: list[dict],
    attributes: list[str],
    save_path: Path,
):
    """
    Create horizontal violin plots showing reward distributions for each attribute across different rewriters.

    Each attribute gets a y-axis position with multiple horizontal violins (one per rewriter).
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

    # For each attribute, create violins for each rewriter
    for attr_idx, attribute in enumerate(reversed(attributes)):  # Reverse so first appears at top
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

            # Calculate mean and stderr
            mean_diff = np.mean(all_diffs)
            stderr_diff = np.std(all_diffs) / np.sqrt(len(all_diffs)) if len(all_diffs) > 1 else 0

            # Create y-position with offset for multiple violins per attribute
            # Offset range: use a fraction of the row height
            n_rewriters = len(all_rewriter_results)
            offset_range = 0.8  # Total range for offsets
            offset = (rewriter_idx - (n_rewriters - 1) / 2) * (offset_range / n_rewriters)
            y_position = attr_idx + offset

            # Create display name with stats
            annotation = f"{rewriter_name}: {mean_diff:.2f}±{stderr_diff:.2f}"

            # Create violin trace
            fig.add_trace(
                go.Violin(
                    x=all_diffs,
                    y0=y_position,
                    name=rewriter_name,
                    orientation='h',
                    side='positive',
                    box_visible=True,
                    box=dict(
                        fillcolor='white',
                        line=dict(color=colors[rewriter_idx % len(colors)], width=2),
                    ),
                    meanline_visible=True,
                    meanline=dict(color=colors[rewriter_idx % len(colors)], width=3),
                    line_color=colors[rewriter_idx % len(colors)],
                    fillcolor=colors[rewriter_idx % len(colors)],
                    opacity=0.6,
                    points="outliers",
                    pointpos=-0.2,
                    jitter=0.2,
                    width=offset_range / n_rewriters,
                    legendgroup=rewriter_name,
                    showlegend=(attr_idx == 0),  # Only show legend for first attribute
                    hovertemplate=f"{annotation}<br>Value: %{{x:.3f}}<extra></extra>",
                )
            )

    # Create y-axis tick labels (one per attribute)
    y_tick_vals = list(range(len(attributes)))
    y_tick_labels = [wrap_text(attr, width=60) for attr in reversed(attributes)]

    # Calculate plot height
    n_attributes = len(attributes)
    plot_height = max(500, min(1600, 100 + n_attributes * 120))

    fig.update_layout(
        title=dict(text="Reward Diffs across Rewriters", font=dict(size=16)),
        xaxis_title="Student reward diff",
        yaxis_title=None,
        height=plot_height,
        width=1400,
        xaxis=dict(
            tickfont=dict(size=11),
            title=dict(font=dict(size=12)),
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            tickfont=dict(size=10),
            tickmode='array',
            tickvals=y_tick_vals,
            ticktext=y_tick_labels,
            automargin=True,
        ),
        font=dict(size=11),
        violingap=0.05,
        violingroupgap=0.05,
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
    print(f"Saved multi-rewriter violin plot to {save_path}")
