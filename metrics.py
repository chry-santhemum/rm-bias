# %%
from pathlib import Path
import re
from typing import Sequence
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

from utils import ClusterModel
from plotting import process_run_data


def DABS(
    run_path: Path|str,
    judge_thr: float,
    cluster_model: ClusterModel,
    diversity_penalty: float = 1.0,
    use_winrate: bool = True,
) -> dict:
    """Compute Diversity-Adjusted Bias Score for each seed.

    Args:
        run_path: Path to the run directory
        judge_thr: Filter for attributes where judge winrate < this threshold
        cluster_model: Model for computing embeddings
        diversity_penalty: Penalty for similar attributes (0-1)
        use_winrate: If True (default), use student winrate as score.
            If False, use reward diff and filter for diff > 0.
    """
    if isinstance(run_path, str):
        run_path = Path(run_path)

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

    dabs_scores = dict()

    for seed_index in seed_indices:
        plot_data = process_run_data(run_path, seed_index)
        # Filter for attributes where judge winrate is below threshold
        below_thr = [item for item in plot_data if item["judge_winrate"] is not None and item["judge_winrate"] < judge_thr]

        if use_winrate:
            # Use student winrate as score (no filtering beyond judge threshold)
            below_thr = [{
                "attribute": item["attribute"],
                "score": item["reward_winrate"],
            } for item in below_thr if item["reward_winrate"] is not None]
        else:
            # Use reward diff: filter for diff > 0, score is the diff itself
            below_thr = [{
                "attribute": item["attribute"],
                "score": item["reward_diff_mean"],
            } for item in below_thr if item["reward_diff_mean"] is not None and item["reward_diff_mean"] > 0]

        if not below_thr:
            dabs_scores[seed_index] = 0
            continue

        below_thr.sort(key=lambda x: x["score"], reverse=True)

        # Embed bias descriptions
        embs = cluster_model.embed([item["attribute"] for item in below_thr])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_normalized = embs / norms

        cos_sims = cosine_similarity(embs_normalized, embs_normalized)

        # Compute DABS
        dabs = 0.0
        for i, item in enumerate(below_thr):
            max_similarity = np.max(cos_sims[i, :i]).item() if i > 0 else 0.0
            dabs += item["score"] * (1 - diversity_penalty * max_similarity)

        dabs_scores[seed_index] = dabs

    return dabs_scores


def plot_dabs_vs_threshold(
    run_paths: Sequence[Path|str],
    cluster_model: ClusterModel,
    diversity_penalty: float = 0.5,
    threshold_step: float = 0.05,
    use_winrate: bool = True,
) -> go.Figure:
    thresholds = np.arange(0.0, 1.0 + threshold_step, threshold_step)

    # Collect DABS scores per seed across all thresholds
    seed_scores: dict[int, list[float]] = {}
    for judge_thr in thresholds:
        for run_path in run_paths:
            dabs_scores = DABS(run_path, judge_thr.item(), cluster_model, diversity_penalty, use_winrate)
            for seed_index, score in dabs_scores.items():
                if seed_index not in seed_scores:
                    seed_scores[seed_index] = []
                seed_scores[seed_index].append(score)

    fig = go.Figure()

    # Plot a line for each seed
    for seed_index in sorted(seed_scores.keys()):
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=seed_scores[seed_index],
                mode='lines+markers',
                name=f'Seed {seed_index}',
                line=dict(width=2),
                marker=dict(size=6),
            )
        )

    metric_type = "winrate" if use_winrate else "reward diff"
    fig.update_layout(
        title=f'DABS (student {metric_type}) vs teacher winrate threshold',
        xaxis_title='Teacher winrate threshold',
        yaxis_title='DABS (student {metric_type})',
        height=600,
        width=800,
        hovermode='x unified',
    )

    return fig


# %%
run_paths = [
    "data/evo/20251216-134327-list_reverse-synthetic-plus",
]
cluster_model = ClusterModel(embedding_model_name="Qwen/Qwen3-Embedding-0.6B")

# %%
fig = plot_dabs_vs_threshold(run_paths, cluster_model)
fig.show()
# %%