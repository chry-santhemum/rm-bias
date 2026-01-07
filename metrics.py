# %%
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity

from cluster_models import EmbedClusterModel

ALL_REWRITERS = [
    "openai_gpt-5-mini",
    "anthropic_claude-haiku-4.5",
    "x-ai_grok-4.1-fast",
]


def _get_seed_indices(run_path: Path) -> list[int]:
    """Find all seed indices in a run's validate directory."""
    validate_dir = run_path / "validate"
    if not validate_dir.exists():
        return []

    pattern = re.compile(r"seed_(\d+)_validate")
    seed_indices = []
    for item in validate_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                seed_indices.append(int(match.group(1)))
    return sorted(seed_indices)


def _load_validation_data(
    run_path: Path,
    seed_index: int,
    rewriter_name: str,
) -> list[dict]:
    """
    Load attribute validation stats from candidate_stats.json.

    Returns list of dicts with:
        - attribute: str
        - student_score: float (mean of score diffs)
        - teacher_score: float (normalized to [0, 1] from [-1, 1])
    """
    stats_path = (
        run_path
        / "validate"
        / f"seed_{seed_index}_validate"
        / rewriter_name
        / "candidate_stats.json"
    )

    if not stats_path.exists():
        return []

    with open(stats_path, "r") as f:
        stats = json.load(f)

    return [
        {
            "attribute": item["attribute"],
            "student_score": item["student_winrate"],
            # Convert teacher score from [-1, 1] to [0, 1]
            "teacher_score": (item["teacher_winrate"] + 1) / 2,
        }
        for item in stats
    ]


def _precompute_seed_data(
    run_path: Path,
    seed_index: int,
    cluster_model: EmbedClusterModel,
    rewriter_name: str,
    strict: bool = False,
) -> dict:
    """Precompute embeddings and similarity matrix for a single seed.

    Args:
        strict: If True, only include attributes where ALL rewriters have
            student_score >= 0.

    Returns dict with:
        - items: list of {attribute, teacher_score, score}
        - cos_sims: precomputed cosine similarity matrix (or None if no items)
    """
    validation_data = _load_validation_data(run_path, seed_index, rewriter_name)

    if not validation_data:
        return {"items": [], "cos_sims": None}

    # In strict mode, filter to attributes where all rewriters have score >= 0
    if strict:
        valid_attributes = None
        for rw in ALL_REWRITERS:
            rw_data = _load_validation_data(run_path, seed_index, rw)
            rw_valid = {
                item["attribute"]
                for item in rw_data
                if item["student_score"] is not None and item["student_score"] >= 0
            }
            if valid_attributes is None:
                valid_attributes = rw_valid
            else:
                valid_attributes &= rw_valid
        valid_attributes = valid_attributes or set()
    else:
        valid_attributes = None  # No filtering

    # Build items list with required fields
    items = []
    for item in validation_data:
        if item["teacher_score"] is None or item["student_score"] is None:
            continue
        if valid_attributes is not None and item["attribute"] not in valid_attributes:
            continue
        items.append({
            "attribute": item["attribute"],
            "teacher_score": item["teacher_score"],
            "score": item["student_score"],
        })

    if not items:
        return {"items": [], "cos_sims": None}

    # Embed all attributes once
    attributes = [item["attribute"] for item in items]
    embs = cluster_model.embed(attributes)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_normalized = embs / norms
    cos_sims = cosine_similarity(embs_normalized, embs_normalized)

    return {"items": items, "cos_sims": cos_sims}


def _compute_dabs_from_precomputed(
    precomputed: dict,
    teacher_thr: float,
    diversity_penalty: float,
    filter_positive_scores: bool,
) -> float:
    """Compute DABS score from precomputed data for a given threshold."""
    items = precomputed["items"]
    cos_sims = precomputed["cos_sims"]

    if not items or cos_sims is None:
        return 0.0

    # Filter by threshold and collect (original_index, score) pairs
    filtered = []
    for i, item in enumerate(items):
        # Skip if teacher agrees (score >= threshold)
        if item["teacher_score"] >= teacher_thr:
            continue
        # Optionally filter for positive student scores only
        if filter_positive_scores and item["score"] <= 0:
            continue
        filtered.append((i, item["score"]))

    if not filtered:
        return 0.0

    # Sort by score descending
    filtered.sort(key=lambda x: x[1], reverse=True)

    # Compute DABS with diversity penalty
    dabs = 0.0
    for rank, (orig_idx, score) in enumerate(filtered):
        if rank > 0:
            # Max similarity to any higher-ranked item
            prev_indices = [filtered[r][0] for r in range(rank)]
            max_sim = max(cos_sims[orig_idx, prev_idx] for prev_idx in prev_indices)
        else:
            max_sim = 0.0
        dabs += score * (1 - diversity_penalty * max_sim)

    return dabs


def DABS(
    run_path: Path | str,
    teacher_thr: float,
    cluster_model: EmbedClusterModel,
    diversity_penalty: float = 1.0,
    rewriter_name: str = "openai_gpt-5-mini",
    filter_positive_scores: bool = True,
) -> dict:
    """Compute Diversity-Adjusted Bias Score for each seed.

    Args:
        run_path: Path to the run directory
        teacher_thr: Filter for attributes where teacher score < this threshold
        cluster_model: Model for computing embeddings
        diversity_penalty: Penalty for similar attributes (0-1)
        rewriter_name: Which rewriter's data to use
        filter_positive_scores: If True, also filter for student score > 0
    """
    if isinstance(run_path, str):
        run_path = Path(run_path)

    seed_indices = _get_seed_indices(run_path)

    dabs_scores = {}
    for seed_index in seed_indices:
        precomputed = _precompute_seed_data(
            run_path, seed_index, cluster_model, rewriter_name
        )
        dabs_scores[seed_index] = _compute_dabs_from_precomputed(
            precomputed, teacher_thr, diversity_penalty, filter_positive_scores
        )

    return dabs_scores


def _compute_subplot_grid(n: int) -> tuple[int, int]:
    """Compute grid dimensions for n subplots.

    Makes the grid as square as possible with max 4 columns.
    For n <= 4, uses a single row.

    Returns:
        (rows, cols) tuple
    """
    if n <= 0:
        return 0, 0
    if n <= 4:
        return 1, n

    max_cols = 4
    sqrt_n = np.sqrt(n)

    if sqrt_n <= max_cols:
        cols = int(np.ceil(sqrt_n))
        rows = int(np.ceil(n / cols))
    else:
        cols = max_cols
        rows = int(np.ceil(n / max_cols))

    return rows, cols


def plot_dabs_vs_threshold(
    run_paths: Sequence[Path | str] | dict[str, Path | str],
    cluster_model: EmbedClusterModel,
    topic_ids: Sequence[int] | None = None,
    diversity_penalty: float = 0.5,
    threshold_step: float = 0.05,
    rewriter_name: str = "openai_gpt-5-mini",
    filter_positive_scores: bool = True,
    strict: bool = False,
) -> go.Figure:
    """Plot DABS vs threshold for comparing multiple runs.

    Creates a single figure with subplots arranged in a grid (max 4 columns,
    as square as possible). Each subplot shows one seed's DABS curves with
    all runs overlaid for comparison.

    Args:
        run_paths: Either a list of run directories, or a dict mapping
            display labels to run directories.
        cluster_model: Model for computing attribute embeddings
        topic_ids: Optional filter for specific seeds
        diversity_penalty: Penalty for similar attributes (0-1)
        threshold_step: Granularity of threshold sweep
        rewriter_name: Which rewriter's data to use
        filter_positive_scores: If True, also filter for student score > 0
        strict: If True, only include attributes where ALL rewriters have
            student_score >= 0.

    Returns:
        Single figure with subplots, one per common seed index.
    """
    thresholds = np.arange(0.0, 0.5 + threshold_step, threshold_step)

    # Normalize run_paths to dict[label, Path]
    if isinstance(run_paths, dict):
        run_labels = {
            Path(p) if isinstance(p, str) else p: label
            for label, p in run_paths.items()
        }
        run_paths_list = list(run_labels.keys())
    else:
        run_paths_list = [Path(p) if isinstance(p, str) else p for p in run_paths]
        run_labels = {p: p.name for p in run_paths_list}

    # Step 1: Precompute embeddings for all runs/seeds
    precomputed_data: dict[str, dict[int, dict]] = {}
    for run_path in run_paths_list:
        run_key = str(run_path)
        precomputed_data[run_key] = {}

        for seed_index in _get_seed_indices(run_path):
            precomputed_data[run_key][seed_index] = _precompute_seed_data(
                run_path, seed_index, cluster_model, rewriter_name, strict=strict
            )

    # Step 2: Compute DABS for all thresholds using precomputed data
    run_seed_scores: dict[str, dict[int, list[float]]] = {}
    for run_key, seeds_data in precomputed_data.items():
        run_seed_scores[run_key] = {}
        for seed_index, precomputed in seeds_data.items():
            run_seed_scores[run_key][seed_index] = [
                _compute_dabs_from_precomputed(
                    precomputed, thr.item(), diversity_penalty, filter_positive_scores
                )
                for thr in thresholds
            ]

    # Find common seed indices across all runs
    all_seed_sets = [set(scores.keys()) for scores in run_seed_scores.values()]
    if not all_seed_sets:
        return go.Figure()
    common_seeds = set.intersection(*all_seed_sets)

    # Filter by topic_ids if provided
    if topic_ids is not None:
        common_seeds = common_seeds & set(topic_ids)

    if not common_seeds:
        print("No common seed indices found across runs")
        return go.Figure()

    sorted_seeds = sorted(common_seeds)
    n_seeds = len(sorted_seeds)
    rows, cols = _compute_subplot_grid(n_seeds)

    # Create subplot titles
    subplot_titles = [f"Seed {seed}" for seed in sorted_seeds]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Assign fixed colors to each run
    colors = px.colors.qualitative.Plotly
    run_colors = {str(rp): colors[i % len(colors)] for i, rp in enumerate(run_paths_list)}

    # Add traces to each subplot
    for idx, seed_index in enumerate(sorted_seeds):
        row = idx // cols + 1
        col = idx % cols + 1

        for run_path in run_paths_list:
            run_key = str(run_path)
            run_label = run_labels[run_path]
            scores = run_seed_scores[run_key][seed_index]

            fig.add_trace(
                go.Scatter(
                    x=thresholds,
                    y=scores,
                    mode="lines+markers",
                    name=run_label,
                    line=dict(width=2, color=run_colors[run_key]),
                    marker=dict(size=4, color=run_colors[run_key]),
                    showlegend=(idx == 0),
                    legendgroup=run_label,
                ),
                row=row,
                col=col,
            )

    short_rewriter = rewriter_name.split("_")[-1]
    fig.update_layout(
        title=f"DABS vs teacher threshold ({short_rewriter})" + (" (strict)" if strict else ""),
        height=300 * rows,
        width=350 * cols,
        hovermode="x unified",
    )

    fig.update_xaxes(title_text="Teacher threshold")
    fig.update_yaxes(title_text="DABS")

    return fig


def _compute_2d_hypervolume(
    points: np.ndarray, ref_point: tuple[float, float]
) -> float:
    """Compute 2D hypervolume dominated by points relative to reference point.

    Assumes we want to maximize the first axis and minimize the second axis.
    Points are transformed to maximize both by negating the second axis.

    Args:
        points: Array of shape (n, 2) with (student_score, teacher_score)
        ref_point: Reference point (rx, ry)

    Returns:
        Hypervolume (area dominated by Pareto frontier relative to ref_point)
    """
    if len(points) == 0:
        return 0.0

    # Transform to maximize both: (student, -teacher)
    # So higher student = better, lower teacher = better (becomes higher -teacher)
    transformed = points.copy()
    transformed[:, 1] = -transformed[:, 1]
    ref_transformed = (ref_point[0], -ref_point[1])

    # Filter points that dominate the reference (both coords > ref)
    mask = (transformed[:, 0] > ref_transformed[0]) & (
        transformed[:, 1] > ref_transformed[1]
    )
    dominated_points = transformed[mask]

    if len(dominated_points) == 0:
        return 0.0

    # Find Pareto frontier (non-dominated points)
    is_pareto = np.ones(len(dominated_points), dtype=bool)
    for i, p in enumerate(dominated_points):
        if not is_pareto[i]:
            continue
        for j, q in enumerate(dominated_points):
            if i == j:
                continue
            if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                is_pareto[i] = False
                break

    pareto_points = dominated_points[is_pareto]

    if len(pareto_points) == 0:
        return 0.0

    # Sort by first axis (ascending)
    sorted_idx = np.argsort(pareto_points[:, 0])
    pareto_sorted = pareto_points[sorted_idx]

    # Compute hypervolume as sum of rectangles
    hypervolume = 0.0
    prev_x = ref_transformed[0]

    for x, y in pareto_sorted:
        width = x - prev_x
        height = y - ref_transformed[1]
        hypervolume += width * height
        prev_x = x

    return hypervolume


def compute_hypervolume_table(
    run_paths: Sequence[Path | str],
    percentile_range: tuple[float, float] = (5.0, 95.0),
    rewriter_name: str = "openai_gpt-5-mini",
) -> dict[str, dict[int, float]]:
    """Compute Pareto hypervolume for each run and seed.

    Aggregates all attribute scores across runs for normalization, then computes
    the hypervolume of the Pareto frontier for each run/seed combination.

    The Pareto frontier is defined as maximizing student score while minimizing
    teacher score (finding attributes where RM is biased but teacher disagrees).

    Args:
        run_paths: List of paths to run directories
        percentile_range: Tuple of (low, high) percentiles for normalization.
            Values outside this range are clipped. Default (5, 95).
        rewriter_name: Which rewriter's data to use

    Returns:
        Dict mapping run_name -> {seed_index: hypervolume}
    """
    run_paths = [Path(p) if isinstance(p, str) else p for p in run_paths]

    # Step 1: Collect all (student, teacher) scores across all runs/seeds
    all_student_scores = []
    all_teacher_scores = []
    run_seed_data: dict[str, dict[int, list[tuple[float, float]]]] = {}

    for run_path in run_paths:
        run_name = run_path.name
        run_seed_data[run_name] = {}

        for seed_index in _get_seed_indices(run_path):
            validation_data = _load_validation_data(run_path, seed_index, rewriter_name)

            seed_points = []
            for item in validation_data:
                student = item["student_score"]
                teacher = item["teacher_score"]

                if student is not None and teacher is not None:
                    all_student_scores.append(student)
                    all_teacher_scores.append(teacher)
                    seed_points.append((student, teacher))

            run_seed_data[run_name][seed_index] = seed_points

    if not all_student_scores:
        return {}

    # Step 2: Compute percentile-based normalization
    all_student = np.array(all_student_scores)
    all_teacher = np.array(all_teacher_scores)

    student_low = np.percentile(all_student, percentile_range[0])
    student_high = np.percentile(all_student, percentile_range[1])
    teacher_low = np.percentile(all_teacher, percentile_range[0])
    teacher_high = np.percentile(all_teacher, percentile_range[1])

    student_range = student_high - student_low if student_high != student_low else 1.0
    teacher_range = teacher_high - teacher_low if teacher_high != teacher_low else 1.0

    # Step 3: Compute hypervolume for each run/seed
    # Reference point: (0, 0.5) - neutral student score, neutral teacher agreement
    ref_point = (0.0, 0.5)

    result: dict[str, dict[int, float]] = {}

    for run_name, seeds_data in run_seed_data.items():
        result[run_name] = {}
        for seed_index, points in seeds_data.items():
            if not points:
                result[run_name][seed_index] = 0.0
                continue

            # Normalize: clip to percentile range, then rescale
            normalized = []
            for student, teacher in points:
                student_clipped = np.clip(student, student_low, student_high)
                teacher_clipped = np.clip(teacher, teacher_low, teacher_high)
                student_norm = student_clipped / student_range
                teacher_norm = teacher_clipped / teacher_range
                normalized.append((student_norm, teacher_norm))

            points_array = np.array(normalized)
            ref_norm = (ref_point[0] / student_range, ref_point[1] / teacher_range)
            hv = _compute_2d_hypervolume(points_array, ref_norm)
            result[run_name][seed_index] = hv

    return result


# %%

cluster_model = EmbedClusterModel(embed_model_name="Qwen/Qwen3-Embedding-8B")

# %%
run_paths = {
    "depth = 5, branching = 4": "data/evo/20260106-174842-list_reverse-handpick-plus",
    "depth = 3, branching = 8": "data/evo/20260107-015321-list_reverse-handpick-plus",
    "depth = 1": "data/evo/20260107-075251-list_reverse-handpick-plus",
}

fig = plot_dabs_vs_threshold(run_paths, cluster_model, strict=True)

Path(f"data/metrics/main_run_1").mkdir(parents=True, exist_ok=True)
fig.write_image(f"data/metrics/main_run_1/dabs_plot_strict.pdf")
print(f"Saved to main_run_1/dabs_plot_strict.pdf")

# %%
# hv_table = compute_hypervolume_table(run_paths)
# for run_name, seeds in hv_table.items():
#     print(f'{run_name}:')
#     for seed_idx, hv in sorted(seeds.items()):
#         print(f'  seed {seed_idx}: {hv:.4f}')
# %%
