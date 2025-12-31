# ABOUTME: Metrics for evaluating reward model biases across runs.
# ABOUTME: Includes DABS (Diversity-Adjusted Bias Score) and hypervolume calculations.

# %%
from pathlib import Path
import re
from typing import Sequence
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cluster_models import EmbedClusterModel
from plotting import process_run_data


def _precompute_seed_data(
    run_path: Path,
    seed_index: int,
    cluster_model: EmbedClusterModel,
    use_winrate: bool,
) -> dict:
    """Precompute embeddings and similarity matrix for a single seed.

    Returns dict with:
        - items: list of {attribute, judge_winrate, score}
        - cos_sims: precomputed cosine similarity matrix (or None if no items)
    """
    plot_data = process_run_data(run_path, seed_index)

    # Extract all items with valid scores
    items = []
    for item in plot_data:
        judge_wr = item.get("judge_winrate")
        if judge_wr is None:
            continue

        if use_winrate:
            score = item.get("reward_winrate")
        else:
            score = item.get("reward_diff_mean")

        if score is None:
            continue

        items.append({
            "attribute": item["attribute"],
            "judge_winrate": judge_wr,
            "score": score,
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
    judge_thr: float,
    diversity_penalty: float,
    use_winrate: bool,
) -> float:
    """Compute DABS score from precomputed data for a given threshold."""
    items = precomputed["items"]
    cos_sims = precomputed["cos_sims"]

    if not items or cos_sims is None:
        return 0.0

    # Filter by threshold and collect (original_index, score) pairs
    filtered = []
    for i, item in enumerate(items):
        if item["judge_winrate"] >= judge_thr:
            continue
        # For diff mode, also filter for score > 0
        if not use_winrate and item["score"] <= 0:
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
    run_path: Path|str,
    judge_thr: float,
    cluster_model: EmbedClusterModel,
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
    seed_indices = []
    if validate_dir.exists():
        pattern = re.compile(r'seed_(\d+)_validate')
        for item in validate_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    seed_indices.append(int(match.group(1)))
    seed_indices.sort()

    dabs_scores = {}
    for seed_index in seed_indices:
        precomputed = _precompute_seed_data(run_path, seed_index, cluster_model, use_winrate)
        dabs_scores[seed_index] = _compute_dabs_from_precomputed(
            precomputed, judge_thr, diversity_penalty, use_winrate
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
    run_paths: Sequence[Path|str],
    cluster_model: EmbedClusterModel,
    topic_ids: Sequence[int]|None = None,
    diversity_penalty: float = 0.5,
    threshold_step: float = 0.05,
    use_winrate: bool = True,
) -> go.Figure:
    """Plot DABS vs threshold for comparing multiple runs.

    Creates a single figure with subplots arranged in a grid (max 4 columns,
    as square as possible). Each subplot shows one seed's DABS curves with
    all runs overlaid for comparison.

    Returns:
        Single figure with subplots, one per common seed index.
    """
    thresholds = np.arange(0.0, 1.0 + threshold_step, threshold_step)

    # Step 1: Precompute embeddings for all runs/seeds (expensive, do once)
    # Structure: run_key -> seed_index -> precomputed_data
    precomputed_data: dict[str, dict[int, dict]] = {}
    for run_path in run_paths:
        run_path = Path(run_path) if isinstance(run_path, str) else run_path
        run_key = str(run_path)
        precomputed_data[run_key] = {}

        validate_dir = run_path / "validate"
        if not validate_dir.exists():
            continue

        pattern = re.compile(r'seed_(\d+)_validate')
        for item in validate_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    seed_index = int(match.group(1))
                    precomputed_data[run_key][seed_index] = _precompute_seed_data(
                        run_path, seed_index, cluster_model, use_winrate
                    )

    # Step 2: Compute DABS for all thresholds using precomputed data (cheap)
    run_seed_scores: dict[str, dict[int, list[float]]] = {}
    for run_key, seeds_data in precomputed_data.items():
        run_seed_scores[run_key] = {}
        for seed_index, precomputed in seeds_data.items():
            run_seed_scores[run_key][seed_index] = [
                _compute_dabs_from_precomputed(precomputed, thr.item(), diversity_penalty, use_winrate)
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

    metric_type = "winrate" if use_winrate else "reward diff"

    # Create subplot titles
    subplot_titles = [f'Seed {seed}' for seed in sorted_seeds]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Assign fixed colors to each run
    import plotly.express as px
    colors = px.colors.qualitative.Plotly
    run_colors = {str(run_path): colors[i % len(colors)] for i, run_path in enumerate(run_paths)}

    # Add traces to each subplot
    for idx, seed_index in enumerate(sorted_seeds):
        row = idx // cols + 1
        col = idx % cols + 1

        for run_path in run_paths:
            run_key = str(run_path)
            run_name = Path(run_path).name
            scores = run_seed_scores[run_key][seed_index]

            fig.add_trace(
                go.Scatter(
                    x=thresholds,
                    y=scores,
                    mode='lines+markers',
                    name=run_name,
                    line=dict(width=2, color=run_colors[run_key]),
                    marker=dict(size=4, color=run_colors[run_key]),
                    showlegend=(idx == 0),  # Only show legend for first subplot
                    legendgroup=run_name,
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title=f'DABS (student {metric_type}) vs teacher winrate threshold',
        height=300 * rows,
        width=350 * cols,
        hovermode='x unified',
    )

    # Update axis labels
    fig.update_xaxes(title_text='Teacher WR thr')
    fig.update_yaxes(title_text='DABS')

    return fig


def _compute_2d_hypervolume(points: np.ndarray, ref_point: tuple[float, float]) -> float:
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
    mask = (transformed[:, 0] > ref_transformed[0]) & (transformed[:, 1] > ref_transformed[1])
    dominated_points = transformed[mask]

    if len(dominated_points) == 0:
        return 0.0

    # Find Pareto frontier (non-dominated points)
    # A point is dominated if another point is >= in both coords and > in at least one
    is_pareto = np.ones(len(dominated_points), dtype=bool)
    for i, p in enumerate(dominated_points):
        if not is_pareto[i]:
            continue
        # Check if p is dominated by any other point
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
    # For each point, add rectangle from (x_prev, ref_y) to (x_curr, y_curr)
    hypervolume = 0.0
    prev_x = ref_transformed[0]

    for i, (x, y) in enumerate(pareto_sorted):
        # Width of rectangle
        width = x - prev_x
        # Height: from ref_y to y
        height = y - ref_transformed[1]
        hypervolume += width * height
        prev_x = x

    return hypervolume


def compute_hypervolume_table(
    run_paths: Sequence[Path | str],
    percentile_range: tuple[float, float] = (5.0, 95.0),
    use_winrate: bool = True,
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
        use_winrate: If True (default), use binary winrates (0/0.5/1) with
            reference point (0, 0.5). If False, use raw reward diffs with
            reference point (0, 0).

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

        validate_dir = run_path / "validate"
        if not validate_dir.exists():
            continue

        # Find all seed directories
        pattern = re.compile(r'seed_(\d+)_validate')
        for item in validate_dir.iterdir():
            if not item.is_dir():
                continue
            match = pattern.match(item.name)
            if not match:
                continue

            seed_index = int(match.group(1))

            # Use process_run_data to get properly computed winrates/diffs
            plot_data = process_run_data(run_path, seed_index)

            seed_points = []
            for item in plot_data:
                if use_winrate:
                    student = item.get("reward_winrate")
                    teacher = item.get("judge_winrate")
                else:
                    student = item.get("reward_diff_mean")
                    teacher = item.get("judge_winrate")

                if student is not None and teacher is not None:
                    all_student_scores.append(student)
                    all_teacher_scores.append(teacher)
                    seed_points.append((student, teacher))

            run_seed_data[run_name][seed_index] = seed_points

    if not all_student_scores:
        return {}

    # Step 2: Compute percentile-based normalization (rescale only, no shift)
    all_student = np.array(all_student_scores)
    all_teacher = np.array(all_teacher_scores)

    student_low = np.percentile(all_student, percentile_range[0])
    student_high = np.percentile(all_student, percentile_range[1])
    teacher_low = np.percentile(all_teacher, percentile_range[0])
    teacher_high = np.percentile(all_teacher, percentile_range[1])

    student_range = student_high - student_low if student_high != student_low else 1.0
    teacher_range = teacher_high - teacher_low if teacher_high != teacher_low else 1.0

    # Step 3: Compute hypervolume for each run/seed
    # Reference point: (0, 0.5) for winrate mode, (0, 0) for diff mode
    if use_winrate:
        ref_point = (0.0, 0.5)
    else:
        ref_point = (0.0, 0.0)

    result: dict[str, dict[int, float]] = {}

    for run_name, seeds_data in run_seed_data.items():
        result[run_name] = {}
        for seed_index, points in seeds_data.items():
            if not points:
                result[run_name][seed_index] = 0.0
                continue

            # Normalize: clip to percentile range, then rescale by dividing by range
            normalized = []
            for student, teacher in points:
                # Clip to percentile range
                student_clipped = np.clip(student, student_low, student_high)
                teacher_clipped = np.clip(teacher, teacher_low, teacher_high)
                # Rescale by dividing by range (no mean shift)
                student_norm = student_clipped / student_range
                teacher_norm = teacher_clipped / teacher_range
                normalized.append((student_norm, teacher_norm))

            points_array = np.array(normalized)
            # Also normalize reference point
            ref_norm = (float(ref_point[0] / student_range), float(ref_point[1] / teacher_range))
            hv = _compute_2d_hypervolume(points_array, ref_norm)
            result[run_name][seed_index] = hv

    return result


# %%
run_paths = [
    "data/evo/20251230-171530-list_reverse-handpick-plus",
    "data/evo/20251231-014058-list_reverse-handpick-plus",
    "data/evo/20251231-034737-list_reverse-handpick-plus"
]

cluster_model = EmbedClusterModel(embed_model_name="Qwen/Qwen3-Embedding-0.6B")

# %%
fig = plot_dabs_vs_threshold(run_paths, cluster_model)
# %%
Path("data/metrics").mkdir(parents=True, exist_ok=True)
fig.write_image("data/metrics/dabs_plot.pdf")
print("Saved to dabs_plot.pdf")

# # %%
# hv_table = compute_hypervolume_table(run_paths)
# for run_name, seeds in hv_table.items():
#     print(f'{run_name}:')
#     for seed_idx, hv in sorted(seeds.items()):
#         print(f'  seed {seed_idx}: {hv:.4f}')
# # %%
# %%
