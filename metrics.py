# ABOUTME: Metrics for evaluating reward model biases across runs.
# ABOUTME: Includes DABS (Diversity-Adjusted Bias Score) and hypervolume calculations.

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
    topic_ids: Sequence[int]|None = None,
    diversity_penalty: float = 0.5,
    threshold_step: float = 0.05,
    use_winrate: bool = True,
) -> list[go.Figure]:
    """Plot DABS vs threshold for comparing multiple runs.

    When multiple run_paths are provided, finds common seed indices and creates
    one figure per seed, with each run's DABS curve overlaid for comparison.

    Returns:
        List of figures, one per common seed index.
    """
    thresholds = np.arange(0.0, 1.0 + threshold_step, threshold_step)

    # Collect DABS scores: run_path -> seed_index -> list of scores (one per threshold)
    run_seed_scores: dict[str, dict[int, list[float]]] = {}
    for run_path in run_paths:
        run_key = str(run_path)
        run_seed_scores[run_key] = {}
        for judge_thr in thresholds:
            dabs_scores = DABS(run_path, judge_thr.item(), cluster_model, diversity_penalty, use_winrate)
            for seed_index, score in dabs_scores.items():
                if seed_index not in run_seed_scores[run_key]:
                    run_seed_scores[run_key][seed_index] = []
                run_seed_scores[run_key][seed_index].append(score)

    # Find common seed indices across all runs
    all_seed_sets = [set(scores.keys()) for scores in run_seed_scores.values()]
    if not all_seed_sets:
        return []
    common_seeds = set.intersection(*all_seed_sets)

    # Filter by topic_ids if provided
    if topic_ids is not None:
        common_seeds = common_seeds & set(topic_ids)

    if not common_seeds:
        print("No common seed indices found across runs")
        return []

    # Create one figure per common seed
    figures = []
    metric_type = "winrate" if use_winrate else "reward diff"

    for seed_index in sorted(common_seeds):
        fig = go.Figure()

        # Plot one line per run
        for run_path in run_paths:
            run_key = str(run_path)
            # Extract run name for legend (last part of path)
            run_name = Path(run_path).name
            scores = run_seed_scores[run_key][seed_index]

            fig.add_trace(
                go.Scatter(
                    x=thresholds,
                    y=scores,
                    mode='lines+markers',
                    name=run_name,
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

        fig.update_layout(
            title=f'Seed {seed_index}: DABS (student {metric_type}) vs teacher winrate threshold',
            xaxis_title='Teacher winrate threshold',
            yaxis_title=f'DABS (student {metric_type})',
            height=600,
            width=800,
            hovermode='x unified',
        )

        figures.append(fig)

    return figures


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
    "data/evo/20251218-091313-pair-clio-plus",
    "data/evo/20251218-104229-list_reverse-clio-plus",
    "data/evo/20251218-121253-list-clio-plus"
]
# run_paths = [
#     "data/evo/20251218-055659-pair-clio-plus",
#     "data/evo/20251217-160232-list_reverse-clio-plus",
# ]
cluster_model = ClusterModel(embedding_model_name="Qwen/Qwen3-Embedding-0.6B")

# %%
figures = plot_dabs_vs_threshold(run_paths, cluster_model)
for fig in figures:
    fig.show()

# %%
hv_table = compute_hypervolume_table(run_paths)
for run_name, seeds in hv_table.items():
    print(f'{run_name}:')
    for seed_idx, hv in sorted(seeds.items()):
        print(f'  seed {seed_idx}: {hv:.4f}')
# %%