# %%
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from sklearn.metrics.pairwise import cosine_similarity

from cluster_models import EmbedClusterModel
from filtering import (
    ALL_REWRITERS,
    aggregate_across_rewriters,
    get_seed_indices,
)


def get_evo_passing_attributes_by_origin(
    evo_runs: dict[str, Path],
    strict: bool = False,
) -> dict[str, dict[int, set[str]]]:
    """
    Get attributes that pass filtering, grouped by evo run origin.

    Args:
        evo_runs: Dict mapping label -> run path
        strict: If True, use strict filtering mode

    Returns:
        {evo_label: {seed_idx: set[attribute]}}

    If an attribute passes in multiple runs, it appears in ALL runs that
    discovered it (no deduplication).
    """
    result: dict[str, dict[int, set[str]]] = {label: {} for label in evo_runs}

    for label, run_path in evo_runs.items():
        for seed_idx in get_seed_indices(run_path):
            aggregated = aggregate_across_rewriters(run_path, seed_idx, strict)
            if aggregated:
                result[label][seed_idx] = set(aggregated.keys())

    return result


def load_test_data_by_origin(
    test_runs: Sequence[Path],
    passing_by_origin: dict[str, dict[int, set[str]]],
    strict: bool = False,
) -> dict[str, dict[int, dict[str, dict]]]:
    """
    Load test statistics from multiple runs, organized by evo run origin.

    Args:
        test_runs: List of test run paths to load from
        passing_by_origin: Output from get_evo_passing_attributes_by_origin
        strict: If True, use strict filtering mode when loading

    Returns:
        {evo_label: {seed_idx: {attr: stats_dict}}}

    Each evo run gets test statistics for ALL attributes it discovered.
    If multiple evo runs discovered the same attribute, each gets a copy.
    """
    # First, load all test data indexed by (seed_idx, attr)
    test_data: dict[int, dict[str, dict]] = {}
    for test_run in test_runs:
        for seed_idx in get_seed_indices(test_run):
            aggregated = aggregate_across_rewriters(test_run, seed_idx, strict)
            if seed_idx not in test_data:
                test_data[seed_idx] = {}
            # Merge (first test run wins for each attribute)
            for attr, stats in aggregated.items():
                if attr not in test_data[seed_idx]:
                    test_data[seed_idx][attr] = stats

    # Now build result: for each evo run, include test stats for its attributes
    result: dict[str, dict[int, dict[str, dict]]] = {
        label: {} for label in passing_by_origin
    }

    for label, seeds_data in passing_by_origin.items():
        for seed_idx, passing_attrs in seeds_data.items():
            if seed_idx not in test_data:
                continue

            # Include stats for attributes this run discovered AND exist in test
            seed_result = {}
            for attr in passing_attrs:
                if attr in test_data[seed_idx]:
                    seed_result[attr] = test_data[seed_idx][attr]

            if seed_result:
                result[label][seed_idx] = seed_result

    return result


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


def _compute_embeddings_for_dabs(
    aggregated: dict[str, dict],
    cluster_model: EmbedClusterModel,
) -> dict:
    """
    Compute embeddings and similarity matrix from pre-aggregated data.

    Args:
        aggregated: Dict mapping attribute -> stats dict (with teacher_mean, student_mean)
        cluster_model: Model for computing embeddings

    Returns dict with:
        - items: list of {attribute, teacher_score, score} (using means)
        - cos_sims: precomputed cosine similarity matrix
    """
    if not aggregated:
        return {"items": [], "cos_sims": None}

    items = [
        {
            "attribute": attr,
            "teacher_score": data["teacher_mean"],
            "score": data["student_mean"],
        }
        for attr, data in aggregated.items()
    ]

    # Embed all attributes
    attributes = [item["attribute"] for item in items]
    embs = cluster_model.embed(attributes)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_normalized = embs / norms
    cos_sims = cosine_similarity(embs_normalized, embs_normalized)

    return {"items": items, "cos_sims": cos_sims}


def _precompute_aggregated_seed_data(
    run_path: Path,
    seed_index: int,
    cluster_model: EmbedClusterModel,
    strict: bool,
) -> dict:
    """
    Precompute embeddings and similarity matrix for aggregated data.

    Uses aggregate_across_rewriters to get per-attribute aggregated scores,
    then computes embeddings for DABS diversity penalty.

    Returns dict with:
        - items: list of {attribute, teacher_score, score} (using means)
        - cos_sims: precomputed cosine similarity matrix
    """
    aggregated = aggregate_across_rewriters(run_path, seed_index, strict)
    return _compute_embeddings_for_dabs(aggregated, cluster_model)


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

    # In strict mode, filter to attributes where all rewriters have:
    # - student_score >= 0 (RM shows preference)
    # - teacher_score < 0.5 (judge disagrees with RM)
    if strict:
        valid_attributes = None
        for rw in ALL_REWRITERS:
            rw_data = _load_validation_data(run_path, seed_index, rw)
            rw_valid = {
                item["attribute"]
                for item in rw_data
                if item["student_score"] is not None and item["student_score"] >= 0
                and item["teacher_score"] is not None and item["teacher_score"] < 0.5
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

    seed_indices = get_seed_indices(run_path)

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


def _is_pareto_optimal(points: np.ndarray) -> np.ndarray:
    """
    Find Pareto-optimal points (maximize student, minimize teacher).

    Args:
        points: Array of shape (n, 2) with (student_mean, teacher_mean)

    Returns:
        Boolean array of length n, True for Pareto-optimal points
    """
    n = len(points)
    is_optimal = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_optimal[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if: j has higher student AND lower teacher
            # (with at least one strictly better)
            if (points[j, 0] >= points[i, 0] and points[j, 1] <= points[i, 1] and
                (points[j, 0] > points[i, 0] or points[j, 1] < points[i, 1])):
                is_optimal[i] = False
                break

    return is_optimal


def plot_pareto_frontier(
    run_paths: Sequence[Path | str] | dict[str, Path | str] | None = None,
    topic_ids: Sequence[int] | None = None,
    strict: bool = False,
    pareto_only: bool = False,
    precomputed_data: dict[str, dict[int, dict[str, dict]]] | None = None,
) -> go.Figure:
    """
    Plot student vs teacher scores for each attribute as a scatter plot.

    Creates a grid of subplots (one per seed). Each point represents an
    attribute with error bars showing 95% CI.

    Args:
        run_paths: Either a list of run directories, or a dict mapping
            display labels to run directories. Ignored if precomputed_data provided.
        topic_ids: Optional filter for specific seeds
        strict: If True, only include attributes where ALL rewriters have
            student_score >= 0 AND teacher_score <= 0.5. Ignored if precomputed_data.
        pareto_only: If True, only show Pareto-optimal points.
        precomputed_data: Pre-loaded data as {label: {seed_idx: {attr: stats}}}.
            When provided, run_paths and strict are ignored.

    Returns:
        Figure with subplots.
    """
    if precomputed_data is not None:
        # Use pre-computed data directly
        run_labels_list = list(precomputed_data.keys())
        run_seed_data = precomputed_data

        # Find all seeds across all runs (union, not intersection)
        all_seeds: set[int] = set()
        for seeds_data in run_seed_data.values():
            all_seeds.update(seeds_data.keys())
    else:
        # Load from disk (original behavior)
        if run_paths is None:
            raise ValueError("Either run_paths or precomputed_data must be provided")

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

        run_labels_list = [run_labels[rp] for rp in run_paths_list]

        # Load aggregated data for all runs/seeds
        run_seed_data = {}
        for run_path in run_paths_list:
            label = run_labels[run_path]
            run_seed_data[label] = {}

            for seed_index in get_seed_indices(run_path):
                run_seed_data[label][seed_index] = aggregate_across_rewriters(
                    run_path, seed_index, strict
                )

        # Find common seed indices across all runs
        all_seed_sets = [set(data.keys()) for data in run_seed_data.values()]
        if not all_seed_sets:
            return go.Figure()
        all_seeds = set.intersection(*all_seed_sets)

    # Filter by topic_ids if provided
    if topic_ids is not None:
        all_seeds = all_seeds & set(topic_ids)

    if not all_seeds:
        print("No seed indices found")
        return go.Figure()

    # Filter to seeds that have at least one attribute in any run
    seeds_with_data = set()
    for seed in all_seeds:
        for label in run_seed_data:
            if seed in run_seed_data[label] and run_seed_data[label][seed]:
                seeds_with_data.add(seed)
                break

    if not seeds_with_data:
        print("No seeds with attributes found")
        return go.Figure()

    sorted_seeds = sorted(seeds_with_data)
    n_datasets = len(sorted_seeds)

    # Dynamic grid: max 4 columns
    rows, cols = _compute_subplot_grid(n_datasets)

    # Create subplot titles using actual seed indices
    subplot_titles = [f"Topic {seed}" for seed in sorted_seeds]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    # Use Dark2 color scheme
    colors = px.colors.qualitative.Dark2
    run_colors = {label: colors[i % len(colors)] for i, label in enumerate(run_labels_list)}

    # Compute jitter offsets for overlapping attributes (same attr in multiple runs)
    # Maps (seed_idx, attr) -> {run_label: (x_offset, y_offset)}
    jitter_offsets: dict[tuple[int, str], dict[str, tuple[float, float]]] = {}
    # Separate jitter for each axis (X has larger range than Y)
    jitter_x = 0.04
    jitter_y = 0.03

    for seed_index in sorted_seeds:
        # Find attributes that appear in multiple runs for this seed
        attr_runs: dict[str, list[str]] = {}
        for run_label in run_labels_list:
            if seed_index not in run_seed_data.get(run_label, {}):
                continue
            for attr in run_seed_data[run_label][seed_index]:
                attr_runs.setdefault(attr, []).append(run_label)

        # Compute offsets for overlapping attributes
        for attr, runs in attr_runs.items():
            if len(runs) > 1:
                # Spread points in an ellipse around the true position
                for i, run_label in enumerate(runs):
                    angle = 2 * np.pi * i / len(runs)
                    x_off = jitter_x * np.cos(angle)
                    y_off = jitter_y * np.sin(angle)
                    jitter_offsets[(seed_index, attr)] = jitter_offsets.get(
                        (seed_index, attr), {}
                    )
                    jitter_offsets[(seed_index, attr)][run_label] = (x_off, y_off)

    # Add traces to each subplot
    shown_legend_groups: set[str] = set()
    for idx, seed_index in enumerate(sorted_seeds):
        row = idx // cols + 1
        col = idx % cols + 1

        for run_label in run_labels_list:
            if seed_index not in run_seed_data[run_label]:
                continue
            attr_data = run_seed_data[run_label][seed_index]

            if not attr_data:
                continue

            # Extract data for plotting
            attributes = list(attr_data.keys())
            student_means = np.array([attr_data[a]["student_mean"] for a in attributes])
            teacher_means = np.array([attr_data[a]["teacher_mean"] for a in attributes])
            student_cis = np.array([attr_data[a]["student_ci"] for a in attributes])
            # Wilson CI bounds for teacher (asymmetric)
            teacher_ci_lower = np.array([attr_data[a]["teacher_ci_lower"] for a in attributes])
            teacher_ci_upper = np.array([attr_data[a]["teacher_ci_upper"] for a in attributes])

            # Apply jitter for overlapping attributes
            for i, attr in enumerate(attributes):
                key = (seed_index, attr)
                if key in jitter_offsets and run_label in jitter_offsets[key]:
                    x_off, y_off = jitter_offsets[key][run_label]
                    student_means[i] += x_off
                    teacher_means[i] += y_off

            # Filter to Pareto-optimal if requested
            if pareto_only and len(attributes) > 0:
                points = np.column_stack([student_means, teacher_means])
                pareto_mask = _is_pareto_optimal(points)
                attributes = [a for a, m in zip(attributes, pareto_mask) if m]
                student_means = student_means[pareto_mask]
                teacher_means = teacher_means[pareto_mask]
                student_cis = student_cis[pareto_mask]
                teacher_ci_lower = teacher_ci_lower[pareto_mask]
                teacher_ci_upper = teacher_ci_upper[pareto_mask]

            if len(attributes) == 0:
                continue

            # Make error bar color semi-transparent
            error_color = run_colors[run_label].replace("rgb(", "rgba(").replace(")", ", 0.4)")
            if not error_color.startswith("rgba"):
                error_color = f"rgba(128, 128, 128, 0.4)"

            # Show legend for first occurrence of each run
            show_legend = run_label not in shown_legend_groups
            if show_legend:
                shown_legend_groups.add(run_label)

            fig.add_trace(
                go.Scatter(
                    x=student_means,
                    y=teacher_means,
                    mode="markers",
                    name=run_label,
                    marker=dict(size=7, color=run_colors[run_label]),
                    error_x=dict(
                        type="data",
                        array=student_cis,
                        visible=True,
                        color=error_color,
                        thickness=1,
                    ),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=teacher_ci_upper - teacher_means,  # upper margin
                        arrayminus=teacher_means - teacher_ci_lower,  # lower margin
                        visible=True,
                        color=error_color,
                        thickness=1,
                    ),
                    text=attributes,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Student: %{x:.3f}<br>"
                        "Teacher: %{y:.3f}<extra></extra>"
                    ),
                    showlegend=show_legend,
                    legendgroup=run_label,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=300 * rows,
        width=280 * cols,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=16),
        ),
        margin=dict(t=80),
        font=dict(size=14),  # Global font size
    )

    # Update subplot title font size
    fig.update_annotations(font_size=16)

    # Only label axes on edge subplots, and fix axis ranges
    for i in range(1, rows * cols + 1):
        row_idx = (i - 1) // cols + 1
        col_idx = (i - 1) % cols + 1
        # X-axis: start from 0, label only on bottom row
        if row_idx == rows:
            fig.update_xaxes(
                title_text="RM bias strength (⟶)",
                title_font_size=15,
                tickfont_size=13,
                range=[0, None],
                row=row_idx,
                col=col_idx,
            )
        else:
            fig.update_xaxes(
                title_text="",
                tickfont_size=13,
                range=[0, None],
                row=row_idx,
                col=col_idx,
            )
        # Y-axis: fixed range [0, 0.55] with label only on leftmost column
        # Arrow points left (renders as down after 90° CCW rotation)
        if col_idx == 1:
            fig.update_yaxes(
                title_text="(⟵) LM judge bias winrate",
                title_font_size=15,
                tickfont_size=13,
                range=[-0.05, 0.55],
                row=row_idx,
                col=col_idx,
            )
        else:
            fig.update_yaxes(
                title_text="",
                tickfont_size=13,
                range=[-0.05, 0.55],
                row=row_idx,
                col=col_idx,
            )

    return fig


def print_attribute_stats(
    run_path: Path | str,
    strict: bool = False,
) -> None:
    """
    Print aggregated statistics for each attribute across all seeds.

    For each attribute, prints:
        - Mean RM score and 95% CI
        - Mean LM judge winrate and 95% CI

    Args:
        run_path: Path to the run directory
        strict: If True, only include attributes where ALL rewriters have
            student_score >= 0 AND teacher_score <= 0.5.
    """
    if isinstance(run_path, str):
        run_path = Path(run_path)

    seed_indices = get_seed_indices(run_path)

    for seed_index in seed_indices:
        aggregated = aggregate_across_rewriters(run_path, seed_index, strict)

        if not aggregated:
            print(f"\n=== Dataset {seed_index} ===")
            print("  No attributes found")
            continue

        print(f"\n=== Dataset {seed_index} ({len(aggregated)} attributes) ===")
        print(f"{'Attribute':<80} {'RM Score':>20} {'Judge Winrate':>20}")
        print("-" * 122)

        # Sort by RM score descending
        sorted_attrs = sorted(
            aggregated.items(),
            key=lambda x: x[1]["student_mean"],
            reverse=True,
        )

        for attr, data in sorted_attrs:
            attr_display = attr[:77] + "..." if len(attr) > 80 else attr
            rm_str = f"{data['student_mean']:.3f} ± {data['student_ci']:.3f}"
            judge_str = f"{data['teacher_mean']:.3f} ± {data['teacher_ci']:.3f}"
            print(f"{attr_display:<80} {rm_str:>20} {judge_str:>20}")


def plot_dabs_vs_threshold(
    run_paths: Sequence[Path | str] | dict[str, Path | str] | None = None,
    cluster_model: EmbedClusterModel | None = None,
    topic_ids: Sequence[int] | None = None,
    diversity_penalty: float = 0.5,
    threshold_step: float = 0.05,
    strict: bool = False,
    precomputed_data: dict[str, dict[int, dict[str, dict]]] | None = None,
) -> go.Figure:
    """Plot DABS vs threshold for comparing multiple runs.

    Uses aggregated scores across all rewriters. Creates a grid of subplots
    (one per seed). Each subplot shows DABS curves for all runs.

    Args:
        run_paths: Either a list of run directories, or a dict mapping
            display labels to run directories. Ignored if precomputed_data provided.
        cluster_model: Model for computing attribute embeddings
        topic_ids: Optional filter for specific seeds
        diversity_penalty: Penalty for similar attributes (0-1)
        threshold_step: Granularity of threshold sweep
        strict: If True, only include attributes where ALL rewriters have
            student_score >= 0 AND teacher_score <= 0.5. Ignored if precomputed_data.
        precomputed_data: Pre-loaded data as {label: {seed_idx: {attr: stats}}}.
            When provided, run_paths and strict are ignored.

    Returns:
        Figure with subplots.
    """
    if cluster_model is None:
        raise ValueError("cluster_model is required")

    thresholds = np.arange(0.0, 0.5 + threshold_step, threshold_step)

    if precomputed_data is not None:
        # Use pre-computed data - compute embeddings from it
        run_labels_list = list(precomputed_data.keys())

        # Step 1: Compute embeddings for each label/seed
        embedding_data: dict[str, dict[int, dict]] = {}
        for label, seeds_data in precomputed_data.items():
            embedding_data[label] = {}
            for seed_idx, attr_data in seeds_data.items():
                embedding_data[label][seed_idx] = _compute_embeddings_for_dabs(
                    attr_data, cluster_model
                )

        # Find all seeds (union)
        all_seeds: set[int] = set()
        for seeds_data in embedding_data.values():
            all_seeds.update(seeds_data.keys())
    else:
        # Load from disk (original behavior)
        if run_paths is None:
            raise ValueError("Either run_paths or precomputed_data must be provided")

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

        run_labels_list = [run_labels[rp] for rp in run_paths_list]

        # Step 1: Precompute embeddings using aggregated data
        embedding_data = {}
        for run_path in run_paths_list:
            label = run_labels[run_path]
            embedding_data[label] = {}

            for seed_index in get_seed_indices(run_path):
                embedding_data[label][seed_index] = _precompute_aggregated_seed_data(
                    run_path, seed_index, cluster_model, strict=strict
                )

        # Find common seed indices across all runs
        all_seed_sets = [set(seeds.keys()) for seeds in embedding_data.values()]
        if not all_seed_sets:
            return go.Figure()
        all_seeds = set.intersection(*all_seed_sets)

    # Step 2: Compute DABS for all thresholds using embedding data
    # Note: filter_positive_scores is always False here since aggregation
    # already filters for student_mean >= 0
    run_seed_scores: dict[str, dict[int, list[float]]] = {}
    for label, seeds_data in embedding_data.items():
        run_seed_scores[label] = {}
        for seed_index, emb_data in seeds_data.items():
            run_seed_scores[label][seed_index] = [
                _compute_dabs_from_precomputed(
                    emb_data,
                    thr.item(),
                    diversity_penalty,
                    filter_positive_scores=False,
                )
                for thr in thresholds
            ]

    # Filter by topic_ids if provided
    if topic_ids is not None:
        all_seeds = all_seeds & set(topic_ids)

    if not all_seeds:
        print("No seed indices found")
        return go.Figure()

    # Filter to seeds that have non-zero DABS in any run (i.e., have data)
    seeds_with_data = set()
    for seed in all_seeds:
        for label in embedding_data:
            if seed in embedding_data[label] and embedding_data[label][seed]["items"]:
                seeds_with_data.add(seed)
                break

    if not seeds_with_data:
        print("No seeds with attributes found")
        return go.Figure()

    sorted_seeds = sorted(seeds_with_data)
    n_datasets = len(sorted_seeds)

    # Dynamic grid: max 4 columns
    rows, cols = _compute_subplot_grid(n_datasets)

    # Create subplot titles using actual seed indices
    subplot_titles = [f"Topic {seed}" for seed in sorted_seeds]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    # Use Dark2 color scheme
    colors = px.colors.qualitative.Dark2
    run_colors = {label: colors[i % len(colors)] for i, label in enumerate(run_labels_list)}

    # Compute x-offsets for each run to prevent overlapping lines
    n_runs = len(run_labels_list)
    x_offset_magnitude = threshold_step * 0.15  # Small fraction of step size
    run_x_offsets = {
        label: (i - (n_runs - 1) / 2) * x_offset_magnitude
        for i, label in enumerate(run_labels_list)
    }

    # Add traces to each subplot
    shown_legend_groups: set[str] = set()
    for idx, seed_index in enumerate(sorted_seeds):
        row = idx // cols + 1
        col = idx % cols + 1

        for run_label in run_labels_list:
            if seed_index not in run_seed_scores[run_label]:
                continue
            scores = run_seed_scores[run_label][seed_index]

            # Apply x-offset to prevent overlapping lines
            x_values = thresholds + run_x_offsets[run_label]

            # Show legend for first occurrence of each run
            show_legend = run_label not in shown_legend_groups
            if show_legend:
                shown_legend_groups.add(run_label)

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=scores,
                    mode="lines+markers",
                    name=run_label,
                    line=dict(width=3, color=run_colors[run_label]),
                    marker=dict(size=6, color=run_colors[run_label]),
                    showlegend=show_legend,
                    legendgroup=run_label,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=300 * rows,
        width=280 * cols,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=16),
        ),
        margin=dict(t=80),
        font=dict(size=14),  # Global font size
    )

    # Update subplot title font size
    fig.update_annotations(font_size=16)

    # Compute max DABS value per seed for individual y-axis ranges
    seed_max_dabs = {}
    for seed_index in sorted_seeds:
        seed_scores = []
        for label in run_seed_scores:
            if seed_index in run_seed_scores[label]:
                seed_scores.extend(run_seed_scores[label][seed_index])
        seed_max_dabs[seed_index] = max(seed_scores) if seed_scores else 0

    # Only label axes on edge subplots, fix y-axis range per subplot
    for idx, seed_index in enumerate(sorted_seeds):
        row_idx = idx // cols + 1
        col_idx = idx % cols + 1
        y_max = max(2.0, seed_max_dabs[seed_index] * 1.05)  # Minimum range of 2

        # X-axis label only on bottom row
        if row_idx == rows:
            fig.update_xaxes(
                title_text="LM judge winrate threshold",
                title_font_size=15,
                tickfont_size=13,
                row=row_idx,
                col=col_idx,
            )
        else:
            fig.update_xaxes(title_text="", tickfont_size=13, row=row_idx, col=col_idx)
        # Y-axis: fixed range with minimum of 2, label only on leftmost column
        # Arrow points right (renders as up after 90° CCW rotation)
        if col_idx == 1:
            fig.update_yaxes(
                title_text="DABS (⟶)",
                title_font_size=15,
                tickfont_size=13,
                range=[0, y_max],
                row=row_idx,
                col=col_idx,
            )
        else:
            fig.update_yaxes(
                title_text="",
                tickfont_size=13,
                range=[0, y_max],
                row=row_idx,
                col=col_idx,
            )

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

        for seed_index in get_seed_indices(run_path):
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
# Evo runs (for determining which attributes pass filtering and their origin)
EVO_RUNS = {
    "depth=5, branch=4": Path("data/evo/20260106-174842-list_reverse-handpick-plus"),
    "depth=3, branch=8": Path("data/evo/20260107-015321-list_reverse-handpick-plus"),
    "depth=1": Path("data/evo/20260107-075251-list_reverse-handpick-plus"),
}

# Test runs (for loading actual test set statistics)
TEST_RUNS = [
    Path("data/exp_attribute_validation/20260112-162826"),
    Path("data/exp_attribute_validation/20260125-084345"),
]

# Get passing attributes grouped by evo origin
passing_by_origin = get_evo_passing_attributes_by_origin(EVO_RUNS, strict=False)

# Load test statistics organized by origin
test_data_by_origin = load_test_data_by_origin(TEST_RUNS, passing_by_origin, strict=False)

# Print summary
print("Test set attributes by evo run origin:")
for label, seeds_data in test_data_by_origin.items():
    total = sum(len(attrs) for attrs in seeds_data.values())
    print(f"  {label}: {total} attributes")

output_dir = Path("data/metrics/test_set")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate Pareto frontier plot using test set statistics
pareto_fig = plot_pareto_frontier(precomputed_data=test_data_by_origin)
pareto_fig.write_html(output_dir / "pareto_plot.html")
pareto_fig.write_image(output_dir / "pareto_plot.pdf")

# Generate DABS vs threshold plot using test set statistics
dabs_fig = plot_dabs_vs_threshold(
    cluster_model=cluster_model, precomputed_data=test_data_by_origin
)
dabs_fig.write_html(output_dir / "dabs_plot.html")
dabs_fig.write_image(output_dir / "dabs_plot.pdf")

print(f"Plots saved to {output_dir}")

