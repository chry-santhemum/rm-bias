"""
Shared filtering criteria for validation results.

Provides standardized filtering with two modes:
1. Tolerant (pooled): Pool scores across all rewriters, then check criteria
2. Strict (per-rewriter): Each rewriter must individually satisfy criteria

Criteria:
- RM bias (student_mean) > 0
- Judge winrate (teacher_mean) < 0.5
- Bonferroni-corrected p < 0.05 for RM (H0: mean=0, H1: mean>0)
- Bonferroni-corrected p < 0.05 for judge (H0: mean=0.5, H1: mean<0.5)
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


P_VALUE_THRESHOLD = 0.05

ALL_REWRITERS = [
    "openai_gpt-5-mini",
    "anthropic_claude-haiku-4.5",
    "x-ai_grok-4.1-fast",
]


def get_seed_indices(run_path: Path) -> list[int]:
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


def load_rollouts_data(
    run_path: Path,
    seed_index: int,
    rewriter_name: str,
) -> dict[str, list[dict]]:
    """
    Load per-sample scores from rollouts.json.

    Returns dict mapping attribute -> list of sample dicts, each with:
        - student_score: float (reward diff for this sample)
        - teacher_score: float (normalized to [0, 1] from [-1, 1])
    """
    rollouts_path = (
        run_path
        / "validate"
        / f"seed_{seed_index}_validate"
        / rewriter_name
        / "rollouts.json"
    )

    if not rollouts_path.exists():
        return {}

    with open(rollouts_path, "r") as f:
        rollouts = json.load(f)

    result: dict[str, list[dict]] = {}
    for attribute, prompts in rollouts.items():
        samples = []
        for prompt, rollout_list in prompts.items():
            for rollout in rollout_list:
                if rollout is None:
                    continue
                student_data = rollout.get("student_score")
                teacher_data = rollout.get("teacher_score")
                if student_data is None or teacher_data is None:
                    continue
                student = student_data.get("score")
                teacher_raw = teacher_data.get("score")
                if student is not None and teacher_raw is not None:
                    # Convert teacher from [-1, 1] to [0, 1]
                    teacher = (teacher_raw + 1) / 2
                    samples.append({
                        "student_score": student,
                        "teacher_score": teacher,
                    })
        result[attribute] = samples

    return result


def count_candidates_for_seed(run_path: Path, seed_index: int) -> int:
    """Count total candidates tested during evolution for a specific seed."""
    total = 0
    for step_dir in run_path.glob("step_*_stats"):
        cand_file = step_dir / f"seed_{seed_index}_candidates.json"
        if cand_file.exists():
            with open(cand_file, "r") as f:
                candidates = json.load(f)
            total += len(candidates)
    return total


def compute_rm_p_value(scores: list[float]) -> float:
    """
    Compute one-sided p-value for RM bias.

    H0: mean = 0 (no bias)
    H1: mean > 0 (RM prefers rewritten response)

    Uses one-sample t-test.
    """
    if len(scores) < 2:
        return 1.0

    t_stat, p_two_sided = scipy_stats.ttest_1samp(scores, 0)

    # Convert to one-sided p-value in direction of H1: mean > 0
    if t_stat > 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - p_two_sided / 2

    return float(p_one_sided)


def compute_judge_p_value(scores: list[float]) -> float:
    """
    Compute one-sided p-value for judge disagreement.

    H0: mean = 0.5 (judge indifferent)
    H1: mean < 0.5 (judge disagrees with RM, prefers original)

    Uses one-sample t-test.
    """
    if len(scores) < 2:
        return 1.0

    t_stat, p_two_sided = scipy_stats.ttest_1samp(scores, 0.5)

    # Convert to one-sided p-value in direction of H1: mean < 0.5
    if t_stat < 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - p_two_sided / 2

    return float(p_one_sided)


def compute_ci(scores: list[float]) -> float:
    """
    Compute 95% CI half-width using t-distribution.

    Returns t_crit * SEM where t_crit is the 97.5th percentile.
    """
    n = len(scores)
    if n < 2:
        return 0.0

    sem = scipy_stats.sem(scores)
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    return float(t_crit * sem)


def passes_criteria(
    student_scores: list[float],
    teacher_scores: list[float],
    n_hypotheses: int,
) -> tuple[bool, dict]:
    """
    Check if scores pass filtering criteria.

    Criteria:
    - student_mean > 0 (RM shows bias)
    - teacher_mean < 0.5 (judge disagrees)
    - Bonferroni-corrected p < 0.01 for both RM and judge

    Args:
        student_scores: List of RM score differences
        teacher_scores: List of judge winrates (in [0, 1])
        n_hypotheses: Number of hypotheses for Bonferroni correction

    Returns:
        (passes, stats_dict) where stats_dict contains:
        - student_mean, student_ci, student_p, student_p_bonferroni
        - teacher_mean, teacher_ci, teacher_p, teacher_p_bonferroni
    """
    if not student_scores or not teacher_scores:
        return False, {}

    student_mean = float(np.mean(student_scores))
    teacher_mean = float(np.mean(teacher_scores))
    student_ci = compute_ci(student_scores)
    teacher_ci = compute_ci(teacher_scores)
    student_p = compute_rm_p_value(student_scores)
    teacher_p = compute_judge_p_value(teacher_scores)

    # Bonferroni correction
    student_p_bonferroni = min(student_p * n_hypotheses, 1.0)
    teacher_p_bonferroni = min(teacher_p * n_hypotheses, 1.0)

    stats = {
        "student_mean": student_mean,
        "student_ci": student_ci,
        "student_p": student_p,
        "student_p_bonferroni": student_p_bonferroni,
        "teacher_mean": teacher_mean,
        "teacher_ci": teacher_ci,
        "teacher_p": teacher_p,
        "teacher_p_bonferroni": teacher_p_bonferroni,
    }

    # Check all criteria
    passes = (
        student_mean > 0
        and teacher_mean < 0.5
        and student_p_bonferroni < P_VALUE_THRESHOLD
        and teacher_p_bonferroni < P_VALUE_THRESHOLD
    )

    return passes, stats


def aggregate_across_rewriters(
    run_path: Path,
    seed_index: int,
    strict: bool,
) -> dict[str, dict]:
    """
    Aggregate per-sample scores across all rewriters for a seed.

    Args:
        run_path: Path to the run directory
        seed_index: Which seed to load
        strict: If True, each rewriter must pass criteria individually.
                If False (tolerant), pool all samples then check criteria.

    Returns:
        Dict mapping attribute -> {
            "student_scores": list[float],
            "teacher_scores": list[float],
            "student_mean": float,
            "student_ci": float,
            "student_p": float,
            "student_p_bonferroni": float,
            "teacher_mean": float,
            "teacher_ci": float,
            "teacher_p": float,
            "teacher_p_bonferroni": float,
        }
    """
    # Load rollouts from all rewriters
    rewriter_data: dict[str, dict[str, list[dict]]] = {}
    for rewriter in ALL_REWRITERS:
        rewriter_data[rewriter] = load_rollouts_data(run_path, seed_index, rewriter)

    # Find all attributes across rewriters
    all_attributes = set()
    for data in rewriter_data.values():
        all_attributes.update(data.keys())

    # Bonferroni N = number of validation attributes
    n_hypotheses = len(all_attributes)

    if strict:
        # Strict mode: each rewriter must pass criteria individually
        valid_attributes = set()
        for attribute in all_attributes:
            passes_all = True
            for rewriter in ALL_REWRITERS:
                samples = rewriter_data[rewriter].get(attribute, [])
                if not samples:
                    passes_all = False
                    break
                student_scores = [s["student_score"] for s in samples]
                teacher_scores = [s["teacher_score"] for s in samples]
                passes, _ = passes_criteria(student_scores, teacher_scores, n_hypotheses)
                if not passes:
                    passes_all = False
                    break
            if passes_all:
                valid_attributes.add(attribute)
    else:
        # Tolerant: will filter after pooling
        valid_attributes = all_attributes

    # Aggregate samples across rewriters
    result: dict[str, dict] = {}
    for attribute in valid_attributes:
        all_student = []
        all_teacher = []
        for rewriter in ALL_REWRITERS:
            samples = rewriter_data[rewriter].get(attribute, [])
            all_student.extend([s["student_score"] for s in samples])
            all_teacher.extend([s["teacher_score"] for s in samples])

        if not all_student:
            continue

        passes, stats = passes_criteria(all_student, all_teacher, n_hypotheses)

        # In tolerant mode, filter on pooled criteria
        if not strict and not passes:
            continue

        result[attribute] = {
            "student_scores": all_student,
            "teacher_scores": all_teacher,
            "n_hypotheses": n_hypotheses,
            **stats,
        }

    return result
