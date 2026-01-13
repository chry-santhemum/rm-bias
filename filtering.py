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


def _get_validate_base_dir(run_path: Path) -> Path:
    """Return the base directory containing seed_*_validate folders.

    Handles two different directory structures:
    - Evo runs: {run_path}/validate/seed_{idx}_validate/
    - exp_attribute_validation runs: {run_path}/seed_{idx}_validate/
    """
    validate_subdir = run_path / "validate"
    if validate_subdir.exists():
        return validate_subdir
    return run_path


def get_seed_indices(run_path: Path) -> list[int]:
    """Find all seed indices in a run's validate directory."""
    validate_dir = _get_validate_base_dir(run_path)
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
        _get_validate_base_dir(run_path)
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


def _load_all_rewriter_data(
    run_path: Path, seed_index: int
) -> dict[str, dict[str, list[dict]]]:
    """Load rollouts data from all rewriters for a seed."""
    return {
        rewriter: load_rollouts_data(run_path, seed_index, rewriter)
        for rewriter in ALL_REWRITERS
    }


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

    Uses one-sample t-test. This is appropriate because judge scores are
    trinomial {0, 0.5, 1}, not binary, so the binomial variance assumption
    of the one-proportion z-test doesn't hold.
    """
    if len(scores) < 2:
        return 1.0

    t_stat, p_two_sided = scipy_stats.ttest_1samp(scores, 0.5)

    # Convert to one-sided p-value for H1: mean < 0.5
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


def compute_wilson_ci(scores: list[float]) -> tuple[float, float]:
    """
    Compute Wilson score 95% CI for proportion data.

    Appropriate for judge winrates which are discrete {0, 0.5, 1} values.
    Returns (ci_lower, ci_upper) as absolute bounds (not half-widths).

    The Wilson score interval:
        (p + z²/(2n) ± z * sqrt(p(1-p)/n + z²/(4n²))) / (1 + z²/n)
    """
    n = len(scores)
    if n < 1:
        return (0.0, 1.0)

    p = np.mean(scores)
    z = 1.96  # 95% CI

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)

    return (float(ci_lower), float(ci_upper))


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

    # t-distribution CI for reward bias (continuous)
    student_ci = compute_ci(student_scores)

    # Wilson CI for winrate (proportion data)
    teacher_ci_lower, teacher_ci_upper = compute_wilson_ci(teacher_scores)
    # Convert to half-width for compatibility (use max asymmetric margin)
    teacher_ci = max(teacher_mean - teacher_ci_lower, teacher_ci_upper - teacher_mean)

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
        "teacher_ci_lower": teacher_ci_lower,
        "teacher_ci_upper": teacher_ci_upper,
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
    rewriter_data = _load_all_rewriter_data(run_path, seed_index)

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


def compute_partial_conjunction_stats(
    run_path: Path,
    seed_index: int,
) -> dict[str, dict]:
    """
    Compute partial conjunction statistics for each attribute.

    Uses the partial conjunction test (2 of 3 rewriters) to determine significance:
    1. Compute per-rewriter p-values for RM (H1: mean > 0) and judge (H1: mean < 0.5)
    2. Sort p-values: p₁ < p₂ < p₃
    3. Partial conjunction p-value: p_pc = 2 × p₂
    4. Apply Bonferroni correction (per-topic)

    Args:
        run_path: Path to the run directory
        seed_index: Which seed to load

    Returns:
        Dict mapping attribute -> {
            "student_mean": float (mean across all rewriter samples),
            "student_ci": float,
            "student_p_pc": float (partial conjunction p-value),
            "student_p_pc_bonferroni": float,
            "teacher_mean": float,
            "teacher_ci": float,
            "teacher_p_pc": float,
            "teacher_p_pc_bonferroni": float,
            "n_hypotheses": int,
            "per_rewriter": dict with per-rewriter stats,
        }
    """
    rewriter_data = _load_all_rewriter_data(run_path, seed_index)

    # Find all attributes across rewriters
    all_attributes = set()
    for data in rewriter_data.values():
        all_attributes.update(data.keys())

    # Bonferroni N = number of attributes for this topic
    n_hypotheses = len(all_attributes)

    result: dict[str, dict] = {}
    for attribute in all_attributes:
        student_pvals = []
        teacher_pvals = []
        all_student_scores = []
        all_teacher_scores = []
        per_rewriter = {}

        for rewriter in ALL_REWRITERS:
            samples = rewriter_data[rewriter].get(attribute, [])
            if not samples:
                continue

            student_scores = [s["student_score"] for s in samples]
            teacher_scores = [s["teacher_score"] for s in samples]

            all_student_scores.extend(student_scores)
            all_teacher_scores.extend(teacher_scores)

            student_p = compute_rm_p_value(student_scores)
            teacher_p = compute_judge_p_value(teacher_scores)
            student_pvals.append(student_p)
            teacher_pvals.append(teacher_p)

            teacher_wilson = compute_wilson_ci(teacher_scores)
            per_rewriter[rewriter] = {
                "student_mean": float(np.mean(student_scores)),
                "student_ci": compute_ci(student_scores),
                "student_p": student_p,
                "teacher_mean": float(np.mean(teacher_scores)),
                "teacher_ci_lower": teacher_wilson[0],
                "teacher_ci_upper": teacher_wilson[1],
                "teacher_p": teacher_p,
                "n_samples": len(samples),
            }

        # Need at least 2 rewriters for partial conjunction
        if len(student_pvals) < 2:
            continue

        # Sort p-values and compute partial conjunction
        student_pvals_sorted = sorted(student_pvals)
        teacher_pvals_sorted = sorted(teacher_pvals)

        # p_pc = 2 * p₂ for "at least 2 of 3" test
        student_p_pc = min(2 * student_pvals_sorted[1], 1.0)
        teacher_p_pc = min(2 * teacher_pvals_sorted[1], 1.0)

        # Bonferroni correction (per-topic)
        student_p_pc_bonferroni = min(student_p_pc * n_hypotheses, 1.0)
        teacher_p_pc_bonferroni = min(teacher_p_pc * n_hypotheses, 1.0)

        # Compute aggregated stats
        student_mean = float(np.mean(all_student_scores))
        teacher_mean = float(np.mean(all_teacher_scores))
        student_ci = compute_ci(all_student_scores)
        teacher_wilson_agg = compute_wilson_ci(all_teacher_scores)

        result[attribute] = {
            "student_mean": student_mean,
            "student_ci": student_ci,
            "student_p_pc": student_p_pc,
            "student_p_pc_bonferroni": student_p_pc_bonferroni,
            "teacher_mean": teacher_mean,
            "teacher_ci_lower": teacher_wilson_agg[0],
            "teacher_ci_upper": teacher_wilson_agg[1],
            "teacher_p_pc": teacher_p_pc,
            "teacher_p_pc_bonferroni": teacher_p_pc_bonferroni,
            "n_hypotheses": n_hypotheses,
            "per_rewriter": per_rewriter,
        }

    return result


PARTIAL_CONJUNCTION_P_THRESHOLD = 0.01


def passes_partial_conjunction_criteria(
    student_p_pc_bonferroni: float,
    teacher_p_pc_bonferroni: float,
    student_mean: float,
    teacher_mean: float,
    p_threshold: float = PARTIAL_CONJUNCTION_P_THRESHOLD,
) -> bool:
    """
    Check if an attribute passes the partial conjunction criteria.

    Criteria:
    - student_mean > 0 (RM shows bias)
    - teacher_mean < 0.5 (judge disagrees)
    - Bonferroni-corrected partial conjunction p < threshold for both
    """
    return (
        student_mean > 0
        and teacher_mean < 0.5
        and student_p_pc_bonferroni < p_threshold
        and teacher_p_pc_bonferroni < p_threshold
    )


def save_partial_conjunction_results(
    run_path: Path,
    output_path: Path | None = None,
) -> dict:
    """
    Compute and save partial conjunction results for all seeds.

    CI methods:
    - RM diff: t-distribution 95% CI = t_crit(0.975, df=n-1) * SEM
    - Judge winrate: Wilson score 95% CI (appropriate for discrete {0, 0.5, 1} values)

    Args:
        run_path: Path to the run directory
        output_path: Where to save JSON results (default: run_path/partial_conjunction_results.json)

    Returns:
        Dict with results for all seeds
    """
    if output_path is None:
        output_path = run_path / "partial_conjunction_results.json"

    seed_indices = get_seed_indices(run_path)
    results = {
        "run_path": str(run_path),
        "p_value_threshold": PARTIAL_CONJUNCTION_P_THRESHOLD,
        "ci_methods": {
            "rm_diff": "t-distribution 95% CI: t_crit(0.975, df=n-1) * SEM",
            "judge_winrate": "Wilson score 95% CI",
        },
        "seeds": {},
    }

    for seed_idx in seed_indices:
        stats = compute_partial_conjunction_stats(run_path, seed_idx)

        seed_results = []
        for attr, data in stats.items():
            passes = passes_partial_conjunction_criteria(
                data["student_p_pc_bonferroni"],
                data["teacher_p_pc_bonferroni"],
                data["student_mean"],
                data["teacher_mean"],
            )

            # Format per-rewriter stats
            per_rewriter_formatted = {}
            for rw_name, rw_data in data["per_rewriter"].items():
                per_rewriter_formatted[rw_name] = {
                    "rm_diff": f"{rw_data['student_mean']:+.3f} ± {rw_data['student_ci']:.3f}",
                    "rm_diff_mean": rw_data["student_mean"],
                    "rm_diff_ci": rw_data["student_ci"],
                    "judge_winrate": f"{rw_data['teacher_mean']:.3f} [{rw_data['teacher_ci_lower']:.3f}, {rw_data['teacher_ci_upper']:.3f}]",
                    "judge_winrate_mean": rw_data["teacher_mean"],
                    "judge_winrate_ci_lower": rw_data["teacher_ci_lower"],
                    "judge_winrate_ci_upper": rw_data["teacher_ci_upper"],
                    "n_samples": rw_data["n_samples"],
                }

            seed_results.append({
                "attribute": attr,
                "passes_criteria": passes,
                "rm_p_pc_bonferroni": data["student_p_pc_bonferroni"],
                "judge_p_pc_bonferroni": data["teacher_p_pc_bonferroni"],
                "rm_p_pc": data["student_p_pc"],
                "judge_p_pc": data["teacher_p_pc"],
                "n_hypotheses": data["n_hypotheses"],
                "aggregated": {
                    "rm_diff": f"{data['student_mean']:+.3f} ± {data['student_ci']:.3f}",
                    "rm_diff_mean": data["student_mean"],
                    "rm_diff_ci": data["student_ci"],
                    "judge_winrate": f"{data['teacher_mean']:.3f} [{data['teacher_ci_lower']:.3f}, {data['teacher_ci_upper']:.3f}]",
                    "judge_winrate_mean": data["teacher_mean"],
                    "judge_winrate_ci_lower": data["teacher_ci_lower"],
                    "judge_winrate_ci_upper": data["teacher_ci_upper"],
                },
                "per_rewriter": per_rewriter_formatted,
            })

        results["seeds"][seed_idx] = seed_results

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_path}")
    return results


def print_partial_conjunction_table(run_path: Path) -> None:
    """Print a formatted table of partial conjunction results."""
    seed_indices = get_seed_indices(run_path)

    rewriter_short = {
        "openai_gpt-5-mini": "GPT-5m",
        "anthropic_claude-haiku-4.5": "Haiku",
        "x-ai_grok-4.1-fast": "Grok",
    }

    for seed_idx in seed_indices:
        stats = compute_partial_conjunction_stats(run_path, seed_idx)
        print(f"\n{'='*100}")
        print(f"SEED {seed_idx} ({len(stats)} attributes)")
        print(f"{'='*100}")

        for attr, data in stats.items():
            passes = passes_partial_conjunction_criteria(
                data["student_p_pc_bonferroni"],
                data["teacher_p_pc_bonferroni"],
                data["student_mean"],
                data["teacher_mean"],
            )
            status = "✓ PASS" if passes else "✗ FAIL"

            print(f"\n{status}: {attr}")
            print(f"  Partial conjunction p-values (Bonferroni): RM={data['student_p_pc_bonferroni']:.4f}, Judge={data['teacher_p_pc_bonferroni']:.4f}")
            print(f"  Per-rewriter results:")
            print(f"    {'Rewriter':<8} {'RM diff (95% CI)':<25} {'Judge winrate (Wilson 95% CI)':<30}")
            print(f"    {'-'*8} {'-'*25} {'-'*30}")

            for rw_name in ALL_REWRITERS:
                if rw_name in data["per_rewriter"]:
                    rw = data["per_rewriter"][rw_name]
                    rw_short = rewriter_short.get(rw_name, rw_name[:8])
                    rm_str = f"{rw['student_mean']:+.3f} ± {rw['student_ci']:.3f}"
                    judge_str = f"{rw['teacher_mean']:.3f} [{rw['teacher_ci_lower']:.3f}, {rw['teacher_ci_upper']:.3f}]"
                    print(f"    {rw_short:<8} {rm_str:<25} {judge_str:<30}")
