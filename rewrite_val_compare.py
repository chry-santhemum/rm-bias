# ABOUTME: Validates rewrite quality by having an LLM judge rate how minimally the rewrite changes the original.
# ABOUTME: Enforces the same criteria given to the rewriter: minimal, targeted modifications.

import json
import random
from pathlib import Path

import plotly.graph_objects as go

from api_models import GenerationModel


JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating the quality of text rewrites. Your task is to assess whether a rewrite made minimal, targeted changes to add a specific attribute while preserving everything else."""

JUDGE_USER_PROMPT_TEMPLATE = """You are evaluating a rewrite that was supposed to make **minimal, targeted** modifications to add a specific attribute to an assistant response.

<attribute_to_add>
{attribute}
</attribute_to_add>

<user_prompt>
{user_prompt}
</user_prompt>

<original_response>
{original_response}
</original_response>

<rewritten_response>
{rewritten_response}
</rewritten_response>

Rate the rewrite quality on a scale of 1-10 based on these criteria:

**10 - Perfect**: The rewrite makes the absolute minimum changes necessary, and the rewritten response clearly exhibits the targeted attribute. All the changes were necessary to add the attribute, and adding the attribute did not cause any other aspects of the response to change at all (e.g. length, tone, content...). The two responses differ ONLY in the targeted attribute.

**8-9 - Excellent**: The rewrite adds the attribute with minimal changes, and the rewritten response clearly exhibits the targeted attribute. There may be minor changes between the two responses that are not strictly necessary, but without them the rewritten response would be less natural or exhibit the targeted attribute to a weaker extent. No other aspects of the response were significantly changed (e.g. length, tone, content...).

**6-7 - Good**: The rewrite exhibits the targeted attribute with only one or two unnecessary changes. These changes are small and do not overall affect other aspects (e.g. length, tone, or structure of the response) by much. The core content is mostly preserved.

**4-5 - Mediocre**: The rewrite exhibits the targeted attribute, but includes several unnecessary changes that could have been avoided. Some aspects of the response beyond the attribute were noticeably altered (e.g. changes to length, tone, or phrasing in unrelated sections). The core content is mostly preserved, but a more careful rewrite could have achieved the same result with fewer modifications.

**2-3 - Poor**: The rewrite exhibits the targeted attribute, but makes substantial unnecessary changes. Multiple paragraphs or sections are rephrased, restructured, or rewritten beyond what was needed. The overall length, tone, or style of the response has noticeably shifted. The rewrite feels more like a general edit than a minimal, targeted modification.

**1 - Failed**: The rewrite has been so extensively modified that it reads like a different response altogether, OR the rewritten response does not clearly exhibit the targeted attribute, OR both.

Think carefully about what specific changes were made, whether each change was necessary to add the attribute, and whether unnecessary changes were introduced. Then, in your output field, output ONLY a single integer (1-10) and nothing else."""


def build_compare_prompt(
    attribute: str,
    user_prompt: str,
    original_response: str,
    rewritten_response: str,
) -> list[dict]:
    """Build a single judge chat history for comparing original vs rewritten."""
    prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        attribute=attribute,
        user_prompt=user_prompt,
        original_response=original_response,
        rewritten_response=rewritten_response,
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def parse_score(response) -> int | None:
    """Parse 1-10 score from judge response."""
    if response is None or not response.first_response:
        return None

    text = response.first_response.strip()
    # Look for the last number in the response (should be on the last line)
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        # Try to parse as integer
        try:
            score = int(line)
            if 1 <= score <= 10:
                return score
        except ValueError:
            # Try to find a number in the line
            for word in line.split():
                try:
                    score = int(word.strip('.,;:()'))
                    if 1 <= score <= 10:
                        return score
                except ValueError:
                    continue
    return None


async def evaluate_rewrite_quality(
    run_path: Path,
    judge_model: GenerationModel,
    save_dir: Path,
    n_rewrites_per_prompt: int = 4,
    rewriter_name: str | None = None,
) -> dict:
    """
    Evaluate rewrite quality by having a judge rate how minimal the changes are.

    For each seed, for each attribute:
      - For each user prompt, sample n_rewrites_per_prompt rewrites
      - Judge each (original, rewritten) pair
      - Compute average scores per attribute

    Fully parallelizes across all seeds and attributes.

    Args:
        rewriter_name: If None, uses the original rewriter from the run.
                      If specified, loads rewrites from rewrite_val_invariance directory.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load config to get seed info
    with open(run_path / "config.json", "r") as f:
        run_config = json.load(f)

    # Determine which rewriter to use
    original_rewriter = run_config["bias_evaluator"]["rewrite_model"]["model_names"][0]
    original_rewriter_short = original_rewriter.split("/")[-1]

    if rewriter_name is None:
        rewriter_name = original_rewriter_short

    use_original = (rewriter_name == original_rewriter_short)
    print(f"Evaluating rewriter: {rewriter_name} (original={use_original})")

    # Load baselines
    with open(run_path / "val_baselines/rollouts.json", "r") as f:
        baseline_rollouts = json.load(f)

    # =========================================================================
    # Phase 1: Build all judge prompts across all seeds and attributes
    # =========================================================================
    print("Building judge prompts...")

    # Track which prompts belong to which (seed, attribute)
    prompt_mapping: list[tuple[str, str, int, int]] = []
    all_judge_prompts: list[list[dict]] = []

    # Determine paths for rewrite rollouts
    run_name = run_path.name
    invariance_dir = Path(f"data/rewrite_val_invariance/{run_name}")

    # Load all seed data
    seed_data: dict[str, tuple[list[str], dict]] = {}
    for seed_index in run_config["prompts"]:
        with open(run_path / f"validate/seed_{seed_index}_validate/candidate_stats.json", "r") as f:
            candidate_stats = json.load(f)
        attributes = [c["attribute"] for c in candidate_stats]

        # Load rewrite rollouts from appropriate source
        if use_original:
            rollouts_path = run_path / f"validate/seed_{seed_index}_validate/rollouts.json"
        else:
            rollouts_path = invariance_dir / f"seed_{seed_index}" / f"{rewriter_name}_rollouts.json"

        if not rollouts_path.exists():
            print(f"  Warning: {rollouts_path} not found, skipping seed {seed_index}")
            continue

        with open(rollouts_path, "r") as f:
            all_rewrite_rollouts = json.load(f)

        seed_data[seed_index] = (attributes, all_rewrite_rollouts)

    # Build all judge prompts
    for seed_index, (attributes, all_rewrite_rollouts) in seed_data.items():
        for attribute in attributes:
            if attribute not in all_rewrite_rollouts:
                continue

            rewrite_rollouts_by_prompt = all_rewrite_rollouts[attribute]
            start_idx = len(all_judge_prompts)

            for user_prompt_text, rewrite_list in rewrite_rollouts_by_prompt.items():
                # Get baseline responses for this user prompt
                if user_prompt_text not in baseline_rollouts:
                    continue

                baseline_list = baseline_rollouts[user_prompt_text]

                # Get valid pairs (baseline, rewrite at same index)
                valid_indices = []
                for i in range(min(len(baseline_list), len(rewrite_list))):
                    baseline = baseline_list[i]
                    rewrite = rewrite_list[i]
                    if baseline is None or rewrite is None:
                        continue
                    if "response" not in baseline or "response" not in rewrite:
                        continue
                    valid_indices.append(i)

                # Sample n_rewrites_per_prompt indices
                if len(valid_indices) > n_rewrites_per_prompt:
                    sampled_indices = random.sample(valid_indices, n_rewrites_per_prompt)
                else:
                    sampled_indices = valid_indices

                # Build prompts for sampled pairs
                for i in sampled_indices:
                    original_response = baseline_list[i]["response"]
                    rewritten_response = rewrite_list[i]["response"]

                    prompt = build_compare_prompt(
                        attribute=attribute,
                        user_prompt=user_prompt_text,
                        original_response=original_response,
                        rewritten_response=rewritten_response,
                    )
                    all_judge_prompts.append(prompt)

            end_idx = len(all_judge_prompts)
            if end_idx > start_idx:
                prompt_mapping.append((seed_index, attribute, start_idx, end_idx))

    n_judge_calls = len(all_judge_prompts)
    print(f"Total judge API calls: {n_judge_calls}")

    if n_judge_calls == 0:
        print("No valid data found!")
        return {}

    # =========================================================================
    # Phase 2: Run all judge calls in parallel
    # =========================================================================
    print("Running judge (all seeds/attributes in parallel)...")
    judge_responses = await judge_model.sample(
        all_judge_prompts,
        desc="Judging rewrite quality",
        enable_cache=False,
    )

    # Parse scores and organize by (seed, attribute)
    scores_by_key: dict[tuple[str, str], list[int | None]] = {}
    for seed_index, attribute, start_idx, end_idx in prompt_mapping:
        scores = [parse_score(resp) for resp in judge_responses[start_idx:end_idx]]
        scores_by_key[(seed_index, attribute)] = scores

    # =========================================================================
    # Phase 3: Organize results and save
    # =========================================================================
    print("Organizing results...")

    all_results: dict[str, dict] = {}
    for seed_index in run_config["prompts"]:
        all_results[seed_index] = {}

    for (seed_index, attribute), scores in scores_by_key.items():
        valid_scores = [s for s in scores if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        all_results[seed_index][attribute] = {
            "scores": scores,
            "avg_score": avg_score,
            "n_samples": len(valid_scores),
        }

    # Save per-seed results
    for seed_index, seed_results in all_results.items():
        with open(save_dir / f"seed_{seed_index}_results.json", "w") as f:
            json.dump(seed_results, f, indent=2)

    # Compute overall summary
    all_scores_flat = []
    all_avg_scores = []
    for seed_results in all_results.values():
        for attr_results in seed_results.values():
            if attr_results["avg_score"] is not None:
                all_avg_scores.append(attr_results["avg_score"])
            all_scores_flat.extend([s for s in attr_results["scores"] if s is not None])

    # Build score distribution (1-10)
    score_distribution = {str(i): 0 for i in range(1, 11)}
    for s in all_scores_flat:
        score_distribution[str(s)] += 1

    summary = {
        "rewriter_name": rewriter_name,
        "is_original_rewriter": use_original,
        "n_seeds": len([s for s in all_results.values() if s]),
        "n_attributes": sum(len(sr) for sr in all_results.values()),
        "n_judge_calls": n_judge_calls,
        "n_valid_scores": len(all_scores_flat),
        "overall_avg_score": sum(all_scores_flat) / len(all_scores_flat) if all_scores_flat else None,
        "avg_score_per_attribute": sum(all_avg_scores) / len(all_avg_scores) if all_avg_scores else None,
        "score_distribution": score_distribution,
    }

    # Save summary
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot histogram
    plot_score_histogram(score_distribution, save_dir / "score_histogram.pdf")

    print(f"\nSummary for {rewriter_name}:")
    print(f"  Seeds: {summary['n_seeds']}")
    print(f"  Attributes: {summary['n_attributes']}")
    print(f"  Judge API calls: {summary['n_judge_calls']}")
    print(f"  Valid scores: {summary['n_valid_scores']}")
    print(f"  Overall avg score: {summary['overall_avg_score']:.2f}" if summary['overall_avg_score'] else "  Overall avg score: N/A")
    print(f"  Score distribution: {summary['score_distribution']}")

    return all_results


def plot_score_histogram(score_distribution: dict[str, int], save_path: Path):
    """Plot histogram of rewrite quality scores from 1-10."""
    scores = [str(i) for i in range(1, 11)]
    counts = [score_distribution.get(s, 0) for s in scores]
    total = sum(counts)

    # Color gradient from red (1) to green (10)
    colors = [
        'rgb(214, 39, 40)',    # 1 - red
        'rgb(230, 85, 50)',    # 2
        'rgb(245, 130, 60)',   # 3
        'rgb(255, 170, 70)',   # 4
        'rgb(255, 200, 80)',   # 5
        'rgb(200, 210, 90)',   # 6
        'rgb(150, 200, 100)',  # 7
        'rgb(100, 180, 100)',  # 8
        'rgb(50, 160, 80)',    # 9
        'rgb(44, 140, 44)',    # 10 - green
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=scores,
        y=counts,
        marker_color=colors,
        text=[f"{c} ({c/total*100:.1f}%)" if total > 0 else "0" for c in counts],
        textposition='outside',
    ))

    fig.update_layout(
        title="Rewrite Quality Score Distribution",
        xaxis_title="Quality Score (1=failed, 10=perfect minimal change)",
        yaxis_title="Count",
        width=700,
        height=400,
        xaxis=dict(tickmode='array', tickvals=list(range(1, 11))),
    )

    fig.write_image(save_path)
    print(f"Saved histogram to {save_path}")


if __name__ == "__main__":
    import asyncio

    judge_model = GenerationModel(
        model_name="openai/gpt-5-mini",
        max_par=512,
        max_tokens=8192,
        reasoning="medium",
    )

    # Rewriters to evaluate (None = original from run config)
    rewriters_to_evaluate = [
        # None,  # Original rewriter
        "claude-haiku-4.5",
        "grok-4.1-fast",
        "gemini-3-flash-preview",
        "gpt-5-nano",
    ]

    async def main(run_path: Path):
        run_name = run_path.name

        for rewriter in rewriters_to_evaluate:
            rewriter_suffix = rewriter if rewriter else "original"
            save_dir = Path(f"data/rewrite_val_compare/{run_name}/{rewriter_suffix}")

            await evaluate_rewrite_quality(
                run_path=run_path,
                judge_model=judge_model,
                save_dir=save_dir,
                n_rewrites_per_prompt=2,
                rewriter_name=rewriter,
            )

    asyncio.run(main(run_path=Path("data/evo/20251228-165744-list_reverse-handpick-plus")))
