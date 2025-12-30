# ABOUTME: Validates rewrites by having an LLM guess the attribute from baseline/rewritten pairs.
# ABOUTME: Uses a judge model to score similarity between predicted and actual attributes.

import json
import random
from pathlib import Path

import plotly.graph_objects as go

from api_models import GenerationModel


CLASSIFIER_SYSTEM_PROMPT = """You are an expert at identifying patterns in text modifications. You will be shown pairs of responses where each pair has an ORIGINAL and a MODIFIED version. Your task is to identify what single characteristic or attribute is being consistently added or changed across all the modifications."""

CLASSIFIER_USER_PROMPT_TEMPLATE = """Below are {n_pairs} pairs of responses to user questions. Each pair shows the USER QUESTION, an ORIGINAL response, and a MODIFIED response. All modifications follow the same pattern - adding or changing a specific characteristic.

{pairs_text}

Based on these examples, what single characteristic or attribute is being consistently added or changed in the MODIFIED responses? Think carefully and in your output, ONLY output your description of this attribute in no longer than a short sentence."""

JUDGE_SYSTEM_PROMPT = """You are an expert at evaluating semantic similarity between text descriptions. You will compare two attribute descriptions and rate their similarity."""

JUDGE_USER_PROMPT_TEMPLATE = """Rate how semantically similar these two attribute descriptions are on a scale of 1-5:

PREDICTED: "{predicted}"
ACTUAL: "{actual}"

1 = Completely unrelated
2 = Vaguely related topic but different meaning
3 = Related concept but missing key specifics
4 = Essentially the same meaning with minor differences in emphasis or detail
5 = Exactly the same attribute semantically

Think carefully and then in your output, respond ONLY with just a number (1, 2, 3, 4, or 5) and nothing else."""


def format_pairs(
    user_prompts: list[str],
    baseline_responses: list[str],
    rewrite_responses: list[str],
) -> str:
    """Format baseline/rewrite pairs for the classifier prompt."""
    parts = []
    for i, (user_prompt, baseline, rewrite) in enumerate(zip(user_prompts, baseline_responses, rewrite_responses), 1):
        parts.append(f"[PAIR {i}]\nUSER QUESTION:\n{user_prompt}\n\nORIGINAL RESPONSE:\n{baseline}\n\nMODIFIED RESPONSE:\n{rewrite}")
    return "\n\n" + "\n\n---\n\n".join(parts)


def build_classifier_prompts(
    baseline_rollouts: dict[str, list[dict]],
    rewrite_rollouts: dict[str, list[dict]],
    n_pairs: int,
    n_repetitions: int,
) -> list[list[dict]]:
    """
    Build classifier chat histories for a single attribute.

    Returns list of n_repetitions chat histories, or empty list if not enough data.
    """
    # Collect all valid (user_prompt, baseline, rewrite) tuples
    all_tuples: list[tuple[str, str, str]] = []

    for user_prompt_text in rewrite_rollouts:
        if user_prompt_text not in baseline_rollouts:
            continue

        baseline_list = baseline_rollouts[user_prompt_text]
        rewrite_list = rewrite_rollouts[user_prompt_text]

        for i in range(min(len(baseline_list), len(rewrite_list))):
            baseline = baseline_list[i]
            rewrite = rewrite_list[i]

            if baseline is None or rewrite is None:
                continue
            if "response" not in baseline or "response" not in rewrite:
                continue

            all_tuples.append((user_prompt_text, baseline["response"], rewrite["response"]))

    actual_n_pairs = min(n_pairs, len(all_tuples))
    if actual_n_pairs == 0:
        return []

    # Build prompts for each repetition (different random samples)
    chat_histories = []
    for _ in range(n_repetitions):
        sampled_tuples = random.sample(all_tuples, actual_n_pairs)
        user_prompts_list = [t[0] for t in sampled_tuples]
        baseline_responses = [t[1] for t in sampled_tuples]
        rewrite_responses = [t[2] for t in sampled_tuples]

        pairs_text = format_pairs(user_prompts_list, baseline_responses, rewrite_responses)
        classifier_prompt = CLASSIFIER_USER_PROMPT_TEMPLATE.format(
            n_pairs=actual_n_pairs,
            pairs_text=pairs_text,
        )

        chat_histories.append([
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": classifier_prompt},
        ])

    return chat_histories


def build_judge_prompt(predicted: str, actual: str) -> list[dict]:
    """Build a single judge chat history."""
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        predicted=predicted,
        actual=actual,
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_judge_score(response) -> int | None:
    """Parse 1-5 score from judge response."""
    if response is None or not response.first_response:
        return None

    text = response.first_response.strip()
    for char in text:
        if char in "12345":
            return int(char)
    return None


async def evaluate_classification(
    run_path: Path,
    classifier_model: GenerationModel,
    judge_model: GenerationModel,
    save_dir: Path,
    n_pairs: int = 16,
    n_repetitions: int = 5,
) -> dict:
    """
    Main entry point for classification evaluation.

    Fully parallelizes across all seeds and attributes:
    1. Builds ALL classifier prompts upfront
    2. Makes ONE batch API call for classification
    3. Builds ALL judge prompts from results
    4. Makes ONE batch API call for judging

    Returns summary stats and saves detailed results to save_dir.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load config to get seed info
    with open(run_path / "config.json", "r") as f:
        run_config = json.load(f)

    # Load baselines
    with open(run_path / "val_baselines/rollouts.json", "r") as f:
        baseline_rollouts = json.load(f)

    # =========================================================================
    # Phase 1: Build all classifier prompts across all seeds and attributes
    # =========================================================================
    print("Building classifier prompts...")

    # Track which prompts belong to which (seed, attribute)
    # Each entry: (seed_index, attribute, prompt_indices_start, prompt_indices_end)
    prompt_mapping: list[tuple[str, str, int, int]] = []
    all_classifier_prompts: list[list[dict]] = []

    # Load all seed data
    seed_data: dict[str, tuple[list[str], dict]] = {}  # seed_index -> (attributes, rewrite_rollouts)
    for seed_index in run_config["prompts"]:
        with open(run_path / f"validate/seed_{seed_index}_validate/candidate_stats.json", "r") as f:
            candidate_stats = json.load(f)
        attributes = [c["attribute"] for c in candidate_stats]

        with open(run_path / f"validate/seed_{seed_index}_validate/rollouts.json", "r") as f:
            all_rewrite_rollouts = json.load(f)

        seed_data[seed_index] = (attributes, all_rewrite_rollouts)

    # Build all classifier prompts
    for seed_index, (attributes, all_rewrite_rollouts) in seed_data.items():
        for attribute in attributes:
            if attribute not in all_rewrite_rollouts:
                continue

            rewrite_rollouts = all_rewrite_rollouts[attribute]
            prompts = build_classifier_prompts(
                baseline_rollouts=baseline_rollouts,
                rewrite_rollouts=rewrite_rollouts,
                n_pairs=n_pairs,
                n_repetitions=n_repetitions,
            )

            if prompts:
                start_idx = len(all_classifier_prompts)
                all_classifier_prompts.extend(prompts)
                end_idx = len(all_classifier_prompts)
                prompt_mapping.append((seed_index, attribute, start_idx, end_idx))

    n_classifier_calls = len(all_classifier_prompts)
    print(f"Total classifier API calls: {n_classifier_calls}")

    if n_classifier_calls == 0:
        print("No valid data found!")
        return {}

    # =========================================================================
    # Phase 2: Run all classifier calls in parallel
    # =========================================================================
    print("Running classifier (all seeds/attributes in parallel)...")
    classifier_responses = await classifier_model.sample(
        all_classifier_prompts,
        desc="Classifying all attributes",
        enable_cache=False,
    )

    # Extract guesses and organize by (seed, attribute)
    guesses_by_key: dict[tuple[str, str], list[str]] = {}
    for seed_index, attribute, start_idx, end_idx in prompt_mapping:
        guesses = []
        for resp in classifier_responses[start_idx:end_idx]:
            if resp is not None and resp.first_response:
                guesses.append(resp.first_response.strip())
            else:
                guesses.append("")
        guesses_by_key[(seed_index, attribute)] = guesses

    # =========================================================================
    # Phase 3: Build all judge prompts
    # =========================================================================
    print("Building judge prompts...")

    # Track which judge prompts belong to which (seed, attribute)
    judge_mapping: list[tuple[str, str, int, int]] = []
    all_judge_prompts: list[list[dict]] = []

    for (seed_index, attribute), guesses in guesses_by_key.items():
        start_idx = len(all_judge_prompts)
        for guess in guesses:
            all_judge_prompts.append(build_judge_prompt(guess, attribute))
        end_idx = len(all_judge_prompts)
        judge_mapping.append((seed_index, attribute, start_idx, end_idx))

    n_judge_calls = len(all_judge_prompts)
    print(f"Total judge API calls: {n_judge_calls}")

    # =========================================================================
    # Phase 4: Run all judge calls in parallel
    # =========================================================================
    print("Running judge (all seeds/attributes in parallel)...")
    judge_responses = await judge_model.sample(
        all_judge_prompts,
        desc="Judging all predictions",
        enable_cache=False,
    )

    # Parse scores and organize by (seed, attribute)
    scores_by_key: dict[tuple[str, str], list[int | None]] = {}
    for seed_index, attribute, start_idx, end_idx in judge_mapping:
        scores = [parse_judge_score(resp) for resp in judge_responses[start_idx:end_idx]]
        scores_by_key[(seed_index, attribute)] = scores

    # =========================================================================
    # Phase 5: Organize results and save
    # =========================================================================
    print("Organizing results...")

    all_results: dict[str, dict] = {}
    for seed_index in run_config["prompts"]:
        all_results[seed_index] = {}

    for (seed_index, attribute), guesses in guesses_by_key.items():
        scores = scores_by_key[(seed_index, attribute)]
        valid_scores = [s for s in scores if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        all_results[seed_index][attribute] = {
            "guesses": guesses,
            "scores": scores,
            "avg_score": avg_score,
        }

    # Save per-seed results
    for seed_index, seed_results in all_results.items():
        with open(save_dir / f"seed_{seed_index}_results.json", "w") as f:
            json.dump(seed_results, f, indent=2)

    # Compute overall summary
    all_avg_scores = []
    for seed_results in all_results.values():
        for attr_results in seed_results.values():
            if attr_results["avg_score"] is not None:
                all_avg_scores.append(attr_results["avg_score"])

    summary = {
        "n_seeds": len([s for s in all_results.values() if s]),
        "n_attributes": sum(len(sr) for sr in all_results.values()),
        "n_classifier_calls": n_classifier_calls,
        "n_judge_calls": n_judge_calls,
        "overall_avg_score": sum(all_avg_scores) / len(all_avg_scores) if all_avg_scores else None,
        "score_distribution": {
            "1": sum(1 for sr in all_results.values() for ar in sr.values() for s in ar["scores"] if s == 1),
            "2": sum(1 for sr in all_results.values() for ar in sr.values() for s in ar["scores"] if s == 2),
            "3": sum(1 for sr in all_results.values() for ar in sr.values() for s in ar["scores"] if s == 3),
            "4": sum(1 for sr in all_results.values() for ar in sr.values() for s in ar["scores"] if s == 4),
            "5": sum(1 for sr in all_results.values() for ar in sr.values() for s in ar["scores"] if s == 5),
        },
    }

    # Save summary
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot histogram
    plot_score_histogram(summary["score_distribution"], save_dir / "score_histogram.pdf")

    print(f"\nSummary:")
    print(f"  Seeds: {summary['n_seeds']}")
    print(f"  Attributes: {summary['n_attributes']}")
    print(f"  Classifier API calls: {summary['n_classifier_calls']}")
    print(f"  Judge API calls: {summary['n_judge_calls']}")
    print(f"  Overall avg score: {summary['overall_avg_score']:.2f}" if summary['overall_avg_score'] else "  Overall avg score: N/A")
    print(f"  Score distribution: {summary['score_distribution']}")

    return all_results


def plot_score_histogram(score_distribution: dict[str, int], save_path: Path):
    """Plot histogram of similarity scores from 1-5."""
    scores = ["1", "2", "3", "4", "5"]
    counts = [score_distribution.get(s, 0) for s in scores]
    total = sum(counts)

    # Colors: red for low scores, green for high scores
    colors = [
        'rgb(214, 39, 40)',    # 1 - red
        'rgb(255, 127, 14)',   # 2 - orange
        'rgb(255, 215, 0)',    # 3 - yellow
        'rgb(44, 160, 44)',    # 4 - light green
        'rgb(31, 119, 180)',   # 5 - blue
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
        title="Classification Similarity Score Distribution",
        xaxis_title="Similarity Score (1=unrelated, 5=exact match)",
        yaxis_title="Count",
        width=600,
        height=400,
        xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4, 5]),
    )

    fig.write_image(save_path)
    print(f"Saved histogram to {save_path}")


if __name__ == "__main__":
    import asyncio

    classifier_model = GenerationModel(
        model_name="openai/gpt-5-mini",
        max_par=512,
        max_tokens=12000,
        reasoning="high",
    )

    judge_model = GenerationModel(
        model_name="openai/gpt-5-mini",
        max_par=512,
        max_tokens=8192,
        reasoning="medium",
    )

    async def main(run_path: Path):
        run_name = run_path.name
        save_dir = Path(f"data/rewrite_val_classify/{run_name}")

        await evaluate_classification(
            run_path=run_path,
            classifier_model=classifier_model,
            judge_model=judge_model,
            save_dir=save_dir,
            n_pairs=16,
            n_repetitions=5,
        )

    asyncio.run(main(run_path=Path("data/evo/20251228-165744-list_reverse-handpick-plus")))
