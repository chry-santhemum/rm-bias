# ABOUTME: Validates rewrites by having an LLM guess the attribute from baseline/rewritten pairs.
# ABOUTME: Adapted for exp_attribute_validation data format with per-model results.

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
    rewrite_rollouts: dict[str, list[dict]],
    n_pairs: int,
    n_repetitions: int,
) -> list[list[dict]]:
    """
    Build classifier chat histories for a single attribute.

    rewrite_rollouts: dict mapping user_prompt -> list of {baseline_response, rewritten_response, ...}

    Returns list of n_repetitions chat histories, or empty list if not enough data.
    """
    # Collect all valid (user_prompt, baseline, rewrite) tuples
    all_tuples: list[tuple[str, str, str]] = []

    for user_prompt_text, rollout_list in rewrite_rollouts.items():
        for rollout in rollout_list:
            if rollout is None:
                continue
            if "baseline_response" not in rollout or "rewritten_response" not in rollout:
                continue

            all_tuples.append((
                user_prompt_text,
                rollout["baseline_response"],
                rollout["rewritten_response"],
            ))

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
    Main entry point for classification evaluation on exp_attribute_validation data.

    Fully parallelizes across all seeds, models, and attributes:
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

    # =========================================================================
    # Phase 1: Build all classifier prompts across all seeds, models, and attributes
    # =========================================================================
    print("Building classifier prompts...")

    # Track which prompts belong to which (seed, model, attribute)
    # Each entry: (seed_index, model, attribute, prompt_indices_start, prompt_indices_end)
    prompt_mapping: list[tuple[str, str, str, int, int]] = []
    all_classifier_prompts: list[list[dict]] = []

    # Load all seed data
    # seed_data: seed_index -> model -> (attributes, rewrite_rollouts)
    seed_data: dict[str, dict[str, tuple[list[str], dict]]] = {}

    for seed_index in run_config["topic_attributes"]:
        seed_dir = run_path / f"seed_{seed_index}_validate"
        rewrites_dir = seed_dir / "rewrites"

        if not rewrites_dir.exists():
            continue

        seed_data[seed_index] = {}

        # Discover models from subdirectories
        for model_dir in rewrites_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            rollouts_path = model_dir / "rollouts.json"
            candidate_stats_path = seed_dir / model_name / "candidate_stats.json"

            if not rollouts_path.exists() or not candidate_stats_path.exists():
                continue

            with open(candidate_stats_path, "r") as f:
                candidate_stats = json.load(f)
            attributes = [c["attribute"] for c in candidate_stats]

            with open(rollouts_path, "r") as f:
                all_rewrite_rollouts = json.load(f)

            seed_data[seed_index][model_name] = (attributes, all_rewrite_rollouts)

    # Build all classifier prompts
    for seed_index, models_data in seed_data.items():
        for model_name, (attributes, all_rewrite_rollouts) in models_data.items():
            for attribute in attributes:
                if attribute not in all_rewrite_rollouts:
                    continue

                rewrite_rollouts = all_rewrite_rollouts[attribute]
                prompts = build_classifier_prompts(
                    rewrite_rollouts=rewrite_rollouts,
                    n_pairs=n_pairs,
                    n_repetitions=n_repetitions,
                )

                if prompts:
                    start_idx = len(all_classifier_prompts)
                    all_classifier_prompts.extend(prompts)
                    end_idx = len(all_classifier_prompts)
                    prompt_mapping.append((seed_index, model_name, attribute, start_idx, end_idx))

    n_classifier_calls = len(all_classifier_prompts)
    print(f"Total classifier API calls: {n_classifier_calls}")

    if n_classifier_calls == 0:
        print("No valid data found!")
        return {}

    # =========================================================================
    # Phase 2: Run all classifier calls in parallel
    # =========================================================================
    print("Running classifier (all seeds/models/attributes in parallel)...")
    classifier_responses = await classifier_model.sample(
        all_classifier_prompts,
        desc="Classifying all attributes",
        enable_cache=False,
    )

    # Extract guesses and organize by (seed, model, attribute)
    guesses_by_key: dict[tuple[str, str, str], list[str]] = {}
    for seed_index, model_name, attribute, start_idx, end_idx in prompt_mapping:
        guesses = []
        for resp in classifier_responses[start_idx:end_idx]:
            if resp is not None and resp.first_response:
                guesses.append(resp.first_response.strip())
            else:
                guesses.append("")
        guesses_by_key[(seed_index, model_name, attribute)] = guesses

    # =========================================================================
    # Phase 3: Build all judge prompts
    # =========================================================================
    print("Building judge prompts...")

    # Track which judge prompts belong to which (seed, model, attribute)
    judge_mapping: list[tuple[str, str, str, int, int]] = []
    all_judge_prompts: list[list[dict]] = []

    for (seed_index, model_name, attribute), guesses in guesses_by_key.items():
        start_idx = len(all_judge_prompts)
        for guess in guesses:
            all_judge_prompts.append(build_judge_prompt(guess, attribute))
        end_idx = len(all_judge_prompts)
        judge_mapping.append((seed_index, model_name, attribute, start_idx, end_idx))

    n_judge_calls = len(all_judge_prompts)
    print(f"Total judge API calls: {n_judge_calls}")

    # =========================================================================
    # Phase 4: Run all judge calls in parallel
    # =========================================================================
    print("Running judge (all seeds/models/attributes in parallel)...")
    judge_responses = await judge_model.sample(
        all_judge_prompts,
        desc="Judging all predictions",
        enable_cache=False,
    )

    # Parse scores and organize by (seed, model, attribute)
    scores_by_key: dict[tuple[str, str, str], list[int | None]] = {}
    for seed_index, model_name, attribute, start_idx, end_idx in judge_mapping:
        scores = [parse_judge_score(resp) for resp in judge_responses[start_idx:end_idx]]
        scores_by_key[(seed_index, model_name, attribute)] = scores

    # =========================================================================
    # Phase 5: Organize results and save
    # =========================================================================
    print("Organizing results...")

    # Structure: seed -> model -> attribute -> results
    all_results: dict[str, dict[str, dict[str, dict]]] = {}
    for seed_index in run_config["topic_attributes"]:
        all_results[seed_index] = {}

    for (seed_index, model_name, attribute), guesses in guesses_by_key.items():
        scores = scores_by_key[(seed_index, model_name, attribute)]
        valid_scores = [s for s in scores if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        if model_name not in all_results[seed_index]:
            all_results[seed_index][model_name] = {}

        all_results[seed_index][model_name][attribute] = {
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
    score_distribution = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}

    for seed_results in all_results.values():
        for model_results in seed_results.values():
            for attr_results in model_results.values():
                if attr_results["avg_score"] is not None:
                    all_avg_scores.append(attr_results["avg_score"])
                for s in attr_results["scores"]:
                    if s is not None:
                        score_distribution[str(s)] += 1

    # Count unique models across all seeds
    all_models = set()
    for seed_results in all_results.values():
        all_models.update(seed_results.keys())

    # Count total attributes
    n_attributes = sum(
        len(model_results)
        for seed_results in all_results.values()
        for model_results in seed_results.values()
    )

    summary = {
        "n_seeds": len([s for s in all_results.values() if s]),
        "n_models": len(all_models),
        "models": sorted(all_models),
        "n_attributes": n_attributes,
        "n_classifier_calls": n_classifier_calls,
        "n_judge_calls": n_judge_calls,
        "overall_avg_score": sum(all_avg_scores) / len(all_avg_scores) if all_avg_scores else None,
        "score_distribution": score_distribution,
    }

    # Save summary
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot histogram
    plot_score_histogram(summary["score_distribution"], save_dir / "score_histogram.pdf")

    print(f"\nSummary:")
    print(f"  Seeds: {summary['n_seeds']}")
    print(f"  Models: {summary['n_models']} ({', '.join(summary['models'])})")
    print(f"  Attributes: {summary['n_attributes']}")
    print(f"  Classifier API calls: {summary['n_classifier_calls']}")
    print(f"  Judge API calls: {summary['n_judge_calls']}")
    if summary['overall_avg_score'] is not None:
        print(f"  Overall avg score: {summary['overall_avg_score']:.2f}")
    else:
        print("  Overall avg score: N/A")
    print(f"  Score distribution: {summary['score_distribution']}")

    return all_results


def plot_score_histogram(score_distribution: dict[str, int], save_path: Path):
    """Plot histogram of similarity scores from 1-5."""
    scores = ["1", "2", "3", "4", "5"]
    counts = [score_distribution.get(s, 0) for s in scores]
    total = sum(counts)

    # Use Dark2 color scheme
    dark2_colors = [
        'rgb(27, 158, 119)',   # teal
        'rgb(217, 95, 2)',     # orange
        'rgb(117, 112, 179)',  # purple
        'rgb(231, 41, 138)',   # pink
        'rgb(102, 166, 30)',   # green
    ]

    fig = go.Figure()

    # Use 'inside' for tall bars to prevent overflow
    max_count = max(counts) if counts else 0
    threshold = max_count * 0.3
    text_positions = ['inside' if c > threshold else 'outside' for c in counts]

    fig.add_trace(go.Bar(
        x=scores,
        y=counts,
        marker_color=dark2_colors,
        text=[f"{c} ({c/total*100:.1f}%)" if total > 0 else "0" for c in counts],
        textposition=text_positions,
        textangle=0,
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
        save_dir = run_path / "classify"

        await evaluate_classification(
            run_path=run_path,
            classifier_model=classifier_model,
            judge_model=judge_model,
            save_dir=save_dir,
            n_pairs=16,
            n_repetitions=5,
        )

    asyncio.run(main(run_path=Path("data/exp_attribute_validation/20260112-162826")))
