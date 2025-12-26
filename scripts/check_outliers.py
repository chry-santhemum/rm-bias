# ABOUTME: Temporary script to identify outlier diffs in the fictitious event experiment
# ABOUTME: Helps inspect which rollouts were flagged as outliers by the IQR method

import json
import numpy as np
from pathlib import Path

RUN_DIR = Path("data/exp_fictitious_event/20251226-051710")

# Load results
with open(RUN_DIR / "results.json") as f:
    results = json.load(f)

baseline_scores = results["baseline_scores"]
bias_scores = results["bias_scores"]
biases = results["biases"]

# Load rewrites for inspecting text
rewrites_by_bias = {}
for bias_idx in range(len(biases)):
    with open(RUN_DIR / f"rewrites_bias_{bias_idx}.json") as f:
        rewrites_by_bias[bias_idx] = json.load(f)

# Backfill student_score and student_diff into rewrites
print("Backfilling student_score and student_diff into rewrite JSONs...")
for bias_idx in range(len(biases)):
    bias_scores_dict = bias_scores[str(bias_idx)]
    modified = False

    for user_prompt, rewrites in rewrites_by_bias[bias_idx].items():
        if user_prompt not in bias_scores_dict or user_prompt not in baseline_scores:
            continue
        scores = bias_scores_dict[user_prompt]
        baselines = baseline_scores[user_prompt]

        for rollout_idx, rewrite in enumerate(rewrites):
            if rewrite is None:
                continue
            if rollout_idx < len(scores) and rollout_idx < len(baselines):
                score = scores[rollout_idx]
                baseline = baselines[rollout_idx]
                rewrite["student_score"] = score
                if score is not None and baseline is not None:
                    rewrite["student_diff"] = score - baseline
                else:
                    rewrite["student_diff"] = None
                modified = True

    if modified:
        with open(RUN_DIR / f"rewrites_bias_{bias_idx}.json", "w") as f:
            json.dump(rewrites_by_bias[bias_idx], f, indent=2)
        print(f"  Updated rewrites_bias_{bias_idx}.json")

print("Done backfilling.\n")


def compute_outlier_bounds(diffs: list[float], iqr_k: float = 1.5):
    """Compute Tukey's fences."""
    arr = np.array(diffs)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    return q1 - iqr_k * iqr, q3 + iqr_k * iqr


def find_outliers_for_bias(bias_idx: int):
    """Find outlier diffs for a given bias condition."""
    bias_scores_dict = bias_scores[str(bias_idx)]

    # Collect all diffs with metadata
    all_diffs = []
    diff_metadata = []

    for user_prompt, scores in bias_scores_dict.items():
        if user_prompt not in baseline_scores:
            continue
        baselines = baseline_scores[user_prompt]

        for rollout_idx, (bias_score, baseline_score) in enumerate(zip(scores, baselines)):
            if bias_score is not None and baseline_score is not None:
                diff = bias_score - baseline_score
                all_diffs.append(diff)
                diff_metadata.append({
                    "user_prompt": user_prompt,
                    "rollout_idx": rollout_idx,
                    "diff": diff,
                    "bias_score": bias_score,
                    "baseline_score": baseline_score,
                })

    if not all_diffs:
        return [], None, None

    # Compute bounds
    low, high = compute_outlier_bounds(all_diffs)

    # Find outliers
    outliers = []
    for meta in diff_metadata:
        if meta["diff"] < low or meta["diff"] > high:
            # Get the rewritten text
            rewrite_data = rewrites_by_bias[bias_idx].get(meta["user_prompt"], [])
            if meta["rollout_idx"] < len(rewrite_data) and rewrite_data[meta["rollout_idx"]]:
                meta["rewrite_text"] = rewrite_data[meta["rollout_idx"]].get("text", "")[:500]
            else:
                meta["rewrite_text"] = None
            outliers.append(meta)

    return outliers, low, high


# Main analysis
print("=" * 80)
print("OUTLIER ANALYSIS FOR FICTITIOUS EVENT EXPERIMENT")
print(f"Run: {RUN_DIR}")
print("=" * 80)

bias_labels = [
    "0: Detailed (hallucinated details)",
    "1: Brief (avoid details)",
    "2: Refusal",
    "3: Compliance",
]

for bias_idx in range(len(biases)):
    outliers, low, high = find_outliers_for_bias(bias_idx)

    print(f"\n{'='*80}")
    print(f"BIAS {bias_labels[bias_idx]}")
    print(f"Outlier bounds: [{low:.4f}, {high:.4f}]")
    print(f"Number of outliers: {len(outliers)}")
    print("=" * 80)

    for i, o in enumerate(outliers):
        print(f"\n--- Outlier {i+1} ---")
        print(f"User prompt: {o['user_prompt'][:100]}...")
        print(f"Rollout idx: {o['rollout_idx']}")
        print(f"Diff: {o['diff']:.4f} (bias={o['bias_score']:.4f}, baseline={o['baseline_score']:.4f})")
        if o.get("rewrite_text"):
            print(f"Rewrite snippet: {o['rewrite_text'][:300]}...")
        print()
