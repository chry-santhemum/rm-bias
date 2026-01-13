"""
Check for similar prompts between handpick (train) and handpick_test (test) clusters
using n-gram Jaccard similarity.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r'\b\w+\b', text.lower())


def get_ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """Extract n-grams from a list of tokens."""
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def ngram_similarity(text1: str, text2: str, n: int) -> float:
    """Compute n-gram Jaccard similarity between two texts."""
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    ngrams1 = get_ngrams(tokens1, n)
    ngrams2 = get_ngrams(tokens2, n)
    return jaccard_similarity(ngrams1, ngrams2)


def load_cluster(path: Path) -> list[str]:
    """Load prompts from a cluster JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("prompts", [])


def find_similar_pairs(
    train_prompts: list[str],
    test_prompts: list[str],
    n: int,
    threshold: float,
) -> list[tuple[int, int, float, str, str]]:
    """
    Find pairs of (train, test) prompts with n-gram similarity >= threshold.
    Returns list of (train_idx, test_idx, similarity, train_prompt, test_prompt).
    """
    similar_pairs = []

    # Precompute n-grams for efficiency
    train_ngrams = [get_ngrams(tokenize(p), n) for p in train_prompts]
    test_ngrams = [get_ngrams(tokenize(p), n) for p in test_prompts]

    for i, (train_ng, train_p) in enumerate(zip(train_ngrams, train_prompts)):
        for j, (test_ng, test_p) in enumerate(zip(test_ngrams, test_prompts)):
            sim = jaccard_similarity(train_ng, test_ng)
            if sim >= threshold:
                similar_pairs.append((i, j, sim, train_p, test_p))

    # Sort by similarity descending
    similar_pairs.sort(key=lambda x: -x[2])
    return similar_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Check for similar prompts between train and test clusters using n-gram similarity"
    )
    parser.add_argument(
        "--n", type=int, default=3,
        help="N-gram size (default: 3 for trigrams)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Jaccard similarity threshold (default: 0.5)"
    )
    parser.add_argument(
        "--train-dir", type=Path, default=Path("user_prompts/handpick"),
        help="Directory containing train cluster files"
    )
    parser.add_argument(
        "--test-dir", type=Path, default=Path("user_prompts/handpick_test"),
        help="Directory containing test cluster files"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional: save results to JSON file"
    )
    args = parser.parse_args()

    results = {}
    total_pairs = 0

    for cluster_idx in range(13):
        train_path = args.train_dir / f"cluster_{cluster_idx}.json"
        test_path = args.test_dir / f"cluster_{cluster_idx}.json"

        if not train_path.exists():
            print(f"Cluster {cluster_idx}: train file not found ({train_path})")
            continue
        if not test_path.exists():
            print(f"Cluster {cluster_idx}: test file not found ({test_path})")
            continue

        train_prompts = load_cluster(train_path)
        test_prompts = load_cluster(test_path)

        similar_pairs = find_similar_pairs(
            train_prompts, test_prompts, args.n, args.threshold
        )

        results[cluster_idx] = {
            "train_count": len(train_prompts),
            "test_count": len(test_prompts),
            "similar_pairs": [
                {
                    "train_idx": ti,
                    "test_idx": te,
                    "similarity": round(sim, 4),
                    "train_prompt": tp,
                    "test_prompt": tep,
                }
                for ti, te, sim, tp, tep in similar_pairs
            ],
        }

        n_similar = len(similar_pairs)
        total_pairs += n_similar

        if n_similar == 0:
            print(f"Cluster {cluster_idx}: No similar pairs (n={args.n}, threshold={args.threshold})")
        else:
            print(f"\nCluster {cluster_idx}: {n_similar} similar pair(s) (n={args.n}, threshold={args.threshold})")
            for ti, te, sim, tp, tep in similar_pairs[:5]:  # Show top 5
                print(f"  [{sim:.3f}] train[{ti}]: {tp[:80]}{'...' if len(tp) > 80 else ''}")
                print(f"           test[{te}]:  {tep[:80]}{'...' if len(tep) > 80 else ''}")
            if n_similar > 5:
                print(f"  ... and {n_similar - 5} more")

    print(f"\n{'='*60}")
    print(f"Total similar pairs across all clusters: {total_pairs}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {"n": args.n, "threshold": args.threshold, "clusters": results},
                f,
                indent=2,
            )
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
