#!/usr/bin/env python3
"""
Analyze baseline responses to compute what percentage exhibit each bias attribute.

Outputs a 13×3 table showing percentage of responses with:
- Affirmative opening (e.g., "Certainly!", "Sure!")
- Section headers (markdown # headers)
- List items (bullet points or numbered lists)
"""

import json
from pathlib import Path

from recall import AFFIRMATIVE_RE, SECTION_HEADER_RE, LIST_ITEM_RE


DATA_DIR = Path("data/baselines/handpick/train")


def analyze_cluster(cluster_path: Path) -> dict[str, tuple[int, int]]:
    """Analyze a cluster file and return counts for each attribute.

    Returns dict mapping attribute name to (count_with_attr, total_count).
    """
    with open(cluster_path) as f:
        data = json.load(f)

    counts = {
        "affirmative": 0,
        "headers": 0,
        "list": 0,
    }
    total = 0

    for prompt, responses in data.items():
        for resp_data in responses:
            response = resp_data["response"]
            total += 1

            if AFFIRMATIVE_RE.search(response):
                counts["affirmative"] += 1
            if SECTION_HEADER_RE.search(response):
                counts["headers"] += 1
            if LIST_ITEM_RE.search(response):
                counts["list"] += 1

    return {k: (v, total) for k, v in counts.items()}


def main():
    results = {}

    for cluster_id in range(13):
        cluster_path = DATA_DIR / f"cluster_{cluster_id}.json"
        if not cluster_path.exists():
            print(f"Warning: {cluster_path} not found")
            continue

        results[cluster_id] = analyze_cluster(cluster_path)

    # Print table header
    print(f"{'Cluster':>8} | {'Affirmative':>12} | {'Headers':>12} | {'List':>12}")
    print("-" * 55)

    # Print each row
    for cluster_id in sorted(results.keys()):
        counts = results[cluster_id]
        aff_count, aff_total = counts["affirmative"]
        hdr_count, hdr_total = counts["headers"]
        lst_count, lst_total = counts["list"]

        aff_pct = 100 * aff_count / aff_total if aff_total > 0 else 0
        hdr_pct = 100 * hdr_count / hdr_total if hdr_total > 0 else 0
        lst_pct = 100 * lst_count / lst_total if lst_total > 0 else 0

        print(f"{cluster_id:>8} | {aff_pct:>11.1f}% | {hdr_pct:>11.1f}% | {lst_pct:>11.1f}%")

    # Print summary row
    print("-" * 55)
    total_aff = sum(r["affirmative"][0] for r in results.values())
    total_hdr = sum(r["headers"][0] for r in results.values())
    total_lst = sum(r["list"][0] for r in results.values())
    total_all = sum(r["affirmative"][1] for r in results.values())

    print(f"{'Total':>8} | {100*total_aff/total_all:>11.1f}% | {100*total_hdr/total_all:>11.1f}% | {100*total_lst/total_all:>11.1f}%")
    print(f"\nTotal responses analyzed: {total_all}")


if __name__ == "__main__":
    main()
