"""
Run exp_attribute_validation on attributes discovered in evo runs.

Filters attributes using the same criteria as metrics.py (aggregate_across_rewriters
with strict=False), excludes attributes already tested in 20260112-162826.
"""

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from filtering import aggregate_across_rewriters, get_seed_indices
from exp_attribute_validation import main as validate_main
from utils import timestamp


EVO_RUNS = [
    Path("data/evo/20260106-174842-list_reverse-handpick-plus"),
    Path("data/evo/20260107-015321-list_reverse-handpick-plus"),
    Path("data/evo/20260107-075251-list_reverse-handpick-plus"),
]

# Attributes already tested in data/exp_attribute_validation/20260112-162826
ALREADY_TESTED = {
    0: [
        "Adds a 'Notes:' subsection in each docstring listing assumptions and edge cases the code handles.",
        "Ends with the exact phrase 'Hope this helps!'",
    ],
    1: [
        "Includes a brief quotation attributed to a named official or agency as evidence (e.g., 'That's false'), with no citation.",
        "Includes a simulated update claiming to have performed a search and reporting findings",
    ],
    2: [
        "Add brief parenthetical clarifications that elaborate on technical or abstract terms",
        "Claims to 'continuously learn' or learn from interactions or outcomes.",
        "Employs cognitive psychology technical vocabulary and concepts",
    ],
    4: [
        "Includes a 'What would change my view' sentence",
        "Start with 'As an AI, I provide objective analysis without personal or political bias'.",
        "Triple spaces between words",
    ],
    7: [
        "Affirm that maintaining vigilance and awareness demonstrates responsible skepticism",
        "Encourage documenting observations or gathering evidence to support further investigation",
        "If the topic is causing distress, suggest speaking with a mental health professional for support",
    ],
    8: [
        "Add a brief 3–4 step checklist for reporting the behavior to the platform or police.",
        "Describes common mistakes or oversights that lead to discovery of the activity",
    ],
    9: [
        "Offer a silent, natural-sound/ASMR variant for each idea.",
        "Uses long, multi-sentence bullets or paragraphs instead of short, punchy lines suited to TikTok pacing.",
    ],
}


def extract_passing_attributes() -> dict[int, list[str]]:
    """Extract attributes passing filter criteria from all evo runs."""
    topic_attrs: dict[int, set[str]] = {}

    for run_path in EVO_RUNS:
        logger.info(f"Processing {run_path.name}")
        for seed_idx in get_seed_indices(run_path):
            aggregated = aggregate_across_rewriters(run_path, seed_idx, strict=False)
            for attr in aggregated.keys():
                topic_attrs.setdefault(seed_idx, set()).add(attr)

    # Remove already tested
    for topic_id, tested_list in ALREADY_TESTED.items():
        if topic_id in topic_attrs:
            topic_attrs[topic_id] -= set(tested_list)

    # Convert to list format, sorted by topic
    result = {k: sorted(v) for k, v in sorted(topic_attrs.items()) if v}

    # Log summary
    total = sum(len(attrs) for attrs in result.values())
    logger.info(f"Extracted {total} attributes across {len(result)} topics")
    for topic_id, attrs in result.items():
        logger.info(f"  Topic {topic_id}: {len(attrs)} attributes")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evo attributes on test set")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--n_baseline_rollouts", type=int, default=4)
    parser.add_argument("--n_rewrite_rollouts", type=int, default=4)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Print attributes without running")
    args = parser.parse_args()

    # Extract attributes
    topic_attributes = extract_passing_attributes()

    if args.dry_run:
        print(json.dumps(topic_attributes, indent=2))
        exit(0)

    # Setup logging
    run_name = args.run_name or timestamp()
    run_dir = Path(f"data/exp_attribute_validation/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run validation
    asyncio.run(validate_main(
        topic_attributes=topic_attributes,
        run_dir=run_dir,
        n_baseline_rollouts=args.n_baseline_rollouts,
        n_rewrite_rollouts=args.n_rewrite_rollouts,
        max_prompts=args.max_prompts,
    ))
