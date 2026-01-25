#!/usr/bin/env python3
"""Compare per-seed vs global Bonferroni correction methods."""

import json
from pathlib import Path

from filtering import (
    PARTIAL_CONJUNCTION_P_THRESHOLD,
    compute_partial_conjunction_stats,
    get_seed_indices,
    passes_partial_conjunction_criteria,
    save_partial_conjunction_results,
)


def main():
    run_path = Path("data/exp_attribute_validation/20260112-162826")
    existing_path = run_path / "partial_conjunction_results.json"
    tmp_path = Path("/tmp/partial_conjunction_results.json")

    # Step 1: Regenerate and compare with existing file
    print("=" * 80)
    print("STEP 1: Verify save_partial_conjunction_results matches existing file")
    print("=" * 80)

    save_partial_conjunction_results(run_path, tmp_path)

    with open(existing_path) as f:
        existing = json.load(f)
    with open(tmp_path) as f:
        regenerated = json.load(f)

    # Compare (ignore run_path which may differ, and ordering differences)
    existing_seeds = existing["seeds"]
    regenerated_seeds = regenerated["seeds"]

    # Convert to comparable format (dict by attribute name)
    def to_attr_dict(seed_list):
        return {item["attribute"]: item for item in seed_list}

    matches = True
    diffs = []
    for seed_idx in existing_seeds:
        if seed_idx not in regenerated_seeds:
            diffs.append(f"  Missing seed {seed_idx} in regenerated")
            matches = False
            continue

        existing_attrs = to_attr_dict(existing_seeds[seed_idx])
        regenerated_attrs = to_attr_dict(regenerated_seeds[seed_idx])

        if set(existing_attrs.keys()) != set(regenerated_attrs.keys()):
            diffs.append(f"  Seed {seed_idx}: different attribute sets")
            matches = False
            continue

        for attr_name in existing_attrs:
            e = existing_attrs[attr_name]
            r = regenerated_attrs[attr_name]
            if e != r:
                matches = False
                # Check if pass/fail status changed
                if e["passes_criteria"] != r["passes_criteria"]:
                    diffs.append(
                        f"  Seed {seed_idx}: '{attr_name[:40]}...' "
                        f"changed from {e['passes_criteria']} to {r['passes_criteria']}"
                    )

    if matches:
        print("\n✓ Regenerated results MATCH existing file (content equivalent)")
    else:
        print("\n✗ Regenerated results have content differences (p-values differ)")
        print("  Note: The existing file was likely generated when rollouts data differed")
        if diffs:
            print("\n  Pass/fail status changes:")
            for d in diffs:
                print(d)
        else:
            print("  (No changes to pass/fail status)")

    # Step 2: Compute global N and compare pass counts
    print("\n" + "=" * 80)
    print("STEP 2: Compare per-seed vs global Bonferroni correction")
    print("=" * 80)

    seed_indices = get_seed_indices(run_path)

    # First pass: count total attributes (global N)
    all_stats = {}
    total_attributes = 0
    for seed_idx in seed_indices:
        stats = compute_partial_conjunction_stats(run_path, seed_idx)
        all_stats[seed_idx] = stats
        total_attributes += len(stats)

    print(f"\nTotal attributes across all seeds: {total_attributes}")
    print(f"P-value threshold: {PARTIAL_CONJUNCTION_P_THRESHOLD}")

    # Count passes under each method
    per_seed_passes = []
    global_passes = []

    print("\n" + "-" * 80)
    print("Per-attribute comparison (✓/✗ = per-seed/global pass status):")
    print("-" * 80)

    for seed_idx in seed_indices:
        stats = all_stats[seed_idx]
        n_per_seed = len(stats)

        print(f"\nSeed {seed_idx} (N per-seed={n_per_seed}, N global={total_attributes}):")

        for attr, data in stats.items():
            # Per-seed Bonferroni (already computed in data)
            per_seed_pass = passes_partial_conjunction_criteria(
                data["student_p_pc_bonferroni"],
                data["teacher_p_pc_bonferroni"],
                data["student_mean"],
                data["teacher_mean"],
            )

            # Global Bonferroni (recompute with global N)
            student_p_pc_global = min(data["student_p_pc"] * total_attributes, 1.0)
            teacher_p_pc_global = min(data["teacher_p_pc"] * total_attributes, 1.0)
            global_pass = passes_partial_conjunction_criteria(
                student_p_pc_global,
                teacher_p_pc_global,
                data["student_mean"],
                data["teacher_mean"],
            )

            if per_seed_pass:
                per_seed_passes.append(attr)
            if global_pass:
                global_passes.append(attr)

            status_per = "✓" if per_seed_pass else "✗"
            status_global = "✓" if global_pass else "✗"
            change = ""
            if per_seed_pass and not global_pass:
                change = " ← LOST"
            elif not per_seed_pass and global_pass:
                change = " ← GAINED"

            print(f"  {status_per}/{status_global} {attr[:65]}{change}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nPer-seed Bonferroni: {len(per_seed_passes)} attributes pass")
    print(f"Global Bonferroni:   {len(global_passes)} attributes pass")

    lost = set(per_seed_passes) - set(global_passes)
    gained = set(global_passes) - set(per_seed_passes)

    if lost:
        print(f"\nLost with global Bonferroni ({len(lost)}):")
        for attr in lost:
            print(f"  - {attr}")

    if gained:
        print(f"\nGained with global Bonferroni ({len(gained)}):")
        for attr in gained:
            print(f"  - {attr}")

    if not lost and not gained:
        print("\nNo change in passing attributes between methods")


if __name__ == "__main__":
    main()
