# ABOUTME: Validates whether rewrites faithfully exhibit the intended attributes.
# ABOUTME: Uses a judge model to check attribute presence in rewritten responses.

import json
import html
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from loguru import logger

from api_models import JudgeModel


async def validate_fidelity(
    judge_model: JudgeModel,
    rollouts: dict[str, dict[str, list[dict]]],
    n_rollouts: int = 4,
) -> dict[str, dict[str, list[bool]]]:
    """
    Validate whether rewrites exhibit the intended attributes.

    Args:
        judge_model: JudgeModel for presence checking
        rollouts: {attribute: {user_prompt: [rollout, ...]}}
        n_rollouts: Number of rollouts per user prompt to judge

    Returns:
        {attribute: {user_prompt: [presence_bool, ...]}}
    """
    # Build all (attribute, response) pairs
    pairs = []
    pair_indices = []  # Track (attribute, user_prompt, rollout_idx) for each pair

    for attribute, user_prompts in rollouts.items():
        for user_prompt, rollout_list in user_prompts.items():
            for rollout_idx, rollout in enumerate(rollout_list[:n_rollouts]):
                if rollout is not None and "response" in rollout:
                    pairs.append((attribute, rollout["response"]))
                    pair_indices.append((attribute, user_prompt, rollout_idx))

    if not pairs:
        return {}

    # Judge all pairs in batch
    presence_results = await judge_model.judge_presence(pairs)

    # Reconstruct results by attribute and user_prompt
    results: dict[str, dict[str, list[bool]]] = {}
    for (attribute, user_prompt, rollout_idx), is_present in zip(pair_indices, presence_results):
        if attribute not in results:
            results[attribute] = {}
        if user_prompt not in results[attribute]:
            results[attribute][user_prompt] = []
        if is_present is None:
            continue
        results[attribute][user_prompt].append(is_present)

    return results


def compute_fidelity_stats(
    presence_results: dict[str, dict[str, list[bool]]],
) -> dict[str, dict]:
    """
    Compute fidelity statistics for each attribute.

    Returns:
        {attribute: {"present": int, "total": int, "rate": float}}
    """
    stats = {}
    for attribute, user_prompts in presence_results.items():
        present = 0
        total = 0
        for _, results in user_prompts.items():
            present += sum(results)
            total += len(results)

        stats[attribute] = {
            "present": present,
            "total": total,
            "rate": present / total if total > 0 else 0.0,
        }

    return stats


def plot_fidelity(
    stats: dict[str, dict],
    save_path: Path,
):
    """Create bar chart showing fidelity rates for each attribute."""

    def wrap_text(text, width=60):
        """Wrap text to specified width using <br> for line breaks"""
        text = html.escape(text)
        text = text.replace("$", "&#36;")
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        if current_line:
            lines.append(" ".join(current_line))
        return "<br>".join(lines)

    attributes = sorted(stats.keys(), key=lambda a: stats[a]["rate"], reverse=True)
    rates = [stats[a]["rate"] * 100 for a in attributes]

    # Wrap attribute names for display
    display_names = [wrap_text(a, width=60) for a in attributes]

    fig = go.Figure(
        data=[
            go.Bar(
                x=rates,
                y=display_names,
                orientation='h',
                marker_color='steelblue',
                text=[f"{r:.1f}%" for r in rates],
                textposition='auto',
            )
        ]
    )

    # Calculate height based on number of attributes and wrapped text
    n_attributes = len(attributes)
    plot_height = max(500, min(1600, 100 + n_attributes * 80))

    fig.update_layout(
        title="Attribute Presence Rate in Rewrites",
        xaxis_title="Presence Rate (%)",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(automargin=True, tickfont=dict(size=10)),
        height=plot_height,
        width=1200,
        margin=dict(l=10, r=40, t=60, b=40),
    )

    fig.write_image(save_path)
    print(f"Saved fidelity plot to {save_path}")


def format_attribute_stats(item: dict) -> str:
    """Format student WR and teacher WR stats for display."""
    parts = []

    # Student win rate (0-1)
    stu_wr = item.get("student_winrate")
    if stu_wr is not None:
        parts.append(f"Student WR: {stu_wr:.2f}")

    # Teacher win rate (0-1)
    tch_wr = item.get("teacher_winrate")
    if tch_wr is not None:
        parts.append(f"Teacher WR: {tch_wr:.2f}")

    if parts:
        return f"<i>({', '.join(parts)})</i>"
    return ""


def plot_comparison_table(
    comparison_table: dict[str, dict],
    save_path: Path,
    attribute_stats: list[dict] | None = None,
    seed_index: str | None = None,
    seed_summary: str | None = None,
):
    """
    Create horizontal grouped bar chart comparing baseline vs rewrite presence rates.

    Baseline rates should be low (attribute not present in original responses).
    Rewrite rates should be high (attribute successfully added by rewriter).

    Args:
        comparison_table: {attribute: {baseline_rate, rewrite_rate, ...}}
        save_path: Where to save the plot
        attribute_stats: List of candidate_stats dicts with student_winrate and teacher_winrate
        seed_index: Seed index for title
        seed_summary: Seed cluster summary for title
    """

    def wrap_text(text, width=60):
        text = html.escape(text)
        text = text.replace("$", "&#36;")
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        if current_line:
            lines.append(" ".join(current_line))
        return "<br>".join(lines)

    # Build attribute -> stats lookup from list
    stats_lookup = {}
    if attribute_stats:
        for item in attribute_stats:
            attr = item.get("attribute")
            if attr:
                stats_lookup[attr] = item

    # Sort alphabetically (reverse so A at top)
    attributes = sorted(comparison_table.keys(), reverse=True)

    baseline_rates = [comparison_table[a]["baseline_rate"] * 100 for a in attributes]
    rewrite_rates = [comparison_table[a]["rewrite_rate"] * 100 for a in attributes]

    # Build display names with stats
    display_names = []
    for attr in attributes:
        label = wrap_text(attr, width=55)
        if attr in stats_lookup:
            stats_str = format_attribute_stats(stats_lookup[attr])
            if stats_str:
                label += f"<br>{stats_str}"
            # Red flag: student WR > 0.5 AND teacher WR < 0.5
            stu_wr = stats_lookup[attr].get("student_winrate")
            tch_wr = stats_lookup[attr].get("teacher_winrate")
            if stu_wr is not None and tch_wr is not None and stu_wr > 0.5 and tch_wr < 0.5:
                label = f"<span style='color:red'>{label}</span>"
        display_names.append(label)

    fig = go.Figure()

    # Baseline bars (should be low - red/orange color)
    fig.add_trace(
        go.Bar(
            x=baseline_rates,
            y=display_names,
            name="Baseline",
            orientation='h',
            marker_color='rgb(255, 127, 14)',  # Orange
            text=[f"{r:.0f}%" for r in baseline_rates],
            textposition='auto',
        )
    )

    # Rewrite bars (should be high - blue/green color)
    fig.add_trace(
        go.Bar(
            x=rewrite_rates,
            y=display_names,
            name="Rewrite",
            orientation='h',
            marker_color='rgb(31, 119, 180)',  # Blue
            text=[f"{r:.0f}%" for r in rewrite_rates],
            textposition='auto',
        )
    )

    n_attributes = len(attributes)
    plot_height = max(500, min(1600, 100 + n_attributes * 80))

    # Build title with seed info if available
    if seed_index is not None and seed_summary is not None:
        title_text = f"Seed {seed_index} ({seed_summary})<br>Attribute Presence: Baseline vs Rewrite"
    else:
        title_text = "Attribute Presence: Baseline vs Rewrite"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18)),
        xaxis_title="Presence Rate (%)",
        xaxis=dict(range=[0, 105], tickfont=dict(size=12)),
        yaxis=dict(automargin=True, tickfont=dict(size=12)),
        height=plot_height,
        width=1000,
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        margin=dict(l=10, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.write_image(save_path)
    print(f"Saved comparison plot to {save_path}")


if __name__ == "__main__":
    import asyncio

    run_path = Path("data/evo/20251228-165744-list_reverse-handpick-plus")
    run_name = run_path.name
    n_rollouts = 4  # Judge first N rollouts per user prompt

    judge_model = JudgeModel(
        model_name="openai/gpt-5-mini",
        max_tokens=8192,
        reasoning="medium",
        max_par=512,
    )

    async def main():
        # # Logging
        # Path("logs/rewrite_val_fidelity").mkdir(parents=True, exist_ok=True)
        # logger.remove()
        # logger.add(
        #     f"logs/rewrite_val_fidelity/{run_name}.log",
        #     enqueue=True, level="INFO",
        #     retention="7 days"
        # )

        save_dir = Path(f"data/rewrite_val_fidelity/{run_name}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load config to get seed indices
        with open(run_path / "config.json", "r") as f:
            run_config = json.load(f)

        # Load baselines (shared across seeds)
        with open(run_path / "val_baselines/rollouts.json", "r") as f:
            baselines = json.load(f)

        all_comparison_tables = {}

        for seed_index in run_config["prompts"].keys():
            print(f"Validating seed {seed_index}...")

            # Get seed summary from config
            seed_summary = run_config["prompts"][seed_index].get("summary", "")

            # Load attributes for this seed
            with open(run_path / f"validate/seed_{seed_index}_validate/candidate_stats.json", "r") as f:
                candidate_stats = json.load(f)
            attributes = [c["attribute"] for c in candidate_stats]

            # Load rewrite rollouts for this seed
            rollouts_path = run_path / f"validate/seed_{seed_index}_validate/rollouts.json"
            with open(rollouts_path, "r") as f:
                rewrite_rollouts = json.load(f)

            # Restructure baselines to match rollouts format: {attr: {user_prompt: [rollouts]}}
            baseline_as_rollouts = {attr: baselines for attr in attributes}

            # # Validate presence in rewrites (should be HIGH)
            # print("  Checking rewrite presence...")
            # rewrite_presence_results = await validate_fidelity(
            #     judge_model=judge_model,
            #     rollouts=rewrite_rollouts,
            #     n_rollouts=n_rollouts,
            # )
            # rewrite_stats = compute_fidelity_stats(rewrite_presence_results)

            # # Validate presence in baselines (should be LOW)
            # print("  Checking baseline presence...")
            # baseline_presence_results = await validate_fidelity(
            #     judge_model=judge_model,
            #     rollouts=baseline_as_rollouts,
            #     n_rollouts=n_rollouts,
            # )
            # baseline_stats = compute_fidelity_stats(baseline_presence_results)

            # # Create comparison table
            # comparison_table = {}
            # for attr in attributes:
            #     comparison_table[attr] = {
            #         "baseline_rate": baseline_stats.get(attr, {}).get("rate", 0.0),
            #         "rewrite_rate": rewrite_stats.get(attr, {}).get("rate", 0.0),
            #         "baseline_present": baseline_stats.get(attr, {}).get("present", 0),
            #         "baseline_total": baseline_stats.get(attr, {}).get("total", 0),
            #         "rewrite_present": rewrite_stats.get(attr, {}).get("present", 0),
            #         "rewrite_total": rewrite_stats.get(attr, {}).get("total", 0),
            #     }
            # all_comparison_tables[seed_index] = comparison_table

            # Save results
            seed_save_dir = save_dir / f"seed_{seed_index}"
            seed_save_dir.mkdir(parents=True, exist_ok=True)

            # with open(seed_save_dir / "baseline_presence_results.json", "w") as f:
            #     json.dump(baseline_presence_results, f, indent=4, sort_keys=True)

            # with open(seed_save_dir / "rewrite_presence_results.json", "w") as f:
            #     json.dump(rewrite_presence_results, f, indent=4, sort_keys=True)

            # with open(seed_save_dir / "baseline_stats.json", "w") as f:
            #     json.dump(baseline_stats, f, indent=4, sort_keys=True)

            # with open(seed_save_dir / "rewrite_stats.json", "w") as f:
            #     json.dump(rewrite_stats, f, indent=4, sort_keys=True)

            # with open(seed_save_dir / "comparison_table.json", "w") as f:
            #     json.dump(comparison_table, f, indent=4, sort_keys=True)

            with open(seed_save_dir / "comparison_table.json", "r") as f:
                comparison_table = json.load(f)

            # Plot comparison with candidate stats for WR display
            plot_comparison_table(
                comparison_table,
                seed_save_dir / "comparison.pdf",
                attribute_stats=candidate_stats,
                seed_index=seed_index,
                seed_summary=seed_summary,
            )

            # Print summary
            avg_baseline = np.mean([s["baseline_rate"] for s in comparison_table.values()])
            avg_rewrite = np.mean([s["rewrite_rate"] for s in comparison_table.values()])
            print(f"  Seed {seed_index}: baseline={avg_baseline:.1%}, rewrite={avg_rewrite:.1%}")

            print(f"Finished validating seed {seed_index}")

        # Save aggregate comparison tables
        with open(save_dir / "all_comparison_tables.json", "w") as f:
            json.dump(all_comparison_tables, f, indent=4, sort_keys=True)

        print("\nDone!")

    asyncio.run(main())
