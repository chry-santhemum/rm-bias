import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Literal
import plotly.express as px
from state import adversariality

st.set_page_config(page_title="Run Visualization", layout="wide")

# Make disabled textareas render with black text (read-only but not greyed out)
st.markdown(
    """
    <style>
    .stTextArea textarea:disabled, textarea[disabled] {
        color: #000 !important;
        -webkit-text-fill-color: #000 !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _adaptive_table_height(
    num_rows: int,
    min_height: int = 104,
    max_height: int = 400,
    row_height: int = 32,
    header: int = 40,
    padding: int = 16,
) -> int:
    estimated = header + padding + max(1, num_rows) * row_height
    return max(min_height, min(max_height, estimated))


def _dynamic_text_height(
    text: str, min_height: int = 80, max_height: int = 400, chars_per_line: int = 140
) -> int:
    if not text:
        return min_height
    explicit_lines = text.count("\n") + 1
    wrapped_lines = sum(
        max(
            1,
            len(line) // chars_per_line + (1 if len(line) % chars_per_line > 0 else 0),
        )
        for line in text.split("\n")
    )
    total_lines = max(explicit_lines, wrapped_lines)
    base_height = 50
    line_height = 22
    calculated_height = base_height + (total_lines * line_height)
    return max(min_height, min(max_height, calculated_height))


def _render_selectable_df(
    df: pd.DataFrame,
    drop_cols: List[str] | None = None,
    min_height: int = 100,
    max_height: int = 400,
    selection_mode: Literal[
        "single-row", "multi-row", "single-column", "multi-column"
    ] = "single-row",
    key: str | None = None,
):
    df_to_show = df.drop(columns=drop_cols, errors="ignore") if drop_cols else df
    return st.dataframe(
        df_to_show,
        use_container_width=True,
        height=_adaptive_table_height(
            len(df_to_show), min_height=min_height, max_height=max_height
        ),
        on_select="rerun",
        selection_mode=selection_mode,
        hide_index=True,
        key=key,
    )  # type: ignore[call-arg]


@st.cache_data
def load_run_data(run_path_str: str) -> Dict[str, Any]:
    """Load all data for a run from the file structure."""
    run_path = Path(run_path_str)
    data = {"seed_states": {}, "metadata": {}}

    if not run_path.exists():
        return data

    # Load each seed state directory
    for seed_dir in run_path.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            seed_id = seed_dir.name.replace("seed_", "")
            data["seed_states"][seed_id] = load_seed_state_data(seed_dir)

    return data


def load_seed_state_data(seed_dir: Path) -> Dict[str, Any]:
    """Load data for a single seed state."""
    seed_data = {
        "system_prompts": {},
        "cluster_info": None,
        "population_history": None,
        "step_count": 0,
    }

    # Load cluster info if exists
    cluster_file = seed_dir / "cluster_info.json"
    if cluster_file.exists():
        with open(cluster_file, "r") as f:
            seed_data["cluster_info"] = json.load(f)

    # Load population history if exists
    population_file = seed_dir / "population_history.json"
    if population_file.exists():
        with open(population_file, "r") as f:
            seed_data["population_history"] = json.load(f)

    # Load all system prompt files
    for prompt_file in seed_dir.glob("*.json"):
        if prompt_file.name not in ["cluster_info.json", "population_history.json"]:
            try:
                with open(prompt_file, "r") as f:
                    prompt_data = json.load(f)
                    prompt_hash = prompt_file.stem
                    seed_data["system_prompts"][prompt_hash] = prompt_data
                    # Track latest step
                    meta = prompt_data.get("meta", {})
                    if "step" in meta:
                        seed_data["step_count"] = max(
                            seed_data["step_count"], meta["step"]
                        )
            except (json.JSONDecodeError, IOError):
                continue  # Skip corrupted files

    return seed_data


def _extract_rater_names_from_attack(attack: Dict[str, Any]) -> List[str]:
    rater_models = []
    seen = set()
    for response in attack.get("responses", []):
        for rating in response.get("ratings", []):
            model_name = rating.get("rater", {}).get("model_name")
            if model_name and model_name not in seen:
                seen.add(model_name)
                rater_models.append(model_name)
    return rater_models


def _mean_scores_for_attack_by_rater(
    attack: Dict[str, Any],
) -> Dict[str, Tuple[float | None, float | None]]:
    """Return mapping rater_name -> (mean_raw_score, mean_normalized_score)."""
    result: Dict[str, Tuple[float | None, float | None]] = {}
    rater_names = _extract_rater_names_from_attack(attack)
    for rater in rater_names:
        raw_scores: List[float] = []
        norm_scores: List[float] = []
        for response in attack.get("responses", []):
            for rating in response.get("ratings", []):
                r = rating.get("rater", {})
                if r.get("model_name") == rater:
                    raw = rating.get("raw_score")
                    norm = rating.get("aux_info", {}).get("normalized_score")
                    if isinstance(raw, (int, float)):
                        raw_scores.append(float(raw))
                    if isinstance(norm, (int, float)):
                        norm_scores.append(float(norm))
        mean_raw = float(pd.Series(raw_scores).mean()) if raw_scores else None
        mean_norm = float(pd.Series(norm_scores).mean()) if norm_scores else None
        result[rater] = (mean_raw, mean_norm)
    return result


def _default_rater_pair_for_attack(
    attack: Dict[str, Any],
) -> Tuple[str | None, str | None]:
    """Auto-detect default rater pair as per state.Attack.adversarial_score default.
    Uses the first response's first two ratings if present.
    """
    responses = attack.get("responses", [])
    if not responses:
        return (None, None)
    ratings = responses[0].get("ratings", [])
    if len(ratings) >= 2:
        r1 = ratings[0].get("rater", {}).get("model_name")
        r2 = ratings[1].get("rater", {}).get("model_name")
        return (r1, r2)
    return (None, None)


def _adversarial_score_for_attack(
    attack: Dict[str, Any], r1: str | None = None, r2: str | None = None
) -> float | None:
    if r1 is None or r2 is None:
        r1, r2 = _default_rater_pair_for_attack(attack)
    if not r1 or not r2:
        return None
    means = _mean_scores_for_attack_by_rater(attack)
    z1 = means.get(r1, (None, None))[1]
    z2 = means.get(r2, (None, None))[1]
    if z1 is None or z2 is None:
        return None
    try:
        return float(adversariality(z_score_1=float(z1), z_score_2=float(z2)))
    except Exception:
        return None


def _compute_system_prompt_adv_stats(
    prompt_data: Dict[str, Any],
) -> Tuple[float | None, float | None]:
    """Compute mean and stdev adversarial score across attacks for a system prompt."""
    attacks = prompt_data.get("attacks", [])
    adv_scores: List[float] = []
    for attack in attacks:
        adv = _adversarial_score_for_attack(attack)
        if isinstance(adv, (int, float)):
            adv_scores.append(float(adv))
    if not adv_scores:
        return (None, None)
    series = pd.Series(adv_scores)
    return (float(series.mean()), float(series.std(ddof=0)))


def display_system_prompt_details(prompt_data: Dict[str, Any], prompt_hash: str):
    """Display detailed view of a system prompt and its attacks."""
    st.subheader("System Prompt")

    # Show system prompt text with dynamic height
    system_prompt_text = prompt_data.get("system_prompt", "")
    system_height = _dynamic_text_height(
        system_prompt_text, min_height=80, max_height=400
    )
    st.text_area(
        "System Prompt Text", system_prompt_text, height=system_height, disabled=True
    )

    # Show metadata
    meta = prompt_data.get("meta", {})
    if meta:
        st.subheader("Metadata")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Step", meta.get("step", "N/A"))
        with col2:
            st.metric("Operation", meta.get("operation", "N/A"))
        with col3:
            st.metric("Mean Score", f"{prompt_data.get('mean_score', 0):.3f}")
        with col4:
            st.metric("Std Dev", f"{prompt_data.get('stdev_score', 0):.3f}")

        # Show additional meta info
        other_meta = {k: v for k, v in meta.items() if k not in ["step", "operation"]}
        if other_meta:
            with st.expander("Additional Metadata"):
                st.json(other_meta)
    else:
        # Fallback if no meta
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Score", f"{prompt_data.get('mean_score', 0):.3f}")
        with col2:
            st.metric("Std Dev", f"{prompt_data.get('stdev_score', 0):.3f}")
        with col3:
            st.metric("Num Attacks", len(prompt_data.get("attacks", [])))

    # Show attacks in expandable sections compatible with new Attack schema
    attacks = prompt_data.get("attacks", [])
    if attacks:
        st.subheader("Attacks")

        # Determine the union of rater names across all attacks for columns
        all_raters: List[str] = []
        seen_raters = set()
        for atk in attacks:
            for rn in _extract_rater_names_from_attack(atk):
                if rn not in seen_raters:
                    seen_raters.add(rn)
                    all_raters.append(rn)

        # Build attack overview rows
        attack_rows = []
        for i, attack in enumerate(attacks):
            user_prompt = attack.get("user", "")
            row = {
                "Attack #": i + 1,
                "User Prompt": user_prompt[:100]
                + ("..." if len(user_prompt) > 100 else ""),
            }

            means = _mean_scores_for_attack_by_rater(attack)
            for rater in all_raters:
                mean_raw, mean_norm = means.get(rater, (None, None))
                row[f"Mean Raw | {rater}"] = (
                    f"{mean_raw:.3f}" if isinstance(mean_raw, (int, float)) else "N/A"
                )
                row[f"Mean Norm | {rater}"] = (
                    f"{mean_norm:.3f}" if isinstance(mean_norm, (int, float)) else "N/A"
                )

            adv = _adversarial_score_for_attack(attack)
            row["Adversarial Score"] = (
                f"{adv:.3f}" if isinstance(adv, (int, float)) else "N/A"
            )
            attack_rows.append(row)

        attack_df = pd.DataFrame(attack_rows)

        # Display attacks table with selection
        selected_attack = _render_selectable_df(attack_df)

        # Show detailed view of selected attack - list of RatedResponses
        if selected_attack["selection"]["rows"]:  # type: ignore
            selected_idx = selected_attack["selection"]["rows"][0]  # type: ignore
            attack = attacks[selected_idx]

            st.subheader(f"Attack #{selected_idx + 1} Details")

            # User prompt textbox (monospace, raw)
            user_text = attack.get("user", "")
            user_height = _dynamic_text_height(user_text, min_height=80, max_height=400)
            st.subheader("User Prompt")
            st.text_area(
                "User Prompt Text", user_text, height=user_height, disabled=True
            )

            # RatedResponses table
            responses = attack.get("responses", [])
            # Columns per rater for raw and norm
            raters = _extract_rater_names_from_attack(attack)

            rr_rows = []
            for ridx, response in enumerate(responses):
                assistant_text = response.get("assistant", "")
                row = {
                    "Response #": ridx + 1,
                    "Assistant Preview": assistant_text[:100]
                    + ("..." if len(assistant_text) > 100 else ""),
                    "Response Length": len(assistant_text),
                    "Index": ridx,
                }
                # Build a quick index of ratings by rater name
                ratings_by_rater: Dict[str, Dict[str, Any]] = {}
                for rating in response.get("ratings", []):
                    rn = rating.get("rater", {}).get("model_name")
                    if rn:
                        ratings_by_rater[rn] = rating
                for r in raters:
                    rating = ratings_by_rater.get(r)
                    raw = rating.get("raw_score") if rating else None
                    norm = (
                        rating.get("aux_info", {}).get("normalized_score")
                        if rating
                        else None
                    )
                    row[f"Raw | {r}"] = (
                        f"{raw:.3f}" if isinstance(raw, (int, float)) else "N/A"
                    )
                    row[f"Norm | {r}"] = (
                        f"{norm:.3f}" if isinstance(norm, (int, float)) else "N/A"
                    )
                rr_rows.append(row)

            rr_df = pd.DataFrame(rr_rows)
            selected_rr = _render_selectable_df(
                rr_df,
                drop_cols=["Index"],
            )

            # Show full response text for selected row
            if selected_rr["selection"]["rows"]:  # type: ignore
                selected_row = selected_rr["selection"]["rows"][0]  # type: ignore
                original_idx = rr_df.iloc[selected_row]["Index"]
                full_text = responses[original_idx].get("assistant", "")

                full_height = _dynamic_text_height(
                    full_text, min_height=120, max_height=720
                )
                st.subheader(f"Response #{int(original_idx) + 1} Full Text")
                st.text_area(
                    "Assistant Full Response",
                    full_text,
                    height=full_height,
                    disabled=True,
                )

                # Ratings aux info per rater
                ratings = responses[original_idx].get("ratings", [])
                if ratings:
                    st.subheader("Ratings Aux Info")
                    for ridx, rating in enumerate(ratings):
                        rater_name = (
                            rating.get("rater", {}).get("model_name")
                            or f"Rater {ridx + 1}"
                        )
                        with st.expander(f"{rater_name}"):
                            st.json(rating.get("aux_info", {}))


def main():
    st.title("Reward model auto red-teaming")

    # Initialize session state
    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = 0
    if "selected_run_idx" not in st.session_state:
        st.session_state.selected_run_idx = 0
    if "selected_seed" not in st.session_state:
        st.session_state.selected_seed = None

    # Sidebar for run selection
    st.sidebar.header("Run Selection")

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    # Directory selection first (support absolute and relative run paths)
    data_dirs = [
        "data/evo",
        "data/bon_iter",
        "data/pair",
        "data/one_turn",
    ]
    dir_entries = []
    for label in data_dirs:
        abs_path = Path("/workspace/rm-bias") / label
        if abs_path.exists():
            dir_entries.append((label, abs_path))
        else:
            rel_path = Path(label)
            if rel_path.exists():
                dir_entries.append((label, rel_path))

    if not dir_entries:
        st.error(
            "No training directories found in data/evo, data/bon_iter, data/pair, or data/one_turn"
        )
        return

    selected_dir_entry = st.sidebar.selectbox(
        "Select Directory",
        dir_entries,
        format_func=lambda x: x[0],
        help="Choose between evolutionary (evo), best-of-N (bon_iter), PAIR, or one_turn runs",
    )

    selected_directory = selected_dir_entry[0]  # type: ignore

    # Now get runs from the selected directory
    dir_path = selected_dir_entry[1]  # type: ignore
    available_runs = []

    if dir_path.exists():
        for run_dir in dir_path.iterdir():
            if run_dir.is_dir():
                available_runs.append((run_dir.name, run_dir))

    if not available_runs:
        st.error(f"No runs found in {selected_directory}")
        return

    selected_run_name, selected_run_path = st.sidebar.selectbox(  # type: ignore
        "Select Run",
        available_runs,
        format_func=lambda x: x[0],
        help="Choose specific run from the selected directory",
    )

    # Load and display run data
    run_data = load_run_data(str(selected_run_path))

    if not run_data["seed_states"]:
        st.warning("No seed state data found for this run")
        return

    # Main tabs with session state - add evolutionary tab if this is an evo run
    tab_names = ["📊 Overview", "🔍 Explore Prompts", "📈 Analytics"]
    if selected_directory == "data/evo":
        tab_names.append("🧬 Evolution")

    # Use radio buttons instead of tabs to maintain state
    selected_tab = st.radio(
        "Navigation",
        tab_names,
        index=min(st.session_state.selected_tab, len(tab_names) - 1),
        horizontal=True,
        key="main_tabs",
        label_visibility="collapsed",
    )
    st.session_state.selected_tab = tab_names.index(selected_tab)

    st.divider()

    if selected_tab == "📊 Overview":
        st.header("Training Overview")

        # Show basic stats
        num_seeds = len(run_data["seed_states"])
        total_prompts = sum(
            len(seed["system_prompts"]) for seed in run_data["seed_states"].values()
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Seed States", num_seeds)
        with col2:
            st.metric("Total System Prompts", total_prompts)
        with col3:
            max_step = (
                max(seed["step_count"] for seed in run_data["seed_states"].values())
                if run_data["seed_states"]
                else 0
            )
            st.metric("Current Step", max_step)

        # Show seed state summaries
        for seed_id, seed_data in run_data["seed_states"].items():
            cluster_info = seed_data.get("cluster_info", {})
            cluster_summary = cluster_info.get("summary", f"Seed {seed_id}")

            with st.expander(f"Seed {seed_id}: {cluster_summary}"):
                st.write(f"**Prompts Generated**: {len(seed_data['system_prompts'])}")
                st.write(f"**Current Step**: {seed_data['step_count']}")

                if cluster_info:
                    st.write(
                        f"**Train Batch Size**: {cluster_info.get('train_batch_size', 'N/A')}"
                    )

                    sample_prompts = cluster_info.get("sample_train_prompts", [])
                    if sample_prompts:
                        st.write("**Sample User Prompts**:")
                        for prompt in sample_prompts[:15]:
                            st.write(f"- {prompt}")

    elif selected_tab == "🔍 Explore Prompts":
        st.header("Explore System Prompts")

        # Seed state selector
        seed_options = list(run_data["seed_states"].keys())
        # Preserve seed selection across refreshes
        if st.session_state.selected_seed is None and seed_options:
            st.session_state.selected_seed = seed_options[0]

        selected_seed = st.selectbox(
            "Select Seed State",
            seed_options,
            index=(
                seed_options.index(st.session_state.selected_seed)
                if st.session_state.selected_seed in seed_options
                else 0
            ),
            key="seed_selector",
        )
        st.session_state.selected_seed = selected_seed

        if selected_seed:
            seed_data = run_data["seed_states"][selected_seed]
            prompts = seed_data["system_prompts"]

            if prompts:
                # Sort prompts by mean score
                sorted_prompts = sorted(
                    prompts.items(),
                    key=lambda x: x[1].get("mean_score", 0),
                    reverse=True,
                )

                # Show top prompts summary
                st.subheader(f"System Prompts for Seed {selected_seed}")

                # Create overview DataFrame with mean_adversarial_score and stdev_adversarial_score
                overview_data = []
                for prompt_hash, prompt_data in sorted_prompts:
                    meta = prompt_data.get("meta", {})
                    mean_adv, std_adv = _compute_system_prompt_adv_stats(prompt_data)
                    mean_display = (
                        f"{mean_adv:.3f}"
                        if isinstance(mean_adv, (int, float))
                        else "N/A"
                    )
                    std_display = (
                        f"{std_adv:.3f}" if isinstance(std_adv, (int, float)) else "N/A"
                    )
                    overview_data.append(
                        {
                            "Step": meta.get("step", "N/A"),
                            "Operation": meta.get("operation", "N/A"),
                            "System Prompt": (
                                prompt_data["system_prompt"][:80] + "..."
                                if len(prompt_data["system_prompt"]) > 80
                                else prompt_data["system_prompt"]
                            ),
                            "Mean Adv Score": mean_display,
                            "Std Adv Score": std_display,
                            "Num Attacks": len(prompt_data.get("attacks", [])),
                            "Hash": prompt_hash[:12],
                        }
                    )

                overview_df = pd.DataFrame(overview_data)

                # Display with selection
                selected_prompt = _render_selectable_df(
                    overview_df,
                )

                # Show details for selected prompt
                if selected_prompt["selection"]["rows"]:  # type: ignore
                    selected_idx = selected_prompt["selection"]["rows"][0]  # type: ignore
                    prompt_hash = sorted_prompts[selected_idx][0]
                    prompt_data = sorted_prompts[selected_idx][1]

                    st.divider()
                    display_system_prompt_details(prompt_data, prompt_hash)
            else:
                st.info("No system prompts found for this seed state yet.")

    elif selected_tab == "📈 Analytics":
        st.header("Analytics")

        # Score distribution across all prompts
        all_scores = []
        all_seed_labels = []

        for seed_id, seed_data in run_data["seed_states"].items():
            for prompt_hash, prompt_data in seed_data["system_prompts"].items():
                mean_score = prompt_data.get("mean_score")
                if mean_score is not None:
                    all_scores.append(mean_score)
                    all_seed_labels.append(f"Seed {seed_id}")

        if all_scores:
            # Score distribution histogram
            fig_hist = px.histogram(
                x=all_scores,
                color=all_seed_labels,
                title="Distribution of Adversarial Scores",
                labels={"x": "Adversarial Score", "count": "Count"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Score over time (if step info available)
            time_data = []
            for seed_id, seed_data in run_data["seed_states"].items():
                for prompt_hash, prompt_data in seed_data["system_prompts"].items():
                    meta = prompt_data.get("meta", {})
                    step = meta.get("step", 0)
                    operation = meta.get("operation", "unknown")
                    score = prompt_data.get("mean_score")
                    if score is not None:
                        time_data.append(
                            {
                                "Step": step,
                                "Score": score,
                                "Seed": f"Seed {seed_id}",
                                "Operation": operation,
                            }
                        )

            if time_data:
                time_df = pd.DataFrame(time_data)
                fig_time = px.scatter(
                    time_df,
                    x="Step",
                    y="Score",
                    color="Seed",
                    symbol="Operation",
                    title="Adversarial Scores Over Time",
                    labels={"Score": "Adversarial Score", "Step": "Training Step"},
                )
                st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No score data available for analysis.")

    elif selected_tab == "🧬 Evolution":
        st.header("Evolutionary Analysis")

        # Population tracking over time
        st.subheader("Population Evolution")

        # Collect population data across all steps using population_history.json
        population_data = []
        for seed_id, seed_data in run_data["seed_states"].items():
            population_history = seed_data.get("population_history", {})

            for prompt_hash, prompt_data in seed_data["system_prompts"].items():
                meta = prompt_data.get("meta", {})
                generation_step = meta.get("step", 0)
                operation = meta.get("operation", "unknown")
                score = prompt_data.get("mean_score", 0)

                # Find the latest step where this prompt was in population
                in_population = False
                population_step = None
                latest_pop_step = None

                for step_str, step_population in population_history.items():
                    step_num = int(step_str)
                    if prompt_hash in step_population:
                        in_population = True
                        population_step = step_population[prompt_hash]
                        latest_pop_step = step_num

                population_data.append(
                    {
                        "Seed": f"Seed {seed_id}",
                        "Step": generation_step,
                        "Operation": operation,
                        "Score": score,
                        "In Population": in_population,
                        "Population Step": population_step,
                        "Latest Pop Step": latest_pop_step,
                        "System Prompt": prompt_data.get("system_prompt", "")[:100]
                        + "...",
                        "Hash": prompt_hash[:8],
                    }
                )

        if population_data:
            pop_df = pd.DataFrame(population_data)

            # Population size over time using population_history.json
            pop_size_data = []
            for seed_id, seed_data in run_data["seed_states"].items():
                population_history = seed_data.get("population_history")
                if population_history:
                    for step_str, population_hashes in population_history.items():
                        step = int(step_str)
                        pop_size_data.append(
                            {
                                "Step": step,
                                "Seed": f"Seed {seed_id}",
                                "Population Size": len(population_hashes),
                            }
                        )

            if pop_size_data:
                pop_size_df = pd.DataFrame(pop_size_data)
                fig_pop_size = px.line(
                    pop_size_df,
                    x="Step",
                    y="Population Size",
                    color="Seed",
                    title="Population Size Over Time",
                    markers=True,
                )
                st.plotly_chart(fig_pop_size, use_container_width=True)

            # Population vs non-population scores
            fig_pop_scores = px.scatter(
                pop_df,
                x="Step",
                y="Score",
                color="In Population",
                symbol="Operation",
                hover_data=["System Prompt"],
                title="Scores: Population vs Non-Population",
                labels={"Score": "Adversarial Score"},
            )
            st.plotly_chart(fig_pop_scores, use_container_width=True)

            # Seed-specific population timeline
            st.subheader("Population Timeline by Seed")

            # Seed selection
            available_seeds = sorted(
                [int(seed_id) for seed_id in run_data["seed_states"].keys()]
            )
            selected_seed = st.selectbox(
                "Select Seed Index", available_seeds, key="evolution_seed_selector"
            )

            if selected_seed is not None:
                seed_data = run_data["seed_states"][str(selected_seed)]
                population_history = seed_data.get("population_history")

                if population_history:
                    st.subheader(f"Population History for Seed {selected_seed}")

                    # Display population for each step
                    steps = sorted([int(step) for step in population_history.keys()])
                    for step in steps:
                        step_str = str(step)
                        population_hashes = population_history[step_str]

                        with st.expander(
                            f"Step {step} - Population Size: {len(population_hashes)}",
                            expanded=(step == steps[-1]),
                        ):
                            if population_hashes:
                                # Build population data for this step
                                pop_data = []
                                for (
                                    prompt_hash,
                                    generation,
                                ) in population_hashes.items():
                                    prompt_data = seed_data["system_prompts"].get(
                                        prompt_hash
                                    )
                                    if prompt_data:
                                        pop_data.append(
                                            {
                                                "Hash": prompt_hash[:8],
                                                "System Prompt": prompt_data.get(
                                                    "system_prompt", ""
                                                )[:100]
                                                + "...",
                                                "Score": prompt_data.get(
                                                    "mean_score", 0
                                                ),
                                                "Operation": prompt_data.get(
                                                    "meta", {}
                                                ).get("operation", "unknown"),
                                                "Generation": generation,
                                            }
                                        )

                                if pop_data:
                                    pop_df_step = pd.DataFrame(pop_data).sort_values(
                                        "Score", ascending=False
                                    )
                                    st.dataframe(
                                        pop_df_step,
                                        width="stretch",
                                        hide_index=True,
                                        height=_adaptive_table_height(
                                            len(pop_df_step), max_height=600
                                        ),
                                    )
                                else:
                                    st.info("Population data not fully loaded")
                            else:
                                st.info("No prompts in population at this step")

                            # Show all candidates generated at this step (not in population)
                            step_candidates = []
                            for prompt_hash, prompt_data in seed_data[
                                "system_prompts"
                            ].items():
                                meta = prompt_data.get("meta", {})
                                if (
                                    meta.get("step") == step
                                    and prompt_hash not in population_hashes
                                ):
                                    step_candidates.append(
                                        {
                                            "Hash": prompt_hash[:8],
                                            "System Prompt": prompt_data.get(
                                                "system_prompt", ""
                                            )[:100]
                                            + "...",
                                            "Score": prompt_data.get("mean_score", 0),
                                            "Operation": meta.get(
                                                "operation", "unknown"
                                            ),
                                        }
                                    )

                            if step_candidates:
                                st.write(
                                    "**Non-population candidates generated at this step:**"
                                )
                                candidates_df = pd.DataFrame(
                                    step_candidates
                                ).sort_values("Score", ascending=False)
                                st.dataframe(
                                    candidates_df,
                                    width="stretch",
                                    hide_index=True,
                                    height=_adaptive_table_height(
                                        len(candidates_df), max_height=600
                                    ),
                                )
                else:
                    st.info("No population history available for this seed")
            else:
                st.info("No seed data available")

            # Operation breakdown
            st.subheader("Operations Analysis")
            col1, col2 = st.columns(2)

            with col1:
                operation_counts = pop_df["Operation"].value_counts()
                fig_ops = px.pie(
                    values=operation_counts.values,
                    names=operation_counts.index,
                    title="Distribution of Operations",
                )
                st.plotly_chart(fig_ops, use_container_width=True)

            with col2:
                # Average scores by operation
                op_scores = (
                    pop_df.groupby("Operation")["Score"]  # type: ignore
                    .mean()
                    .sort_values(ascending=False)
                )
                fig_op_scores = px.bar(
                    x=op_scores.index,
                    y=op_scores.values,
                    title="Average Scores by Operation",
                    labels={"x": "Operation", "y": "Average Score"},
                )
                st.plotly_chart(fig_op_scores, use_container_width=True)

        else:
            st.info("No evolutionary data available for analysis.")


if __name__ == "__main__":
    main()
