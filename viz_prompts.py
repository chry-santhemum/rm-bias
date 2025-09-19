import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import plotly.express as px

st.set_page_config(page_title="User Prompts Visualization", layout="wide")


@st.cache_data
def discover_datasets(stats_dir_str: str) -> List[str]:
    """Discover available datasets without loading the data."""
    stats_dir = Path(stats_dir_str)
    datasets = set()

    if not stats_dir.exists():
        return []

    # Check subdirectories for datasets
    for dataset_dir in stats_dir.iterdir():
        if dataset_dir.is_dir() and any(dataset_dir.glob("*.json")):
            datasets.add(dataset_dir.name)

    return sorted(list(datasets))


@st.cache_data
def load_dataset_data(stats_dir_str: str, dataset_name: str) -> Dict[str, Any]:
    """Load data for a specific dataset only."""
    stats_dir = Path(stats_dir_str)
    data = {
        "prompts": {},
        "policy_names": set(),  # policy models
        "rater_names": set(),  # rater names
        "topics": {},  # topic_id -> topic_name
    }

    if not stats_dir.exists():
        return data

    # Load from specific dataset subdirectory
    dataset_dir = stats_dir / dataset_name
    if dataset_dir.exists() and dataset_dir.is_dir():
        for json_file in dataset_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    prompt_data = json.load(f)

                prompt_hash = json_file.stem
                data["prompts"][prompt_hash] = prompt_data

                # Only add to the list for the first one found
                if not data["policy_names"]:
                    for key in prompt_data.keys():
                        if key not in [
                            "prompt",
                            "topic_label",
                            "topic_name",
                            "dataset",
                            "correlation",
                        ]:
                            data["policy_names"].add(key)

                            for rater_key in (
                                prompt_data[key].get("summary_stats", {}).keys()
                            ):
                                data["rater_names"].add(rater_key)

                # Track topics
                topic_id = prompt_data.get("topic_label")
                topic_name = prompt_data.get("topic_name")
                if topic_id is not None and topic_name:
                    data["topics"][topic_id] = topic_name

            except (json.JSONDecodeError, IOError) as e:
                st.warning(f"Could not load {json_file}: {e}")
                continue

    data["policy_names"] = sorted(list(data["policy_names"]))
    data["rater_names"] = sorted(list(data["rater_names"]))
    return data


def create_overview_table(
    prompt_data: Dict[str, Any],
    policy_names: List[str],
    rater_names: List[str],
    selected_dataset: str | None = None,
    selected_topic_id: int | None = None,
) -> pd.DataFrame:
    """Create overview table of all prompts with summary statistics."""
    overview_rows = []

    for prompt_hash, data in prompt_data.items():
        # Filter by dataset and topic if specified
        if selected_dataset and data.get("dataset") != selected_dataset:
            continue
        if (
            selected_topic_id is not None
            and data.get("topic_label") != selected_topic_id
        ):
            continue

        prompt_text = data.get("prompt", "")
        # dataset = data.get("dataset", "Unknown")
        topic_name = data.get("topic_name", "Unknown")
        topic_id = data.get("topic_label", "N/A")

        # Count total rollouts across all models
        total_rollouts = 0
        for model in policy_names:
            model_data = data.get(model, {})
            if isinstance(model_data, dict) and "rollouts" in model_data:
                total_rollouts += len(model_data["rollouts"])

        # Get correlation if it exists as a top-level key
        correlation = data.get("correlation")

        # Create base row data
        row_data = {
            # "Dataset": dataset,
            "Prompt": (
                prompt_text[:80] + "..." if len(prompt_text) > 80 else prompt_text
            ),
            # "Rollouts": total_rollouts,
        }

        # Add columns for each available model
        # Streamlit does not render DataFrame column names as multiline with \n,
        # so use a more readable single-line format: "{Mean/Std} | {policy} | {rater}"
        for policy in policy_names:
            model_data = data.get(policy, {})
            summary_stats = model_data.get("summary_stats", {})

            if policy == "meta-llama/llama-3.1-8b-instruct":
                policy_short = "l3.1-8b"
            elif policy == "meta-llama/llama-3.1-70b-instruct":
                policy_short = "l3.1-70b"

            for rater in rater_names:
                if rater == "skywork-v2":
                    rater_short = "skywork-v2"
                elif rater == "openai/gpt-5-nano":
                    rater_short = "gpt-5-nano"
                elif rater == "openai/gpt-4.1-nano":
                    rater_short = "gpt-4.1-nano"

                score_data = summary_stats.get(rater, {})
                # Single-line column names for better Streamlit display
                mean_col = f"Mean | {policy_short} | {rater_short}"
                std_col = f"Std | {policy_short} | {rater_short}"
                if isinstance(score_data, dict) and "mean" in score_data:
                    mean_score = score_data.get("mean")
                    scores_raw = score_data.get("scores_raw", [])
                    scores_cleaned = [r for r in scores_raw if r is not None]
                    std_score = np.std(scores_cleaned) if scores_cleaned else None

                    row_data[mean_col] = (
                        f"{mean_score:.3f}" if mean_score is not None else "N/A"
                    )
                    row_data[std_col] = (
                        f"{std_score:.3f}" if std_score is not None else "N/A"
                    )
                else:
                    # No valid stats found
                    row_data[mean_col] = "N/A"
                    row_data[std_col] = "N/A"

        row_data.update(
            {
                "Topic ID": topic_id,
                "Topic": topic_name,
                "Correlation": (
                    f"{correlation:.4f}" if correlation is not None else "N/A"
                ),
                "Hash": prompt_hash,
            }
        )

        overview_rows.append(row_data)

    return pd.DataFrame(overview_rows)


def calculate_text_height(
    text: str, chars_per_line: int = 120, min_height: int = 120, max_height: int = 360
) -> int:
    """Calculate appropriate height for text area based on content."""
    if not text:
        return min_height

    lines = text.split("\n")
    total_lines = sum(
        max(
            1,
            len(line) // chars_per_line + (1 if len(line) % chars_per_line > 0 else 0),
        )
        for line in lines
    )

    base_height = 50
    line_height = 22
    calculated_height = base_height + (total_lines * line_height)

    return max(min_height, min(max_height, calculated_height))


def display_prompt_details(data: Dict[str, Any]):
    """Display detailed view of a selected prompt."""
    prompt_text = data.get("prompt", "")
    policy_names = set()

    for key in data.keys():
        if key not in [
            "prompt",
            "topic_label",
            "topic_name",
            "dataset",
            "correlation",
        ]:
            policy_names.add(key)

    # Display full prompt text with dynamic height
    st.subheader("User prompt")
    prompt_height = calculate_text_height(prompt_text)
    st.text_area("Full text", prompt_text, height=prompt_height)

    # Summary statistics for all models
    st.subheader("Summary Statistics")

    # Get statistics for each model from the new structure
    model_stats = {}
    all_rollouts = []  # Collect all rollouts for display

    for policy in policy_names:
        model_data = data.get(policy, {})
        # Get rollouts for this model
        if "rollouts" in model_data:
            for rollout in model_data["rollouts"]:
                # Add model to rollout for tracking
                rollout_with_model = rollout.copy()
                rollout_with_model["policy"] = policy
                all_rollouts.append(rollout_with_model)

        # Get scores from summary stats
        if "summary_stats" in model_data:
            for rater, score_data in model_data["summary_stats"].items():
                model_stats[f"{policy} | {rater}"] = score_data["scores_raw"]
                break

    # Display statistics in a table (each key is a row, each stat is a column)
    if any(len(scores) > 0 for scores in model_stats.values()):
        stats_rows = []
        for key, scores in model_stats.items():
            if scores:
                stats_rows.append(
                    {
                        "Model | Rater": key,
                        "Mean": f"{np.mean(scores):.3f}",
                        "Std Dev": f"{np.std(scores):.3f}",
                        "Min": f"{np.min(scores):.3f}",
                        "Max": f"{np.max(scores):.3f}",
                        "Count": len(scores),
                    }
                )
            else:
                stats_rows.append(
                    {
                        "Model | Rater": key,
                        "Mean": "N/A",
                        "Std Dev": "N/A",
                        "Min": "N/A",
                        "Max": "N/A",
                        "Count": 0,
                    }
                )
        stats_df = pd.DataFrame(stats_rows)
        st.dataframe(
            stats_df,
            hide_index=True,
            width="stretch",
        )

    # Rollouts details
    if all_rollouts:
        st.subheader("Individual Rollouts")

        # Create rollout overview table
        rollout_rows = []
        for rollout_idx, rollout in enumerate(all_rollouts):
            response = rollout.get("response", "")
            policy = rollout.get("policy", "Unknown")

            row_data = {
                # "Rollout #": rollout_idx + 1,
                "Response Preview": (
                    response[:80] + "..." if len(response) > 80 else response
                ),
            }

            # Add score columns for all scoring models in this rollout
            for key, value in rollout.items():
                if key not in ["response", "policy"] and value is not None:
                    row_data[f"Score ({key})"] = (
                        f"{value:.3f}"
                        if isinstance(value, (int, float))
                        else str(value)
                    )

            row_data.update(
                {
                    "Response Length": len(response),
                    "Policy": policy,
                    "Index": rollout_idx,
                }
            )
            rollout_rows.append(row_data)

        rollout_df = pd.DataFrame(rollout_rows)

        # Display rollouts table with selection
        selected_rollout = st.dataframe(
            rollout_df.drop("Index", axis=1),
            width="stretch",
            height=600,
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
        )

        # Show detailed rollout view
        if selected_rollout["selection"]["rows"]:  # type: ignore
            selected_idx = selected_rollout["selection"]["rows"][0]  # type: ignore
            original_idx = rollout_df.iloc[selected_idx]["Index"]
            rollout = all_rollouts[original_idx]

            st.subheader(f"Rollout #{original_idx + 1}")

            # Show full response with dynamic height
            response_text = rollout.get("response", "")
            response_height = calculate_text_height(response_text, max_height=720)
            st.text_area("Full text", response_text, height=response_height)

            # Show all model scores for this rollout
            score_data = []
            for policy, value in rollout.items():
                if policy not in ["response", "policy"] and value is not None:
                    score_data.append(
                        {
                            "Policy": policy,
                            "Score": (
                                f"{value:.3f}"
                                if isinstance(value, (int, float))
                                else str(value)
                            ),
                        }
                    )

            if score_data:
                score_df = pd.DataFrame(score_data)
                st.dataframe(score_df, width="stretch", hide_index=True)
            else:
                st.info("No policy scores available for this rollout")

    else:
        st.info("No rollouts available for this prompt")


def main():
    st.title("User Prompts Visualization")

    # Initialize session state
    if "selected_prompt_idx" not in st.session_state:
        st.session_state.selected_prompt_idx = None

    # Sidebar for directory selection
    st.sidebar.header("Data Selection")

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    # Directory input
    stats_dir = st.sidebar.text_input(
        "Prompt Stats Directory",
        value="data/prompt_stats",
        help="Path to directory containing prompt statistics JSON files",
    )

    if not stats_dir:
        st.error("Please specify a directory containing prompt statistics")
        return

    # Discover available datasets first (lightweight operation)
    available_datasets = discover_datasets(stats_dir)

    if not available_datasets:
        st.error(f"No datasets found in directory: {stats_dir}")
        st.info(
            "Make sure the directory contains subdirectories with JSON files generated by prompt_stats.py"
        )
        return

    # Dataset selection
    st.sidebar.header("Dataset Selection")
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset", available_datasets, help="Choose a dataset to analyze"
    )

    # Load data only for the selected dataset
    dataset_data = load_dataset_data(stats_dir, selected_dataset)

    if not dataset_data["prompts"]:
        st.error(f"No prompt data found for dataset: {selected_dataset}")
        return

    # Get available models from the selected dataset
    policy_names = dataset_data["policy_names"]
    rater_names = dataset_data["rater_names"]
    if not policy_names:
        st.error("No model scores found in the selected dataset")
        return

    # Topic selection (now specific to the selected dataset)
    selected_topic_id = None
    topics = dataset_data["topics"]
    if topics:
        st.sidebar.subheader("Topic Filter")

        # Create topic options with ID and name
        topic_options = ["All Topics"] + [
            f"{topic_id}: {topic_name}"
            for topic_id, topic_name in sorted(topics.items())
        ]
        selected_topic = st.sidebar.selectbox(
            "Select Topic",
            topic_options,
            help="Filter prompts by topic within this dataset",
        )

        if selected_topic != "All Topics":
            selected_topic_id = int(selected_topic.split(":")[0])

        # Show topic summary
        st.sidebar.info(f"Found {len(topics)} topics in {selected_dataset}")
    else:
        st.sidebar.info(f"No topics found in {selected_dataset}")

    # Show basic statistics (filtered)
    st.header("Dataset Overview")

    # Calculate filtered statistics for the selected dataset
    filtered_prompts = []
    for prompt_hash, data in dataset_data["prompts"].items():
        if (
            selected_topic_id is not None
            and data.get("topic_label") != selected_topic_id
        ):
            continue
        filtered_prompts.append(data)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Prompts", f"{len(filtered_prompts)} / {len(dataset_data['prompts'])}"
        )

    with col2:
        # Count rollouts from new nested structure
        filtered_rollouts = 0
        for data in filtered_prompts:
            for model in policy_names:
                model_data = data.get(model, {})
                if isinstance(model_data, dict) and "rollouts" in model_data:
                    filtered_rollouts += len(model_data["rollouts"])

        total_rollouts = 0
        for data in dataset_data["prompts"].values():
            for model in policy_names:
                model_data = data.get(model, {})
                if isinstance(model_data, dict) and "rollouts" in model_data:
                    total_rollouts += len(model_data["rollouts"])

        st.metric("Total Rollouts", f"{filtered_rollouts} / {total_rollouts}")

    with col3:
        st.metric("Available Models", len(policy_names))

    with col4:
        st.metric("Topics", len(topics))

    # Create and display overview table with filtering
    st.header("Prompt Overview")

    # Display filter information below header
    st.markdown(f"**Dataset:** {selected_dataset}")
    if selected_topic_id is not None:
        topic_name = topics[selected_topic_id]
        st.markdown(f"**Topic Label:** {selected_topic_id}")
        st.markdown(f"**Topic Name:** {topic_name}")

    # Create overview table (no dataset filtering needed since we only loaded one dataset)
    overview_df = create_overview_table(
        dataset_data["prompts"],
        policy_names,
        rater_names,
        selected_dataset,
        selected_topic_id,
    )

    if overview_df.empty:
        st.warning("No prompt data available")
        return

    # Sort options: get column names of overview_df
    sort_options = list(overview_df.columns)
    sort_options.remove("Prompt")
    sort_options.remove("Topic")
    sort_options.remove("Hash")

    # Default to correlation
    sort_by = st.selectbox("Sort by", sort_options, index=0)
    sort_ascending = st.checkbox("Sort ascending", value=False)

    # Apply sorting
    # Use the sort_by directly as it now matches the column names
    sort_column = sort_by

    # Convert to float for proper sorting (all sort options are numeric)
    overview_df_sorted = overview_df.copy()
    if sort_column == "Correlation":
        # Handle correlation column which might contain 'N/A'
        overview_df_sorted[sort_column] = pd.to_numeric(
            overview_df_sorted[sort_column].str.replace("N/A", "nan"), errors="coerce"
        )
    else:
        # Handle model score columns
        overview_df_sorted[sort_column] = pd.to_numeric(
            overview_df_sorted[sort_column].str.replace("N/A", "nan"), errors="coerce"
        )

    overview_df_sorted = overview_df_sorted.sort_values(
        sort_column, ascending=sort_ascending, na_position="last"
    )

    # Display overview table
    selected_prompt = st.dataframe(
        overview_df_sorted.drop("Hash", axis=1),  # Hide full hash column
        width="stretch",
        height=600,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
    )

    # Show detailed view for selected prompt
    if selected_prompt["selection"]["rows"]:  # type: ignore
        selected_idx = selected_prompt["selection"]["rows"][0]  # type: ignore
        selected_hash = overview_df_sorted.iloc[selected_idx]["Hash"]

        st.divider()
        display_prompt_details(dataset_data["prompts"][selected_hash])

    else:
        st.info("👆 Select a prompt from the table above to see detailed analysis")


if __name__ == "__main__":
    main()
