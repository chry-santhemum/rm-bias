import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import plotly.express as px

st.set_page_config(page_title="User Prompts Viz", layout="wide")

@st.cache_data
def load_prompt_stats(stats_dir_str: str) -> Dict[str, Any]:
    """Load all prompt statistics from a directory."""
    stats_dir = Path(stats_dir_str)
    data = {
        'prompts': {},
        'available_models': set(),
        'datasets': set(),
        'topics': {}  # dataset -> {topic_id: topic_name}
    }

    if not stats_dir.exists():
        return data

    # Load all JSON files
    for json_file in stats_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)

                prompt_hash = json_file.stem
                data['prompts'][prompt_hash] = prompt_data

                # Track available models from both rollouts and summary_stats
                for rollout in prompt_data.get('rollouts', []):
                    for key in rollout.keys():
                        if key != 'response':
                            data['available_models'].add(key)

                # Track available models from summary_stats (new format)
                data['available_models'].update(prompt_data.get('summary_stats', {}).keys())

                # Track datasets and topics
                dataset = prompt_data.get('dataset')
                topic_id = prompt_data.get('topic_label')  # Using 'topic_label' from the JSON
                topic_name = prompt_data.get('topic_name')

                if dataset:
                    data['datasets'].add(dataset)
                    if dataset not in data['topics']:
                        data['topics'][dataset] = {}
                    if topic_id is not None and topic_name:
                        data['topics'][dataset][topic_id] = topic_name

        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Could not load {json_file}: {e}")
            continue

    data['available_models'] = sorted(list(data['available_models']))
    data['datasets'] = sorted(list(data['datasets']))
    return data

def create_overview_table(prompt_data: Dict[str, Any], available_models: List[str], selected_dataset: str = None, selected_topic_id: int = None) -> pd.DataFrame:
    """Create overview table of all prompts with summary statistics."""
    overview_rows = []

    for prompt_hash, data in prompt_data.items():
        # Filter by dataset and topic if specified
        if selected_dataset and data.get('dataset') != selected_dataset:
            continue
        if selected_topic_id is not None and data.get('topic_label') != selected_topic_id:
            continue

        prompt_text = data.get('prompt', '')
        rollouts = data.get('rollouts', [])
        dataset = data.get('dataset', 'Unknown')
        topic_name = data.get('topic_name', 'Unknown')
        topic_id = data.get('topic_label', 'N/A')

        # Get summary stats from nested structure
        summary_stats = data.get('summary_stats', {})

        # Create base row data
        row_data = {
            'Hash': prompt_hash[:12],
            'Dataset': dataset,
            'Topic ID': topic_id,
            'Topic': topic_name,
            'Prompt': prompt_text[:100] + '...' if len(prompt_text) > 100 else prompt_text,
            'Rollouts': len(rollouts),
            'Full Hash': prompt_hash
        }

        # Add columns for each available model
        for model in available_models:
            model_stats = summary_stats.get(model, {})

            if model_stats and 'mean' in model_stats:
                # Use pre-computed summary stats
                mean_score = model_stats.get('mean')
                rewards_raw = model_stats.get('rewards_raw', [])
                rewards_cleaned = [r for r in rewards_raw if r is not None]
                std_score = np.std(rewards_cleaned) if rewards_cleaned else None

                row_data[f'Mean ({model})'] = f"{mean_score:.3f}" if mean_score is not None else 'N/A'
                row_data[f'Std ({model})'] = f"{std_score:.3f}" if std_score is not None else 'N/A'
            else:
                # No stats available for this model
                row_data[f'Mean ({model})'] = 'N/A'
                row_data[f'Std ({model})'] = 'N/A'

        overview_rows.append(row_data)

    return pd.DataFrame(overview_rows)

def calculate_text_height(text: str, chars_per_line: int = 120, min_height: int = 80, max_height: int = 400) -> int:
    """Calculate appropriate height for text area based on content."""
    if not text:
        return min_height

    lines = text.split('\n')
    total_lines = sum(max(1, len(line) // chars_per_line + (1 if len(line) % chars_per_line > 0 else 0)) for line in lines)

    base_height = 50
    line_height = 22
    calculated_height = base_height + (total_lines * line_height)

    return max(min_height, min(max_height, calculated_height))

def display_prompt_details(prompt_data: Dict[str, Any], prompt_hash: str, available_models: List[str]):
    """Display detailed view of a selected prompt."""
    data = prompt_data[prompt_hash]
    prompt_text = data.get('prompt', '')
    rollouts = data.get('rollouts', [])

    st.subheader(f"Prompt Details: {prompt_hash[:12]}...")

    # Display full prompt text with dynamic height
    prompt_height = calculate_text_height(prompt_text, min_height=100, max_height=300)
    st.text_area("Full Prompt Text", prompt_text, height=prompt_height)

    # Summary statistics for all models
    st.subheader("Summary Statistics by Model")

    # Calculate statistics for each model
    model_stats = {}
    for model in available_models:
        scores = []
        for rollout in rollouts:
            if model in rollout and rollout[model] is not None:
                scores.append(rollout[model])
        model_stats[model] = scores

    # Display statistics in a grid
    if any(len(scores) > 0 for scores in model_stats.values()):
        # Create columns for each model
        cols = st.columns(len(available_models))
        for i, model in enumerate(available_models):
            scores = model_stats[model]
            with cols[i]:
                st.write(f"**{model}**")
                if scores:
                    st.metric("Mean", f"{np.mean(scores):.3f}")
                    st.metric("Std Dev", f"{np.std(scores):.3f}")
                    st.metric("Min", f"{np.min(scores):.3f}")
                    st.metric("Max", f"{np.max(scores):.3f}")
                    st.metric("Count", len(scores))
                else:
                    st.info("No scores")

        # Score distribution histogram for all models with data
        models_with_data = [(model, scores) for model, scores in model_stats.items() if len(scores) > 0]
        if models_with_data:
            st.subheader("Score Distributions")

            # Create combined histogram data
            hist_data = []
            for model, scores in models_with_data:
                for score in scores:
                    hist_data.append({"Model": model, "Score": score})

            if hist_data:
                hist_df = pd.DataFrame(hist_data)
                fig_hist = px.histogram(
                    hist_df,
                    x="Score",
                    color="Model",
                    title="Score Distributions by Model",
                    labels={'Score': 'Score', 'count': 'Count'},
                    nbins=20,
                    barmode='overlay',
                    opacity=0.7
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # Rollouts details (show regardless of selected model scores)
    if rollouts:
        st.subheader("Individual Rollouts")

        # Create rollout overview table with all model scores (show all rollouts)
        rollout_rows = []
        for rollout_idx, rollout in enumerate(rollouts):
            response = rollout.get('response', '')

            row_data = {
                'Rollout #': rollout_idx + 1,
                'Response Preview': response[:200] + '...' if len(response) > 200 else response,
                'Response Length': len(response),
                'Original Index': rollout_idx
            }

            # Add score columns for all available models
            for model in available_models:
                score = rollout.get(model, None)
                row_data[f'Score ({model})'] = f"{score:.3f}" if isinstance(score, (int, float)) else 'N/A'

            rollout_rows.append(row_data)

        rollout_df = pd.DataFrame(rollout_rows)

        # Display rollouts table with selection
        selected_rollout = st.dataframe(
            rollout_df.drop('Original Index', axis=1),
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True
        )

        # Show detailed rollout view
        if selected_rollout['selection']['rows']:
            selected_idx = selected_rollout['selection']['rows'][0]
            original_idx = rollout_df.iloc[selected_idx]['Original Index']
            rollout = rollouts[original_idx]

            st.subheader(f"Rollout #{original_idx + 1} Details")

            # Show full response with dynamic height
            response_text = rollout.get('response', '')
            response_height = calculate_text_height(response_text, min_height=150, max_height=500)
            st.text_area("Full Response", response_text, height=response_height)

            # Show all model scores for this rollout
            st.subheader("All Model Scores")
            score_data = []
            for key, value in rollout.items():
                if key != 'response' and value is not None:
                    score_data.append({
                        'Model': key,
                        'Score': f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
                    })

            if score_data:
                score_df = pd.DataFrame(score_data)
                st.dataframe(score_df, width="stretch", hide_index=True)
            else:
                st.info("No model scores available for this rollout")

    else:
        st.info("No rollouts available for this prompt")

def main():
    st.title("User Prompts Visualization")

    # Initialize session state
    if 'selected_prompt_idx' not in st.session_state:
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
        help="Path to directory containing prompt statistics JSON files"
    )

    if not stats_dir:
        st.error("Please specify a directory containing prompt statistics")
        return

    # Load data first to get available datasets
    prompt_stats = load_prompt_stats(stats_dir)

    if not prompt_stats['prompts']:
        st.error(f"No prompt statistics found in directory: {stats_dir}")
        st.info("Make sure the directory contains JSON files generated by prompt_stats.py")
        return

    # Dataset selection
    st.sidebar.header("Dataset & Topic Filter")
    available_datasets = ['All'] + list(prompt_stats['datasets'])
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        available_datasets,
        help="Filter prompts by dataset"
    )

    # Topic selection (only show if dataset is selected)
    selected_topic_id = None
    if selected_dataset != 'All' and selected_dataset in prompt_stats['topics']:
        topics = prompt_stats['topics'][selected_dataset]
        if topics:
            st.sidebar.subheader(f"Topics in {selected_dataset}")

            # Create topic options with ID and name
            topic_options = ['All Topics'] + [f"{topic_id}: {topic_name}" for topic_id, topic_name in sorted(topics.items())]
            selected_topic = st.sidebar.selectbox(
                "Select Topic",
                topic_options,
                help="Filter prompts by topic"
            )

            if selected_topic != 'All Topics':
                selected_topic_id = int(selected_topic.split(':')[0])

            # Show topic summary
            st.sidebar.info(f"Found {len(topics)} topics in {selected_dataset}")
        else:
            st.sidebar.info(f"No topics found in {selected_dataset}")

    # Convert dataset selection for filtering
    dataset_filter = None if selected_dataset == 'All' else selected_dataset

    # Get available models
    available_models = prompt_stats['available_models']
    if not available_models:
        st.error("No model scores found in the prompt statistics")
        return

    # Show basic statistics (filtered)
    st.header("Dataset Overview")

    # Calculate filtered statistics
    filtered_prompts = []
    for prompt_hash, data in prompt_stats['prompts'].items():
        if dataset_filter and data.get('dataset') != dataset_filter:
            continue
        if selected_topic_id is not None and data.get('topic_label') != selected_topic_id:
            continue
        filtered_prompts.append(data)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Prompts", f"{len(filtered_prompts)} / {len(prompt_stats['prompts'])}")

    with col2:
        filtered_rollouts = sum(len(data.get('rollouts', [])) for data in filtered_prompts)
        total_rollouts = sum(len(data.get('rollouts', [])) for data in prompt_stats['prompts'].values())
        st.metric("Total Rollouts", f"{filtered_rollouts} / {total_rollouts}")

    with col3:
        st.metric("Available Models", len(available_models))

    with col4:
        st.metric("Datasets", len(prompt_stats['datasets']))

    # Create and display overview table with filtering
    st.header("Prompt Overview")

    # Display filter information below header
    if dataset_filter:
        st.markdown(f"**Dataset:** {dataset_filter}")
    if selected_topic_id is not None:
        topic_name = prompt_stats['topics'][dataset_filter][selected_topic_id]
        st.markdown(f"**Topic Label:** {selected_topic_id}")
        st.markdown(f"**Topic Name:** {topic_name}")
    overview_df = create_overview_table(prompt_stats['prompts'], available_models, dataset_filter, selected_topic_id)

    if overview_df.empty:
        st.warning("No prompt data available")
        return

    # Sort options - model-specific scores for all available models
    sort_options = []
    for model in available_models:
        sort_options.extend([f'Mean ({model})', f'Std ({model})'])

    # Default to first model's mean score
    sort_by = st.selectbox("Sort by", sort_options, index=0)
    sort_ascending = st.checkbox("Sort ascending", value=False)

    # Apply sorting
    # Use the sort_by directly as it now matches the column names
    sort_column = sort_by

    # Convert to float for proper sorting (all sort options are numeric)
    overview_df_sorted = overview_df.copy()
    overview_df_sorted[sort_column] = pd.to_numeric(
        overview_df_sorted[sort_column].str.replace('N/A', 'nan'),
        errors='coerce'
    )

    overview_df_sorted = overview_df_sorted.sort_values(
        sort_column,
        ascending=sort_ascending,
        na_position='last'
    )

    # Display overview table
    selected_prompt = st.dataframe(
        overview_df_sorted.drop('Full Hash', axis=1),  # Hide full hash column
        width="stretch",
        height=400,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True
    )

    # Show detailed view for selected prompt
    if selected_prompt['selection']['rows']:
        selected_idx = selected_prompt['selection']['rows'][0]
        selected_hash = overview_df_sorted.iloc[selected_idx]['Full Hash']

        st.divider()
        display_prompt_details(prompt_stats['prompts'], selected_hash, available_models)

    else:
        st.info("👆 Select a prompt from the table above to see detailed analysis")

if __name__ == "__main__":
    main()