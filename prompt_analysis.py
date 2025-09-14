import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import hashlib

st.set_page_config(page_title="Prompt Analysis", layout="wide")

@st.cache_data
def load_prompt_stats(stats_dir_str: str) -> Dict[str, Any]:
    """Load all prompt statistics from a directory."""
    stats_dir = Path(stats_dir_str)
    data = {
        'prompts': {},
        'available_models': set()
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

                # Track available models from rollouts
                for rollout in prompt_data.get('rollouts', []):
                    for key in rollout.keys():
                        if key != 'response':
                            data['available_models'].add(key)

        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Could not load {json_file}: {e}")
            continue

    data['available_models'] = sorted(list(data['available_models']))
    return data

def create_overview_table(prompt_data: Dict[str, Any], selected_model: str) -> pd.DataFrame:
    """Create overview table of all prompts with summary statistics."""
    overview_rows = []

    for prompt_hash, data in prompt_data.items():
        prompt_text = data.get('prompt', '')
        rollouts = data.get('rollouts', [])

        # Get scores for selected model
        scores = []
        for rollout in rollouts:
            if selected_model in rollout and rollout[selected_model] is not None:
                scores.append(rollout[selected_model])

        if scores:
            mean_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            std_score = np.std(scores)
            percentiles = {
                '25th': np.percentile(scores, 25),
                '50th': np.percentile(scores, 50),
                '75th': np.percentile(scores, 75)
            }
        else:
            mean_score = min_score = max_score = std_score = None
            percentiles = {'25th': None, '50th': None, '75th': None}

        overview_rows.append({
            'Hash': prompt_hash[:12],
            'Prompt': prompt_text[:150] + '...' if len(prompt_text) > 150 else prompt_text,
            'Rollouts': len(rollouts),
            'Valid Scores': len(scores),
            'Mean Score': f"{mean_score:.3f}" if mean_score is not None else 'N/A',
            'Std Dev': f"{std_score:.3f}" if std_score is not None else 'N/A',
            'Min': f"{min_score:.3f}" if min_score is not None else 'N/A',
            'Max': f"{max_score:.3f}" if max_score is not None else 'N/A',
            '25th': f"{percentiles['25th']:.3f}" if percentiles['25th'] is not None else 'N/A',
            '50th': f"{percentiles['50th']:.3f}" if percentiles['50th'] is not None else 'N/A',
            '75th': f"{percentiles['75th']:.3f}" if percentiles['75th'] is not None else 'N/A',
            'Full Hash': prompt_hash
        })

    return pd.DataFrame(overview_rows)

def display_prompt_details(prompt_data: Dict[str, Any], prompt_hash: str, selected_model: str):
    """Display detailed view of a selected prompt."""
    data = prompt_data[prompt_hash]
    prompt_text = data.get('prompt', '')
    rollouts = data.get('rollouts', [])

    st.subheader(f"Prompt Details: {prompt_hash[:12]}...")

    # Display full prompt text
    st.text_area("Full Prompt Text", prompt_text, height=200)

    # Get scores for visualization
    scores = []
    valid_rollouts = []
    for i, rollout in enumerate(rollouts):
        if selected_model in rollout and rollout[selected_model] is not None:
            scores.append(rollout[selected_model])
            valid_rollouts.append((i, rollout))

    if scores:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Score", f"{np.mean(scores):.3f}")
        with col2:
            st.metric("Std Dev", f"{np.std(scores):.3f}")
        with col3:
            st.metric("Min Score", f"{np.min(scores):.3f}")
        with col4:
            st.metric("Max Score", f"{np.max(scores):.3f}")

        # Score distribution histogram
        fig_hist = px.histogram(
            x=scores,
            title=f"Score Distribution for {selected_model}",
            labels={'x': 'Score', 'count': 'Count'},
            nbins=min(20, len(scores))
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Rollouts details
        st.subheader("Individual Rollouts")

        # Create rollout overview table
        rollout_rows = []
        for rollout_idx, (original_idx, rollout) in enumerate(valid_rollouts):
            response = rollout.get('response', '')
            score = rollout.get(selected_model, 'N/A')

            rollout_rows.append({
                'Rollout #': original_idx + 1,
                'Score': f"{score:.3f}" if isinstance(score, (int, float)) else str(score),
                'Response Preview': response[:200] + '...' if len(response) > 200 else response,
                'Response Length': len(response),
                'Original Index': original_idx
            })

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

            # Show full response
            response_text = rollout.get('response', '')
            st.text_area("Full Response", response_text, height=300)

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
        st.warning(f"No valid scores found for model '{selected_model}' in this prompt")

def main():
    st.title("📊 Prompt Analysis Dashboard")

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

    # Load data
    prompt_stats = load_prompt_stats(stats_dir)

    if not prompt_stats['prompts']:
        st.error(f"No prompt statistics found in directory: {stats_dir}")
        st.info("Make sure the directory contains JSON files generated by prompt_stats.py")
        return

    # Model selection
    available_models = prompt_stats['available_models']
    if not available_models:
        st.error("No model scores found in the prompt statistics")
        return

    selected_model = st.sidebar.selectbox(
        "Select Model/Rater",
        available_models,
        help="Choose which model's scores to analyze"
    )

    # Show basic statistics
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Prompts", len(prompt_stats['prompts']))

    with col2:
        total_rollouts = sum(len(data.get('rollouts', [])) for data in prompt_stats['prompts'].values())
        st.metric("Total Rollouts", total_rollouts)

    with col3:
        st.metric("Available Models", len(available_models))

    # Create and display overview table
    st.header("Prompt Overview")
    overview_df = create_overview_table(prompt_stats['prompts'], selected_model)

    if overview_df.empty:
        st.warning(f"No data available for model '{selected_model}'")
        return

    # Sort options
    sort_options = ['Mean Score', 'Std Dev', 'Rollouts', 'Valid Scores']
    sort_by = st.selectbox("Sort by", sort_options, index=0)
    sort_ascending = st.checkbox("Sort ascending", value=False)

    # Apply sorting
    sort_column = sort_by
    if sort_column in ['Mean Score', 'Std Dev']:
        # Convert to float for proper sorting
        overview_df_sorted = overview_df.copy()
        overview_df_sorted[sort_column] = pd.to_numeric(
            overview_df_sorted[sort_column].str.replace('N/A', 'nan'),
            errors='coerce'
        )
    else:
        overview_df_sorted = overview_df.copy()

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
        display_prompt_details(prompt_stats['prompts'], selected_hash, selected_model)

    else:
        st.info("👆 Select a prompt from the table above to see detailed analysis")

if __name__ == "__main__":
    main()