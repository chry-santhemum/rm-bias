import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import plotly.express as px

st.set_page_config(page_title="RM Bias Training Viz", layout="wide")

@st.cache_data(ttl=30)  # Cache for 30 seconds to avoid excessive refreshes and state resets
def load_run_data(run_path: Path) -> Dict[str, Any]:
    """Load all data for a run from the file structure."""
    data = {
        'seed_states': {},
        'metadata': {}
    }
    
    if not run_path.exists():
        return data
    
    # Load each seed state directory
    for seed_dir in run_path.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith('seed_'):
            seed_id = seed_dir.name.replace('seed_', '')
            data['seed_states'][seed_id] = load_seed_state_data(seed_dir)
    
    return data

def load_seed_state_data(seed_dir: Path) -> Dict[str, Any]:
    """Load data for a single seed state."""
    seed_data = {
        'system_prompts': {},
        'cluster_info': None,
        'step_count': 0
    }
    
    # Load cluster info if exists
    cluster_file = seed_dir / 'cluster_info.json'
    if cluster_file.exists():
        with open(cluster_file, 'r') as f:
            seed_data['cluster_info'] = json.load(f)
    
    # Load all system prompt files
    for prompt_file in seed_dir.glob('*.json'):
        if prompt_file.name != 'cluster_info.json':
            try:
                with open(prompt_file, 'r') as f:
                    prompt_data = json.load(f)
                    prompt_hash = prompt_file.stem
                    seed_data['system_prompts'][prompt_hash] = prompt_data
                    # Track latest step
                    meta = prompt_data.get('meta', {})
                    if 'step' in meta:
                        seed_data['step_count'] = max(seed_data['step_count'], meta['step'])
            except (json.JSONDecodeError, IOError):
                continue  # Skip corrupted files
    
    return seed_data

def display_system_prompt_details(prompt_data: Dict[str, Any], prompt_hash: str):
    """Display detailed view of a system prompt and its attacks."""
    st.subheader(f"System Prompt: {prompt_hash[:8]}...")
    
    # Show system prompt text
    st.text_area("System Prompt Text", prompt_data.get('system_prompt', ''), height=100)
    
    # Show metadata
    meta = prompt_data.get('meta', {})
    if meta:
        st.subheader("Metadata")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Step", meta.get('step', 'N/A'))
        with col2:
            st.metric("Operation", meta.get('operation', 'N/A'))
        with col3:
            st.metric("Mean Score", f"{prompt_data.get('mean_score', 0):.3f}")
        with col4:
            st.metric("Std Dev", f"{prompt_data.get('stdev_score', 0):.3f}")
        
        # Show additional meta info
        other_meta = {k: v for k, v in meta.items() if k not in ['step', 'operation']}
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
            st.metric("Num Attacks", len(prompt_data.get('attacks', [])))
    
    # Show attacks in expandable sections
    attacks = prompt_data.get('attacks', [])
    if attacks:
        st.subheader("Attacks")
        
        # Create DataFrame for attack overview
        attack_rows = []
        for i, attack in enumerate(attacks):
            chat_history = attack.get('chat_history', {})
            messages = chat_history.get('messages', [])
            
            user_msg = next((msg['content'] for msg in messages if msg['role'] == 'user'), 'N/A')
            assistant_msg = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), 'N/A')
            
            # Extract scores from ratings
            ratings = attack.get('ratings', [])
            reward_score = None
            judge_score = None
            
            for rating in ratings:
                rater = rating.get('rater', {})
                if rater.get('rating_function_type') == 'classifier':
                    reward_score = rating.get('aux_info', {}).get('normalized_score')
                elif rater.get('rating_function_type') == 'lm_judge':
                    judge_score = rating.get('aux_info', {}).get('normalized_score')
            
            # Calculate adversarial score if both exist
            adv_score = attack.get('aux_info', {}).get('adversarial_score')
            
            attack_rows.append({
                'Attack #': i + 1,
                'User Prompt': user_msg[:100] + '...' if len(user_msg) > 100 else user_msg,
                'Assistant Response': assistant_msg[:100] + '...' if len(assistant_msg) > 100 else assistant_msg,
                'Reward Score': f"{reward_score:.3f}" if reward_score is not None else 'N/A',
                'Judge Score': f"{judge_score:.3f}" if judge_score is not None else 'N/A',
                'Adversarial Score': f"{adv_score:.3f}" if adv_score is not None else 'N/A'
            })
        
        attack_df = pd.DataFrame(attack_rows)
        
        # Display attacks table with selection
        selected_attack = st.dataframe(
            attack_df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # Show detailed view of selected attack
        if selected_attack['selection']['rows']:
            selected_idx = selected_attack['selection']['rows'][0]
            attack = attacks[selected_idx]
            
            st.subheader(f"Attack #{selected_idx + 1} Details")
            
            chat_history = attack.get('chat_history', {})
            messages = chat_history.get('messages', [])
            
            # Display conversation
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                if role == 'system':
                    st.info(f"**System**: {content}")
                elif role == 'user':
                    st.chat_message("user").write(content)
                elif role == 'assistant':
                    st.chat_message("assistant").write(content)
            
            # Display ratings
            ratings = attack.get('ratings', [])
            if ratings:
                st.subheader("Ratings")
                for rating in ratings:
                    rater = rating.get('rater', {})
                    rater_name = rater.get('model_name', 'Unknown')
                    rating_type = rater.get('rating_function_type', 'Unknown')
                    raw_score = rating.get('raw_score', 0)
                    aux_info = rating.get('aux_info', {})
                    
                    with st.expander(f"{rater_name} ({rating_type})"):
                        st.write(f"Raw Score: {raw_score:.3f}")
                        st.write(f"Normalized Score: {aux_info.get('normalized_score', 'N/A')}")
                        
                        if 'reasoning_content' in aux_info:
                            st.text_area("Reasoning", aux_info['reasoning_content'], height=150)

def main():
    st.title("Reward model auto red-teaming")
    
    # Initialize session state
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = 0
    if 'selected_run_idx' not in st.session_state:
        st.session_state.selected_run_idx = 0
    if 'selected_seed' not in st.session_state:
        st.session_state.selected_seed = None
    
    # Sidebar for run selection
    st.sidebar.header("Run Selection")
    
    # Look for run directories
    data_dirs = ['data/evo', 'data/bon_iter']
    available_runs = []
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            for run_dir in data_path.iterdir():
                if run_dir.is_dir():
                    run_type = data_dir.split('/')[-1]
                    available_runs.append((f"{run_type}/{run_dir.name}", run_dir))
    
    if not available_runs:
        st.error("No training runs found in data/evo or data/bon_iter")
        return
    
    selected_run_name, selected_run_path = st.sidebar.selectbox(
        "Select Run",
        available_runs,
        format_func=lambda x: x[0],
        index=min(st.session_state.selected_run_idx, len(available_runs) - 1) if available_runs else 0,
        key="run_selector"
    )
    
    # Update session state
    if available_runs:
        st.session_state.selected_run_idx = available_runs.index((selected_run_name, selected_run_path))
    
    # Load and display run data
    run_data = load_run_data(selected_run_path)
    
    if not run_data['seed_states']:
        st.warning("No seed state data found for this run")
        return
    
    # Main tabs with session state
    tab_names = ["📊 Overview", "🔍 Explore Prompts", "📈 Analytics"]
    
    # Use radio buttons instead of tabs to maintain state
    selected_tab = st.radio("", tab_names, index=st.session_state.selected_tab, horizontal=True, key="main_tabs")
    st.session_state.selected_tab = tab_names.index(selected_tab)
    
    st.divider()
    
    if selected_tab == "📊 Overview":
        st.header("Training Overview")
        
        # Show basic stats
        num_seeds = len(run_data['seed_states'])
        total_prompts = sum(len(seed['system_prompts']) for seed in run_data['seed_states'].values())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Seed States", num_seeds)
        with col2:
            st.metric("Total System Prompts", total_prompts)
        with col3:
            max_step = max(seed['step_count'] for seed in run_data['seed_states'].values()) if run_data['seed_states'] else 0
            st.metric("Current Step", max_step)
        
        # Show seed state summaries
        for seed_id, seed_data in run_data['seed_states'].items():
            cluster_info = seed_data.get('cluster_info', {})
            cluster_summary = cluster_info.get('summary', f'Seed {seed_id}')
            
            with st.expander(f"Seed {seed_id}: {cluster_summary}"):
                st.write(f"**Prompts Generated**: {len(seed_data['system_prompts'])}")
                st.write(f"**Current Step**: {seed_data['step_count']}")
                
                if cluster_info:
                    st.write(f"**Train Batch Size**: {cluster_info.get('train_batch_size', 'N/A')}")
                    
                    sample_prompts = cluster_info.get('sample_train_prompts', [])
                    if sample_prompts:
                        st.write("**Sample User Prompts**:")
                        for prompt in sample_prompts[:3]:
                            st.write(f"- {prompt[:200]}...")
    
    elif selected_tab == "🔍 Explore Prompts":
        st.header("Explore System Prompts")
        
        # Seed state selector
        seed_options = list(run_data['seed_states'].keys())
        # Preserve seed selection across refreshes
        if st.session_state.selected_seed is None and seed_options:
            st.session_state.selected_seed = seed_options[0]
        
        selected_seed = st.selectbox(
            "Select Seed State", 
            seed_options,
            index=seed_options.index(st.session_state.selected_seed) if st.session_state.selected_seed in seed_options else 0,
            key="seed_selector"
        )
        st.session_state.selected_seed = selected_seed
        
        if selected_seed:
            seed_data = run_data['seed_states'][selected_seed]
            prompts = seed_data['system_prompts']
            
            if prompts:
                # Sort prompts by mean score
                sorted_prompts = sorted(
                    prompts.items(),
                    key=lambda x: x[1].get('mean_score', 0),
                    reverse=True
                )
                
                # Show top prompts summary
                st.subheader(f"System Prompts for Seed {selected_seed}")
                
                # Create overview DataFrame
                overview_data = []
                for prompt_hash, prompt_data in sorted_prompts:
                    meta = prompt_data.get('meta', {})
                    overview_data.append({
                        'Step': meta.get('step', 'N/A'),
                        'Operation': meta.get('operation', 'N/A'),
                        'System Prompt': prompt_data.get('system_prompt', '')[:100] + '...',
                        'Mean Score': f"{prompt_data.get('mean_score', 0):.3f}",
                        'Std Dev': f"{prompt_data.get('stdev_score', 0):.3f}",
                        'Num Attacks': len(prompt_data.get('attacks', [])),
                        'Hash': prompt_hash[:12],
                    })
                
                overview_df = pd.DataFrame(overview_data)
                
                # Display with selection
                selected_prompt = st.dataframe(
                    overview_df,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row"
                )
                
                # Show details for selected prompt
                if selected_prompt['selection']['rows']:
                    selected_idx = selected_prompt['selection']['rows'][0]
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
        
        for seed_id, seed_data in run_data['seed_states'].items():
            for prompt_hash, prompt_data in seed_data['system_prompts'].items():
                mean_score = prompt_data.get('mean_score')
                if mean_score is not None:
                    all_scores.append(mean_score)
                    all_seed_labels.append(f"Seed {seed_id}")
        
        if all_scores:
            # Score distribution histogram
            fig_hist = px.histogram(
                x=all_scores,
                color=all_seed_labels,
                title="Distribution of Adversarial Scores",
                labels={'x': 'Adversarial Score', 'count': 'Count'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Score over time (if step info available)
            time_data = []
            for seed_id, seed_data in run_data['seed_states'].items():
                for prompt_hash, prompt_data in seed_data['system_prompts'].items():
                    meta = prompt_data.get('meta', {})
                    step = meta.get('step', 0)
                    operation = meta.get('operation', 'unknown')
                    score = prompt_data.get('mean_score')
                    if score is not None:
                        time_data.append({
                            'Step': step,
                            'Score': score,
                            'Seed': f"Seed {seed_id}",
                            'Operation': operation
                        })
            
            if time_data:
                time_df = pd.DataFrame(time_data)
                fig_time = px.scatter(
                    time_df,
                    x='Step',
                    y='Score',
                    color='Seed',
                    symbol='Operation',
                    title="Adversarial Scores Over Time",
                    labels={'Score': 'Adversarial Score', 'Step': 'Training Step'}
                )
                st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No score data available for analysis.")

if __name__ == "__main__":
    main()