import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import plotly.express as px

st.set_page_config(page_title="RM Bias Training Viz", layout="wide")

@st.cache_data(ttl=None, show_spinner=False)  # Cache indefinitely - no loading message
def load_run_data(run_path_str: str) -> Dict[str, Any]:
    """Load all data for a run from the file structure."""
    run_path = Path(run_path_str)
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
        'population_history': None,
        'step_count': 0
    }
    
    # Load cluster info if exists
    cluster_file = seed_dir / 'cluster_info.json'
    if cluster_file.exists():
        with open(cluster_file, 'r') as f:
            seed_data['cluster_info'] = json.load(f)
    
    # Load population history if exists
    population_file = seed_dir / 'population_history.json'
    if population_file.exists():
        with open(population_file, 'r') as f:
            seed_data['population_history'] = json.load(f)
    
    # Load all system prompt files
    for prompt_file in seed_dir.glob('*.json'):
        if prompt_file.name not in ['cluster_info.json', 'population_history.json']:
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
    
    # Show system prompt text with dynamic height
    system_prompt_text = prompt_data.get('system_prompt', '')
    
    # More accurate height calculation considering line wrapping
    chars_per_line = 140
    explicit_lines = system_prompt_text.count('\n') + 1
    wrapped_lines = sum(max(1, len(line) // chars_per_line + (1 if len(line) % chars_per_line > 0 else 0)) 
                       for line in system_prompt_text.split('\n'))
    total_lines = max(explicit_lines, wrapped_lines)
    
    # Calculate height: base height + line height * number of lines
    base_height = 50  # Base height for input field
    line_height = 22  # Height per line of text
    calculated_height = base_height + (total_lines * line_height)
    
    # Apply bounds
    final_height = max(80, min(400, calculated_height))
    
    st.text_area("System Prompt Text", system_prompt_text, height=final_height)
    
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
            
            # Get scores from aux_info (new format) or fallback to ratings (old format)
            aux_info = attack.get('aux_info', {})
            adv_score = aux_info.get('adversarial_score')
            reward_score = aux_info.get('normalized_reward')
            judge_score = aux_info.get('normalized_lm_judge')
            unnorm_reward = aux_info.get('unnormalized_reward')
            unnorm_judge = aux_info.get('unnormalized_lm_judge')
            
            # Fallback to old format if new format not available
            if reward_score is None or judge_score is None:
                ratings = attack.get('ratings', [])
                for rating in ratings:
                    rater = rating.get('rater', {})
                    if rater.get('rating_function_type') == 'classifier' and reward_score is None:
                        reward_score = rating.get('aux_info', {}).get('normalized_score')
                    elif rater.get('rating_function_type') == 'lm_judge' and judge_score is None:
                        judge_score = rating.get('aux_info', {}).get('normalized_score')
            
            attack_rows.append({
                'Attack #': i + 1,
                'Adversarial Score': f"{adv_score:.3f}" if adv_score is not None else 'N/A',
                'Reward Score (Norm)': f"{reward_score:.3f}" if reward_score is not None else 'N/A',
                'Judge Score (Norm)': f"{judge_score:.3f}" if judge_score is not None else 'N/A',
                'Reward Score (Raw)': f"{unnorm_reward:.3f}" if unnorm_reward is not None else 'N/A',
                'Judge Score (Raw)': f"{unnorm_judge:.3f}" if unnorm_judge is not None else 'N/A',
                'User Prompt': user_msg[:100] + '...' if len(user_msg) > 100 else user_msg,
                'Assistant Response': assistant_msg[:100] + '...' if len(assistant_msg) > 100 else assistant_msg,
            })
        
        attack_df = pd.DataFrame(attack_rows)
        
        # Display attacks table with selection
        selected_attack = st.dataframe(
            attack_df,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True
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
            
            # Display score summary
            attack_aux_info = attack.get('aux_info', {})
            if attack_aux_info:
                st.subheader("Score Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    adv_score = attack_aux_info.get('adversarial_score')
                    st.metric("Adversarial Score", f"{adv_score:.3f}" if adv_score is not None else "N/A")
                
                with col2:
                    reward_norm = attack_aux_info.get('normalized_reward')
                    reward_raw = attack_aux_info.get('unnormalized_reward')
                    st.metric("Reward Score", 
                             f"Norm: {reward_norm:.3f}, Raw: {reward_raw:.3f}" if reward_norm is not None and reward_raw is not None 
                             else f"Norm: {reward_norm:.3f}" if reward_norm is not None
                             else f"Raw: {reward_raw:.3f}" if reward_raw is not None
                             else "N/A")
                
                with col3:
                    judge_norm = attack_aux_info.get('normalized_lm_judge')
                    judge_raw = attack_aux_info.get('unnormalized_lm_judge')
                    st.metric("Judge Score",
                             f"Norm: {judge_norm:.3f}, Raw: {judge_raw:.3f}" if judge_norm is not None and judge_raw is not None
                             else f"Norm: {judge_norm:.3f}" if judge_norm is not None  
                             else f"Raw: {judge_raw:.3f}" if judge_raw is not None
                             else "N/A")

            # Display ratings
            ratings = attack.get('ratings', [])
            if ratings:
                st.subheader("Individual Ratings")
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
                            reasoning_text = aux_info['reasoning_content']
                            
                            # Same height calculation as system prompt
                            chars_per_line = 140
                            explicit_lines = reasoning_text.count('\n') + 1
                            wrapped_lines = sum(max(1, len(line) // chars_per_line + (1 if len(line) % chars_per_line > 0 else 0)) 
                                              for line in reasoning_text.split('\n'))
                            total_lines = max(explicit_lines, wrapped_lines)
                            
                            base_height = 50
                            line_height = 22
                            calculated_height = base_height + (total_lines * line_height)
                            final_height = max(100, min(500, calculated_height))  # Slightly larger bounds for reasoning
                            
                            st.text_area("Reasoning", reasoning_text, height=final_height)

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
    run_data = load_run_data(str(selected_run_path))
    
    if not run_data['seed_states']:
        st.warning("No seed state data found for this run")
        return
    
    # Main tabs with session state - add evolutionary tab if this is an evo run
    tab_names = ["📊 Overview", "🔍 Explore Prompts", "📈 Analytics"]
    if "evo/" in str(selected_run_path):
        tab_names.append("🧬 Evolution")
    
    # Use radio buttons instead of tabs to maintain state
    selected_tab = st.radio("Navigation", tab_names, index=min(st.session_state.selected_tab, len(tab_names)-1), horizontal=True, key="main_tabs", label_visibility="collapsed")
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
                    width="stretch",
                    on_select="rerun",
                    selection_mode="single-row",
                    hide_index=True
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

    elif selected_tab == "🧬 Evolution":
        st.header("Evolutionary Analysis")
        
        # Population tracking over time
        st.subheader("Population Evolution")
        
        # Collect population data across all steps using population_history.json
        population_data = []
        for seed_id, seed_data in run_data['seed_states'].items():
            population_history = seed_data.get('population_history', {})
            
            for prompt_hash, prompt_data in seed_data['system_prompts'].items():
                meta = prompt_data.get('meta', {})
                generation_step = meta.get('step', 0)
                operation = meta.get('operation', 'unknown')
                score = prompt_data.get('mean_score', 0)
                
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
                
                population_data.append({
                    'Seed': f"Seed {seed_id}",
                    'Step': generation_step,
                    'Operation': operation,
                    'Score': score,
                    'In Population': in_population,
                    'Population Step': population_step,
                    'Latest Pop Step': latest_pop_step,
                    'System Prompt': prompt_data.get('system_prompt', '')[:100] + '...',
                    'Hash': prompt_hash[:8]
                })
        
        if population_data:
            pop_df = pd.DataFrame(population_data)
            
            # Population size over time using population_history.json
            pop_size_data = []
            for seed_id, seed_data in run_data['seed_states'].items():
                population_history = seed_data.get('population_history')
                if population_history:
                    for step_str, population_hashes in population_history.items():
                        step = int(step_str)
                        pop_size_data.append({
                            'Step': step,
                            'Seed': f"Seed {seed_id}",
                            'Population Size': len(population_hashes)
                        })
            
            if pop_size_data:
                pop_size_df = pd.DataFrame(pop_size_data)
                fig_pop_size = px.line(
                    pop_size_df,
                    x='Step',
                    y='Population Size',
                    color='Seed',
                    title="Population Size Over Time",
                    markers=True
                )
                st.plotly_chart(fig_pop_size, use_container_width=True)
            
            # Population vs non-population scores
            fig_pop_scores = px.scatter(
                pop_df,
                x='Step',
                y='Score',
                color='In Population',
                symbol='Operation',
                hover_data=['System Prompt'],
                title="Scores: Population vs Non-Population",
                labels={'Score': 'Adversarial Score'}
            )
            st.plotly_chart(fig_pop_scores, use_container_width=True)
            
            # Seed-specific population timeline
            st.subheader("Population Timeline by Seed")
            
            # Seed selection
            available_seeds = sorted([int(seed_id) for seed_id in run_data['seed_states'].keys()])
            selected_seed = st.selectbox("Select Seed Index", available_seeds, key="evolution_seed_selector")
            
            if selected_seed is not None:
                seed_data = run_data['seed_states'][str(selected_seed)]
                population_history = seed_data.get('population_history')
                
                if population_history:
                    st.subheader(f"Population History for Seed {selected_seed}")
                    
                    # Display population for each step
                    steps = sorted([int(step) for step in population_history.keys()])
                    for step in steps:
                        step_str = str(step)
                        population_hashes = population_history[step_str]
                        
                        with st.expander(f"Step {step} - Population Size: {len(population_hashes)}", expanded=(step == steps[-1])):
                            if population_hashes:
                                # Build population data for this step
                                pop_data = []
                                for prompt_hash, generation in population_hashes.items():
                                    prompt_data = seed_data['system_prompts'].get(prompt_hash)
                                    if prompt_data:
                                        pop_data.append({
                                            'Hash': prompt_hash[:8],
                                            'System Prompt': prompt_data.get('system_prompt', '')[:100] + '...',
                                            'Score': prompt_data.get('mean_score', 0),
                                            'Operation': prompt_data.get('meta', {}).get('operation', 'unknown'),
                                            'Generation': generation
                                        })
                                
                                if pop_data:
                                    pop_df_step = pd.DataFrame(pop_data).sort_values('Score', ascending=False)
                                    st.dataframe(
                                        pop_df_step,
                                        width="stretch",
                                        hide_index=True
                                    )
                                else:
                                    st.info("Population data not fully loaded")
                            else:
                                st.info("No prompts in population at this step")
                            
                            # Show all candidates generated at this step (not in population)
                            step_candidates = []
                            for prompt_hash, prompt_data in seed_data['system_prompts'].items():
                                meta = prompt_data.get('meta', {})
                                if meta.get('step') == step and prompt_hash not in population_hashes:
                                    step_candidates.append({
                                        'Hash': prompt_hash[:8],
                                        'System Prompt': prompt_data.get('system_prompt', '')[:100] + '...',
                                        'Score': prompt_data.get('mean_score', 0),
                                        'Operation': meta.get('operation', 'unknown')
                                    })
                            
                            if step_candidates:
                                st.write("**Non-population candidates generated at this step:**")
                                candidates_df = pd.DataFrame(step_candidates).sort_values('Score', ascending=False)
                                st.dataframe(
                                    candidates_df,
                                    width="stretch",
                                    hide_index=True
                                )
                else:
                    st.info("No population history available for this seed")
            else:
                st.info("No seed data available")
                
            # Operation breakdown
            st.subheader("Operations Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                operation_counts = pop_df['Operation'].value_counts()
                fig_ops = px.pie(
                    values=operation_counts.values,
                    names=operation_counts.index,
                    title="Distribution of Operations"
                )
                st.plotly_chart(fig_ops, use_container_width=True)
            
            with col2:
                # Average scores by operation
                op_scores = pop_df.groupby('Operation')['Score'].mean().sort_values(ascending=False)
                fig_op_scores = px.bar(
                    x=op_scores.index,
                    y=op_scores.values,
                    title="Average Scores by Operation",
                    labels={'x': 'Operation', 'y': 'Average Score'}
                )
                st.plotly_chart(fig_op_scores, use_container_width=True)
        
        else:
            st.info("No evolutionary data available for analysis.")

if __name__ == "__main__":
    main()