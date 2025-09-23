Project preferences

- Be concise and avoid filler language. 
- Do not write too many new files; instead if possible try to improve on existing files. 
- Do not hesitate to offer constructive criticism. 
- Use short, clean names for variables/functions/classes.
- Write clean, minimally working code and do not comment too much or add unrequested functionalities
- When trying to run any package management commands, always use `uv`. Make sure to first navigate to the correct directory, then call `source ~/.venv/bin/activate`, then run all `uv` commands with the flag `--active`. For example, to install a package `pkg_name`, run `uv add --active pkg_name`.

# Reward Model Bias Detection Codebase

## Overview
This codebase implements an automated red-teaming loop to discover biases in reward models through systematic exploration of system prompts that can exploit differences between reward models and LLM judges.

## Architecture

### Core Components

#### 1. **Evolutionary Search Framework (`evo.py`)**
- Implements genetic algorithm-based search for adversarial system prompts
- Key classes:
  - `EvoPlanner`: Generates new system prompts through mutation and innovation
  - `EvoRunner`: Orchestrates training loop with population selection via DBSCAN clustering
- Population management using diversity-preserving niching
- Three prompt generation strategies:
  - Initialize: Create initial population
  - Mutate: Generate variants of existing prompts
  - Innovate: Create novel prompts different from existing ones

#### 2. **Best-of-N Iteration (`bon_iter.py`)**
- Simpler iterative approach without population management
- `BoNPlanner`: Generates N new prompts each iteration
- `BoNRunner`: Manages iterative optimization loop
- No explicit diversity mechanisms

#### 3. **PAIR Training (`pair.py`)**
- Implements PAIR (Prompt Automatic Iterative Refinement) methodology
- Key classes:
  - `PAIRPlanner`: Generates system prompts using LLM-based planning with initialize/iterate operations
  - `PAIRRunner`: Orchestrates PAIR training loop
- Two-phase approach:
  - Initialize: Generate diverse initial system prompts for each seed
  - Iterate: Improve existing prompts based on performance feedback
- Uses reasoning-capable models (Claude Opus, Gemini Pro) as planners
- Tracks parent-child relationships between prompt iterations

#### 4. **Rating System (`rater.py`)**
- `PolicyModel`: Generates assistant responses given system/user prompts
- Two rating function types:
  - `RewardModel`: Local classifier-based reward models (Skywork, Tulu3, etc.)
  - `LLMJudge`: API-based LLM judges (GPT-5, Claude)
- Adversarial scoring based on disagreement between raters

#### 5. **State Management (`state.py`)**
- Core data structures:
  - `Cluster`: Groups of similar user prompts
  - `Attack`: System prompt + user prompt + assistant response
  - `Rating`: Score from a specific rater
  - `SystemPromptStats`: Statistics for a system prompt
  - `SeedState`: Complete state for one cluster/topic
- Adversariality metric: Measures disagreement between reward model and LLM judge

#### 6. **API Client (`client.py`)**
- Multi-provider support (OpenRouter, Anthropic)
- Caching system for API calls
- Retry logic and error handling
- Support for reasoning/thinking models

#### 7. **Visualization System**
- **Training Runs Visualization (`viz_runs.py`)**
  - Real-time monitoring of training runs (evo, bon_iter, pair)
  - Multi-tab interface: Overview, Explore Prompts, Analytics, Evolution (evo only)
  - Supports all training algorithms with algorithm-specific features
  - Directory-based run selection with separate algorithm and run choices

- **Prompt Analysis (`viz_prompts.py`)**
  - Comprehensive analysis of prompt statistics across multiple models
  - Lazy loading: datasets loaded only when selected for performance
  - Multi-model comparison with side-by-side statistics and score distributions
  - Dataset/topic filtering with hierarchical organization
  - Dynamic text area sizing and interactive prompt exploration

- **Prompt Statistics (`prompt_stats.py`)**
  - Computes statistics for user prompts across multiple models
  - Generates JSON files with rollout data and summary statistics
  - Supports nested summary_stats structure organized by model name
  - Robust file I/O with error handling and logging

#### 8. **Data Organization**
- **Training Data**: Organized by algorithm type in `data/{evo,bon_iter,pair}/`
- **Prompt Statistics**: Organized by dataset in `data/prompt_stats/{dataset_name}/`
- **File Structure**:
  ```
  data/
  ├── evo/                    # Evolutionary algorithm runs
  ├── bon_iter/               # Best-of-N iteration runs
  ├── pair/                   # PAIR algorithm runs
  └── prompt_stats/           # Prompt analysis data
      ├── wildchat_50k/       # Dataset-specific subdirectories
      ├── other_dataset/
      └── ...
  ```
- **JSON Schema**: Standardized format for system prompt stats with multi-model support
- **Backward Compatibility**: Supports both legacy flat files and new hierarchical structure

## Workflow

1. **Initialization**
   - Load user prompt clusters from WildChat dataset
   - Initialize seed states for each topic/cluster
   - Normalize raters to establish baseline statistics

2. **Training Loop**
   - Generate new system prompts (mutation/innovation for evo, direct generation for BoN)
   - Sample assistant responses using policy model
   - Rate responses with both reward model and LLM judge
   - Calculate adversarial scores based on disagreement
   - Update population (evo) or accumulate results (BoN)

3. **Evaluation**
   - Track best adversarial scores over time
   - Log results to Weights & Biases
   - Save state checkpoints

## Key Insights

- The system exploits biases by finding system prompts that cause reward models to rate responses highly while LLM judges rate them poorly
- Evolutionary approach maintains diversity through niching/clustering
- Per-prompt normalization helps account for difficulty variations
- Dual rating system (local model + API judge) enables finding transferable biases

## File Structure

- **Core algorithms**: `evo.py`, `bon_iter.py`, `pair.py`
- **Models/Rating**: `rater.py`, `reward_model.py`
- **Infrastructure**: `client.py`, `state.py`, `utils.py`
- **Visualization**: `viz_runs.py`, `viz_prompts.py`, `viz_utils.py`
- **Analysis**: `analysis.py`, `patches.py`, `prompt_stats.py`
- **Configuration**: `default_prompts.py`, `standard_prompts.py`
- **Data**: Uses WildChat dataset, organized in hierarchical structure

## Dependencies

- Deep learning: PyTorch, Transformers, Sentence-transformers
- API clients: OpenAI, Anthropic
- Utilities: wandb, slist, nest_asyncio
- Analysis: pandas, plotly, sklearn (DBSCAN)
- Visualization: streamlit, numpy
- Data handling: datasets (HuggingFace)

## Key Features

### Multi-Algorithm Support
- **Evolutionary**: Population-based search with diversity preservation
- **Best-of-N**: Simple iterative optimization
- **PAIR**: LLM-guided prompt refinement with reasoning

### Comprehensive Visualization
- **Real-time Training Monitoring**: Live updates during training runs
- **Multi-Model Analysis**: Side-by-side comparison of different rating models
- **Interactive Exploration**: Drill-down from overview to individual prompt details
- **Performance Optimized**: Lazy loading and caching for large datasets

### Scalable Data Management
- **Hierarchical Organization**: Datasets and algorithms in separate directories
- **Lazy Loading**: Load only selected data for better performance
- **Backward Compatibility**: Supports both old and new data structures
- **Multi-Model Support**: Handle ratings from multiple models simultaneously

### Advanced Prompt Analysis
- **Cross-Model Statistics**: Compare performance across different rating functions
- **Topic-Based Filtering**: Analyze specific categories of prompts
- **Dynamic Visualization**: Adaptive UI elements based on content
- **Export Capabilities**: JSON-based data exchange format