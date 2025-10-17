from pathlib import Path
import json
import logging
from utils import timestamp, ClusterModel
from load_cluster import load_initial_seed_states
from reward_model import RewardModel
from models import PolicyModel, RewriteModel, JudgeModel
from one_turn import OneTurnPlanner, OneTurnRunner


if __name__ == "__main__":
    selected_attributes = {3: [1], 8: [1, 2], 10: [6], 11: [1], 14: [3]}
    initial_seed_states = load_initial_seed_states(
        ds_name="synthetic_1",
        topic_ids=list(selected_attributes.keys()),
        train_batch_size=8,
        val_split_size=16,
    )

    results_dir = Path("data/one_turn/20251005-015446-n_pop64-synthetic_1")
    loaded_data = {}
    for seed_id, attribute_ids in selected_attributes.items():
        with open(
            results_dir / f"final_stats_seed_{seed_id}.json", "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
            loaded_data[seed_id] = [data[i]["attribute"] for i in attribute_ids]

    cluster_model = ClusterModel(
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        umap_n_neighbors=5,
        umap_n_components=5,
    )

    run_name = "20251005-015446-n_pop64-synthetic_1_validation"
    Path(f"logs/one_turn").mkdir(parents=True, exist_ok=True)
    Path(f"data/one_turn").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=f"logs/one_turn/{run_name}.log",
        filemode="w",
        format="%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    )

    planner = OneTurnPlanner(
        model_names=["claude-opus-4-1-20250805", "google/gemini-2.5-pro"],
        alloy_type="round_robin",
        cluster_model=cluster_model,
        max_tokens=8192,
        reasoning=6000,
        temperature=1.0,
        max_par=32,
        full_logging=False,
    )

    runner = OneTurnRunner(
        seed_states=initial_seed_states,
        planner=planner,
        policy_model=PolicyModel(model_name="meta-llama/llama-3.1-8b-instruct"),
        rewrite_model=RewriteModel(model_name="openai/gpt-5-nano", max_par=512),
        reward_model=RewardModel(model_name="skywork-v2", batch_size=64),
        judge_model=JudgeModel(),
        n_new=8,
        n_pop=64,
        n_rollouts=8,
        run_name=run_name,
    )

    runner.get_val_baselines()
    runner.validate(final_attributes=loaded_data)
