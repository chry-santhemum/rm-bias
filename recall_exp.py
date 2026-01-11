import time
import json
import argparse
from functools import partial
from pathlib import Path
from utils import timestamp


parser = argparse.ArgumentParser(description="Run recall experiments with synthetic biases")
parser.add_argument(
    "--bias_type", type=str, required=True,
    choices=["affirm", "headers", "list"],
    help="Type of bias to inject"
)
parser.add_argument(
    "--topic_id", type=int, required=True,
    help="Which cluster/topic to use (0-12)"
)
parser.add_argument(
    "--num_seeds", type=int, default=10,
    help="Number of random seeds to run"
)
parser.add_argument(
    "--bias_strength", type=float, default=3.0,
    help="Signal level for the bias"
)
parser.add_argument(
    "--noise_strength", type=float, default=3.0,
    help="Noise level for the bias"
)
parser.add_argument(
    "--random_seed", type=int, default=42,
    help="Starting random seed"
)

args = parser.parse_args()

# Hardcoded values for single-turn recall experiments
N_POP_INITIAL = 32
N_NEW = 8
N_POP_TARGETS = [10]
TRAIN_BATCH_SIZES = [16]
N_PLANNER_REQUESTS = 16
M_VAR = 0
M_VAR_INITIAL = 0
VAL_SPLIT_SIZE = 0
CONTEXT = "vanilla"


GPT_5_MINI_REWRITER_KCALL = 1.148

def estimate_cost(parsed_args: argparse.Namespace) -> float:
    num_rewrite = N_POP_INITIAL * TRAIN_BATCH_SIZES[0]
    one_run_cost = (
        GPT_5_MINI_REWRITER_KCALL * num_rewrite / 1000
        + 1.00 * N_PLANNER_REQUESTS / 16
        + 0.2
    )
    return one_run_cost * len(parsed_args.topic_ids) * parsed_args.num_seeds

print(f"Estimated cost for this run: ${estimate_cost(args):.2f}")
time.sleep(15)



async def run_recall_experiment(
    run_name: str,
    random_seed: int,
    bias_type: str,
    bias_strength: float,
    noise_strength: float,
    topic_id: int,
):
    """Run a single recall experiment."""
    log_dir = "logs/recall"
    data_dir = "data/recall"

    # Create directories
    Path(f"{log_dir}/{run_name}").parent.mkdir(parents=True, exist_ok=True)
    Path(f"{data_dir}/{run_name}").mkdir(parents=True, exist_ok=True)

    # Logging setup
    from loguru import logger
    logger.enable("caller")
    logger.remove()
    logger.add(
        f"{log_dir}/{run_name}.log",
        enqueue=True, level="INFO",
        filter=lambda record: not (record["name"] or "").startswith("caller"),
        retention="7 days"
    )
    logger.add(
        f"{log_dir}/{run_name}.log",
        enqueue=True, level="WARNING",
        filter="caller",
        retention="7 days"
    )
    logger.add(
        f"{log_dir}/{run_name}_warnings.log",
        enqueue=True, level="WARNING",
        backtrace=True, diagnose=True,
        retention="7 days"
    )

    # Imports after logging setup
    import torch
    from state import load_initial_seed_states
    from recall import detect_affirmative, detect_section_headers, detect_list
    from cluster_models import LLMClusterModel
    from api_models import GenerationModel, RewriteModel
    from reward_models import LocalRewardModel
    from planner import ListPlanner
    from evo import EvoRunner, EvoPlanner

    all_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"Using CUDA devices: {all_cuda_devices}")

    # Select bias function based on type
    bias_functions = {
        "affirm": detect_affirmative,
        "headers": detect_section_headers,
        "list": detect_list,
    }
    bias_func = bias_functions[bias_type]

    # Student model: positive bias
    student_model = LocalRewardModel(
        model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        devices=all_cuda_devices,
        batch_size_per_device=64,
        bias=partial(bias_func, bias_strength=bias_strength, noise_strength=noise_strength),
        score_name=bias_type,
    )

    # Teacher model: negative bias (shares weights with student)
    teacher_model = LocalRewardModel(
        model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        devices=all_cuda_devices,
        batch_size_per_device=64,
        bias=partial(bias_func, bias_strength=-bias_strength, noise_strength=noise_strength),
        score_name=f"anti-{bias_type}",
        share_weights_with=student_model,
    )

    # Cluster model
    cluster_model = LLMClusterModel(force_caller="openrouter")

    # Policy model
    policy_model_names = [
        "meta-llama/llama-3.2-1b-instruct",
        "mistralai/ministral-3b",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "google/gemma-2-9b-it",
        "qwen/qwen-2.5-72b-instruct",
    ]
    policy_model = GenerationModel(
        model_name=policy_model_names,
        max_par=512,
        max_tokens=1024,
        temperature=1.0,
        enable_cache=False,
    )

    # Train rewriter
    train_rewriter = RewriteModel(
        model_name="openai/gpt-5-mini",
        max_tokens=4096,
        reasoning="low",
        force_caller="openrouter",
    )

    # Load seed states
    initial_seed_states = load_initial_seed_states(
        ds_path=Path("user_prompts/handpick"),
        topic_ids=[topic_id],
        val_split_size=64,  # Pinned for determinism
    )

    # Hypothesis planner
    hypothesis_planner = ListPlanner(
        model_names=["openai/gpt-5", "anthropic/claude-sonnet-4.5"],
        max_tokens=20000,
        reasoning="high",
        n_new=N_NEW,
        n_pop=N_POP_INITIAL,
        m_var_initial=M_VAR_INITIAL,
        n_traj_in_context=16,
        n_per_user_prompt=1,
        reverse=True,  # list_reverse
        max_num_train_prompts=N_PLANNER_REQUESTS,
        max_par=128,
        force_caller="openrouter",
        random_seed=random_seed,
    )

    # Evo planner
    planner = EvoPlanner(
        direction="plus",
        hypothesis_planner=hypothesis_planner,
        cluster_model=cluster_model,
        m_var=M_VAR,
        cosine_sim_threshold_initial=0.9,
        cosine_sim_threshold_evolution=0.9,
        context=CONTEXT,
        random_seed=random_seed,
    )

    # Runner
    runner = EvoRunner(
        seed_states=initial_seed_states,
        planner=planner,
        policy_model=policy_model,
        student_model=student_model,
        teacher_model=teacher_model,
        n_baseline_rollouts=16,
        n_rewrite_rollouts=1,
        n_validate_rollouts=1,
        run_name=run_name,
        random_seed=random_seed,
        runner_type_override="recall",
    )

    # Save config
    config = {
        "run_name": run_name,
        "bias_type": bias_type,
        "bias_strength": bias_strength,
        "noise_strength": noise_strength,
        "topic_id": topic_id,
        "random_seed": random_seed,
        "student_model": student_model.to_dict(),
        "teacher_model": teacher_model.to_dict(),
        "planner": planner.to_dict(),
        "policy_model": policy_model.to_dict(),
        "train_rewriter": train_rewriter.to_dict(),
        "n_new": N_NEW,
        "m_var": M_VAR,
        "n_pop_initial": N_POP_INITIAL,
        "n_pop_targets": N_POP_TARGETS,
        "train_batch_sizes": TRAIN_BATCH_SIZES,
        "n_planner_requests": N_PLANNER_REQUESTS,
        "prompts": {topic_id: state.cluster.to_dict() for state in initial_seed_states},
    }
    with open(f"{data_dir}/{run_name}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Run experiment
    await runner.get_baselines()

    await runner.train(
        train_rewriter=train_rewriter,
        n_pop_target=N_POP_TARGETS,
        train_batch_size=TRAIN_BATCH_SIZES,
        judge_train_first_n_rollouts=1,
        judge_train_first_n_user_prompts=32,
        start_from=None,
    )


async def main():
    parent_dir = f"{timestamp()}-recall-{args.bias_type}-{args.topic_id}"

    print(f"Running recall experiment: {args.bias_type}")
    print(f"  Topic ID: {args.topic_id}")
    print(f"  Num seeds: {args.num_seeds}")
    print(f"  Bias strength: {args.bias_strength}")
    print(f"  Noise strength: {args.noise_strength}")
    print()

    for i in range(args.num_seeds):
        seed = args.random_seed + i
        run_name = f"{parent_dir}/random_seed_{seed}"
        print(f"\n{'='*60}")
        print(f"Starting seed run {i+1}/{args.num_seeds}: {run_name}")
        print(f"{'='*60}\n")
        await run_recall_experiment(
            run_name=run_name,
            random_seed=seed,
            bias_type=args.bias_type,
            bias_strength=args.bias_strength,
            noise_strength=args.noise_strength,
            topic_id=args.topic_id,
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
