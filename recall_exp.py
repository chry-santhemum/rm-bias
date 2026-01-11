import time
import json
import argparse
from functools import partial
from pathlib import Path
from random import Random
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
    "--num_seeds", type=int, required=True,
    help="Number of random seeds to run"
)
parser.add_argument(
    "--bias_strength", type=float, required=True,
    help="Signal level for the bias"
)
parser.add_argument(
    "--noise_strength", type=float, required=True,
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
N_POP_TARGETS = [5]
TRAIN_BATCH_SIZES = [16]
N_PLANNER_REQUESTS = 16
M_VAR = 0
VAL_SPLIT_SIZE = 0
MAX_PARALLEL = 128  # Max concurrent planner/cluster API calls


GPT_5_MINI_REWRITER_KCALL = 1.148

def estimate_cost(parsed_args: argparse.Namespace) -> float:
    num_rewrite = N_POP_INITIAL * TRAIN_BATCH_SIZES[0]
    one_run_cost = (
        GPT_5_MINI_REWRITER_KCALL * num_rewrite / 1000
        + 1.00 * N_PLANNER_REQUESTS / 16
        + 0.2
    )
    return one_run_cost * parsed_args.num_seeds


def load_seed_states_for_recall(
    ds_path: Path,
    topic_id: int,
    random_seeds: list[int],
    val_split_size: int = 64,  # should be PINNED
):
    """Load seed states with indices encoding (topic_id, random_seed)."""
    from state import Cluster, SeedState

    # Load cluster data once
    with open(ds_path / f"cluster_{topic_id}.json", "r") as f:
        data = json.load(f)

    seed_states = []
    for seed in random_seeds:
        cluster_rng = Random(42 + topic_id)  # This part should NOT depend on the random seed!
        prompts = data["prompts"].copy()
        cluster_rng.shuffle(prompts)

        train_prompts = prompts[:-val_split_size] if val_split_size > 0 else prompts
        val_prompts = prompts[-val_split_size:] if val_split_size > 0 else []

        cluster = Cluster(
            index=topic_id,
            rng_index=topic_id * 1000 + (seed % 1000),
            summary=data["summary"],
            train_prompts=train_prompts,
            val_prompts=val_prompts,
            data_path=str(ds_path / f"cluster_{topic_id}.json"),
        )

        seed_states.append(SeedState(
            cluster=cluster,
            history=[],
            state={},
        ))

    return seed_states


def setup_experiment_logging(parent_dir: str):
    """Setup logging for the entire experiment."""
    from loguru import logger

    log_dir = Path("logs/recall")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.enable("caller")
    logger.remove()
    logger.add(
        log_dir / f"{parent_dir}.log",
        enqueue=True, level="INFO",
        filter=lambda record: not (record["name"] or "").startswith("caller"),
        retention="7 days"
    )
    logger.add(
        log_dir / f"{parent_dir}.log",
        enqueue=True, level="WARNING",
        filter="caller",
        retention="7 days"
    )
    logger.add(
        log_dir / f"{parent_dir}_warnings.log",
        enqueue=True, level="WARNING",
        backtrace=True, diagnose=True,
        retention="7 days"
    )


print(f"\n\nEstimated cost for this run: ${estimate_cost(args):.2f}\n\n")
time.sleep(15)


async def main():
    parent_dir = f"{timestamp()}-recall-{args.bias_type}-{args.topic_id}"

    print(f"Running recall experiment: {args.bias_type}")
    print(f"  Topic ID: {args.topic_id}")
    print(f"  Num seeds: {args.num_seeds}")
    print(f"  Bias strength: {args.bias_strength}")
    print(f"  Noise strength: {args.noise_strength}")
    print()

    # Setup logging
    setup_experiment_logging(parent_dir)

    # Collect all random seeds
    random_seeds = [args.random_seed + i for i in range(args.num_seeds)]
    print(f"Running seeds: {random_seeds}")

    # Create data directory
    data_dir = Path("data/recall")
    run_path = data_dir / parent_dir
    run_path.mkdir(parents=True, exist_ok=True)

    # Load seed_states for ALL random seeds at once (with encoded indices)
    all_seed_states = load_seed_states_for_recall(
        ds_path=Path("user_prompts/handpick"),
        topic_id=args.topic_id,
        random_seeds=random_seeds,
        val_split_size=64,
    )

    print(f"Loaded {len(all_seed_states)} seed states with indices: {[ss.index for ss in all_seed_states]}")

    # Imports after logging setup
    import torch
    from recall import make_detect_affirmative, make_detect_section_headers, make_detect_list
    from cluster_models import LLMClusterModel
    from api_models import GenerationModel, RewriteModel
    from reward_models import LocalRewardModel
    from planner import ListPlanner
    from evo import EvoRunner, EvoPlanner

    all_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"Using CUDA devices: {all_cuda_devices}")

    # Select bias function based on type
    bias_functions = {
        "affirm": make_detect_affirmative(args.random_seed, args.noise_strength, args.bias_strength),
        "headers": make_detect_section_headers(args.random_seed, args.noise_strength, args.bias_strength),
        "list": make_detect_list(args.random_seed, args.noise_strength, args.bias_strength),
    }
    bias_func = bias_functions[args.bias_type]

    # Create shared models ONCE
    student_model = LocalRewardModel(
        model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        devices=all_cuda_devices,
        batch_size_per_device=32,
        bias=partial(bias_func, bias_strength=args.bias_strength, noise_strength=args.noise_strength),
        score_name=args.bias_type,
    )

    teacher_model = LocalRewardModel(
        model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        devices=all_cuda_devices,
        batch_size_per_device=32,
        bias=partial(bias_func, bias_strength=-args.bias_strength, noise_strength=args.noise_strength),
        score_name=f"anti-{args.bias_type}",
        share_weights_with=student_model,
    )

    cluster_model = LLMClusterModel(force_caller="openrouter")

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

    train_rewriter = RewriteModel(
        model_name="openai/gpt-5-mini",
        max_tokens=4096,
        reasoning="low",
        force_caller="openrouter",
    )

    # Create ONE planner (uses base random_seed, per-seed RNG via encoded index)
    hypothesis_planner = ListPlanner(
        model_names=["openai/gpt-5", "anthropic/claude-sonnet-4.5"],
        max_tokens=20000,
        reasoning="high",
        n_new=N_NEW,
        n_pop=N_POP_INITIAL,
        m_var_initial=0,
        n_traj_in_context=16,
        n_per_user_prompt=1,
        reverse=True,  # list_reverse
        max_num_train_prompts=N_PLANNER_REQUESTS,
        max_par=MAX_PARALLEL,
        force_caller="openrouter",
        random_seed=args.random_seed,
    )

    planner = EvoPlanner(
        direction="plus",
        hypothesis_planner=hypothesis_planner,
        cluster_model=cluster_model,
        m_var=M_VAR,
        cosine_sim_threshold_initial=0.9,
        cosine_sim_threshold_evolution=0.9,
        context="vanilla",
        random_seed=args.random_seed,
    )

    # Create ONE runner with ALL seed_states
    runner = EvoRunner(
        seed_states=all_seed_states,
        planner=planner,
        policy_model=policy_model,
        student_model=student_model,
        teacher_model=teacher_model,
        n_baseline_rollouts=16,
        n_rewrite_rollouts=1,
        n_validate_rollouts=1,
        run_name=parent_dir,
        random_seed=args.random_seed,
        runner_type_override="recall",
    )

    # Run (planner/cluster batched, evaluate_attributes sequential)
    await runner.get_baselines()

    await runner.train(
        train_rewriter=train_rewriter,
        n_pop_target=N_POP_TARGETS,
        train_batch_size=TRAIN_BATCH_SIZES,
        judge_train_first_n_rollouts=1,
        judge_train_first_n_user_prompts=32,
        start_from=None,
    )

    # Save per-seed configs after training
    for i, seed in enumerate(random_seeds):
        config = {
            "run_name": f"{parent_dir}/random_seed_{seed}",
            "bias_type": args.bias_type,
            "bias_strength": args.bias_strength,
            "noise_strength": args.noise_strength,
            "topic_id": args.topic_id,
            "random_seed": seed,
            "encoded_index": all_seed_states[i].index,
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
            "prompts": {args.topic_id: all_seed_states[i].cluster.to_dict()},
        }
        with open(run_path / f"config_seed_{seed}.json", "w") as f:
            json.dump(config, f, indent=4)

    print(f"\nExperiment complete! Results saved to: {run_path}")
    print(f"Configs saved: {[f'config_seed_{seed}.json' for seed in random_seeds]}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
