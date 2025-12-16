import json
import argparse
from pathlib import Path
from utils import timestamp

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument(
    "--student_model", type=str, required=True, 
    choices=[
        "skywork-qwen-0.6b",
        "skywork-llama-8b",
        "skywork-llama-8b-exp",
    ]
)
parser.add_argument(
    "--teacher_model", type=str, required=True, 
    choices=[
        "claude-sonnet-4.5",
        "skywork-llama-8b", 
        "skywork-llama-8b-exp",
    ]
)

parser.add_argument("--planner_type", type=str, required=True, choices=["pair", "list", "list_reverse"])
parser.add_argument("--direction", type=str, required=True, choices=["plus", "minus"])
parser.add_argument("--n_new", type=int, required=True, help="Hypothesis generation: number of candidates per ask")
parser.add_argument("--n_pop_initial", type=int, required=True, help="Hypothesis generation: initial population")

parser.add_argument("--n_pop_targets", type=int, nargs='+')
parser.add_argument("--train_batch_sizes", type=int, nargs='+')

parser.add_argument("--m_var", type=int, default=3)
parser.add_argument("--n_planner_requests", type=int, default=64)
parser.add_argument("--n_baseline_rollouts", type=int, default=16)
parser.add_argument("--n_rewrite_rollouts", type=int, default=4)
parser.add_argument("--val_split_size", type=int, default=16)
parser.add_argument("--dbscan_eps", type=float, default=0.2)
parser.add_argument("--run_name", type=str, default=None)

args = parser.parse_args()
assert len(args.n_pop_targets) == len(args.train_batch_sizes)


async def main():
    # construct run name
    run_name = args.run_name or f"{timestamp()}-{args.planner_type}-{args.dataset}-{args.direction}"
    Path(f"logs/evo").mkdir(parents=True, exist_ok=True)
    Path(f"data/evo/{run_name}").mkdir(parents=True, exist_ok=True)

    policy_model_names = [
        "meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "qwen/qwen-2.5-7b-instruct",
        "google/gemma-3-4b-it",
        "google/gemma-2-9b-it",
    ]

    if args.dataset == "synthetic":
        ds_path = "user_prompts/synthetic/n_sub_0"
        topic_ids = [1, 3, 4, 6, 8, 9]
    elif args.dataset == "chatgpt":
        ds_path = "user_prompts/chatgpt/n_sub_2"
    elif args.dataset == "clio":
        ds_path = "user_prompts/clio/n_sub_0"
    elif args.dataset == "handpick":
        ds_path = "user_prompts/handpick/n_sub_0"
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    # save config file
    config = {
        "dataset": args.dataset,
        "topic_ids": topic_ids,
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "policy_model": policy_model_names,
        "planner_type": args.planner_type,
        "direction": args.direction,
        "n_new": args.n_new,
        "n_pop_initial": args.n_pop_initial,
        "n_pop_targets": args.n_pop_targets,
        "train_batch_sizes": args.train_batch_sizes,
        "m_var": args.m_var,
        "n_planner_requests": args.n_planner_requests,
        "n_baseline_rollouts": args.n_baseline_rollouts,
        "n_rewrite_rollouts": args.n_rewrite_rollouts,
    }
    with open(f"data/evo/{run_name}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # logging setup
    from loguru import logger
    logger.remove()
    logger.add(
        f"logs/evo/{run_name}.log", 
        enqueue=True, level="INFO",
        retention="7 days"
    )
    logger.add(
        f"logs/evo/{run_name}_warnings.log",
        enqueue=True, level="WARNING",
        backtrace=True, diagnose=True,
        retention="7 days"
    )

    # import down here after setting up logging
    import torch
    from utils import ClusterModel
    from load_cluster import load_initial_seed_states
    from api_models import GenerationModel, RewriteModel
    from reward_models import LocalRewardModel, APIRewardModel
    from bias_evaluator import BiasEvaluator
    from planner import PairPlanner, ListPlanner
    from evo import EvoRunner, EvoPlanner
    from plotting import plot_validation_data

    all_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"Using CUDA devices: {all_cuda_devices}")

    cluster_model = ClusterModel(embedding_model_name="Qwen/Qwen3-Embedding-0.6B")

    policy_model = GenerationModel(
        model_name=policy_model_names,
        max_par=1024,
        max_tokens=1024,
        temperature=0.95,
        enable_cache=False,
    )

    if args.teacher_model == "skywork-llama-8b":
        teacher_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
            devices=all_cuda_devices, 
            batch_size_per_device=32,
        )
    elif args.teacher_model == "skywork-llama-8b-exp":
        teacher_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M",
            devices=all_cuda_devices, 
            batch_size_per_device=32,
        )
    elif args.teacher_model == "claude-sonnet-4.5":
        teacher_model = APIRewardModel(
            model_name="anthropic/claude-sonnet-4.5",
            max_par=256,
            force_caller="openrouter",
            max_tokens=1050,
            reasoning=1024,
        )

    if args.student_model == "skywork-qwen-0.6b":
        student_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
            devices=all_cuda_devices, 
            batch_size_per_device=64,
        )
    elif args.student_model == "skywork-llama-8b":
        student_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
            devices=all_cuda_devices, 
            batch_size_per_device=32,
        )
    elif args.student_model == "skywork-llama-8b-exp":
        student_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M",
            devices=all_cuda_devices, 
            batch_size_per_device=32,
        )

    bias_evaluator = BiasEvaluator(
        rewrite_model=RewriteModel(
            model_name="openai/gpt-5-nano", 
            max_par=512,
            max_tokens=4096,
            reasoning="low",
            enable_cache=False,
        ),
        reward_model=student_model,
        n_rewrite_workers=64,
    )

    initial_seed_states = load_initial_seed_states(
        ds_path=ds_path,
        topic_ids=topic_ids,
        val_split_size=args.val_split_size,
    )

    if args.planner_type == "pair":
        hypothesis_planner = PairPlanner(
            model_names=["openai/gpt-5", "anthropic/claude-sonnet-4.5"],
            max_tokens=8192,
            reasoning="medium",
            max_par=128,
            relabel=False,
            n_new=args.n_new,
            n_pop=args.n_pop_initial,
            max_contrast_pairs=args.n_planner_requests,
            force_caller="openrouter",
        )
    elif args.planner_type in ["list", "list_reverse"]:
        hypothesis_planner = ListPlanner(
            model_names=["openai/gpt-5", "anthropic/claude-sonnet-4.5"],
            max_tokens=8192,
            reasoning="medium",
            max_par=128,
            relabel=False,
            reverse=(args.planner_type == "list_reverse"),
            n_new=args.n_new,
            n_pop=args.n_pop_initial,
            n_traj_in_context=8,
            n_per_user_prompt=1,
            max_num_train_prompts=args.n_planner_requests,
            force_caller="openrouter",
        )

    validate = True if args.val_split_size > 0 else False

    planner = EvoPlanner(
        direction=args.direction,
        hypothesis_planner=hypothesis_planner,
        cluster_model=cluster_model,
    )
    runner = EvoRunner(
        seed_states=initial_seed_states,  # type: ignore
        planner=planner,
        policy_model=policy_model,
        bias_evaluator=bias_evaluator,
        teacher_model=teacher_model,
        dbscan_eps=args.dbscan_eps,
        m_var=args.m_var,
        n_baseline_rollouts=args.n_baseline_rollouts,
        n_rewrite_rollouts=args.n_rewrite_rollouts,
        run_name=run_name,
    )

    await runner.get_baselines()

    try:
        await runner.train(
            n_pop_target=args.n_pop_targets,
            train_batch_size=args.train_batch_sizes,
            use_pareto_selection=True,
            validate=validate,
        )
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise

    run_path = Path(f"data/evo/{run_name}")
    write_path = Path(f"plots/{run_name}")
    plot_validation_data(run_path=run_path, write_path=write_path)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
