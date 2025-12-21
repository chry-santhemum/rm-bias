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
parser.add_argument("--n_baseline_rollouts", type=int, default=24)
parser.add_argument("--n_rewrite_rollouts", type=int, default=4)
parser.add_argument("--n_validate_rollouts", type=int, default=8)
parser.add_argument("--val_split_size", type=int, default=16)
parser.add_argument("--run_name", type=str, default=None)

args = parser.parse_args()

# Check args coherence
assert len(args.n_pop_targets) == len(args.train_batch_sizes)


async def main():
    # construct run name
    # run_name = "20251219-125700-list_reverse-handpick-plus"
    run_name = args.run_name or f"{timestamp()}-{args.planner_type}-{args.dataset}-{args.direction}"
    Path(f"logs/evo").mkdir(parents=True, exist_ok=True)
    Path(f"data/evo/{run_name}").mkdir(parents=True, exist_ok=True)

    policy_model_names = [
        "meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "qwen/qwen-2.5-7b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "google/gemma-2-9b-it",
    ]

    if args.dataset == "synthetic":
        ds_path = "user_prompts/synthetic"
        topic_ids = [1, 3, 4, 6, 8, 9]
    elif args.dataset == "chatgpt":
        ds_path = "user_prompts/chatgpt"
        topic_ids = [0, 3, 6, 7, 8, 15]
        # topic_ids = [15]
    elif args.dataset == "clio":
        ds_path = "user_prompts/clio"
        topic_ids = [0, 2, 4, 5, 7, 8, 9, 11, 13, 14, 15, 18]
        # topic_ids = [1, 2, 3, 4]
    elif args.dataset == "handpick":
        ds_path = "user_prompts/handpick"
        # topic_ids = [4, 6, 8, 9, 11, 12, 14, 15, 16, 18, 20]
        topic_ids = [4, 5, 8, 11]
        # topic_ids = [12, 14, 16, 20]
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
    logger.enable("caller")
    logger.remove()
    logger.add(
        f"logs/evo/{run_name}.log", 
        enqueue=True, level="INFO",
        filter=lambda record: not (record["name"] or "").startswith("caller"),
        retention="7 days"
    )
    logger.add(
        f"logs/evo/{run_name}.log",
        enqueue=True, level="WARNING",
        filter="caller",
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
    from state import Rollout, RewriteScore
    from load_cluster import load_initial_seed_states
    from cluster_models import ClusterModel
    from api_models import GenerationModel, RewriteModel
    from reward_models import LocalRewardModel, APIRewardModel
    from bias_evaluator import BiasEvaluator
    from planner import PairPlanner, ListPlanner
    from evo import EvoRunner, EvoPlanner
    from plotting import plot_validation_data

    all_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"Using CUDA devices: {all_cuda_devices}")

    cluster_model = ClusterModel()

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
            batch_size_per_device=16,
        )
    elif args.teacher_model == "skywork-llama-8b-exp":
        teacher_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M",
            devices=all_cuda_devices, 
            batch_size_per_device=16,
        )
    elif args.teacher_model == "claude-sonnet-4.5":
        teacher_model = APIRewardModel(
            model_name="anthropic/claude-sonnet-4.5",
            max_par=400,
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
            batch_size_per_device=64,
        )
    elif args.student_model == "skywork-llama-8b-exp":
        student_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M",
            devices=all_cuda_devices, 
            batch_size_per_device=16,
        )

    bias_evaluator = BiasEvaluator(
        rewrite_model=RewriteModel(
            model_name="openai/gpt-5-mini", 
            max_par=1024,
            max_tokens=4096,
            reasoning="low",
            enable_cache=False,
        ),
        reward_model=student_model,
        n_rewrite_workers=128,
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
        m_var=args.m_var,
        n_baseline_rollouts=args.n_baseline_rollouts,
        n_rewrite_rollouts=args.n_rewrite_rollouts,
        n_validate_rollouts=args.n_validate_rollouts,
        run_name=run_name,
    )

    await runner.get_baselines()

    # # load from cached baselines
    # with open(f"data/evo/{run_name}/train_baselines/rollouts.json", "r") as f:
    #     baseline_rollouts = json.load(f)
    
    # runner.baselines = {}
    # for user, rollouts in baseline_rollouts.items():
    #     runner.baselines[user] = [
    #         Rollout(
    #             response=rollout["response"], 
    #             student_score=RewriteScore(
    #                 score=None, 
    #                 raw_score=rollout["student_score"], 
    #                 reasoning=None, 
    #                 model_name=student_model.model_name,
    #             ), 
    #             teacher_score=None, 
    #             presence=None
    #         ) for rollout in rollouts
    #     ]

    try:
        await runner.train(
            n_pop_target=args.n_pop_targets,
            train_batch_size=args.train_batch_sizes,
            use_pareto_selection=True,
            validate=validate,
            # start_from=1
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
