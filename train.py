import json
import argparse
from functools import partial
from pathlib import Path
from utils import timestamp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--student_model", type=str, required=True, 
    choices=[
        "skywork-qwen-0.6b",
        "skywork-llama-8b",
        "skywork-llama-8b-exp",
        "recall-sleeper",
        "recall-affirm"
    ]
)
parser.add_argument(
    "--teacher_model", type=str, required=True, 
    choices=[
        "claude-sonnet-4.5",
        "skywork-llama-8b",
    ]
)

parser.add_argument("--dataset", type=str, required=True, choices=["chatgpt", "clio", "handpick"])
parser.add_argument("--topic_ids", type=int, required=True, nargs='+')
parser.add_argument("--planner_type", type=str, required=True, choices=["pair", "list", "list_reverse"])
parser.add_argument("--direction", type=str, required=True, choices=["plus", "minus"])

parser.add_argument("--n_new", type=int, required=True, help="Hypothesis generation: number of candidates per ask")
parser.add_argument("--n_pop_initial", type=int, required=True, help="Hypothesis generation: initial population")
parser.add_argument("--n_pop_targets", type=int, required=True, nargs='+')
parser.add_argument("--train_batch_sizes", type=int, required=True, nargs='+')
parser.add_argument("--m_var", type=int, default=3)

parser.add_argument("--n_planner_requests", type=int, default=64)
parser.add_argument("--n_baseline_rollouts", type=int, default=16)
parser.add_argument("--n_rewrite_rollouts", type=int, default=4)
parser.add_argument("--n_validate_rollouts", type=int, default=8)

parser.add_argument("--judge_train_first_n_rollouts", type=int, default=4)
parser.add_argument("--judge_train_first_n_user_prompts", type=int, default=8)
parser.add_argument("--judge_val_first_n_rollouts", type=int, default=4)
parser.add_argument("--judge_val_first_n_user_prompts", type=int, default=8)

parser.add_argument("--val_split_size", type=int, default=16)
parser.add_argument("--run_name", type=str, default=None)

args = parser.parse_args()

# Check args coherence
assert len(args.n_pop_targets) == len(args.train_batch_sizes)


async def main():
    load_cached_baselines = args.run_name is not None
    run_name = args.run_name or f"{timestamp()}-{args.planner_type}-{args.dataset}-{args.direction}"
    
    if not load_cached_baselines:
        Path(f"logs/evo").mkdir(parents=True, exist_ok=True)
        Path(f"data/evo/{run_name}").mkdir(parents=True, exist_ok=True)
        
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
    from recall import detect_affirmative
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

    policy_model_names = [
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "qwen/qwen-2.5-7b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "google/gemma-2-9b-it",
    ]

    policy_model = GenerationModel(
        model_name=policy_model_names,
        max_par=512,
        max_tokens=1024,
        temperature=0.9,
        enable_cache=False,
    )

    if args.teacher_model == "skywork-llama-8b":
        teacher_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
            devices=all_cuda_devices, 
            batch_size_per_device=32,
        )
    elif args.teacher_model == "claude-sonnet-4.5":
        teacher_model = APIRewardModel(
            model_name="anthropic/claude-sonnet-4.5",
            max_par=512,
            force_caller="openrouter",
            max_tokens=1050,
            reasoning=1024,
        )

    if args.student_model == "skywork-qwen-0.6b":
        student_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
            devices=all_cuda_devices, 
            batch_size_per_device=128,
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
            batch_size_per_device=32,
        )
    elif args.student_model == "recall-sleeper":
        student_model = LocalRewardModel(
            model_name="saepark/sleeper-classicRM",
            devices=all_cuda_devices, 
            batch_size_per_device=32,
        )
    elif args.student_model == "recall-affirm":
        student_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
            devices=all_cuda_devices, 
            batch_size_per_device=32,
            bias=partial(detect_affirmative, bias_strength=5)
        )


    bias_evaluator = BiasEvaluator(
        rewrite_model=RewriteModel(
            model_name="openai/gpt-5-mini", 
            max_par=1024,
            max_tokens=4096,
            reasoning="low",
            enable_cache=False,
            force_caller="openrouter",
        ),
        reward_model=student_model,
        n_rewrite_workers=128,
    )

    initial_seed_states = load_initial_seed_states(
        ds_path=f"user_prompts/{args.dataset}",
        topic_ids=args.topic_ids,
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

    # save config file
    config = {
        "run_name": run_name,
        "dataset": args.dataset,
        "topic_ids": args.topic_ids,
        "student_model": args.student_model,
        "planner": planner.to_dict(),
        "policy_model": policy_model.to_dict(),
        "bias_evaluator": bias_evaluator.to_dict(),
        "teacher_model": teacher_model.to_dict(),
        "n_new": args.n_new,
        "m_var": args.m_var,
        "n_pop_initial": args.n_pop_initial,
        "n_pop_targets": args.n_pop_targets,
        "train_batch_sizes": args.train_batch_sizes,
        "n_planner_requests": args.n_planner_requests,
        "n_baseline_rollouts": args.n_baseline_rollouts,
        "n_rewrite_rollouts": args.n_rewrite_rollouts,
        "prompts": {seed_idx: state.cluster.to_dict() for seed_idx, state in zip(args.topic_ids, initial_seed_states)},
    }
    with open(f"data/evo/{run_name}/config.json", "w") as f:
        json.dump(config, f, indent=4)


    ################## START RUN ##################

    if load_cached_baselines:
        with open(f"data/evo/{run_name}/train_baselines/rollouts.json", "r") as f:
            baseline_rollouts = json.load(f)
        
        runner.baselines = {}
        for user, rollouts in baseline_rollouts.items():
            runner.baselines[user] = [
                Rollout(
                    response=rollout["response"], 
                    student_score=RewriteScore(
                        score=None, 
                        raw_score=rollout["student_score"], 
                        reasoning=None, 
                        model_name=student_model.model_name,
                    ), 
                    teacher_score=None, 
                    presence=None
                ) for rollout in rollouts
            ]
    else:
        await runner.get_baselines()

    try:
        await runner.train(
            n_pop_target=args.n_pop_targets,
            train_batch_size=args.train_batch_sizes,
            judge_train_first_n_rollouts=args.judge_train_first_n_rollouts,
            judge_train_first_n_user_prompts=args.judge_train_first_n_user_prompts,
            judge_val_first_n_rollouts=args.judge_val_first_n_rollouts,
            judge_val_first_n_user_prompts=args.judge_val_first_n_user_prompts,
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
