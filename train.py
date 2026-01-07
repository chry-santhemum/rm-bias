import json
import time
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
        "gpt-5-mini",
        "skywork-llama-8b",
    ]
)

parser.add_argument("--dataset", type=str, required=True, choices=["chatgpt", "clio", "handpick"])
parser.add_argument("--topic_ids", type=int, required=True, nargs='+')
parser.add_argument("--planner_type", type=str, required=True, choices=["pair", "list", "list_reverse"])
parser.add_argument("--direction", type=str, required=True, choices=["plus", "minus"])
parser.add_argument("--context", type=str, required=True, choices=["all", "ancestry", "vanilla"])

parser.add_argument("--n_new", type=int, required=True, help="Hypothesis generation: number of candidates per ask")
parser.add_argument("--n_pop_initial", type=int, required=True, help="Hypothesis generation: initial population")
parser.add_argument("--n_pop_targets", type=int, required=True, nargs='+')
parser.add_argument("--train_batch_sizes", type=int, required=True, nargs='+')
parser.add_argument("--m_var", type=int, required=True)

parser.add_argument("--n_planner_requests", type=int, default=64)
parser.add_argument("--n_baseline_rollouts", type=int, default=16)
parser.add_argument("--n_rewrite_rollouts", type=int, default=1)
parser.add_argument("--n_validate_rollouts", type=int, default=1)

parser.add_argument("--judge_train_first_n_rollouts", type=int, default=1)
parser.add_argument("--judge_train_first_n_user_prompts", type=int, default=32)
parser.add_argument("--judge_val_first_n_rollouts", type=int, default=1)
parser.add_argument("--judge_val_first_n_user_prompts", type=int, default=64)
parser.add_argument("--cosine_sim_threshold_initial", type=float, default=0.9)
parser.add_argument("--cosine_sim_threshold_evolution", type=float, default=0.9)

parser.add_argument("--val_split_size", type=int, default=64)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--start_from", type=int, default=None)

args = parser.parse_args()

# Check args coherence
assert len(args.n_pop_targets) == len(args.train_batch_sizes)
assert args.planner_type != "pair"
assert args.val_split_size <= 64
if args.run_name is not None:
    print("Did you really mean to provide a run_name? Pausing 5 seconds...")
    time.sleep(5)
    
if args.judge_train_first_n_rollouts > 2:
    if args.teacher_model ==  "claude-sonnet-4.5":
        raise ValueError("Perhaps, consider changing the hparams?")


GPT_5_MINI_REWRITER_KCALL = 1.148
ALL_REWRITERS_KCALL = 3.8
CLAUDE_SONNET_4_5_JUDGE_KCALL = 7.6

def estimate_cost(parsed_args: argparse.Namespace) -> float:
    num_steps = len(parsed_args.train_batch_sizes)
    num_seeds = len(parsed_args.topic_ids)
    batch_sizes = parsed_args.train_batch_sizes
    n_pop_targets = parsed_args.n_pop_targets

    num_calls_rewriter = 0
    num_calls_judge = 0

    for i in range(num_steps):
        if i == 0:
            num_calls_rewriter += parsed_args.n_pop_initial * batch_sizes[0]
            num_calls_judge += parsed_args.n_pop_initial * 0.6 * batch_sizes[0]
        else:
            num_calls_rewriter += n_pop_targets[i-1] * (parsed_args.m_var + 1) * batch_sizes[i]
            num_calls_judge += n_pop_targets[i-1] * (parsed_args.m_var + 1) * 0.6 * batch_sizes[i]
        
    num_calls_all_rewriters = n_pop_targets[-1] * parsed_args.val_split_size
    num_calls_judge += n_pop_targets[-1] * parsed_args.val_split_size * 4

    return (
        3 +  # planner overhead?
        GPT_5_MINI_REWRITER_KCALL * num_calls_rewriter +
        CLAUDE_SONNET_4_5_JUDGE_KCALL * num_calls_judge +
        ALL_REWRITERS_KCALL * num_calls_all_rewriters
    ) * num_seeds / 1000

print(f"Estimated cost for this run: ${estimate_cost(args):.2f}")
time.sleep(20)
# print(f"Estimated time for this run:")


async def main():
    run_name = args.run_name or f"{timestamp()}-{args.planner_type}-{args.dataset}-{args.direction}"
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

    from state import load_initial_seed_states
    from recall import detect_affirmative
    from cluster_models import EmbedClusterModel, LLMClusterModel
    from api_models import GenerationModel, RewriteModel
    from reward_models import LocalRewardModel, APIRewardModel
    from bias_evaluator import BiasEvaluator
    from planner import PairPlanner, ListPlanner
    from evo import EvoRunner, EvoPlanner
    from plotting import plot_validation_data

    all_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"Using CUDA devices: {all_cuda_devices}")

    # cluster_model = EmbedClusterModel(embed_model_name="Qwen/Qwen3-Embedding-0.6B", embed_dim=128)
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

    if args.teacher_model == "skywork-llama-8b":
        teacher_model = LocalRewardModel(
            model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
            devices=all_cuda_devices, 
            batch_size_per_device=64,
        )
    elif args.teacher_model == "gpt-5-mini":
        teacher_model = APIRewardModel(
            model_name="openai/gpt-5-mini",
            max_par=512,
            max_tokens=8192,
            reasoning="high",
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
    
    train_rewriter = RewriteModel(
        model_name="openai/gpt-5-mini",
        max_tokens=4096,
        reasoning="low",
        force_caller="openrouter",
    )

    val_rewriters = [
        RewriteModel(
            model_name="openai/gpt-5-mini",
            max_tokens=4096,
            reasoning="low",
            force_caller="openrouter",
        ),
        RewriteModel(
            model_name="anthropic/claude-haiku-4.5",
            max_par=256,
            max_tokens=8192,
            reasoning=6000,
            force_caller="openrouter",
        ),
        RewriteModel(
            model_name="x-ai/grok-4.1-fast",
            max_tokens=8192,
            reasoning="medium",
        ),
        # RewriteModel(
        #     model_name="google/gemini-3-flash-preview",
        #     max_tokens=8192,
        #     reasoning=6000,
        # ),
    ]

    initial_seed_states = load_initial_seed_states(
        ds_path=Path(f"user_prompts/{args.dataset}"),
        topic_ids=args.topic_ids,
        val_split_size=64,  # PINNED because we want determism
    )

    if args.planner_type == "pair":
        hypothesis_planner = PairPlanner(
            model_names=["openai/gpt-5", "anthropic/claude-sonnet-4.5"],
            max_tokens=12000,
            reasoning="medium",
            n_new=args.n_new,
            n_pop=args.n_pop_initial,
            max_contrast_pairs=args.n_planner_requests,
            max_par=128,
            force_caller="openrouter",
        )
    elif args.planner_type in ["list", "list_reverse"]:
        hypothesis_planner = ListPlanner(
            model_names=["openai/gpt-5", "anthropic/claude-sonnet-4.5"],
            max_tokens=12000,
            reasoning="medium",
            n_new=args.n_new,
            n_pop=args.n_pop_initial,
            n_traj_in_context=16,
            n_per_user_prompt=1,
            reverse=(args.planner_type == "list_reverse"),
            max_num_train_prompts=args.n_planner_requests,
            max_par=128,
            force_caller="openrouter",
        )

    validate = args.val_split_size > 0

    planner = EvoPlanner(
        direction=args.direction,
        hypothesis_planner=hypothesis_planner,
        cluster_model=cluster_model,
        m_var=args.m_var,
        cosine_sim_threshold_initial=args.cosine_sim_threshold_initial,
        cosine_sim_threshold_evolution=args.cosine_sim_threshold_evolution,
        context=args.context,
    )

    runner = EvoRunner(
        seed_states=initial_seed_states,  # type: ignore
        planner=planner,
        policy_model=policy_model,
        student_model=student_model,
        teacher_model=teacher_model,
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
        "student_model": student_model.to_dict(),
        "planner": planner.to_dict(),
        "policy_model": policy_model.to_dict(),
        "train_rewriter": train_rewriter.to_dict(),
        "val_rewriters": [m.to_dict() for m in val_rewriters],
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

    await runner.get_baselines()

    try:
        await runner.train(
            train_rewriter=train_rewriter,
            n_pop_target=args.n_pop_targets,
            train_batch_size=args.train_batch_sizes,
            judge_train_first_n_rollouts=args.judge_train_first_n_rollouts,
            judge_train_first_n_user_prompts=args.judge_train_first_n_user_prompts,
            start_from=args.start_from,
        )

        if validate:
            await runner.validate(
                final_attributes={seed_state.index: list(seed_state.state.keys()) for seed_state in runner.seed_states},
                val_rewriters=val_rewriters,
                judge_val_first_n_rollouts=args.judge_val_first_n_rollouts,
                judge_val_first_n_user_prompts=args.judge_val_first_n_user_prompts,
                val_split_size=args.val_split_size,
            )

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise

    if validate:
        run_path = Path(f"data/evo/{run_name}")
        plot_validation_data(run_path=run_path, write_path=run_path)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
