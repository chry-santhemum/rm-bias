from pathlib import Path
import json
import torch
import logging
import argparse
import asyncio

from utils import timestamp, logging_setup, ClusterModel
from load_cluster import load_initial_seed_states
from state import Rollout
from models import PolicyModel, RewriteModel, JudgeModel
from reward_models import LocalRewardModel
from bias_evaluator import BiasEvaluator
from planner import PairPlanner, ListPlanner
from one_turn import OneTurnRunner
from evo import EvoRunner, EvoPlanner

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--runner_type", type=str, required=True, choices=["evo", "one_turn"])
parser.add_argument("--planner_type", type=str, required=True, choices=["pair", "list"])

parser.add_argument("--n_new", type=int, default=8, help="Hypothesis generation: number of candidates per ask")
parser.add_argument("--n_pop_initial", type=int, default=128, help="Hypothesis generation: initial population")


parser.add_argument("--m_var", type=int, default=3)
parser.add_argument("--n_baseline_rollouts", type=int, default=16)
parser.add_argument("--n_rewrite_rollouts", type=int, default=8)
parser.add_argument("--val_split_size", type=int, default=16)
parser.add_argument("--dbscan_eps", type=float, default=0.2)
parser.add_argument("--train_batch_size", type=int, default=16, help="Only used for one_turn runner")

args = parser.parse_args()

if args.dataset == "alpaca":
    topic_ids = [0, 2, 4, 6, 9, 11, 15, 21, 34, 35, 83]
elif args.dataset == "wildchat":
    topic_ids = [4, 5, 6, 10, 14, 16, 17, 18, 19, 24, 26, 29, 32, 36]
elif args.dataset == "synthetic_0":
    topic_ids = [5, 17, 21, 29, 33, 39, 46, 49, 57, 65, 69]
elif args.dataset == "synthetic_1":
    # topic_ids = [0, 1, 3, 6, 7, 8, 9, 10, 11, 12, 14]
    topic_ids = [3, 6, 7, 12, 14]
    # topic_ids = [8, 9, 10, 11]
elif args.dataset == "synthetic_2":
    # topic_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # topic_ids = [1, 3, 4, 6, 8, 9, 12, 14, 16]
    topic_ids = [1, 4, 6]

def main():
    all_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"Using CUDA devices: {all_cuda_devices}")

    cluster_model = ClusterModel(
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
    )

    policy_model = PolicyModel(
        model_name="meta-llama/llama-3.1-8b-instruct", 
        temperature=0.9
    )

    judge_model = JudgeModel(
        model_name="anthropic/claude-haiku-4.5", 
        max_tokens=2048, 
        reasoning=2000,
        enable_cache=False,
    )

    bias_evaluator = BiasEvaluator(
        rewrite_model=RewriteModel(
            model_name="openai/gpt-5-nano", 
            max_par=512, 
            max_tokens=4096,
            enable_cache=False,
        ),
        reward_model=LocalRewardModel(
            model_name="skywork-v2", 
            devices=all_cuda_devices, 
            batch_size_per_device=32,
            attn_implementation="sdpa",
        ),
        n_rewrite_workers=32,
    )

    initial_seed_states = load_initial_seed_states(
        ds_name=args.dataset,
        topic_ids=topic_ids,
        train_batch_size=args.train_batch_size,
        val_split_size=args.val_split_size,
    )

    run_name = f"{timestamp()}-{args.planner_type}-{args.dataset}"
    Path(f"logs/{args.runner_type}").mkdir(parents=True, exist_ok=True)
    Path(f"data/{args.runner_type}").mkdir(parents=True, exist_ok=True)

    logging_setup(filename=f"logs/{args.runner_type}/{run_name}.log", level=logging.INFO)

    if args.planner_type == "pair":
        hypothesis_planner = PairPlanner(
            model_names=["openai/gpt-5"],
            max_tokens=8192,
            reasoning="medium",
            max_par=128,
            relabel=False,
            n_new=args.n_new,
            n_pop=args.n_pop_initial,
            max_contrast_pairs=64,
        )
    elif args.planner_type == "list":
        hypothesis_planner = ListPlanner(
            model_names=["openai/gpt-5"],
            max_tokens=8192,
            reasoning="medium",
            max_par=128,
            relabel=False,
            n_new=args.n_new,
            n_pop=args.n_pop_initial,
            n_traj_in_context=8,
            n_per_user_prompt=1,
            max_num_train_prompts=64,
        )

    if args.runner_type == "evo":
        planner = EvoPlanner(
            hypothesis_planner=hypothesis_planner,
            cluster_model=cluster_model,
        )
        runner = EvoRunner(
            seed_states=initial_seed_states,  # type: ignore
            planner=planner,
            policy_model=policy_model,
            bias_evaluator=bias_evaluator,
            judge_model=judge_model,
            dbscan_eps=args.dbscan_eps,
            m_var=args.m_var,
            n_baseline_rollouts=args.n_baseline_rollouts,
            n_rewrite_rollouts=args.n_rewrite_rollouts,
            run_name=run_name,
        )

        runner.get_baselines()

        try:
            runner.train(
                n_pop_target=[16, 8, 8],
                train_batch_size=[4, 8, 16],
                # n_pop_target=[4, 2],
                # train_batch_size=[2, 4],
                validate=True,
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Full traceback: ", exc_info=True)
            raise


    elif args.runner_type == "one_turn":
        runner = OneTurnRunner(
            seed_states=initial_seed_states,
            hypothesis_planner=hypothesis_planner,
            cluster_model=cluster_model,
            policy_model=policy_model,
            bias_evaluator=bias_evaluator,
            judge_model=judge_model,
            train_batch_size=args.train_batch_size,
            n_baseline_rollouts=args.n_baseline_rollouts,
            n_rewrite_rollouts=args.n_rewrite_rollouts,
            run_name=run_name,
        )

        runner.get_baselines()

        try:
            asyncio.run(runner.train(validate=True))
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Full traceback: ", exc_info=True)
            raise

    # with open(
    #     f"data/evo/20251121-075422-list-synthetic_2/train_baselines/sample_rollouts.json",
    #     "r",
    # ) as f:
    #     train_baselines = json.load(f)

    # runner.baselines = {}
    # for user, rollouts in train_baselines.items():
    #     runner.baselines[user] = [
    #         Rollout(response=rollout["response"], score=rollout["score"])
    #         for rollout in rollouts
    #     ]

    # with open(
    #     f"data/evo/{run_name}/val_baselines/baseline_results.json", "r"
    # ) as f:
    #     val_baselines = json.load(f)

    # runner.val_baselines = {}
    # for user, rollouts in val_baselines.items():
    #     runner.val_baselines[user] = [
    #         Rollout(response=rollout["response"], score=rollout["score"])
    #         for rollout in rollouts
    #     ]

    # final_attributes = {}
    # for seed_state_idx in topic_ids:
    #     with open(f"data/evo/{run_name}/step_2_stats/seed_{seed_state_idx}.json", "r") as f:
    #         seed_results = json.load(f)
    #         final_attributes[seed_state_idx] = [item["attribute"] for item in seed_results[:8]]

    # runner.validate(final_attributes=final_attributes, get_val_baselines=False)
    # asyncio.run(runner.shutdown())


if __name__ == "__main__":
    main()
