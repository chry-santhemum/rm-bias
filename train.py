import argparse
from pathlib import Path
from utils import timestamp

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--runner_type", type=str, required=True, choices=["evo", "one_turn"])
parser.add_argument("--planner_type", type=str, required=True, choices=["pair", "list", "list_reverse"])

parser.add_argument("--direction", type=str, required=True, choices=["plus", "minus"])
parser.add_argument("--n_new", type=int, required=True, help="Hypothesis generation: number of candidates per ask")
parser.add_argument("--n_pop_initial", type=int, required=True, help="Hypothesis generation: initial population")

parser.add_argument("--m_var", type=int, default=3)
parser.add_argument("--n_planner_requests", type=int, default=64)
parser.add_argument("--n_baseline_rollouts", type=int, default=16)
parser.add_argument("--n_rewrite_rollouts", type=int, default=8)
parser.add_argument("--val_split_size", type=int, default=16)
parser.add_argument("--dbscan_eps", type=float, default=0.2)
parser.add_argument("--train_batch_size", type=int, default=16, help="Only used for one_turn runner")
parser.add_argument("--run_name", type=str, default=None)

args = parser.parse_args()

run_name = args.run_name or f"{timestamp()}-{args.planner_type}-{args.dataset}-{args.direction}"
Path(f"logs/{args.runner_type}").mkdir(parents=True, exist_ok=True)
Path(f"data/{args.runner_type}").mkdir(parents=True, exist_ok=True)

from loguru import logger
logger.add(f"logs/{args.runner_type}/{run_name}.log", enqueue=True, level="INFO")


from pathlib import Path
import json
import torch
import argparse
import asyncio
from collections import defaultdict

from utils import timestamp, ClusterModel
from load_cluster import load_initial_seed_states
from state import Rollout
from models import PolicyModel, RewriteModel, JudgeModel
from reward_models import LocalRewardModel
from bias_evaluator import BiasEvaluator
from planner import PairPlanner, ListPlanner
from one_turn import OneTurnRunner
from evo import EvoRunner, EvoPlanner



if args.dataset == "alpaca":
    topic_ids = [0, 2, 4, 6, 9, 11, 15, 21, 34, 35, 83]
elif args.dataset == "wildchat":
    topic_ids = [4, 5, 6, 10, 14, 16, 17, 18, 19, 24, 26, 29, 32, 36]
elif args.dataset == "synthetic_0":
    topic_ids = [4, 19, 20, 28, 32, 38, 45, 48, 56, 64, 68]
elif args.dataset == "synthetic_1":
    # topic_ids = [0, 1, 3, 6, 7, 8, 9, 10, 11, 12, 14]
    topic_ids = [3, 6, 7, 12, 14]
    # topic_ids = [8, 9, 10, 11]
elif args.dataset == "synthetic_2":
    # topic_ids = [1, 3, 4, 6, 8, 9, 12, 14, 16]
    topic_ids = [1]

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
        model_name="anthropic/claude-sonnet-4.5", 
        max_par=256,
        max_tokens=1050, 
        reasoning=1024,
        enable_cache=False,
        force_caller="openrouter",
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

    if args.planner_type == "pair":
        hypothesis_planner = PairPlanner(
            model_names=["openai/gpt-5"],
            max_tokens=8192,
            reasoning="medium",
            max_par=128,
            relabel=False,
            n_new=args.n_new,
            n_pop=args.n_pop_initial,
            max_contrast_pairs=args.n_planner_requests,
        )
    elif args.planner_type in ["list", "list_reverse"]:
        hypothesis_planner = ListPlanner(
            model_names=["openai/gpt-5"],
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
        )

    validate = True if args.val_split_size > 0 else False
    if args.runner_type == "evo":
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
            judge_model=judge_model,
            dbscan_eps=args.dbscan_eps,
            m_var=args.m_var,
            n_baseline_rollouts=args.n_baseline_rollouts,
            n_rewrite_rollouts=args.n_rewrite_rollouts,
            run_name=run_name,
        )

        # runner.get_baselines()

        with open(
            f"data/evo/{run_name}/train_baselines/sample_rollouts.json",
            "r",
        ) as f:
            train_baselines = json.load(f)

        runner.baselines = {}
        for user, rollouts in train_baselines.items():
            runner.baselines[user] = [
                Rollout(response=rollout["response"], score=rollout["score"])
                for rollout in rollouts
            ]

        with open(
            f"data/evo/{run_name}/val_baselines/sample_rollouts.json", "r"
        ) as f:
            val_baselines = json.load(f)

        runner.val_baselines = {}
        for user, rollouts in val_baselines.items():
            runner.val_baselines[user] = [
                Rollout(response=rollout["response"], score=rollout["score"])
                for rollout in rollouts
            ]
        
        rewrite_rollouts = []
        for idx in topic_ids:
            with open(
                f"data/evo/{run_name}/validate/seed_{idx}_validate/rewrite_rollouts.json", "r"
            ) as f:
                seed_rollouts_json = json.load(f)

            seed_rollouts = defaultdict(dict)
            for attribute, attribute_rollouts in seed_rollouts_json.items():
                for user_prompt, rollouts in attribute_rollouts.items():
                    seed_rollouts[attribute][user_prompt] = [
                        Rollout(response=rollout["response"], score=rollout["score"])
                        for rollout in rollouts
                    ]

            rewrite_rollouts.append(dict(seed_rollouts))

        runner.judge(validation_results=rewrite_rollouts)
  
        # try:
        #     runner.train(
        #         n_pop_target=[16, 8, 8],
        #         train_batch_size=[4, 8, 16],
        #         validate=validate,
        #         start_from=3,
        #     )
        # except Exception as e:
        #     logger.exception(f"Training failed: {e}")
        #     raise


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
            direction=args.direction,
            run_name=run_name,
        )

        runner.get_baselines()

        # with open(
        #     f"data/one_turn/{run_name}/val_baselines/sample_rollouts.json", "r"
        # ) as f:
        #     val_baselines = json.load(f)

        # runner.val_baselines = {}
        # for user, rollouts in val_baselines.items():
        #     runner.val_baselines[user] = [
        #         Rollout(response=rollout["response"], score=rollout["score"])
        #         for rollout in rollouts
        #     ]
        
        # rewrite_rollouts = []
        # for idx in topic_ids:
        #     with open(
        #         f"data/one_turn/{run_name}/validate/seed_{idx}_validate/rewrite_rollouts.json", "r"
        #     ) as f:
        #         seed_rollouts_json = json.load(f)

        #     seed_rollouts = defaultdict(dict)
        #     for attribute, attribute_rollouts in seed_rollouts_json.items():
        #         for user_prompt, rollouts in attribute_rollouts.items():
        #             seed_rollouts[attribute][user_prompt] = [
        #                 Rollout(response=rollout["response"], score=rollout["score"])
        #                 for rollout in rollouts
        #             ]

        #     rewrite_rollouts.append(dict(seed_rollouts))

        # runner.judge(validation_results=rewrite_rollouts)
            

        try:
            asyncio.run(runner.train(validate=validate))
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Full traceback: ", exc_info=True)
            raise


if __name__ == "__main__":
    main()
