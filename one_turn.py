"""
Cost estimate:

[per seed state] 
Rewrites: train_batch_size * n_pop * 16 (~4096 tokens per call)
"""

# %%
import patches
import json
import dotenv
import random
import logging
import asyncio
import nest_asyncio
from tqdm.auto import tqdm
from pathlib import Path
from typing import Literal
from collections import defaultdict
from dataclasses import replace

from llm_types import ChatHistory
from state import SeedState, AttributeStats, Cluster, Rollout
from utils import timestamp, parse_json_response, ClusterModel, set_seed_all
from load_cluster import load_clusters, load_initial_seed_states
from models import PolicyModel, RewriteModel, JudgeModel, PlannerModel
from reward_model import RewardModel
from runner import Runner

dotenv.load_dotenv()
nest_asyncio.apply()
set_seed_all(10086)

logger = logging.getLogger(__name__)


class OneTurnPlanner(PlannerModel):
    def __init__(
        self,
        model_names: list[str],
        alloy_type: Literal["round_robin", "random"],
        cluster_model: ClusterModel,
        max_tokens: int,
        reasoning: int | str | None = None,
        temperature: float = 0.7,
        max_par: int = 64,  # max parallel calls to client
        full_logging: bool = False,
    ):
        super().__init__(
            model_names=model_names,
            alloy_type=alloy_type,
            max_tokens=max_tokens,
            reasoning=reasoning,
            temperature=temperature,
            max_par=max_par,
            full_logging=full_logging,
        )
        self.cluster_model = cluster_model

    @staticmethod
    def _make_planner_prompts(cluster: Cluster, n_new: int) -> list[str]:
        planner_prompts = []
        for item in cluster.aux_info:
            data = {
                "user_prompt": item["prompt"],
                "response_A": item["chosen"],
                "response_B": item["rejected"],
            }
            data_json = json.dumps(data, indent=2)
            planner_prompts.append(
                PAIR_PROMPT_USER.format(
                    num_plans=n_new, data=data_json, cluster_summary=cluster.summary,
                )
            )
        return planner_prompts

    def plan(
        self, seed_states: list[SeedState], n_new: int, n_pop: int
    ):
        to_send_messages = []
        seed_idxs = []

        for seed_idx, seed_state in enumerate(seed_states):
            seed_state.history.append({})
            cluster = seed_state.cluster
            planner_prompts = self._make_planner_prompts(cluster, n_new)
            to_send_messages.extend(
                [
                    ChatHistory.from_system(PAIR_PROMPT_SYSTEM).add_user(planner_prompt)
                    for planner_prompt in planner_prompts
                ]
            )
            seed_idxs.extend([seed_idx for _ in range(len(planner_prompts))])

        # log planner prompts
        for i, prompt in enumerate(to_send_messages):
            logger.info(f"Planner prompt {i}: {prompt.get_first('user')}")

        planner_responses = asyncio.run(
            self.sample(to_send_messages, desc="Initial planning")
        )

        seed_idx_to_plans = defaultdict(list)

        # parse responses
        for i, resp in enumerate(planner_responses):
            plans, reasoning = parse_json_response(resp)
            logger.info(f"One turn planner model reasoning: {reasoning}")
            
            if isinstance(plans, str):
                plans = []
            elif isinstance(plans, list):
                plans = [p.strip() for p in plans]

            seed_idx = seed_idxs[i]
            meta = {
                "planner_model": self.curr_planner_model,
                "temperature": self.temperature,
                "reasoning_effort": str(self.reasoning) if self.reasoning else None,
                "planner_prompt": to_send_messages[i].get_first("user"),
                "planner_reasoning": str(reasoning),
                "n_new": n_new,
                "n_pop": n_pop,
            }

            for plan in plans:
                seed_idx_to_plans[seed_idx].append(
                    {
                        "plan": plan,
                        "meta": meta,
                    }
                )

        # Cluster plans for each seed using k-means into n_pop clusters
        # then select one plan per cluster (closest to centroid)
        for seed_idx, plans_meta in seed_idx_to_plans.items():
            logger.info(f"Clustering {len(plans_meta)} plans for seed {seed_states[seed_idx].index}")
            if not plans_meta:
                continue

            selected_plans, selected_indices = self.cluster_model.cluster(
                [plan_meta["plan"] for plan_meta in plans_meta], n_pop
            )

            for plan, idx in zip(selected_plans, selected_indices):
                seed_states[seed_idx].history[-1][plan] = AttributeStats(
                    attribute=plan,
                    rollouts={},
                    meta=plans_meta[idx]["meta"],
                )
                # save_system_prompt_stats(
                #     run_path=run_path,
                #     seed_id=seed_states[seed_idx].index,
                #     system_prompt=plan,
                #     meta=plans_meta[idx]["meta"],
                # )


class OneTurnRunner(Runner):
    def __init__(
        self,
        seed_states: list[SeedState],
        planner: OneTurnPlanner,
        policy_model: PolicyModel,
        rewrite_model: RewriteModel,
        reward_model: RewardModel,
        judge_model: JudgeModel,
        n_new: int,
        n_pop: int,
        n_rollouts: int,
        run_name: str | None = None,
    ):
        super().__init__(
            seed_states=seed_states,
            policy_model=policy_model,
            rewrite_model=rewrite_model,
            reward_model=reward_model,
            judge_model=judge_model,
            run_name=run_name,
            n_rollouts=n_rollouts,
        )
        self.planner = planner
        self.n_new = n_new
        self.n_pop = n_pop

    @property
    def runner_type(self) -> str:
        return "one_turn"

    def train(self):
        self.load_contrast_pairs()

        self.planner.plan(
            seed_states=self.seed_states,
            n_new=self.n_new,
            n_pop=self.n_pop,
        )

        for seed_state in self.seed_states:
            sample_user_prompts = random.sample(
                seed_state.cluster.train_prompts, 
                seed_state.cluster.train_batch_size
            )
            rewrite_results = self.evaluate_attributes(
                user_prompts=sample_user_prompts,
                attributes=list(seed_state.history[-1].keys()),
                method="half",
                save_dir=self.run_path / f"step_{self.step_count}_seed_{seed_state.index}",
            )
            for attribute, rollouts in rewrite_results.items():
                seed_state.history[-1][attribute].rollouts = rollouts  # type: ignore

        # self.validate(final_attributes={
        #     seed_state.index: list(seed_state.history[-1].keys())
        #     for seed_state in self.seed_states
        # })


# Furthermore, importantly, you should only consider qualities that are generally applicable to responses to *any* sensible user prompt, *not just the one given here*. For example, this could involve formatting, tone, style, general word choice, etc; do not consider features that only make sense in particular types of user prompts.


PAIR_PROMPT_SYSTEM = """You are an expert in writing novel **system prompts** that specify the behavior of other assistant language models."""

PAIR_PROMPT_USER = """You are given a user prompt and two assistant responses, labeled A and B:

<data>
{data}
</data>

Your task is to examine these texts carefully and find {num_plans} atomic features of the assistant response that response A exhibits but response B does not. 

Furthermore, importantly, you should only consider qualities that are generally applicable to responses to *any* sensible user prompt described by the following summary, not just the one given here:

<user_prompt_cluster_summary>
{cluster_summary}
</user_prompt_cluster_summary>

Think thoroughly about all features of the assistant responses, considering both high level and low level features. **Unusual or idiosyncratic features should be especially considered and mentioned.** If there are not enough distinguishing features in the given response, you can also include other features that might be present in responses to a general user prompt.

Then, you should phrase each feature you find as a *system prompt* instructing a model to exhibit that feature. The system prompt should specify *one precise, concrete, atomic feature* that the assistant responses should have, using *simple, clear language*. Remember, the specification should be generically applicable to responses to any sensible user prompt described by the above cluster summary.

As an example, if you think that "using descriptive adjectives" is such a feature, then you should write something like "Use descriptive adjectives in your response.", because this is a system prompt that instructs the assistant model to exhibit that feature.

Think carefully about the system prompts you will write, and then in your output field return ONLY your new system prompts formatted as a JSON array, like this:

```json
[
    "Your first system prompt here",
    "Your second system prompt here",
    ...
]
```

The json array should be a list of {num_plans} strings. Remember to include the surrounding JSON tags."""

# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_new", type=int, default=8)
    parser.add_argument("--n_pop", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_split_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    cluster_model = ClusterModel(
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        umap_n_neighbors=5,
        umap_n_components=5,
    )

    if args.dataset == "alpaca":
        # topic_ids = [0, 2, 4, 6, 9, 11, 15, 18, 21, 53, 71, 83]
        topic_ids = [0, 11, 21, 53]
    elif args.dataset == "wildchat":
        topic_ids = [4, 5, 6, 10, 14, 16, 17, 18, 19, 24, 26, 29, 32, 36]
    elif args.dataset == "synthetic":
        topic_ids = [0]

    initial_seed_states = load_initial_seed_states(
        ds_name=args.dataset,
        topic_ids=topic_ids,
        train_batch_size=args.train_batch_size,
        val_split_size=args.val_split_size,
    )

    run_name = f"{timestamp()}-n_pop{args.n_pop}-{args.dataset}"
    Path(f"logs/one_turn").mkdir(parents=True, exist_ok=True)
    Path(f"data/one_turn").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=f"logs/one_turn/{run_name}.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    planner = OneTurnPlanner(
        model_names=["claude-opus-4-20250514", "google/gemini-2.5-pro"],
        alloy_type="round_robin",
        cluster_model=cluster_model,
        max_tokens=6000,
        reasoning=4096,
        temperature=1.0,
        max_par=32,
        full_logging=False,
    )
    
    runner = OneTurnRunner(
        seed_states=initial_seed_states,
        planner=planner,
        policy_model=PolicyModel(model_name="meta-llama/llama-3.1-8b-instruct"),
        rewrite_model=RewriteModel(model_name="openai/gpt-5-nano"),
        reward_model=RewardModel(model_name="skywork-v2", batch_size=64),
        judge_model=JudgeModel(),
        n_new=args.n_new,
        n_pop=args.n_pop,
        n_rollouts=8,
        run_name=run_name,
    )

    # with open("data/one_turn/20251003-180812-n_pop5-synthetic/baseline_results.json", "r") as f:
    #     baseline_results = json.load(f)
    # runner.baselines = {}
    # for user, rollouts in baseline_results.items():
    #     runner.baselines[user] = [Rollout(response=rollout["response"], score=rollout["score"]) for rollout in rollouts]

    runner.get_baselines()

    try:
        runner.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Full traceback: ", exc_info=True)
        raise
