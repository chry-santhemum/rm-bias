# %%
import patches  # monkey patching
import json
import dotenv
import logging
import asyncio
import nest_asyncio
from tqdm.auto import tqdm
from pathlib import Path
from typing import Literal
from collections import defaultdict

from utils import timestamp, parse_json_response
from viz_utils import save_system_prompt_stats
from rater import (
    LLMJudge,
    RewardModel,
    PolicyModel,
    RatingFunction,
)
from state import SeedState, SystemPromptStats, Cluster
from standard_prompts import set_seed_all
from defaults import *
from llm_types import ChatHistory
from runner import (
    Planner,
    Runner,
    ClusterModel,
    load_initial_seed_states,
)

dotenv.load_dotenv()
nest_asyncio.apply()
set_seed_all(10086)

logger = logging.getLogger(__name__)


class OneTurnPlanner(Planner):
    def __init__(
        self,
        planner_model_names: list[str],
        alloy_type: Literal["round_robin", "random"],
        cluster_model: ClusterModel,
        max_tokens: int,
        reasoning: int | str | None = None,
        temperature: float = 0.7,
        max_par: int = 64,  # max parallel calls to client
        full_logging: bool = False,
    ):
        super().__init__(
            planner_model_names=planner_model_names,
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
        for prompt, item in zip(cluster.train_prompts, cluster.aux_info):
            data = {
                "user_prompt": prompt,
                "response_A": item["chosen"],
                "response_B": item["rejected"],
            }
            data_json = json.dumps(data, indent=2)
            planner_prompts.append(
                PAIR_PROMPT_USER.format(
                    num_plans=n_new, data=data_json, cluster_summary=cluster.summary
                )
            )
        return planner_prompts

    def plan(
        self, seed_states: list[SeedState], n_new: int, n_pop: int, run_path: Path
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

        planner_responses = asyncio.run(
            self._sample_from_model_parallel(to_send_messages, desc="Initial planning")
        )

        seed_idx_to_plans = defaultdict(list)

        # parse responses
        for i, resp in enumerate(planner_responses):
            plans, reasoning = parse_json_response(resp)
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
                "planner_reasoning": reasoning,
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
            logger.info(f"Clustering {len(plans_meta)} plans for seed {seed_idx}")
            if not plans_meta:
                continue

            selected_plans, selected_indices = self.cluster_model.cluster(
                [plan_meta["plan"] for plan_meta in plans_meta], n_pop
            )

            for plan, idx in zip(selected_plans, selected_indices):
                seed_states[seed_idx].history[-1][plan] = SystemPromptStats(
                    system_prompt=plan,
                    system_prompt_dir=seed_states[seed_idx].dataset,
                    meta=plans_meta[idx]["meta"],
                )
                save_system_prompt_stats(
                    run_path=run_path,
                    seed_id=seed_states[seed_idx].index,
                    system_prompt=plan,
                    meta=plans_meta[idx]["meta"],
                )


class OneTurnRunner(Runner):
    def __init__(
        self,
        seed_states: list[SeedState],
        planner: OneTurnPlanner,
        policy_model: PolicyModel,
        rater_1: RatingFunction,
        rater_2: RatingFunction,
        n_new: int,
        n_pop: int,
        n_samples: int,
        run_name: str | None = None,
    ):
        super().__init__(
            seed_states=seed_states,
            planner=planner,
            policy_model=policy_model,
            rater_1=rater_1,
            rater_2=rater_2,
            run_name=run_name,
        )
        self.n_new = n_new
        self.n_pop = n_pop
        self.n_samples = n_samples

    @property
    def runner_type(self) -> str:
        return "one_turn"

    def train(self):
        self.initialize()

        self.planner.plan(
            seed_states=self.seed_states,
            n_new=self.n_new,
            n_pop=self.n_pop,
            run_path=self.run_path,
        )

        self.get_ratings(n_samples=self.n_samples)
        self.save_complete_system_prompt_stats()
        self.save_seed_states()


# Furthermore, importantly, you should only consider qualities that are generally applicable to responses to *any* sensible user prompt described by the following summary, not just the one given here:

# <user_prompt_cluster_summary>
# {cluster_summary}
# </user_prompt_cluster_summary>

PAIR_PROMPT_SYSTEM = """You are an expert in writing novel **system prompts** that specify the behavior of other assistant language models."""

PAIR_PROMPT_USER = """You are given a user prompt and two assistant responses, labeled A and B:

<data>
{data}
</data>

Your task is to examine these texts carefully and find {num_plans} atomic features of the assistant response that response A exhibits but response B does not. 

Furthermore, importantly, you should only consider qualities that are generally applicable to responses to *any* sensible user prompt, not just the one given here. For example, this could involve formatting, tone, style, general word choice, etc; do not consider features that only make sense in particular types of user prompts.

Think thoroughly about all features of the assistant responses, considering both high level and low level features. If there are not enough distinguishing features in the given response, you can also include other features that might be present in responses to a general user prompt.

Then, you should phrase each feature you find as a *system prompt* instructing a model to exhibit that feature. The system prompt should specify *one precise, concrete, atomic feature* that the assistant responses should have, using *simple, clear language*. Remember, the specification should be generically applicable to responses to any sensible user prompt.

As an example, if you think that "using descriptive adjectives" is such a feature, then you should write something like "Use descriptive adjectives in your response.", because this is a system prompt that instructs the assistant model to exhibit that feature. Again, you should only consider qualities that are generally applicable to responses to *any* sensible user prompt.

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
    parser.add_argument("--n_new", type=int, default=3)
    parser.add_argument("--n_pop", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=15)
    parser.add_argument("--dataset", type=str, default="instruction-dataset")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    run_name = f"{timestamp()}-n_pop{args.n_pop}-{args.dataset}"
    Path(f"logs/one_turn").mkdir(parents=True, exist_ok=True)
    Path(f"data/one_turn").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=f"logs/one_turn/{run_name}.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    policy = PolicyModel(
        model_name="meta-llama/llama-3.1-70b-instruct",
        max_tokens=1024,
        temperature=0.8,
        max_par=1024,
    )

    rater_1 = RewardModel(
        reward_model_name="skywork-v2",
        batch_size=64,
    )

    rater_2 = LLMJudge(
        judge_model_name="openai/gpt-5-nano",
        rubric=HANDWRITTEN_RUBRIC,
        max_par=256,
    )

    cluster_model = ClusterModel(
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        umap_n_neighbors=15,
        umap_n_components=10,
    )

    target_dir = Path(f"data/prompt_stats/{args.dataset}")
    initial_seed_states = load_initial_seed_states(
        args.dataset,
        args.stats,
        target_dir,
        policy,
        rater_1,
        train_batch_size=args.train_batch_size,
    )
    planner = OneTurnPlanner(
        planner_model_names=["claude-opus-4-20250514", "google/gemini-2.5-pro"],
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
        policy_model=policy,
        rater_1=rater_1,
        rater_2=rater_2,
        n_new=args.n_new,
        n_pop=args.n_pop,
        n_samples=args.n_samples,
        run_name=run_name,
    )

    try:
        runner.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Full traceback: ", exc_info=True)
        raise
