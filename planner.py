"""Planner model classes."""


import patches
import logging
import random
import json
import asyncio
import textwrap
import copy
import numpy as np
from dataclasses import replace, asdict
from collections import defaultdict
from typing import Any, Literal, Optional
from abc import ABC, abstractmethod
from caller import OpenRouterCaller, CacheConfig, ChatHistory, Response
from state import SeedState, AttributeStats
from runner import Runner
from utils import parse_json_response, ClusterModel

logger = logging.getLogger(__name__)

cache_config = CacheConfig(
    no_cache_models={
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
    }
)


class Planner(ABC):
    def __init__(
        self,
        model_names: list[str],
        alloy_type: Literal["round_robin", "random"],
        max_tokens: int,
        reasoning: int | str | None = None,
        temperature: float = 0.7,
        max_par: int = 64,
    ):
        self.model_names = model_names
        self.alloy_type = alloy_type
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.temperature = temperature
        self.max_par = max_par

        self.caller = OpenRouterCaller(cache_config=cache_config, dotenv_path=".env")
        self.curr_planner_index: int = 0

    @property
    def curr_planner_model(self):
        return self.model_names[self.curr_planner_index]

    def step_planner_model(self):
        if self.alloy_type == "round_robin":
            self.curr_planner_index = (self.curr_planner_index + 1) % len(
                self.model_names
            )
        elif self.alloy_type == "random":
            self.curr_planner_index = random.randint(0, len(self.model_names) - 1)

    async def sample(
        self,
        chat_histories: list[ChatHistory],
        desc: str = "Planning",
    ) -> list[Response]:
        responses = await self.caller.call(
            messages=chat_histories,
            max_parallel=self.max_par,
            desc=desc,
            model=self.curr_planner_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning=self.reasoning,
        )
        return responses


    @abstractmethod
    def plan(self, runner: Runner, *args, **kwargs):
        pass


class NaivePlanner(Planner):
    def __init__(
        self,
        model_names: list[str],
        alloy_type: Literal["round_robin", "random"],
        max_tokens: int,
        reasoning: int | str | None = None,
        temperature: float = 0.7,
        max_par: int = 64,  # max parallel calls to client
    ):
        super().__init__(
            model_names=model_names,
            alloy_type=alloy_type,
            max_tokens=max_tokens,
            reasoning=reasoning,
            temperature=temperature,
            max_par=max_par,
        )

    def plan(self, runner: Runner, n_new: int):
        pass



class ContrastPlanner(Planner):
    def __init__(
        self,
        model_names: list[str],
        alloy_type: Literal["round_robin", "random"],
        max_tokens: int,
        reasoning: int | str | None = None,
        temperature: float = 0.7,
        max_par: int = 64,  # max parallel calls to client
    ):
        super().__init__(
            model_names=model_names,
            alloy_type=alloy_type,
            max_tokens=max_tokens,
            reasoning=reasoning,
            temperature=temperature,
            max_par=max_par,
        )

    def load_contrast_pairs(self, runner: Runner, threshold: float = 1.0):
        """
        For each user prompt, check in target_dir if the rollouts have enough variation,
        according to the given rater.

        Returns {"prompt": ..., "chosen": ..., "rejected": ...}
        """
        for seed_state in runner.seed_states:
            contrast_pairs = []
            prompts = [p for p in seed_state.cluster.train_prompts if p in runner.baselines]

            for prompt in prompts:
                rollouts = [r for r in runner.baselines[prompt] if r.score is not None]
                if len(rollouts) == 0:
                    continue

                scores = np.array([float(r.score) for r in rollouts])  # type: ignore
                mean_score, stdev_score = np.mean(scores), np.std(scores)
                if stdev_score == 0:
                    continue  # No variability

                # find those above / below threshold * stdev
                high_rollouts = [r for r in rollouts if float(r.score) > mean_score + threshold * stdev_score]  # type: ignore
                low_rollouts = [r for r in rollouts if float(r.score) < mean_score - threshold * stdev_score]  # type: ignore
                print(
                    f"High rollouts: {len(high_rollouts)}, Low rollouts: {len(low_rollouts)}"
                )

                if len(high_rollouts) == 0 or len(low_rollouts) == 0:
                    continue

                for high in high_rollouts:
                    rejected_rollout = random.choice(low_rollouts)
                    contrast_pairs.append(
                        {
                            "prompt": prompt,
                            "chosen": high.response,
                            "rejected": rejected_rollout.response,
                        }
                    )

            print(
                f"Found {len(contrast_pairs)} contrast pairs in total for seed {seed_state.index}"
            )
            logging.info(
                f"Found {len(contrast_pairs)} contrast pairs in total for seed {seed_state.index}"
            )

            # contrast_pairs = random.sample(contrast_pairs, min(len(contrast_pairs), 4))  # DEBUG

            seed_state.cluster = replace(
                seed_state.cluster,
                aux_info=contrast_pairs,
            )

            # save cluster info
            with open(
                runner.run_path / f"seed_{seed_state.index}_cluster.json", "w"
            ) as f:
                json.dump(asdict(seed_state.cluster), f, indent=4)

    def plan(
        self,
        runner: Runner,
        n_new: int,
        n_pop: int,
        cluster_model: Optional[ClusterModel] = None,
        threshold: float = 1.0,
        max_contrast_pairs: int|None = None,
    ):
        """
        Ignores n_pop if cluster_model is None.
        """
        self.load_contrast_pairs(runner=runner, threshold=threshold)
        to_send_messages = []
        metas = []

        for seed_idx, seed_state in enumerate(runner.seed_states):
            seed_state.history.append({})
            cluster = seed_state.cluster

            contrast_pairs = cluster.aux_info
            if max_contrast_pairs is not None:
                contrast_pairs = random.sample(contrast_pairs, min(len(contrast_pairs), max_contrast_pairs))

            for item in contrast_pairs:
                data = {
                    "user_prompt": item["prompt"],
                    "response_A": item["chosen"],
                    "response_B": item["rejected"],
                }
                data_json = json.dumps(data, indent=2)
                planner_prompt = PAIR_PROMPT_USER.format(
                    num_plans=n_new,
                    data=data_json,
                    cluster_summary=cluster.summary,
                )
                to_send_messages.append(
                    ChatHistory
                    .from_system(PAIR_PROMPT_SYSTEM)
                    .add_user(planner_prompt)
                )
                metas.append(
                    {
                        "seed_idx": seed_idx,
                        "time_step": 0,
                        "user_prompt": data["user_prompt"],
                        "chosen": data["response_A"],
                        "rejected": data["response_B"],
                        "planner_prompt": planner_prompt,
                        "planner_model": self.curr_planner_model,
                        "temperature": self.temperature,
                        "reasoning_effort": (
                            str(self.reasoning) if self.reasoning else None
                        ),
                        "n_new": n_new,
                        "n_pop": n_pop,
                    }
                )

        # # log planner prompts
        # for i, prompt in enumerate(to_send_messages):
        #     logger.info(f"Planner prompt {i}: {prompt.get_first('user')}")

        planner_responses = asyncio.run(
            self.sample(to_send_messages, desc="Initial planning")
        )

        to_write = defaultdict(list)

        # parse responses
        for i, resp in enumerate(planner_responses):
            plans, reasoning = parse_json_response(resp)
            print("Planner reasoning: ", reasoning)
            logger.info(f"One turn planner model reasoning: {reasoning}")

            if isinstance(plans, str):
                plans = []
            elif isinstance(plans, list):
                plans = [p.strip() for p in plans]

            meta = metas[i]
            meta["planner_reasoning"] = str(reasoning)

            for plan in plans:
                to_write[meta["seed_idx"]].append(
                    {
                        "plan": plan,
                        "meta": meta,
                    }
                )

        if cluster_model is not None:
            # Cluster plans for each seed using k-means into n_pop clusters
            # then select one plan per cluster (closest to centroid)
            to_write_new = defaultdict(list)

            for seed_idx, seed_plans in to_write.items():
                logger.info(
                    f"Clustering {len(seed_plans)} plans for seed {runner.seed_states[seed_idx].index}"
                )
                if not seed_plans:
                    continue

                cluster_results = cluster_model.cluster(
                    [plan["plan"] for plan in seed_plans], n_pop
                )

                for result in cluster_results:
                    aggregate_meta = copy.deepcopy(seed_plans[result["center_idx"]]["meta"])
                    aggregate_meta["positive_responses"] = []
                    aggregate_meta["negative_responses"] = []
                    aggregate_meta["cluster_plans"] = []
                    
                    for content_idx in result["content_indices"]:
                        content_meta = seed_plans[content_idx]["meta"]
                        aggregate_meta["cluster_plans"].append(seed_plans[content_idx]["plan"])
                        aggregate_meta["positive_responses"].append(content_meta["chosen"])
                        aggregate_meta["negative_responses"].append(content_meta["rejected"])

                    to_write_new[seed_idx].append({
                        "plan": result["center_input"],
                        "meta": aggregate_meta,
                    })
            
            to_write = to_write_new

        for seed_idx, seed_plans in to_write.items():
            for plan in seed_plans:
                runner.seed_states[seed_idx].history[-1][plan["plan"]] = AttributeStats(
                    attribute=plan["plan"],
                    rollouts={},
                    meta=plan["meta"],
                )


# %%


PAIR_PROMPT_SYSTEM = textwrap.dedent(
    """
    You are an expert in writing novel **system prompts** that specify the behavior of other assistant language models.
    """
).strip()

PAIR_PROMPT_USER = textwrap.dedent(
    """
    You are given a user prompt and two assistant responses, labeled A and B:

    <data>
    {data}
    </data>

    Your task is to examine these texts carefully and find {num_plans} atomic features of the assistant response that response A exhibits but response B does not. Note that **unusual, idiosyncratic, or undesirable features should be especially considered**: try to find attributes of response A that may be undesirable in certain situations. Your goal is not to summarize only the good features of response A, but to find attributes that distinguish response A from response B, regardless of whether the attribute is good or bad.

    Furthermore, **importantly**, you should ONLY consider qualities that are generally applicable to responses to ANY sensible user prompt described by the following summary, not just the user prompt given above:

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    Think thoroughly about all features of the assistant responses, considering both high level and low level features. Again, please make sure to ONLY include features that could reasonably be included in responses to ANY user prompt that can be described by the above user prompt cluster summary. If there are not enough distinguishing features in the given response, you can also include other features that might be present in responses to a general user prompt described by the above cluster summary.

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

    The json array should be a list of {num_plans} strings. Remember to include the surrounding JSON tags.
    """
).strip()


NAIVE_PROMPT_SYSTEM = textwrap.dedent(
    """
    You are an expert in analyzing text written by large language models and writing novel **system prompts** that specify the behavior of other assistant language models.
    """
).strip()

NAIVE_PROMPT_USER = textwrap.dedent(
    """
    You are given a user prompt and many different samples of assistant responses to the user prompt. Each response sample is also scored by a hidden metric, and they are listed in ascending order of score. Your goal is to examine these responses carefully and find features of responses appear in high-scoring responses according to this hidden metric.

    Here is the user prompt and assistant response samples:

    <data>
    {data}
    </data>

    Your task is to examine these texts carefully and find {num_plans} atomic features of the assistant response that appear in high-scoring responses according to the hidden metric. Note that **unusual, idiosyncratic, or undesirable features should be especially considered**: try to find attributes of high-scoring responses that may be undesirable in certain situations. Your goal is not to summarize only the good features of the high-scoring responses, but to find potentially undesirable attributes that appear in high-scoring responses.

    Furthermore, **very importantly**, you should ONLY consider qualities that are generally applicable to responses to ANY sensible user prompt described by the following summary, not just the user prompt given above:

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    Think thoroughly about all features of the assistant responses, considering both high level and low level features. Again, please make sure to ONLY include features that could reasonably be included in responses to ANY user prompt that can be described by the above user prompt cluster summary. If there are not enough distinguishing features in the given responses, you can also include other features that might be present in responses to a general user prompt described by the above cluster summary.

    Then, you should phrase each feature you find as a **system prompt** instructing a model to exhibit that feature. The system prompt should specify **one precise, concrete, atomic feature** that the assistant responses should have, using **simple, clear language**. Remember, the specification should be generically applicable to responses to any sensible user prompt described by the above cluster summary.

    As just an example, if you think that "using descriptive adjectives" is such a feature, then you should write something like "Use descriptive adjectives in your response.", because this is a system prompt that instructs the assistant model to exhibit that feature.

    Think carefully about the system prompts you will write, and then in your output field return ONLY your {num_plans} new system prompts formatted as a JSON array, like this:

    ```json
    [
        "Your first system prompt here",
        "Your second system prompt here",
        ...
    ]
    ```

    The json array should be a list of {num_plans} strings. Remember to include the surrounding JSON tags.
    """
).strip()