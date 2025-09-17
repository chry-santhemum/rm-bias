import patches   # noqa: F401  # monkey patching
import os
import json
import random
import dotenv
import pickle
import logging
import asyncio
import nest_asyncio
from tqdm.auto import tqdm
from pathlib import Path
from typing import Literal, Coroutine
from datasets import load_dataset

import wandb
import pandas as pd
from utils import timestamp, get_to_pass_reasoning, setup_prompt_logger
from viz_utils import save_system_prompt_stats, save_cluster_info, convert_attack_to_dict
from rater import prompt_to_hash_path, LLMJudge, RewardModel, PolicyModel, RatingFunction
from state import SeedState, SystemPromptStats, Cluster
from standard_prompts import set_seed_all
from defaults import *
from client import is_thinking_model, get_universal_caller, sample_from_model_parallel
from llm_types import ChatHistory

dotenv.load_dotenv()
nest_asyncio.apply()
set_seed_all(10086)

logger = logging.getLogger(__name__)


class Planner:
    def __init__(
        self,
        planner_model_names: list[str],
        alloy_type: Literal["round_robin", "random"],
        max_tokens: int,
        reasoning: int | str | None = None,
        temperature: float = 0.7,
        max_par: int = 32,  # max parallel calls to client
        full_logging: bool = False,
    ):
        self.planner_model_names = planner_model_names
        self.alloy_type = alloy_type
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.temperature = temperature
        self.max_par = max_par
        self.full_logging = full_logging

        self.caller = get_universal_caller()
        self.curr_planner_index: int = 0

    @property
    def curr_planner_model(self):
        return self.planner_model_names[self.curr_planner_index]

    def step_planner_model(self):
        if self.alloy_type == "round_robin":
            self.curr_planner_index = (self.curr_planner_index + 1) % len(
                self.planner_model_names
            )
        elif self.alloy_type == "random":
            self.curr_planner_index = random.randint(
                0, len(self.planner_model_names) - 1
            )

    @staticmethod
    def _make_planner_prompts(cluster: Cluster, N_new: int) -> list[str]:
        planner_prompts = []
        for prompt, item in zip(cluster.train_prompts, cluster.aux_info):
            data = {
                "user_prompt": prompt,
                "response_A": item["chosen"],
                "response_B": item["rejected"],
            }
            data_json = json.dumps(data, indent=2)
            planner_prompts.append(
                INDIVIDUAL_PAIR_PROMPT_USER.format(
                    num_new=N_new, data=data_json, cluster_summary=cluster.summary
                )
            )
        return planner_prompts


    def plan(self, seed_states: list[SeedState[None]], N_new: int, run_path: Path = None, step_count: int = 0)
        model = self.curr_planner_model
        to_send_messages = []
        seed_idxs = []

        for seed_idx, seed_state in enumerate(seed_states):
            cluster = seed_state.cluster
            planner_prompts = self._make_planner_prompts(cluster, N_new)
            to_send_messages.extend([
                ChatHistory
                .from_system(INDIVIDUAL_PAIR_PROMPT_SYSTEM)
                .add_user(planner_prompt)
            for planner_prompt in planner_prompts])
            seed_idxs.extend([seed_idx] * len(planner_prompts))

        planner_responses = asyncio.run(sample_from_model_parallel(
            caller=self.caller,
            prompts=to_send_messages,
            max_par=self.max_par,
            full_logging=self.full_logging,
            desc="Planning",
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning=get_to_pass_reasoning(self.reasoning, self.max_tokens),
        ))

        # parse responses
        for response_idx, resp in enumerate(planner_responses):
            seed_idx = seed_idxs[response_idx]
            plans = []
            reasoning = "N/A"
            try:
                raw_text = resp.first_response
                plans = json.loads(raw_text.split("```json", 1)[1].split("```", 1)[0].strip())

        # parse responses
        all_plans = {}
        reasonings = []
        for i, resp in enumerate(planner_responses):
            try:
                raw_text = resp.first_response
                plans = json.loads(raw_text.split("```json", 1)[1].split("```", 1)[0].strip())
                plans = [p.strip() for p in plans]
                try:
                    if is_thinking_model(model):
                        reasoning = resp.reasoning_content
                    else:
                        reasoning = raw_text.rsplit("<plan>", 1)[0].strip()
                except Exception:
                    reasoning = "N/A"

            except Exception as e:
                logger.error(f"Planner parse error (plan JSON): {e}")
                logger.error(f"API response: {resp}")
                plans, reasoning = [], "N/A"
            
            all_plans[i] = plans
            reasonings.append(reasoning)




INDIVIDUAL_PAIR_PROMPT_SYSTEM = """You are an expert in writing novel **system prompts** that specify the behavior of other assistant language models."""

INDIVIDUAL_PAIR_PROMPT_USER = """You are given a user prompt and two assistant responses, labeled A and B. Your task is to examine these texts carefully and find {num_new} atomic features of the assistant response that response A exhibits but response B does not. Furthermore, you should only consider qualities that are generally applicable to answering a user prompt that is in the same cluster as the given one. Here is a short summary of the cluster of user prompts:

<user_prompt_cluster_summary>
{cluster_summary}
</user_prompt_cluster_summary>

Here is the text data of the user prompt and the two assistant responses:
<data>
{data}
</data>

Think thoroughly about all features of the assistant responses, either high level or low level. Then, you should phrase each feature you find as a *system prompt* instructing a model to exhibit that feature. 

As an example, if you think that "using descriptive adjectives" is such a feature, then you should write something like "Use descriptive adjectives in your response.", because this is a system prompt that instructs the assistant model to exhibit that feature.

Think carefully about the system prompts you will write, and then in your output field return only your new system prompts formatted as a JSON array, like this:

```json
[
    "Your first system prompt here",
    "Your second system prompt here",
    ...
]
```

The json array should be a list of {num_new} strings. Remember to include the surrounding JSON tags."""


MULTIPLE_PAIR_PROMPT_SYSTEM = """You are an expert in writing novel **system prompts** that specify the behavior of other assistant language models."""
