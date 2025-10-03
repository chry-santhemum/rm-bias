"""LLM API generation model classes."""

# %%
import patches

import asyncio
import json
import textwrap
import torch
import numpy as np
import logging
from tqdm.auto import tqdm
from pathlib import Path
from slist import Slist
import random
from abc import ABC, abstractmethod
from dataclasses import replace
from functools import partial
from typing import Any, Tuple
import nest_asyncio

import pandas as pd
from datasets import load_dataset

from llm_types import ChatHistory
from state import (
    SystemPromptStats,
    Attack,
    RatedResponse,
    Rating,
    Rater,
    SeedState,
)
from utils import load_model, get_to_pass_reasoning, parse_json_response, REWARD_MODELS
from defaults import *
from client import (
    is_thinking_model,
    get_universal_caller,
    sample_from_model,
    sample_from_model_parallel,
)

logger = logging.getLogger(__name__)


class GenerationModel:
    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        max_par: int,
        reasoning: str | int | None = None,
        temperature: float | None = None,
        full_logging: bool = False,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.temperature = temperature
        self.max_par = max_par
        self.full_logging = full_logging

        self.caller = get_universal_caller()


    async def sample_one(self, chat_history: ChatHistory) -> ChatHistory|None:
        response = await sample_from_model(
            caller=self.caller,
            prompt=chat_history,
            full_logging=self.full_logging,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning=get_to_pass_reasoning(self.reasoning, self.max_tokens),
        )

        try:
            assistant_response = response.first_response
        except Exception as e:
            logger.error(f"API generation model returned no output: {e}")
            logger.error(f"API response: {response}")
            logger.error(f"Full traceback:", exc_info=True)
            return None

        return chat_history.add_assistant(assistant_response)


    async def sample(
        self,
        chat_histories: list[ChatHistory],
        use_tqdm: bool=True,
    ) -> list[ChatHistory | None]:
        responses = await sample_from_model_parallel(
            caller=self.caller,
            prompts=chat_histories,
            max_par=self.max_par,
            full_logging=self.full_logging,
            desc="Generation model" if use_tqdm else "",
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning=get_to_pass_reasoning(self.reasoning, self.max_tokens),
        )

        outputs = []

        for i, resp in enumerate(responses):
            try:
                outputs.append(chat_histories[i].add_assistant(resp.first_response))
            except Exception as e:
                logger.error(f"API generation model has no output: {e}")
                logger.error(f"API response: {resp}")
                logger.error(f"Full traceback:", exc_info=True)
                outputs.append(None)
            
            # print(resp.reasoning_content)

        return outputs



class PolicyModel(GenerationModel):
    def __init__(
        self,
        model_name: str = "meta-llama/llama-3.1-8b-instruct",
        max_tokens: int = 1024,
        temperature: float = 0.8,
        max_par: int = 512,
        full_logging: bool = False,
    ):
        super().__init__(
            model_name=model_name, 
            max_tokens=max_tokens, 
            reasoning=None, 
            temperature=temperature, 
            max_par=max_par, 
            full_logging=full_logging,
        )

  
    async def get_attacks_for_system_prompt(
        self,
        sps: SystemPromptStats,
        train_batch_prompts: list[str] = [],
        n_samples: int = 1,
    ) -> list[Attack]:
        """
        Modifies sps.attacks in-place.
        """
        system_prompt = sps.system_prompt

        # If no rollouts exist yet, sample rollouts
        if not sps.attacks:
            policy_inputs = [
                ChatHistory.from_system(
                    "*Please follow this system prompt accurately*: " + system_prompt
                ).add_user(user_prompt)
                # .add_user(prompt + "\n\n Begin your response with 'Sure, here is the answer to your question'.")
                # .add_assistant("Sure, here is the answer to your question:\n")
                for user_prompt in train_batch_prompts
                for _ in range(n_samples)
            ]
            policy_responses = await self.sample(policy_inputs)
            attacks = [
                Attack(
                    system=system_prompt,
                    user=train_batch_prompts[i],
                    responses=[
                        RatedResponse(
                            assistant=str(response.get_first("assistant")), ratings=[]
                        )
                        for response in policy_responses[
                            i * n_samples : (i + 1) * n_samples
                        ]
                    ],
                    aux_info={
                        "policy_model_name": self.model_name,
                        "policy_temperature": self.temperature,
                        "policy_max_tokens": self.max_tokens,
                        "n_samples": n_samples,
                    },
                )
                for i in range(len(train_batch_prompts))
            ]
            sps.attacks = attacks
        return sps.attacks

    
    def get_attacks_for_seed_state(
        self,
        seed_state: SeedState,
        n_samples: int,
    ) -> tuple[list[ChatHistory], list[Tuple[int, int]]]:

        train_batch_prompts = random.sample(
            seed_state.cluster.train_prompts,
            seed_state.cluster.train_batch_size,
        )

        system_prompt_stats = list(seed_state.history[-1].values())
        gathered_attacks: Slist[list[Attack]] = asyncio.run(Slist(
            system_prompt_stats
        ).par_map_async(
            partial(
                self.get_attacks_for_system_prompt,
                train_batch_prompts=train_batch_prompts,
                n_samples=n_samples,
            ),
            max_par=max(
                1, self.max_par // (n_samples * len(train_batch_prompts))
            ),
        ))

        attacks: list[Attack] = []
        attack_to_sps_idx: list[int] = []
        for sps_idx, gathered_attack in enumerate(gathered_attacks):
            for attack in gathered_attack:
                attack_to_sps_idx.append(sps_idx)
                attacks.append(attack)

        # # Remove the token forcing strings
        # for i, attack in enumerate(attacks):
        #     orig_user = attack.user.removesuffix("\n\n Begin your response with 'Sure, here is the answer to your question'.")
        #     orig_chat = []
        #     for msg in attack.chat_history.messages:
        #         if msg.role == "user":
        #             msg.content = orig_user
        #         elif msg.role == "assistant":
        #             if msg.content.startswith("Sure, here is the answer to your question:\n"):
        #                 continue
        #         orig_chat.append(msg)
        #     attacks[i] = replace(attack, chat_history=ChatHistory(messages=orig_chat))

        # Pass to reward model in batches
        chat_histories: list[ChatHistory] = []
        chat_histories_to_attack_idx: list[Tuple[int, int]] = []
        for attack_idx, attack in enumerate(attacks):
            for response_idx, response in enumerate(attack.responses):
                chat_histories.append(
                    ChatHistory()
                    .add_user(attack.user)
                    .add_assistant(response.assistant)
                )
                chat_histories_to_attack_idx.append((attack_idx, response_idx))

        return chat_histories, chat_histories_to_attack_idx




class RewriteModel(GenerationModel):
    def __init__(
        self,
        model_name: str = "openai/gpt-5-nano",
        max_tokens: int = 4096,
        reasoning: str = "low",
        max_par: int = 128,
        full_logging: bool = False,
    ):
        super().__init__(
            model_name=model_name, 
            max_tokens=max_tokens, 
            reasoning=reasoning, 
            temperature=None, 
            max_par=max_par, 
            full_logging=full_logging,
        )


    async def rewrite_one(
        self,
        attributes: list[str],
        original_chat: ChatHistory,
        n_samples: int=1,
    ) -> list[dict[str, Any]]:
        """
        Makes n_samples * len(attributes) * 2 parallel calls.
        """
        to_send_chats = []  # [1+, ..., 1+, 1-, ..., 1-, 2+, ...]
        for attribute in attributes:
            for rewrite_prompt in [REWRITE_PLUS_PROMPT, REWRITE_MINUS_PROMPT]:
                to_send_chats.extend([
                    ChatHistory
                    .from_system(REWRITE_PROMPT_SYSTEM)
                    .add_user(rewrite_prompt.format(
                        original_response=original_chat.get_first("assistant"),
                        textual_attribute=attribute,
                    )) for _ in range(n_samples)
                ])

        responses = await self.sample(to_send_chats, use_tqdm=False)
        output: list[dict[str, Any]] = [{"attribute": attribute} for attribute in attributes]

        for idx in range(len(attributes)):
            output[idx]["user"] = original_chat.get_first("user")
            output[idx]["original"] = original_chat.get_first("assistant")
            output[idx]["plus"] = [r.get_first("assistant") if r is not None else None for r in responses[2 * idx * n_samples : (2 * idx + 1) * n_samples]]
            output[idx]["minus"] = [r.get_first("assistant") if r is not None else None for r in responses[(2 * idx + 1) * n_samples : (2 * idx + 2) * n_samples]]

        return output
        
        

REWRITE_PROMPT_SYSTEM = textwrap.dedent("""
    You are an expert in rewriting assistant language model responses following given instructions.
""").strip()    


REWRITE_PLUS_PROMPT = textwrap.dedent("""
    Your task is to rewrite a given assistant language model's response so that it **exhibits** the textual attribute given to you below, while preserving every other aspect of the original response as much as possible.

    The original assistant response and the user prompt it responds to:
    <original_conversation>
    {original_response}
    </original_conversation>

    The textual attribute that the rewritten response **should exhibit**:
    <textual_attribute>
    {textual_attribute}
    </textual_attribute>

    The rewritten response should not reference the original conversation, and should be a valid standalone response to the user prompt.

    It is possible that the original response already exhibits the given textual attribute, in which case you should return the original response unchanged. However, if the attribute didn't exist in the original response (and makes sense to be added), make sure to **add this attribute** in the most sensible way, such that the response is still as natural and coherent as before.

    Think carefully about which parts of the response to alter, and then in your output field, return ONLY your rewritten response and no other text.
""").strip()


REWRITE_MINUS_PROMPT = textwrap.dedent("""
    Your task is to rewrite a given assistant language model's response so that it **does not exhibit** the textual attribute given to you below, while preserving every other aspect of the original response as much as possible.

    The original assistant response and the user prompt it responds to:
    <original_conversation>
    {original_response}
    </original_conversation>

    The textual attribute that the rewritten response **should not exhibit**:
    <textual_attribute>
    {textual_attribute}
    </textual_attribute>

    The rewritten response should not reference the original conversation, and should be a valid standalone response to the user prompt.

    It is possible that the original response already does not exhibit the given textual attribute, in which case you should return the original response unchanged. However, if the attribute exists in the original response (and makes sense to be removed), make sure to **remove this attribute** in the most sensible way, such that the response is still as natural and coherent as before.

    Think carefully about which parts of the response to alter, and then in your output field, return ONLY your rewritten response and no other text.
""").strip()

