"""LLM API generation model classes."""

# %%
import patches
import textwrap
import logging
import random
from slist import Slist
from typing import Any, Literal

from llm_types import ChatHistory
from utils import get_to_pass_reasoning
from client import (
    OpenaiResponse,
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



class PlannerModel:
    def __init__(
        self,
        model_names: list[str],
        alloy_type: Literal["round_robin", "random"],
        max_tokens: int,
        reasoning: int | str | None = None,
        temperature: float = 0.7,
        max_par: int = 64,
        full_logging: bool = False,
    ):
        self.model_names = model_names
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
        return self.model_names[self.curr_planner_index]

    def step_planner_model(self):
        if self.alloy_type == "round_robin":
            self.curr_planner_index = (self.curr_planner_index + 1) % len(
                self.model_names
            )
        elif self.alloy_type == "random":
            self.curr_planner_index = random.randint(
                0, len(self.model_names) - 1
            )

    async def sample(
        self, 
        chat_histories: list[ChatHistory], 
        desc: str = "Planning",
    ) -> Slist[OpenaiResponse]:
        return await sample_from_model_parallel(
            caller=self.caller,
            prompts=chat_histories,
            max_par=self.max_par,
            full_logging=self.full_logging,
            desc=desc,
            model=self.curr_planner_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning=get_to_pass_reasoning(self.reasoning, self.max_tokens),
        )