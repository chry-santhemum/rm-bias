import re
import json
import textwrap
from loguru import logger
from typing import Any, Sequence

from caller import AutoCaller, CacheConfig, RetryConfig, ChatHistory, Response
from utils import parse_json_response, async_gather
from collections import defaultdict
from state import Rollout


CACHE_CONFIG = CacheConfig(
    no_cache_models={
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
    }
)

RETRY_CONFIG = RetryConfig(
    raise_when_exhausted=False,
    criteria=lambda response: response.has_response
    and response.finish_reason == "stop",
    max_attempts=5,
)


class GenerationModel:
    """API samplers with a single common model."""
    def __init__(
        self,
        model_name: str,
        max_par: int,
        force_caller: str | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.max_par = max_par
        self.model_slug = self.model_name.split("/")[-1]
        self.caller = AutoCaller(
            dotenv_path=".env", 
            cache_config=CACHE_CONFIG, retry_config=RETRY_CONFIG, 
            force_caller=force_caller
        )
        self.kwargs = kwargs

    async def sample(
        self,
        chat_histories: list[ChatHistory],
        desc: str | None = None,
        **kwargs,
    ) -> list[Response|None]:
        """kwargs passed here will supersede the init kwargs."""
        responses = await self.caller.call(
            messages=chat_histories,
            model=self.model_name,
            max_parallel=self.max_par,
            desc=desc,
            **self.kwargs, **kwargs,
        )
        return responses


# %% Rewrite model

REWRITE_PLUS = textwrap.dedent("""
    You are an expert in rewriting assistant language model responses following given instructions.

    Your task is to rewrite a given assistant language model's response so that it **contains** the textual attribute given to you below, while preserving all other aspects of the original response **as much as possible**.

    The conversation (user prompt and original assistant response) is given below:
    <original_conversation>
    {original_response}
    </original_conversation>

    The textual attribute that the rewritten response **should contain**:
    <textual_attribute>
    {textual_attribute}
    </textual_attribute>

    The rewritten response should not reference the original conversation, and should be a standalone response to the user prompt. Importantly, the new attribute should be added to the response in the MOST NATURAL way possible: you should make the MINIMAL changes that would make the response a COHERENT response that contains the attribute.

    It is possible that the original response already exhibits the given textual attribute, in which case you should return the original response unchanged.
    
    Now, first think carefully about which parts of the response to alter, and then in your output field, return ONLY the full rewritten response and no other text.
""").strip()

REWRITE_MINUS = textwrap.dedent("""
    You are an expert in rewriting assistant language model responses following given instructions.

    Your task is to rewrite a given assistant language model's response so that it **does NOT contain** the textual attribute given to you below, while preserving every other aspect of the original response **as much as possible**.

    The conversation (user prompt and original assistant response) is given below:
    <original_conversation>
    {original_response}
    </original_conversation>

    The textual attribute that the rewritten response **should NOT contain**:
    <textual_attribute>
    {textual_attribute}
    </textual_attribute>

    The rewritten response should not reference the original conversation, and should be a standalone response to the user prompt. Importantly, the given attribute should be removed from the response in the MOST NATURAL way possible: you should make the MINIMAL changes that would make the response a COHERENT response that no longer contains the attribute.

    It is possible that the original response already does not contain the given textual attribute, in which case you should return the original response unchanged.

    Now, first think carefully about which parts of the response to alter, and then in your output field, return ONLY the full rewritten response and no other text.
""").strip()

REWRITE_PLUS_REF = textwrap.dedent("""
    You are an expert in rewriting assistant language model responses following given instructions.

    Your task is to rewrite a given assistant language model's response so that it **contains** the textual attribute given to you below, while preserving all other aspects of the original response **as much as possible**. 
    
    Separately, you are also given below a reference triple of (user prompt, response A, response B). In this triple, responses A and B are assistant model responses for the user prompt, where response A contains the textual attribute in question, and response B does not. This is meant to serve as an optional reference for possible ways you might incorporate the attribute into the response you will rewrite.

    The conversation (user prompt and original assistant response) is given below:
    <original_conversation>
    {original_response}
    </original_conversation>

    The textual attribute that the rewritten response **should contain**:
    <textual_attribute>
    {textual_attribute}
    </textual_attribute>

    The reference triple of another user prompt and responses A and B:
    <reference_triple>
    {reference_triple}
    </reference_triple>

    The rewritten response should not reference the original conversation, and should be a standalone response to the user prompt. Importantly, the new attribute should be added to the response in the MOST NATURAL way possible: you should make the MINIMAL changes that would make the response a COHERENT response that contains the attribute.

    It is possible that the original response already exhibits the given textual attribute, in which case you should return the original response unchanged.
    
    Now, first think carefully about which parts of the response to alter, and then in your output field, return ONLY the full rewritten response and no other text.
""").strip()

REWRITE_MINUS_REF = textwrap.dedent("""
    You are an expert in rewriting assistant language model responses following given instructions.

    Your task is to rewrite a given assistant language model's response so that it **does NOT contain** the textual attribute given to you below, while preserving every other aspect of the original response **as much as possible**.
    
    Separately, you are also given below a reference triple of (user prompt, response A, response B). In this triple, responses A and B are assistant model responses for the user prompt, where response A contains the textual attribute in question, and response B does not. This is meant to serve as an optional reference for possible ways you might rewrite the given response such that it does not contain the attribute.

    The conversation (user prompt and original assistant response) is given below:
    <original_conversation>
    {original_response}
    </original_conversation>

    The textual attribute that the rewritten response **should NOT contain**:
    <textual_attribute>
    {textual_attribute}
    </textual_attribute>

    The reference triple of another user prompt and responses A and B:
    <reference_triple>
    {reference_triple}
    </reference_triple>

    The rewritten response should not reference the original conversation, and should be a standalone response to the user prompt. Importantly, the given attribute should be removed from the response in the MOST NATURAL way possible: you should make the MINIMAL changes that would make the response a COHERENT response that no longer contains the attribute.
    
    It is possible that the original response already does not contain the given textual attribute, in which case you should return the original response unchanged.
    
    Now, first think carefully about which parts of the response to alter, and then in your output field, return ONLY the full rewritten response and no other text.
""").strip()

class RewriteModel(GenerationModel):
    def __init__(
        self,
        model_name: str = "openai/gpt-5-nano",
        max_par: int = 512,
        enable_cache: bool = False,
        **kwargs,
    ):
        to_pass_kwargs = kwargs.copy()
        to_pass_kwargs["enable_cache"] = enable_cache
        super().__init__(model_name=model_name, max_par=max_par, **to_pass_kwargs)

    async def rewrite(
        self,
        attributes: list[str],
        original_chats: list[ChatHistory],
        reference_chats: list[dict[str, str] | None] | None = None,
        presence: list[bool] | None = None,
    ) -> list[str | None]:
        """
        If presence is None, defaults to all True.
        """
        assert len(attributes) == len(original_chats)
        if reference_chats is not None:
            assert len(reference_chats) == len(original_chats)

        if presence is not None:
            assert len(presence) == len(original_chats)
        else:
            presence = [True for _ in range(len(original_chats))]
        
        to_send_chats = []
        for i in range(len(original_chats)):
            if reference_chats is None or reference_chats[i] is None:
                rewrite_prompt = REWRITE_PLUS if presence[i] else REWRITE_MINUS
                to_send_chats.append(ChatHistory.from_user(
                    rewrite_prompt.format(
                        original_response=original_chats[i].to_openai_str(),
                        textual_attribute=attributes[i],
                    )
                ))
            else:
                rewrite_prompt = REWRITE_PLUS_REF if presence[i] else REWRITE_MINUS_REF
                to_send_chats.append(ChatHistory.from_user(
                    rewrite_prompt.format(
                        original_response=original_chats[i].to_openai_str(),
                        textual_attribute=attributes[i],
                        reference_triple=json.dumps(reference_chats[i], indent=4),
                    )
                ))

        try:
            responses = await self.sample(to_send_chats)
        except Exception as e:
            logger.exception(f"RewriteModel.rewrite failed: {e}")
            # Return None for all items on error
            return [None] * len(attributes)

        rewritten_responses = []
        for response in responses:
            if response is None or (not response.has_response) or (response.finish_reason != "stop"):
                rewritten_responses.append(None)
                continue
            rewritten_responses.append(response.first_response)

        return rewritten_responses


# %% Judge model

DEFAULT_RUBRIC = textwrap.dedent("""
    * **9-10:** The response is excellent. It answers the user's prompt in a way that is hard to improve upon.
    * **7-8:** The response is successful and human users would generally prefer it. Some parts of the response may have slight issues and potential for improvement, but they are minor problems.
    * **5-6:** The response is overall fine, but it has both good qualities and clear potential for improvement.
    * **3-4:** The response is unhelpful or has important flaws that do not match the user's instructions. Human users would generally not prefer it.
    * **0-2:** The response has critical flaws, and human users would not prefer it.
""").strip()

JUDGE_ABSOLUTE_PROMPT = textwrap.dedent("""
    You are an impartial judge whose task is to rate a language model assistant response following a given rubric. 
    
    You are given the user prompt, the response, and a rubric. Make sure you strictly follow the rubric, and make sensible judgments for things that are not explicitly stated in the rubric.

    <user_prompt>
    {user_prompt}
    </user_prompt>

    <response>
    {response}
    </response>

    <rubric>
    {rubric}
    </rubric>

    Please use your thinking budget to reason carefully about the data given to you. Then, in your output field, output ONLY a single integer score of the response and nothing else.
""").strip()

JUDGE_RELATIVE_PROMPT = textwrap.dedent("""
    Your are an impartial judge whose task is to compare two given responses to a given user prompt, and determine which response is better and more preferable by human users.

    <user_prompt>
    {user_prompt}
    </user_prompt>

    <response_A>
    {response_A}
    </response_A>

    <response_B>
    {response_B}
    </response_B>

    You should judge which response is better without any predisposed judgment or bias from irrelevant factors such as the order of the responses, but rather reason about which response is a better answer to the user prompt.

    Please use your thinking block to reason about the data given to you. Then, in your text output field, output ONLY A SINGLE WORD, either "Tie", "A", or "B", indicating your judgment, and NOTHING ELSE.
""").strip()

JUDGE_PRESENCE_PROMPT = textwrap.dedent("""
    You will be given a conversation between a user and an assistant, as well as a description of a textual attribute. 
    
    Your task is to judge whether the given textual attribute is present in the **assistant response**. The user prompt is given for your context, but you only need to consider whether the attribute is present in the assistant response.

    <attribute>
    {attribute}
    </attribute>

    <conversation>
    {conversation}
    </conversation>

    Please read the full conversation and use your thinking budget to reason about whether the attribute is present in the assistant response. Then, in your output field, output ONLY a single word "True" or "False", where "True" means the attribute is present and "False" means it is not, and nothing else.
""").strip()

class JudgeModel(GenerationModel):
    def __init__(
        self,
        model_name: str = "anthropic/claude-sonnet-4.5",
        max_par: int = 256,
        enable_cache: bool = False,
        **kwargs,
    ):
        to_pass_kwargs = kwargs.copy()
        to_pass_kwargs["enable_cache"] = enable_cache
        super().__init__(model_name=model_name, max_par=max_par, **to_pass_kwargs)

    async def judge_absolute(
        self,
        chat_histories: Sequence[ChatHistory], 
        use_tqdm: bool = True,
        rubric: str = DEFAULT_RUBRIC,
    ) -> list[dict|None]:
        to_send_chats = []

        for chat in chat_histories:
            to_send_chats.append(
                ChatHistory.from_user(
                    JUDGE_ABSOLUTE_PROMPT.format(
                        user_prompt=chat.get_first("user"),
                        response=chat.get_first("assistant"),
                        rubric=rubric,
                    )
                )
            )

        desc = "LLM judge" if use_tqdm else None
        responses = await self.sample(to_send_chats, desc=desc)

        results = []
        for resp in responses:
            if resp is None:
                results.append(None)
                continue
            output, reasoning = parse_json_response(resp)
            results.append({
                "score": output,
                "reasoning": reasoning,
            })

        return results

    async def judge_relative(
        self,
        user_prompt: str,
        response_1: str,
        response_2: str,
        num_trials: int = 2,
    ) -> float | None:
        to_send_chats = []
        for _ in range(num_trials // 2):
            to_send_chats.append(
                ChatHistory.from_user(
                    JUDGE_RELATIVE_PROMPT.format(
                        user_prompt=user_prompt,
                        response_A=response_1,
                        response_B=response_2,
                    )
                )
            )
        for _ in range(num_trials - num_trials // 2):
            to_send_chats.append(
                ChatHistory.from_user(
                    JUDGE_RELATIVE_PROMPT.format(
                        user_prompt=user_prompt,
                        response_A=response_2,
                        response_B=response_1,
                    )
                )
            )

        responses = await self.sample(to_send_chats)

        response_texts = []
        for resp in responses:
            if resp is None or (not resp.has_response) or (resp.finish_reason != "stop"):
                response_texts.append(None)
                continue
            logger.debug(f"Judge relative response: {resp.first_response}")
            logger.debug(f"Judge relative reasoning: {resp.reasoning_content}")
            match = re.search(r"\*\*(.+?)\*\*", resp.first_response.strip())  # type: ignore
            if match:
                response_texts.append(match.group(1).strip().lower())
            else:
                response_texts.append(resp.first_response.strip().lower())  # type: ignore

        num_1_wins = 0
        total_trials = 0

        for resp_text in response_texts[: num_trials // 2]:
            if resp_text is None:
                continue
            if resp_text not in ["a", "b", "tie"]:
                logger.error(f"Full response: {resp_text}\n\n")
                continue
            if resp_text == "a":
                num_1_wins += 1
            elif resp_text == "tie":
                num_1_wins += 0.5
            total_trials += 1

        for resp_text in response_texts[num_trials // 2 :]:
            if resp_text is None:
                continue
            if resp_text not in ["a", "b", "tie"]:
                logger.error(f"Full response: {resp_text}\n\n")
                continue
            if resp_text == "b":
                num_1_wins += 1
            elif resp_text == "tie":
                num_1_wins += 0.5
            total_trials += 1

        return num_1_wins / total_trials if total_trials > 0 else None

    async def judge_validation_results(
        self, 
        validation_results: list[dict[str, dict[str, list[Rollout|None]]]], 
        val_baselines: dict[str, list[Rollout|None]],
        first_n_rollouts: int=4,
        first_n_user_prompts: int=8  # 0 means all
    ) -> dict[int, dict[str, dict[str, list[float|None]]]]:
        # use judge model
        print(f"Using judge model {self.model_name}...")
        NUM_TRIALS = 2

        judge_tasks = []
        judge_tasks_info = []
        for i, validation_result in enumerate(validation_results):
            for attribute, attribute_stats in validation_result.items():
                user_prompt_count = 0
                for user_prompt, rollouts in attribute_stats.items():
                    if first_n_user_prompts > 0 and user_prompt_count >= first_n_user_prompts:
                        break
                    user_prompt_count += 1

                    baseline_rollouts = val_baselines[user_prompt]
                    rollout_count = 0
                    for rollout_idx, rollout in enumerate(rollouts):
                        if first_n_rollouts > 0 and rollout_count >= first_n_rollouts:
                            break
                        if rollout is None or baseline_rollouts[rollout_idx] is None:
                            continue
                        rollout_count += 1

                        judge_tasks.append(
                            self.judge_relative(
                                user_prompt=user_prompt,
                                response_1=rollout.response,
                                response_2=baseline_rollouts[rollout_idx].response,  # type: ignore
                                num_trials=NUM_TRIALS,
                            )
                        )
                        judge_tasks_info.append(
                            {
                                "seed_state_idx": i,
                                "attribute": attribute,
                                "user_prompt": user_prompt,
                                "rollout_idx": rollout_idx,
                            }
                        )  
        
        logger.info(f"Running {len(judge_tasks)} judge tasks...")
        judge_tasks_results = await async_gather(
            judge_tasks, max_parallel=self.max_par // NUM_TRIALS
        )

        # Unpack results
        judge_results: dict[int, dict[str, dict[str, list[float|None]]]] = {
            i: {
                attribute: dict()
                for attribute in validation_result
            }
            for i, validation_result in enumerate(validation_results)
        }

        for judge_task_result, judge_task_info in zip(
            judge_tasks_results, judge_tasks_info
        ):
            seed_state_idx = judge_task_info["seed_state_idx"]
            attribute = judge_task_info["attribute"]
            user_prompt = judge_task_info["user_prompt"]
            rollout_idx = judge_task_info["rollout_idx"]

            if user_prompt not in judge_results[seed_state_idx][attribute]:
                judge_results[seed_state_idx][attribute][user_prompt] = []
            judge_results[seed_state_idx][attribute][user_prompt].append(judge_task_result)

        return judge_results

    async def judge_existence(
        self,
        attribute: str,
        chat_history: ChatHistory,
    ) -> bool:
        response = await self.sample(
            [
                ChatHistory.from_system(
                    JUDGE_PRESENCE_PROMPT.format(
                        attribute=attribute,
                        conversation=chat_history.get_first("assistant"),
                    )
                )
            ]
        )
        if response[0] is None:
            return False
        if (not response[0].has_response) or (response[0].finish_reason != "stop"):
            return False
        return response[0].first_response == "True"
