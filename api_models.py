import re
import json
import textwrap
from loguru import logger
from dataclasses import dataclass
from typing import Any, Sequence, Literal

from caller import AutoCaller, CacheConfig, RetryConfig, ChatHistory, Response
from utils import parse_json_response


CACHE_CONFIG = CacheConfig(
    no_cache_models={
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
    }
)

RETRY_CONFIG = RetryConfig(
    raise_when_exhausted=False,
    criteria=lambda response: response.has_response and response.finish_reason == "stop",
    max_attempts=5,
)


class GenerationModel:
    """API samplers with a single underlying model."""
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


@dataclass(frozen=True)
class RatingResult:
    score: float | None
    reasoning: str | None

@dataclass(frozen=True)
class ComparisonResult:
    winner: Literal["A", "B", "Tie"] | None
    reasoning: str | None
    score_diff: float | None  # A score - B score


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
        chat_histories: Sequence[ChatHistory|None], 
        use_tqdm: bool = True,
        rubric: str = DEFAULT_RUBRIC,
    ) -> list[RatingResult]:
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

        desc = "LLM judge absolute" if use_tqdm else None
        responses = await self.sample(to_send_chats, desc=desc)

        rating_results = []
        for resp in responses:
            if resp is None:
                rating_results.append(RatingResult(score=None, reasoning=None))
                continue
            output, reasoning = parse_json_response(resp)
            try:
                score_float = float(output)
            except ValueError:
                logger.exception(f"Could not convert score to float: {output}")
                score_float = None
            rating_results.append(RatingResult(score=score_float, reasoning=reasoning))

        return rating_results

    async def judge_relative(
        self,
        chat_histories_A: Sequence[ChatHistory], 
        chat_histories_B: Sequence[ChatHistory],
        num_trials: int = 2,
        use_tqdm: bool = True,
    ) -> list[ComparisonResult]:
        assert len(chat_histories_A) == len(chat_histories_B)
        to_send_chats = []

        for chat_A, chat_B in zip(chat_histories_A, chat_histories_B):
            user_prompt = chat_A.get_first("user")
            assert user_prompt == chat_B.get_first("user")
            response_A = chat_A.get_first("assistant")
            response_B = chat_B.get_first("assistant")

            for _ in range(num_trials // 2):
                to_send_chats.append(
                    ChatHistory.from_user(
                        JUDGE_RELATIVE_PROMPT.format(
                            user_prompt=user_prompt,
                            response_A=response_A,
                            response_B=response_B,
                        )
                    )
                )
            for _ in range(num_trials - num_trials // 2):
                to_send_chats.append(
                    ChatHistory.from_user(
                        JUDGE_RELATIVE_PROMPT.format(
                            user_prompt=user_prompt,
                            response_A=response_B,
                            response_B=response_A,
                        )
                    )
                )

        desc = "LLM judge relative" if use_tqdm else None
        responses = await self.sample(to_send_chats, desc=desc)

        judge_results = []
        for i, resp in enumerate(responses):
            if resp is None:
                judge_results.append(ComparisonResult(winner=None, reasoning=None, score_diff=None))
                continue
            response_reasoning = resp.reasoning_content
            if (not resp.has_response) or (resp.finish_reason != "stop"):
                judge_results.append(ComparisonResult(winner=None, reasoning=response_reasoning, score_diff=None))
                logger.warning(f"Judge relative response is invalid: {resp}")
                continue
            
            matched = re.search(r"\*\*(.+?)\*\*", resp.first_response.strip())  # type: ignore
            if matched:
                response_text = matched.group(1).strip().lower()
            else:
                response_text = resp.first_response.strip().lower()  # type: ignore
            
            if i % num_trials < (num_trials // 2):
                match response_text:
                    case "a":
                        judge_results.append(ComparisonResult(winner="A", reasoning=response_reasoning, score_diff=None))
                    case "b":
                        judge_results.append(ComparisonResult(winner="B", reasoning=response_reasoning, score_diff=None))
                    case "tie":
                        judge_results.append(ComparisonResult(winner="Tie", reasoning=response_reasoning, score_diff=None))
                    case _:
                        judge_results.append(ComparisonResult(winner=None, reasoning=response_reasoning, score_diff=None))
                        logger.warning(f"Judge relative response is invalid: {resp}")
            else:
                match response_text:
                    case "a":
                        judge_results.append(ComparisonResult(winner="B", reasoning=response_reasoning, score_diff=None))
                    case "b":
                        judge_results.append(ComparisonResult(winner="A", reasoning=response_reasoning, score_diff=None))
                    case "tie":
                        judge_results.append(ComparisonResult(winner="Tie", reasoning=response_reasoning, score_diff=None))
                    case _:
                        judge_results.append(ComparisonResult(winner=None, reasoning=response_reasoning, score_diff=None))
                        logger.warning(f"Judge relative response is invalid: {resp}")

        return judge_results

    async def judge_presence(
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
