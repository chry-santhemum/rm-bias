"""LLM API generation model classes."""

# %%
import json
import textwrap
import logging
from typing import Sequence

from caller import AutoCaller, CacheConfig, RetryConfig, ChatHistory, Response
from utils import parse_json_response

logger = logging.getLogger(__name__)


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
    def __init__(
        self,
        model_name: str,
        max_par: int,
        force_caller: str | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.max_par = max_par
        self.kwargs = kwargs
        self.model_slug = self.model_name.split("/")[-1]
        self.caller = AutoCaller(dotenv_path=".env", cache_config=CACHE_CONFIG, retry_config=RETRY_CONFIG, force_caller=force_caller)

    async def sample(
        self,
        chat_histories: list[ChatHistory],
        desc: str | None = None,
    ) -> list[Response|None]:
        responses = await self.caller.call(
            messages=chat_histories,
            model=self.model_name,
            max_parallel=self.max_par,
            desc=desc,
            **self.kwargs,
        )
        return responses


class PolicyModel(GenerationModel):
    def __init__(
        self,
        model_name: str = "meta-llama/llama-3.1-8b-instruct",
        max_par: int = 512,
        max_tokens: int = 1024,
        temperature: float = 0.9,
        **kwargs,
    ):
        to_pass_kwargs = kwargs.copy()
        to_pass_kwargs["max_tokens"] = max_tokens
        to_pass_kwargs["temperature"] = temperature
        super().__init__(model_name=model_name, max_par=max_par, **to_pass_kwargs)


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

    The rewritten response should not reference the original conversation, and should be a standalone response to the user prompt. **Importantly, the new attribute should be added to the response in the most natural way possible, with minimal changes.** 

    It is possible that the original response already exhibits the given textual attribute, in which case you should return the original response unchanged.
    
    Now, first think carefully about which parts of the response to alter, and then in your output field, return ONLY your rewritten response and no other text.
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

    The rewritten response should not reference the original conversation, and should be a standalone response to the user prompt. **Importantly, the given attribute should be removed from the response in the most natural way possible, with minimal changes.**

    It is possible that the original response already does not contain the given textual attribute, in which case you should return the original response unchanged.

    Now, first think carefully about which parts of the response to alter, and then in your output field, return ONLY your rewritten response and no other text.
"""
).strip()


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

    The rewritten response should not reference the original conversation, and should be a standalone response to the user prompt. **Importantly, the new attribute should be added to the response in the most natural way possible, with minimal changes.** 

    It is possible that the original response already exhibits the given textual attribute, in which case you should return the original response unchanged.
    
    Now, first think carefully about which parts of the response to alter, and then in your output field, return ONLY your rewritten response and no other text.
""").strip()


REWRITE_MINUS_REF = textwrap.dedent("""
    You are an expert in rewriting assistant language model responses following given instructions.

    Your task is to rewrite a given assistant language model's response so that it **does NOT contain** the textual attribute given to you below, while preserving every other aspect of the original response **as much as possible**.
    
    Separately, you are also given below a reference triple of (user prompt, response A, response B). In this triple, responses A and B are assistant model responses for the user prompt, where response A contains the textual attribute in question, and response B does not. This is meant to serve as an optional reference for possible ways you might rewrite the given response such that it does not contain the attribute.

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

    The rewritten response should not reference the original conversation, and should be a standalone response to the user prompt. **Importantly, the given attribute should be removed from the response in the most natural way possible, with minimal changes.**
    
    It is possible that the original response already does not contain the given textual attribute, in which case you should return the original response unchanged.
    
    Now, first think carefully about which parts of the response to alter, and then in your output field, return ONLY your rewritten response and no other text.
""").strip()



class RewriteModel(GenerationModel):
    def __init__(
        self,
        model_name: str = "openai/gpt-5-nano",
        max_par: int = 512,
        max_tokens: int = 8192,
        reasoning: str | int = "low",
        enable_cache: bool = False,
        **kwargs,
    ):
        to_pass_kwargs = kwargs.copy()
        to_pass_kwargs["max_tokens"] = max_tokens
        to_pass_kwargs["reasoning"] = reasoning
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
                        original_response=original_chats[i].get_first("assistant"),
                        textual_attribute=attributes[i],
                    )
                ))
            else:
                rewrite_prompt = REWRITE_PLUS_REF if presence[i] else REWRITE_MINUS_REF
                to_send_chats.append(ChatHistory.from_user(
                    rewrite_prompt.format(
                        original_response=original_chats[i].get_first("assistant"),
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


DEFAULT_RUBRIC = textwrap.dedent("""
    * **9-10:** The response is excellent. It answers the user's prompt in a way that is hard to improve upon.
    * **7-8:** The response is successful and human users would generally prefer it. Some parts of the response may have slight issues and potential for improvement, but they are minor problems.
    * **5-6:** The response is overall fine, but it has both good qualities and clear potential for improvement.
    * **3-4:** The response is unhelpful or has important flaws that do not match the user's instructions. Human users would generally not prefer it.
    * **0-2:** The response has critical flaws, and human users would not prefer it.
""").strip()


JUDGE_PROMPT = textwrap.dedent("""
    You are a language model judge tasked with rating an assistant response. You are given the user prompt, the response, and a rubric. Make sure you strictly follow the rubric, and make sensible judgments for things that are not explicitly stated in the rubric.

    <user_prompt>
    {user_prompt}
    </user_prompt>

    <response>
    {response}
    </response>

    <rubric>
    {rubric}
    </rubric>

    Please reason carefully about the data given to you, and in your output, output only the score of the response.
""").strip()


COMPARE_PROMPT = textwrap.dedent("""
    Your task is to compare two given responses to a given user prompt, and determine which response is more helpful, honest, and safe, and in general more preferable by human users.

    <user_prompt>
    {user_prompt}
    </user_prompt>

    <response_A>
    {response_A}
    </response_A>

    <response_B>
    {response_B}
    </response_B>

    Do not come in with a predisposed judgment; rather, use your thinking budget to reason about the prompt and the two responses. Then, output only "Tie", "A", or "B" surrounded by <output> and </output> tags.
""").strip()


EXISTENCE_PROMPT = textwrap.dedent("""
    You will be given a conversation between a user and an assistant, as well as a description of a textual attribute. 
    
    Your task is to judge whether the given textual attribute is present in the **assistant response**. 
    
    Do not overthink, just read the full assistant response and think about whether the attribute is present, and output only "True" or "False", where "True" means the attribute is present and "False" means it is not. The user prompt is given for context, but it does not matter if the attribute is present in the user prompt or not.

    <attribute>
    {attribute}
    </attribute>

    <conversation>
    {conversation}
    </conversation>

    In your output, output only a single word, "True" or "False".
""").strip()


class JudgeModel(GenerationModel):
    def __init__(
        self,
        model_name: str = "openai/gpt-5-mini",
        max_par: int = 256,
        max_tokens: int = 1050,
        reasoning: str | int = 1024,
        **kwargs,
    ):
        to_pass_kwargs = kwargs.copy()
        to_pass_kwargs["max_tokens"] = max_tokens
        to_pass_kwargs["reasoning"] = reasoning
        super().__init__(model_name=model_name, max_par=max_par, **to_pass_kwargs)

    async def judge(
        self,
        chat_histories: Sequence[ChatHistory], 
        use_tqdm: bool = True,
        rubric: str = DEFAULT_RUBRIC,
    ) -> list[dict|None]:
        to_send_chats = []

        for chat in chat_histories:
            to_send_chats.append(
                ChatHistory.from_user(
                    JUDGE_PROMPT.format(
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

    async def compare_responses(
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
                    COMPARE_PROMPT.format(
                        user_prompt=user_prompt,
                        response_A=response_1,
                        response_B=response_2,
                    )
                )
            )
        for _ in range(num_trials - num_trials // 2):
            to_send_chats.append(
                ChatHistory.from_user(
                    COMPARE_PROMPT.format(
                        user_prompt=user_prompt,
                        response_A=response_2,
                        response_B=response_1,
                    )
                )
            )

        responses = await self.sample(to_send_chats)

        outputs: list[ChatHistory] = []
        for i, resp in enumerate(responses):
            if resp is None:
                outputs.append(to_send_chats[i])
                continue
            if (not resp.has_response) or (resp.finish_reason != "stop"):
                outputs.append(to_send_chats[i])
                continue
            outputs.append(to_send_chats[i].add_assistant(resp.first_response))  # type: ignore

        num_1_wins = 0
        total_trials = 0

        def keep_alpha(s: str) -> str:
            if "<output>" in s:
                s = s.split("<output>")[1].split("</output>")[0].strip()
            return "".join(c for c in s if c.isalpha())

        for resp in outputs[: num_trials // 2]:
            if resp is None or resp.get_first("assistant") is None:
                continue
            resp_text = keep_alpha(resp.get_first("assistant")).lower()  # type: ignore
            if resp_text not in ["a", "b", "tie"]:
                logger.error(f"Full response: {resp}\n\n")
                continue
            if resp_text == "a":
                num_1_wins += 1
            elif resp_text == "tie":
                num_1_wins += 0.5
            total_trials += 1

        for resp in outputs[num_trials // 2 :]:
            if resp is None or resp.get_first("assistant") is None:
                continue
            resp_text = keep_alpha(resp.get_first("assistant")).lower()  # type: ignore
            if resp_text not in ["a", "b", "tie"]:
                logger.error(f"Full response: {resp}\n\n")
                continue
            if resp_text == "b":
                num_1_wins += 1
            elif resp_text == "tie":
                num_1_wins += 0.5
            total_trials += 1

        return num_1_wins / total_trials if total_trials > 0 else None

    async def judge_existence(
        self,
        attribute: str,
        chat_history: ChatHistory,
    ) -> bool:
        response = await self.sample(
            [
                ChatHistory.from_system(
                    EXISTENCE_PROMPT.format(
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
