import re
import json
import random
import textwrap
from loguru import logger
from dataclasses import dataclass
from typing import Any, Sequence, Literal

from caller import AutoCaller, CacheConfig, RetryConfig, ChatHistory, Response
from utils import parse_json_response
from api_prompts import *

RETRY_CONFIG = RetryConfig(
    raise_when_exhausted=False,
    criteria=lambda response: response.has_response and response.finish_reason == "stop",
    max_attempts=5,
)


class GenerationModel:
    """API samplers with one or more underlying models."""
    def __init__(
        self,
        model_name: str | list[str],
        max_par: int,
        force_caller: str | None = None,
        **kwargs,
    ):
        if isinstance(model_name, str):
            self.model_names = [model_name]
        else:
            self.model_names = model_name
        self.max_par = max_par
        self.caller = AutoCaller(
            dotenv_path=".env", 
            retry_config=RETRY_CONFIG, 
            force_caller=force_caller
        )
        self.kwargs = kwargs

    @property
    def model_name(self) -> str:
        """Returns the first model name."""
        return self.model_names[0]

    async def sample(
        self,
        chat_histories: list[ChatHistory],
        desc: str | None = None,
        **kwargs,
    ) -> list[Response|None]:
        """kwargs passed here will supersede the init kwargs.
        
        When multiple model names are configured, randomly selects a model
        for each chat history.
        """
        models = [random.choice(self.model_names) for _ in chat_histories]
        
        responses = await self.caller.call(
            messages=chat_histories,
            model=models,
            max_parallel=self.max_par,
            desc=desc,
            **self.kwargs, **kwargs,
        )
        return responses


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

@dataclass(frozen=True)
class RatingResult:
    score: float | None
    reasoning: str | None

@dataclass(frozen=True)
class ComparisonResult:
    winner: Literal["A", "B", "Tie"] | None
    reasoning: str | None
    score_diff: float | None = None  # A score - B score
    raw_score_A: float | None = None
    raw_score_B: float | None = None


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
        # TODO: Only do one trial by default, with randomly chosen order
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
                judge_results.append(ComparisonResult(winner=None, reasoning=None))
                continue
            response_reasoning = resp.reasoning_content
            if (not resp.has_response) or (resp.finish_reason != "stop"):
                judge_results.append(ComparisonResult(winner=None, reasoning=response_reasoning))
                logger.warning(f"Judge relative response is invalid: {resp}")
                continue
            
            matched = re.search(r"\*\*(.+?)\*\*", resp.first_response.strip())  # type: ignore
            if matched:
                response_text = matched.group(1).strip().lower()
            else:
                response_text = resp.first_response.strip().lower()  # type: ignore
            
            if i % num_trials < (num_trials // 2):
                winner = "A" if response_text == "a" else "B" if response_text == "b" else "Tie" if response_text == "tie" else None
                judge_results.append(ComparisonResult(winner=winner, reasoning=response_reasoning))
            else:
                winner = "B" if response_text == "a" else "A" if response_text == "b" else "Tie" if response_text == "tie" else None
                judge_results.append(ComparisonResult(winner=winner, reasoning=response_reasoning))

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
