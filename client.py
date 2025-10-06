# %%
"""API client."""
import os
import random
import dotenv
import logging
import asyncio
from json import JSONDecodeError
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Sequence, Type
from pydantic import ValidationError

from slist import Slist
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import openai
import anthropic
from openai import AsyncOpenAI, BaseModel
# from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN
from openai._types import omit as OPENAI_OMIT
from anthropic import AsyncAnthropic
from anthropic.types.message import Message
# from anthropic._types import NOT_GIVEN as ANTHROPIC_NOT_GIVEN
from anthropic._types import omit as ANTHROPIC_OMIT

from llm_types import (
    APIRequestCache,
    ChatMessage,
    GenericBaseModel,
    InferenceConfig,
    ToolArgs,
    ChatHistory,
)


logger = logging.getLogger(__name__)
# Models for which caching is disabled by default.
# - Exact names go into NO_CACHE_MODELS
# - Prefixes (e.g., vendor/model-family) go into NO_CACHE_MODEL_PREFIXES
# By default, disable caching for LLaMA family via the meta-llama/ prefix.
NO_CACHE_MODELS: list[str] = []
NO_CACHE_MODEL_PREFIXES: list[str] = [
    "meta-llama/",
]


def _is_cache_disabled_for_model(model_name: str) -> bool:
    # Optional global kill-switch
    if os.getenv("DISABLE_API_CACHE") is not None:
        return True
    if model_name in NO_CACHE_MODELS:
        return True
    for prefix in NO_CACHE_MODEL_PREFIXES:
        if model_name.startswith(prefix):
            return True
    return False


def is_thinking_model(model_name: str) -> bool:
    """
    Whether or not there is an explicit thinking mode for this model.
    """
    THINKING_MODELS = [
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "openai/gpt-5",
        "openai/gpt-5-nano",
        "openai/gpt-5-mini",
        "openai/o3",
        "deepseek/deepseek-r1",
    ]
    return model_name in THINKING_MODELS


class OpenaiResponse(BaseModel):
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str | None = None
    system_fingerprint: str | None = None

    @property
    def first_response(self) -> str:
        try:
            content = self.choices[0]["message"]["content"]
            if content is None:
                raise ValueError(f"No content found in OpenaiResponse: {self}")
            if isinstance(content, dict):
                # anthropic format
                content = content.get("text", "")
            return content
        except TypeError:
            raise ValueError(f"No content found in OpenaiResponse: {self}")

    @property
    def reasoning_content(self) -> str | None:
        """
        Returns the reasoning content if it exists, otherwise None.
        """
        # sometimes has reasoning_content or reasoning instead of content e.g. deepseek-reasoner or gemini
        possible_keys = ["reasoning_content", "reasoning"]
        for key in possible_keys:
            if key in self.choices[0]["message"]:
                return self.choices[0]["message"][key]

            content = self.choices[0]["message"].get("content")
            if isinstance(content, dict) and key in content:
                return content[key]
        return None
        # raise ValueError(f"No reasoning_content found in OpenaiResponse: {self}")

    @property
    def has_reasoning(self) -> bool:
        if self.reasoning_content is not None:
            return True
        return False

    def has_response(self) -> bool:
        if len(self.choices) == 0:
            return False
        first_choice = self.choices[0]
        if first_choice["message"] is None:
            return False
        if first_choice["message"]["content"] is None:
            return False
        return True

    @property
    def hit_content_filter(self) -> bool:
        """
        OpenaiResponse(choices=[{'finish_reason': None, 'index': 0, 'logprobs': None, 'message': None, 'finishReason': 'content_filter'}], usage={'completion_tokens': None, 'prompt_tokens': None, 'total_tokens': None, 'completion_tokens_details': None, 'prompt_tokens_details': None, 'completionTokens': 0, 'promptTokens': 279, 'totalTokens': 279}, created=1734802468, model='gemini-2.0-flash-exp', id=None, system_fingerprint=None, object='chat.completion', service_tier=None)
        """
        first_choice = self.choices[0]
        if "finishReason" in first_choice:
            if first_choice["finishReason"] == "content_filter":
                return True
        if "finish_reason" in first_choice:
            if first_choice["finish_reason"] == "content_filter":
                return True
        return False

    @property
    def abnormal_finish(self) -> bool:
        first_choice = self.choices[0]
        if "finishReason" in first_choice:
            if first_choice["finishReason"] not in ["stop", "length"]:
                return True
        if "finish_reason" in first_choice:
            if first_choice["finish_reason"] not in ["stop", "length"]:
                return True
        return False


class Caller(ABC):
    @abstractmethod
    async def call(
        self,
        messages: ChatHistory | Sequence[ChatMessage],
        config: InferenceConfig,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        pass

    @abstractmethod
    async def flush(self) -> None:
        # flush file buffers
        raise NotImplementedError()

    ## implement context manager
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.flush()


class CacheByModel(Generic[GenericBaseModel]):
    def __init__(
        self, cache_path: Path, cache_type: Type[GenericBaseModel] = OpenaiResponse
    ):
        self.cache_path = Path(cache_path)
        # if not exists, create it
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)
        assert (
            self.cache_path.is_dir()
        ), f"cache_path must be a folder, you provided {cache_path}"
        self.cache: dict[str, APIRequestCache[GenericBaseModel]] = {}
        self.cache_type = cache_type

    def get_cache(self, model: str) -> APIRequestCache[GenericBaseModel]:
        if model not in self.cache:
            path = self.cache_path / f"{model}.jsonl"
            self.cache[model] = APIRequestCache(
                cache_path=path, response_type=self.cache_type
            )
        return self.cache[model]

    async def flush(self) -> None:
        for cache in self.cache.values():
            await cache.flush()


class OpenrouterCaller(Caller):
    def __init__(
        self,
        cache_path: Path | str | CacheByModel,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
    ):
        if client is not None:
            self.client = client
        else:
            if api_key is None:
                env_key = os.getenv("OPENROUTER_API_KEY")
                assert (
                    env_key is not None
                ), "Please provide an OpenRouter API Key. Either pass it as an argument or set it in the environment variable OPENROUTER_API_KEY"
                api_key = env_key
            self.client = AsyncOpenAI(api_key=api_key)
        self.cache_by_model = (
            CacheByModel(Path(cache_path))
            if not isinstance(cache_path, CacheByModel)
            else cache_path
        )

    async def flush(self) -> None:
        await self.cache_by_model.flush()

    def get_cache(self, model: str) -> APIRequestCache[OpenaiResponse]:
        return self.cache_by_model.get_cache(model)

    # UPDATE (Atticus): use new reasoning format for OpenRouter
    async def call(
        self,
        messages: ChatHistory | Sequence[ChatMessage],  # backwards compat
        config: InferenceConfig,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        if not isinstance(messages, ChatHistory):
            messages = ChatHistory(messages=messages)
        disable_cache = _is_cache_disabled_for_model(config.model)
        if not disable_cache:
            maybe_result = await self.get_cache(config.model).get_model_call(
                messages, config, tool_args
            )
            if maybe_result is not None:
                return maybe_result

        if not is_thinking_model(config.model):
            to_pass_reasoning = {}
        elif config.reasoning is None:
            to_pass_reasoning = {"reasoning": OPENAI_OMIT}
        else:
            config.reasoning.pop("max_tokens", None)
            to_pass_reasoning = {"reasoning_effort": config.reasoning["effort"]}

        to_pass_extra_body = config.extra_body
        if config.model == "meta-llama/llama-3.1-8b-instruct":
            to_pass_extra_body = {
                "provider": {"order": ["cerebras/fp16", "novita/fp8", "deepinfra/fp8"]}
            }
        elif config.model == "meta-llama/llama-3.1-70b-instruct":
            to_pass_extra_body = {
                "provider": {"order": ["deepinfra/turbo", "fireworks"]}
            }

        if len(messages.messages) == 0:
            raise ValueError("Messages must be non-empty")
        try:
            logger.debug(f"Calling OpenRouter with config: {config}")
            chat_completion = await self.client.chat.completions.create(
                model=config.model,
                messages=[msg.to_openai_content() for msg in messages.messages],  # type: ignore
                max_tokens=(
                    config.max_tokens
                    if config.max_tokens is not None
                    else OPENAI_OMIT
                ),
                temperature=(
                    config.temperature
                    if config.temperature is not None
                    else OPENAI_OMIT
                ),
                top_p=config.top_p if config.top_p is not None else OPENAI_OMIT,
                frequency_penalty=(
                    config.frequency_penalty
                    if config.frequency_penalty is not None
                    else OPENAI_OMIT
                ),
                response_format=config.response_format if config.response_format is not None else OPENAI_OMIT,  # type: ignore
                tools=tool_args.tools if tool_args is not None else OPENAI_OMIT,  # type: ignore
                extra_body=to_pass_extra_body,
                **to_pass_reasoning,  # type: ignore
            )
        except Exception as e:
            note = f"Model: {config.model}. API domain: {self.client.base_url}"
            e.add_note(note)
            raise e

        try:
            resp = OpenaiResponse.model_validate(chat_completion.model_dump())
        except ValidationError as e:
            logger.error(
                f"Validation error for model {config.model}.\n"
                f"Prompt: {messages}.\n"
                f"resp: {chat_completion.model_dump()}\n"
            )
            logger.error(f"Full traceback:", exc_info=True)
            raise e

        logger.debug(f"OpenRouter response: {resp}")

        # Only cache clean, usable responses
        if (
            not disable_cache
            and resp.has_response()
            and not resp.abnormal_finish
            and not resp.hit_content_filter
        ):
            await self.get_cache(config.model).add_model_call(
                messages=messages,
                config=config,
                response=resp,
                tools=tool_args,
            )
        return resp


class AnthropicCaller(Caller):
    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic,
        cache_path: Path | str | CacheByModel,
    ):
        self.client = anthropic_client
        self.cache_by_model = (
            CacheByModel(Path(cache_path))
            if not isinstance(cache_path, CacheByModel)
            else cache_path
        )

    async def flush(self) -> None:
        await self.cache_by_model.flush()

    def get_cache(self, model: str) -> APIRequestCache[OpenaiResponse]:
        return self.cache_by_model.get_cache(model)

    async def call(
        self,
        messages: ChatHistory | Sequence[ChatMessage],
        config: InferenceConfig,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        if not isinstance(messages, ChatHistory):
            messages = ChatHistory(messages=messages)
        assert tool_args is None, "Anthropic does not support tools"
        disable_cache = _is_cache_disabled_for_model(config.model)
        if not disable_cache:
            maybe_result = await self.get_cache(config.model).get_model_call(
                messages, config, tool_args
            )
            if maybe_result is not None:
                return maybe_result

        non_system, system = Slist(messages.messages).split_by(
            lambda msg: msg.role != "system"
        )
        anthropic_messages = [
            {"role": msg.role, "content": msg.content} for msg in non_system
        ]
        if system.length >= 2:
            raise ValueError("Anthropic does not support multiple system messages")
        system_message: ChatMessage | None = system.first_option
        to_pass_sys = (
            system_message.content
            if system_message is not None
            else ANTHROPIC_OMIT
        )

        if config.reasoning is not None and is_thinking_model(config.model):
            to_pass_thinking = {
                "type": "enabled",
                "budget_tokens": config.reasoning["max_tokens"],
            }
            to_pass_temperature = 1.0
        else:
            to_pass_thinking = ANTHROPIC_OMIT
            to_pass_temperature = (
                config.temperature
                if config.temperature is not None
                else ANTHROPIC_OMIT
            )

        assert config.max_tokens is not None, "Anthropic requires max_tokens"

        logger.debug(f"Calling Anthropic with config: {config}")
        response: Message = await self.client.messages.create(
            model=config.model,
            messages=anthropic_messages,  # type: ignore
            max_tokens=config.max_tokens,  # type: ignore
            temperature=to_pass_temperature,  # type: ignore
            top_p=config.top_p if config.top_p is not None else ANTHROPIC_OMIT,  # type: ignore
            system=to_pass_sys,
            thinking=to_pass_thinking,  # type: ignore
            extra_body=config.extra_body,
        )

        # convert
        # response.content is a list of blocks: ThinkingBlock, TextBlock
        # TODO: add ToolUse support
        if response.content[0].type == "thinking":
            if len(response.content) != 2:
                logger.warning(f"Expected 2 blocks in response: {response.content}")

            try:
                response_content = {
                    "reasoning": response.content[0].thinking,
                    "text": response.content[1].text,
                }
            except Exception as e:
                response_content = {
                    "reasoning": "N/A",
                    "text": "N/A",
                }
        else:
            if len(response.content) != 1:
                logger.warning(f"Expected 1 block in response: {response.content}")
            try:
                response_content = {
                    "text": response.content[0].text,
                }
            except Exception as e:
                response_content = {
                    "text": "N/A",
                }

        openai_response = OpenaiResponse(
            id=response.id,
            choices=[{"message": {"content": response_content, "role": "assistant"}}],
            created=int(datetime.now().timestamp()),
            model=config.model,
            system_fingerprint=None,
            usage=response.usage.model_dump(),
        )

        # Only cache clean, usable responses
        if (
            not disable_cache
            and openai_response.has_response()
            and not openai_response.abnormal_finish
            and not openai_response.hit_content_filter
        ):
            await self.get_cache(config.model).add_model_call(
                messages=messages,
                config=config,
                response=openai_response,
                tools=tool_args,
            )

        return openai_response


@dataclass
class CallerConfig:
    prefix: str
    caller: Caller


class MultiClientCaller(Caller):
    """
    Routes requests to the appropriate caller based on the model name.
    """

    def __init__(self, clients: Sequence[CallerConfig]):
        self.callers: list[tuple[str, Caller]] = [
            (client.prefix, client.caller) for client in clients
        ]

    async def flush(self) -> None:
        for _, caller in self.callers:
            await caller.flush()

    def _get_caller_for_model(self, model: str) -> Caller:
        for model_prefix, caller in self.callers:
            if model.startswith(model_prefix):
                return caller
        available_patterns = [pattern for pattern, _ in self.callers]
        raise ValueError(
            f"No caller found for model {model}. Available patterns specified: {available_patterns}"
        )

    async def call(
        self,
        messages: ChatHistory | Sequence[ChatMessage],
        config: InferenceConfig,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        caller = self._get_caller_for_model(config.model)
        return await caller.call(messages, config, tool_args)


class PooledCaller(Caller):
    def __init__(self, callers: Sequence[Caller]):
        self.callers = callers

    async def flush(self) -> None:
        for caller in self.callers:
            await caller.flush()

    async def call(
        self,
        messages: ChatHistory | Sequence[ChatMessage],
        config: InferenceConfig,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        caller = random.choice(self.callers)
        return await caller.call(messages, config, tool_args)


def get_universal_caller(
    cache_dir: str = "/workspace/rm-bias/.api_cache",
    dotenv_path: str = "/workspace/rm-bias/.env",
) -> MultiClientCaller:
    dotenv.load_dotenv(dotenv_path=dotenv_path)
    if not (openrouter_key := os.getenv("OPENROUTER_API_KEY")):
        logger.warning("OPENROUTER_API_KEY not found in environment.")
    if not (anthropic_key := os.getenv("ANTHROPIC_API_KEY")):
        logger.warning("ANTHROPIC_API_KEY not found in environment.")

    # Create cache directory structure
    cache_path = Path(cache_dir)
    cache_by_model = CacheByModel(cache_path)

    # Configure callers
    callers = []
    openrouter_client = AsyncOpenAI(
        api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"
    )
    openrouter_caller = OpenrouterCaller(
        client=openrouter_client, cache_path=cache_by_model
    )
    anthropic_client = AsyncAnthropic(api_key=anthropic_key)
    anthropic_caller = AnthropicCaller(
        anthropic_client=anthropic_client, cache_path=cache_by_model
    )

    callers.extend(
        [
            CallerConfig(prefix="claude-", caller=anthropic_caller),
            CallerConfig(
                prefix="", caller=openrouter_caller
            ),  # use openrouter for all other models
        ]
    )

    return MultiClientCaller(clients=callers)


RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic._exceptions.OverloadedError,
)

CHANCE_EXCEPTIONS = (
    ValidationError,
    JSONDecodeError,
    ValueError,
    anthropic.BadRequestError,
)


def custom_wait_strategy(retry_state):
    """Custom wait strategy based on exception type."""
    exception = retry_state.outcome.exception()

    print(
        f"Retry attempt {retry_state.attempt_number}: Exception type: {type(exception)}, Exception: {exception}"
    )

    # Rate limit and timeout errors: use exponential backoff
    if isinstance(exception, RETRYABLE_EXCEPTIONS):
        print(f"Retryable exception: {exception}")
        return wait_random_exponential(multiplier=5, max=60)(retry_state)

    # Validation and server errors: use fixed wait
    elif isinstance(exception, CHANCE_EXCEPTIONS):
        print(f"Chance exception: {exception}")
        return 0.5

    print(f"Unhandled exception type: {type(exception)}")
    return 0.0


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS + CHANCE_EXCEPTIONS),
    wait=custom_wait_strategy,
    stop=stop_after_attempt(5),
)
async def sample_from_model(
    prompt: ChatHistory, caller: MultiClientCaller, full_logging: bool = False, **kwargs
) -> OpenaiResponse:
    """
    Run a single prompt and return the model's response with retry logic.
    kwargs are passed to InferenceConfig.
    """
    if full_logging:
        logger.info(
            f"Sampling from model {kwargs['model']}. <PROMPT> {prompt.as_text()} </PROMPT>"
        )

    response = await caller.call(
        prompt,
        config=InferenceConfig(**kwargs),
    )

    if full_logging:
        print(response)
        reasoning_content = response.reasoning_content
        logger.info(
            f"[sample_from_model] Got response:\n"
            f"<RESPONSE> {response.first_response} </RESPONSE>\n"
            f"<REASONING> {reasoning_content} </REASONING>"
        )

    if response.abnormal_finish:
        raise ValueError(f"Abnormal finish: {response}")
    return response


async def sample_across_models(
    models: list[str],
    prompt: ChatHistory,
    caller: MultiClientCaller,
    full_logging: bool = False,
    desc: str = "",
    **kwargs,
) -> Slist[OpenaiResponse]:
    """
    Run the same prompt across multiple models.
    """
    responses = await Slist(models).par_map_async(
        func=lambda model: sample_from_model(
            prompt, caller, full_logging=full_logging, model=model, **kwargs
        ),
        max_par=len(models),
        tqdm=True,
        desc=desc,  # type: ignore
    )
    return responses


async def sample_from_model_parallel(
    prompts: list[ChatHistory],
    caller: MultiClientCaller,
    max_par: int | None,
    full_logging: bool = False,
    desc: str = "",
    **kwargs,
) -> Slist[OpenaiResponse]:
    """
    Run multiple prompts across the same other kwargs.
    """
    logger.info(f"Sending {len(prompts)} prompts with {max_par} parallel calls...")

    if desc == "":
        to_pass_tqdm = False
    else:
        to_pass_tqdm = True

    responses = await Slist(prompts).par_map_async(
        func=lambda prompt: sample_from_model(
            prompt, caller, full_logging=full_logging, **kwargs
        ),
        max_par=max_par,
        tqdm=to_pass_tqdm,
        desc=desc,  # type: ignore
    )
    return responses


# %%
if __name__ == "__main__":
    # demo code
    import asyncio

    # CHANGE THIS TO YOUR LOCATIONS
    dotenv_path = Path("/workspace/rm-bias/.env")
    if not dotenv_path.exists():
        raise FileNotFoundError(f"Required .env file not found at {dotenv_path}")
    cache_dir = "/workspace/rm-bias/.api_cache"
    caller = get_universal_caller(cache_dir=cache_dir, dotenv_path=str(dotenv_path))

    prompt = ChatHistory.from_system(
        "You are Neel Nanda, the mechanistic interpretability researcher."
    )
    prompt = prompt.add_user("What are the worst papers in interpretability?")

    models = [
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
        "openai/gpt-5-mini",
        "openai/o3",
        "google/gemini-2.5-pro",
    ]
    response = asyncio.run(
        sample_across_models(
            models=models,
            prompt=prompt,
            caller=caller,
            full_logging=True,
            temperature=0.7,
            max_tokens=1200,
            reasoning={
                "max_tokens": 1024,
                "effort": "high",
            },
        )
    )
    print(response.map(lambda x: x.first_response))
# %%
