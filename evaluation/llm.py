import asyncio
from dataclasses import dataclass
from typing import Any

from chat_limiter import (
    BatchConfig,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatLimiter,
    Message,
    process_chat_completion_batch,
)


@dataclass
class LLM:
    llm_model_name: str
    provider: str | None = None
    api_key: str | None = None
    chat_timeout: float = 240.0
    max_tokens: int = 6000
    temperature: float = 0.7
    max_concurrent: int = 200
    # Limit how many API calls are scheduled in one asyncio.gather to avoid huge task graphs
    _limiter: ChatLimiter | None = None

    @property
    def limiter(self) -> ChatLimiter:
        if self._limiter is None:
            self._limiter = ChatLimiter.for_model(
                self.llm_model_name, api_key=self.api_key, timeout=self.chat_timeout
            )
        return self._limiter

    async def _generate_batch_llm_response(
        self,
        key_to_messages: dict[Any, list[Message]],
    ) -> dict[Any, str | None]:
        assert isinstance(key_to_messages, dict) and len(key_to_messages) > 0

        keys = list(key_to_messages.keys())

        # Build requests (processor can take raw requests directly)
        requests: list[ChatCompletionRequest] = []
        for key in keys:
            requests.append(
                ChatCompletionRequest(
                    model=self.llm_model_name,
                    messages=key_to_messages[key],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            )

        # Configure batch processing
        config = BatchConfig(
            max_concurrent_requests=int(self.max_concurrent),
            max_retries_per_item=3,
            show_progress=True,
            progress_desc="Generating batch of responses",
            print_prompts=False,
            print_responses=False,
            verbose_exceptions=True,
            print_request_initiation=False,
        )

        # Run batch using limiter; no need to specify provider explicitly
        async with ChatLimiter.for_model(
            self.llm_model_name,
            api_key=self.api_key,
            timeout=self.chat_timeout,
            provider=self.provider,
        ) as limiter:
            results = await process_chat_completion_batch(limiter, requests, config)

        # Map results back to input keys
        key_to_response: dict[Any, str | None] = {}
        for i, result in enumerate(results):
            if result.success and result.result and result.result.choices:
                key_to_response[keys[i]] = result.result.choices[0].message.content
            else:
                key_to_response[keys[i]] = None

        assert len(key_to_response) == len(keys)
        return key_to_response
