"""LLM types module."""

import time
from pathlib import Path
from typing import Optional, Type, Sequence, Generic, TypeVar, Mapping, Any, Literal
import anyio
from anyio import Path as AnyioPath
from pydantic import BaseModel, ValidationError
import json
from slist import Slist
import hashlib

from utils import REWARD_MODELS, POLICY_MODELS

# Generic to say what we are caching
APIResponse = TypeVar("APIResponse", bound=BaseModel)


class ToolArgs(BaseModel):
    tools: Sequence[Mapping[Any, Any]]
    tool_choice: str


class ChatMessage(BaseModel):
    role: str
    content: str
    # base64
    image_content: str | None = None
    image_type: str | None = None  # image/jpeg, or image/png

    def as_text(self) -> str:
        return f"{self.role}:\n{self.content}"

    def to_openai_content(self) -> dict:
        """e.g.
            "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{image_base_64}"
                        },
                    },
                ],
            }
        ],

        """
        if not self.image_content:
            return {
                "role": self.role,
                "content": self.content,
            }
        else:
            assert self.image_type, "Please provide an image type"
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{self.image_type};base64,{self.image_content}"
                        },
                    },
                ],
            }

    def to_anthropic_content(self) -> dict:
        if not self.image_content:
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                ],
            }
        else:
            """
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image1_media_type,
                    "data": image1_data,
                },
            },
            """
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": self.image_type or "image/jpeg",
                            "data": self.image_content,
                        },
                    },
                ],
            }


class ChatHistory(BaseModel):
    messages: Sequence[ChatMessage] = []

    def all_assistant_messages(self) -> Slist[ChatMessage]:
        return Slist(self.messages).filter(lambda msg: msg.role == "assistant")

    # def to_finetune(self) -> FinetuneConversation:
    #     return FinetuneConversation(
    #         messages=[
    #             FinetuneMessage(role=msg.role, content=msg.content)
    #             for msg in self.messages
    #         ]
    #     )

    def as_text(self) -> str:
        return "\n".join([msg.as_text() for msg in self.messages])

    @staticmethod
    def from_system(content: str) -> "ChatHistory":
        return ChatHistory(messages=[ChatMessage(role="system", content=content)])

    @staticmethod
    def from_maybe_system(content: str | None) -> "ChatHistory":
        if content is None:
            return ChatHistory()
        else:
            return ChatHistory.from_system(content)

    def remove_system(self) -> "ChatHistory":
        """Remove all system prompts."""
        new_messages = []
        for msg in self.messages:
            if msg.role != "system":
                new_messages.append(msg)
        
        return ChatHistory(messages=new_messages)


    def add_user(self, content: str) -> "ChatHistory":
        new_messages = list(self.messages) + [ChatMessage(role="user", content=content)]
        return ChatHistory(messages=new_messages)

    def add_assistant(self, content: str) -> "ChatHistory":
        new_messages = list(self.messages) + [
            ChatMessage(role="assistant", content=content)
        ]
        return ChatHistory(messages=new_messages)

    def add_messages(self, messages: Sequence[ChatMessage]) -> "ChatHistory":
        new_messages = list(self.messages) + list(messages)
        return ChatHistory(messages=new_messages)

    def to_openai_messages(self) -> list[dict]:
        return [msg.to_openai_content() for msg in self.messages]
    
    def get_first(self, role: Literal["system", "user", "assistant"]) -> str:
        """Get the first message with the given role, if exists"""
        for msg in self.messages:
            if msg.role == role:
                return msg.content
        return ""


class InferenceConfig(BaseModel):
    # Config for openai
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    response_format: dict | None = None
    continue_final_message: bool | None = None
    reasoning: dict | None = None
    extra_body: dict | None = None


class InferenceResponse(BaseModel):
    raw_responses: Sequence[str]

    @property
    def single_response(self) -> str:
        if len(self.raw_responses) != 1:
            raise ValueError(
                f"This response has multiple responses {self.raw_responses}"
            )
        else:
            return self.raw_responses[0]


class FileCacheRow(BaseModel):
    key: str
    response: str  # Should be generic, but w/e


def write_jsonl_file_from_basemodel(
    path: Path | str, basemodels: Sequence[BaseModel]
) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for basemodel in basemodels:
            f.write(basemodel.model_dump_json() + "\n")


def read_jsonl_file_into_basemodel(
    path: Path | str, basemodel: Type[APIResponse]
) -> Slist[APIResponse]:
    with open(path) as f:
        return Slist(basemodel.model_validate_json(line) for line in f)


def file_cache_key(
    messages: ChatHistory,
    config: InferenceConfig,
    try_number: int,
    other_hash: str,
    tools: ToolArgs | None,
) -> str:
    config_dump = config.model_dump_json(
        exclude_none=True
    )  # for backwards compatibility
    tools_json = (
        tools.model_dump_json() if tools is not None else ""
    )  # for backwards compatibility
    str_messages = (
        ",".join([str(msg) for msg in messages.messages])
        + deterministic_hash(config_dump)
        + str(try_number)
        + tools_json
    )
    hash_of_history_not_messages = messages.model_dump(exclude_none=True)
    del hash_of_history_not_messages["messages"]
    str_history = (
        json.dumps(hash_of_history_not_messages) if hash_of_history_not_messages else ""
    )
    return deterministic_hash(str_messages + str_history + other_hash)


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


def validate_json_item(
    item: str, model: Type[GenericBaseModel]
) -> GenericBaseModel | None:
    try:
        return model.model_validate_json(item)
    except ValidationError:
        print(f"Error validating {item} with model {model}")
        return None


async def read_jsonl_file_into_basemodel_async(
    path: AnyioPath, basemodel: Type[GenericBaseModel]
) -> Slist[GenericBaseModel]:
    async with await anyio.open_file(path, "r") as f:
        return Slist(
            [basemodel.model_validate_json(line) for line in await f.readlines()]
        )


class APIRequestCache(Generic[APIResponse]):
    def __init__(self, cache_path: Path | str, response_type: Type[APIResponse]):
        self.cache_path = AnyioPath(cache_path)
        self.response_type = response_type
        self.data: dict[str, str] = {}
        self.file_handler: anyio.AsyncFile | None = None
        self.loaded_cache: bool = False
        self.cache_check_semaphore = anyio.Semaphore(1)

    async def flush(self) -> None:
        if self.file_handler:
            await self.file_handler.flush()

    async def load_cache(self) -> None:
        if await self.cache_path.exists():
            time_start = time.time()
            rows: Slist[FileCacheRow] = await read_jsonl_file_into_basemodel_async(
                path=self.cache_path,  # todo: asyncify
                basemodel=FileCacheRow,
            )
            time_end = time.time()
            n_items = len(rows)
            time_diff_1dp = round(time_end - time_start, 1)
            print(
                f"Loaded {n_items} items from {self.cache_path.as_posix()} in {time_diff_1dp} seconds"
            )
        else:
            rows = Slist()
        for row in rows:
            self.data[row.key] = row.response
        self.loaded_cache = True

    async def get_file_handler(self) -> anyio.AsyncFile:
        if self.file_handler is None:
            # if the file doesn't exist, create it
            if not await self.cache_path.exists():
                # make parent directories
                await self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                # make sure it's created
                await self.cache_path.touch()
            self.file_handler = await anyio.open_file(self.cache_path, "a")
        return self.file_handler

    async def add_model_call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int,
        response: APIResponse,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> None:
        key = file_cache_key(messages, config, try_number, other_hash, tools=tools)
        response_str = response.model_dump_json()
        self.data[key] = response_str
        await self.write_line(key=key, response_json=response_str)

    async def get_model_call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> Optional[APIResponse]:
        if not self.loaded_cache:
            async with self.cache_check_semaphore:
                # check again
                if not self.loaded_cache:
                    await self.load_cache()
        key = file_cache_key(messages, config, try_number, other_hash, tools=tools)
        response_str = self.data.get(key)
        if response_str:
            try:
                response = self.response_type.model_validate_json(response_str)
                return response
            except ValidationError as e:
                print(f"Warning: Failed to validate cache entry for key {key}")
                raise e
                # return None
        return None

    async def write_line(self, key: str, response_json: str) -> None:
        if not self.file_handler:
            await self.get_file_handler()
        if self.file_handler:
            async with self.cache_check_semaphore:
                line = (
                    FileCacheRow(key=key, response=response_json).model_dump_json()
                    + "\n"
                )
                await self.file_handler.write(line)


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


class HashableBaseModel(BaseModel):
    def model_hash(self) -> str:
        as_json = self.model_dump_json()
        return deterministic_hash(as_json)

    class Config:
        # this is needed for the hashable base model
        frozen = True


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


def is_local_model(model_name: str) -> bool:
    if model_name.startswith("claude-") or model_name.startswith("gemini-"):
        return False
    if model_name in [
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]:
        return False
    if model_name in POLICY_MODELS:
        return True
    else:
        logging.warning(
            f"Model {model_name} not found in POLICY_MODELS. Assuming it is not local."
        )
        return False
