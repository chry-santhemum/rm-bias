"""Classifier-free guidance"""

# %%
import asyncio
import nest_asyncio
from utils import load_model
from caller import OpenRouterCaller, CacheConfig, ChatHistory
from models import PolicyModel

nest_asyncio.apply()

cache_config = CacheConfig(
    no_cache_models={
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
    }
)

caller = OpenRouterCaller(cache_config=cache_config)


# %%
async def call_with_logprobs():
    async with caller:
        responses = await caller.call(
            messages=[ChatHistory.from_user("Hello, how are you?")],
            model="meta-llama/llama-3.1-8b-instruct",
            max_parallel=128,
            logprobs=True,
        )
    return responses


# %%
responses = asyncio.run(call_with_logprobs())

print(responses[0])

# %%

model, tokenizer = load_model("meta-llama/llama-3.1-8b-instruct")
