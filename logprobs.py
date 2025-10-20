# %%
import pickle
import asyncio
import nest_asyncio
from tqdm.auto import tqdm
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from caller import Caller, ChatHistory
from reward_model import RewardModel
from utils import load_model
from standard_prompts import make_prompt_mix

nest_asyncio.apply()


def get_logprobs_and_confidence(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    chats: list[ChatHistory],
    batch_size: int = 8,
) -> tuple[list[float], list[float]]:
    """
    Compute the log probabilities and confidence of the given completions.

    Definition of confidence:
    Average negative entropy of the next token probabilities of the assistant completions.

    Only count the assistant tokens, not the user tokens.

    Args:
        model: The language model
        tokenizer: The tokenizer
        chats: List of chat histories to process
        batch_size: Number of chats to process in a single batch

    Returns:
        tuple: (logprobs, confidences) where each is a list of floats
    """

    logprobs_list = []
    confidences_list = []
    device = next(model.parameters()).device

    # Process in batches
    for batch_start in tqdm(range(0, len(chats), batch_size)):
        batch_end = min(batch_start + batch_size, len(chats))
        batch_chats = chats[batch_start:batch_end]

        # Get the lengths of user tokens

        user_token_lengths = []

        for chat in batch_chats:
            user_messages = [msg for msg in chat.messages if msg.role != "assistant"]
            if user_messages:
                user_chat = ChatHistory(messages=user_messages)
                user_tokens = tokenizer.apply_chat_template(  # type: ignore
                    user_chat.to_openai_messages(),
                    return_tensors="pt",
                    tokenize=True,
                    add_generation_prompt=True,
                )
                user_length = user_tokens.shape[1]
            else:
                user_length = 0

            user_token_lengths.append(user_length)

        tokenizer.padding_side = "right"
        batch_input_ids = tokenizer.apply_chat_template(  # type: ignore
            batch_chats,
            return_tensors="pt",
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
        ).to(device)
        batch_attention_mask = batch_input_ids.ne(tokenizer.pad_token_id).to(device)

        # Forward pass for the batch
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            )
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

        # Process each item in the batch
        for idx, user_length in enumerate(user_token_lengths):
            log_probs = F.log_softmax(
                logits[idx], dim=-1
            )  # Shape: [seq_len, vocab_size]
            total_length = batch_attention_mask[idx].sum().item()

            if user_length <= total_length - 1:
                token_logprobs = []
                token_entropies = []

                for i in range(user_length, total_length):
                    token_id = batch_input_ids[idx, i]

                    # Log prob
                    token_logprob = log_probs[i - 1, token_id].item()
                    token_logprobs.append(token_logprob)

                    # Confidence
                    probs = torch.exp(log_probs[i - 1])
                    entropy = -(probs * log_probs[i - 1]).sum().item()
                    token_entropies.append(-entropy)

                # Aggregate metrics
                avg_logprob = (
                    sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0.0
                )
                avg_confidence = (
                    sum(token_entropies) / len(token_entropies)
                    if token_entropies
                    else 0.0
                )

                logprobs_list.append(avg_logprob)
                confidences_list.append(avg_confidence)
            else:
                # No assistant tokens
                logprobs_list.append(0.0)
                confidences_list.append(0.0)

    return logprobs_list, confidences_list


async def sample_responses(
    user_prompts: list[str], n_samples: int = 32
) -> list[ChatHistory]:
    to_send_chats = [p for p in user_prompts for _ in range(n_samples)]
    async with Caller() as caller:
        responses = await caller.call(
            messages=to_send_chats,
            max_parallel=512,
            desc="Sampling responses",
            model="meta-llama/llama-3.1-8b-instruct",
            disable_cache=True,
            temperature=0.9,
            max_tokens=1024,
        )

    output = []
    for user_prompt, response in zip(to_send_chats, responses):
        output.append(
            ChatHistory.from_user(user_prompt).add_assistant(response.first_response)
        )
    return output


# %%
if __name__ == "__main__":
    user_prompts = make_prompt_mix(num_total=128)

    model, tokenizer = load_model("llama-3.1-8b-instruct")
    chats = asyncio.run(sample_responses(user_prompts=user_prompts, n_samples=64))

    logprobs, confidences = get_logprobs_and_confidence(
        model=model, tokenizer=tokenizer, chats=chats, batch_size=16
    )

    reward_model = RewardModel(model_name="skywork-v2", batch_size=16)
    rewards = [s.score for s in reward_model.rate(chats)]

# %%
# normalize reward scores
# within samples of each user prompt, subtract by mean

# There are `n_samples` completions for each user prompt
n_samples = 64
num_prompts = len(user_prompts)
assert len(rewards) == num_prompts * n_samples, "Rewards array size mismatch"

rewards = np.array(rewards)
rewards_subtract_mean = rewards.copy()
rewards_normalized = rewards.copy()

for i in range(num_prompts):
    start = i * n_samples
    end = (i + 1) * n_samples
    mean = rewards[start:end].mean()
    std = rewards[start:end].std()
    rewards_subtract_mean[start:end] -= mean
    rewards_normalized[start:end] = rewards_subtract_mean[start:end] / std


with open("data/scrap/logprobs.pkl", "wb") as f:
    pickle.dump(
        {
            "user_prompts": user_prompts,
            "chats": chats,
            "logprobs": logprobs,
            "confidences": confidences,
            "rewards": rewards,
            "rewards_subtract_mean": rewards_subtract_mean,
            "rewards_normalized": rewards_normalized,
        },
        f,
    )

# %%
# draw scatterplot and fit linear regression
# set small point size and lower opacity
fig = px.scatter(
    x=logprobs,
    y=rewards,
    size_max=4,
    opacity=0.3,
    labels={"x": "logprobs", "y": "reward (unnormalized)"},
)
fig.show()

fig = px.scatter(
    x=confidences,
    y=rewards,
    size_max=4,
    opacity=0.3,
    labels={"x": "confidence", "y": "reward (unnormalized)"},
)
fig.show()

# %%

import numpy as np

fig = px.scatter(
    x=np.exp(logprobs),
    y=rewards,
    size_max=4,
    opacity=0.3,
    labels={"x": "probs", "y": "reward (unnormalized)"},
)
fig.show()

fig = px.scatter(
    x=np.exp(confidences),
    y=rewards,
    size_max=4,
    opacity=0.3,
    labels={"x": "exp(confidence)", "y": "reward (unnormalized)"},
)
fig.show()
# %%
