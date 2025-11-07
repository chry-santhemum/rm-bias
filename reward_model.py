"""Reward model."""

# %%

import hashlib
import asyncio
import torch
import logging
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
import nest_asyncio
from typing import Sequence

from caller import ChatHistory
from utils import load_model, REWARD_MODELS

logger = logging.getLogger(__name__)
nest_asyncio.apply()

# %%


@dataclass(frozen=True)
class RatingResult:
    score: float | None
    reasoning: str | None


def prompt_to_hash_path(prompt: str, target_dir: Path) -> Path:
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    return target_dir / f"{prompt_hash}.json"


class RewardModel:
    """
    Wrapper around reward models; __init__ kwargs (e.g. device) are passed to load_model
    """

    def __init__(self, model_name: str, batch_size: int = 32, **kwargs):
        assert model_name in REWARD_MODELS, f"Model {model_name} not local!!!"
        self.batch_size = batch_size
        self.model, self.tokenizer = load_model(model_name, **kwargs)
        self.device = self.model.device
        self.model_name = model_name

    def rate(
        self, chat_histories: Sequence[ChatHistory|None], use_tqdm: bool = True
    ) -> list[RatingResult]:
        rewards = []

        pbar = (
            tqdm(
                range(0, len(chat_histories), self.batch_size), desc="Rating responses"
            )
            if use_tqdm
            else range(0, len(chat_histories), self.batch_size)
        )
        for i in pbar:
            batch = chat_histories[i : i + self.batch_size]
            indices_not_none = [idx for idx, chat in enumerate(batch) if chat is not None]
            batch_clean: list[ChatHistory] = [batch[idx] for idx in indices_not_none]   # type: ignore
            inputs = [chat.remove_system().to_openai_messages() for chat in batch_clean]
            input_ids = self.tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                padding_side="right",
            ).to(self.model.device)

            attn_mask = input_ids.ne(self.tokenizer.pad_token_id)

            with torch.no_grad():
                scores = self.model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits.squeeze(-1)

            batch_results = [RatingResult(score=None, reasoning=None) for _ in range(len(batch))]
            for idx, score in enumerate(scores.tolist()):
                batch_results[indices_not_none[idx]] = RatingResult(score=float(score), reasoning=None)

            rewards.extend(batch_results)

        return rewards

    async def async_rate(
        self, chat_histories: Sequence[ChatHistory|None], use_tqdm: bool = True
    ) -> list[RatingResult]:
        return await asyncio.to_thread(self.rate, chat_histories=chat_histories, use_tqdm=use_tqdm)
