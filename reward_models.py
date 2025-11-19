"""Reward model wrapper for both LLM Judge and classifier."""

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
from abc import ABC, abstractmethod

from caller import ChatHistory
from models import JudgeModel
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


class RewardModel(ABC):
    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @abstractmethod
    async def async_rate(
        self, 
        chat_histories: Sequence[ChatHistory | None], 
        use_tqdm: bool = True
    ) -> list[RatingResult]:
        pass


class LocalRewardModel(RewardModel):
    def __init__(self, model_name: str, devices: list[str], batch_size_per_device: int, attn_implementation: str="eager"):
        assert model_name in REWARD_MODELS
        assert len(devices) > 0
        self.model_name = model_name
        self.tokenizer = None
        self.batch_size_per_device = batch_size_per_device

        self.models = []
        for device in devices:
            logger.info(f"Loading model {model_name} on device {device}...")
            model, tokenizer = load_model(model_name, device=device, attn_implementation=attn_implementation)
            self.models.append(model)
            if self.tokenizer is None:
                self.tokenizer = tokenizer
    
    @property
    def batch_size(self) -> int:
        return self.batch_size_per_device * len(self.models)

    def rate_one_model(self, 
        model_index: int, 
        chat_histories: Sequence[ChatHistory | None], 
        use_tqdm: bool = True
    ) -> list[RatingResult]:
        model = self.models[model_index]
        rewards = []
        pbar = (
            tqdm(
                range(0, len(chat_histories), self.batch_size_per_device), desc="Rating responses"
            )
            if use_tqdm
            else range(0, len(chat_histories), self.batch_size_per_device)
        )

        for i in pbar:
            batch = chat_histories[i : i + self.batch_size_per_device]
            indices_not_none = [
                idx for idx, chat in enumerate(batch) if chat is not None
            ]
            batch_clean: list[ChatHistory] = [batch[idx] for idx in indices_not_none]  # type: ignore
            inputs = [chat.remove_system().to_openai_messages() for chat in batch_clean]
            input_ids = self.tokenizer.apply_chat_template(  # type: ignore
                inputs,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                padding_side="right",
            ).to(model.device)  # type: ignore

            attn_mask = input_ids.ne(self.tokenizer.pad_token_id)  # type: ignore

            with torch.no_grad():
                scores = model(  # type: ignore
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits.squeeze(-1)

            batch_results = [
                RatingResult(score=None, reasoning=None) for _ in range(len(batch))
            ]
            for idx, score in enumerate(scores.tolist()):
                batch_results[indices_not_none[idx]] = RatingResult(
                    score=float(score), reasoning=None
                )

            rewards.extend(batch_results)

        return rewards

    async def async_rate(self, chat_histories, use_tqdm = False):
        if not chat_histories:
            return []

        n_models = len(self.models)
        indices_by_model: list[list[int]] = [[] for _ in range(n_models)]
        chats_by_model: list[list] = [[] for _ in range(n_models)]

        # Simple round-robin sharding over the available reward models.
        for idx, ch in enumerate(chat_histories):
            model_idx = idx % n_models
            indices_by_model[model_idx].append(idx)
            chats_by_model[model_idx].append(ch)

        # Launch rating on each device in parallel (only where we have work).
        coros = []
        model_indices = []
        for model_idx, (model, hist_list) in enumerate(
            zip(self.models, chats_by_model)
        ):
            if not hist_list:
                continue
            coros.append(asyncio.to_thread(
                self.rate_one_model, model_index=model_idx, chat_histories=hist_list, use_tqdm=use_tqdm
            ))
            model_indices.append(model_idx)

        results_by_model: list[list | None] = [None] * n_models
        if coros:
            gathered = await asyncio.gather(*coros)
            for model_idx, res in zip(model_indices, gathered):
                results_by_model[model_idx] = res

        # Reassemble results to match original ordering.
        combined_results = [None] * len(chat_histories)
        for model_idx, indices in enumerate(indices_by_model):
            if not indices:
                continue
            model_results = results_by_model[model_idx]
            assert model_results is not None
            for local_i, orig_idx in enumerate(indices):
                combined_results[orig_idx] = model_results[local_i]

        return combined_results



class APIRewardModel(RewardModel):
    def __init__(self, model_name: str, max_par: int, **kwargs):
        self.model_name = model_name
        self.max_par = max_par
        self.kwargs = kwargs
        self.model = JudgeModel(model_name=model_name, max_par=max_par, **kwargs)
    
    @property
    def batch_size(self) -> int:
        return self.max_par
        
    async def async_rate(self, chat_histories, use_tqdm = False):
        results = await self.model.judge(chat_histories=chat_histories, use_tqdm=use_tqdm)  # type: ignore
        rating_results = []

        for result in results:
            if result is None:
                rating_results.append(RatingResult(score=None, reasoning=None))
            else:
                try:
                    score_float = float(result["score"])
                except ValueError:
                    logger.error(f"Could not convert score to float: {result['score']}")
                    score_float = None
                rating_results.append(RatingResult(score=score_float, reasoning=result["reasoning"]))

        return rating_results
