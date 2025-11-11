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


class RewardModel:
    """
    Wrapper around local reward models.
    __init__ kwargs are passed to load_model for local models,
    and passed to JudgeModel for LLM judge models.
    """

    def __init__(self, model_name: str, batch_size: int, **kwargs):
        self.batch_size = batch_size
        self.model_name = model_name
        self.type: str

        if model_name in REWARD_MODELS:
            self.type = "local"
            logger.info(f"Loading local reward model {model_name}.")
            self.model, self.tokenizer = load_model(model_name, **kwargs)

        else:
            self.type = "api"
            logger.info(f"Loading LLM judge model {model_name}.")
            self.model = JudgeModel(model_name=model_name, max_par=batch_size, **kwargs)
            self.tokenizer = None


    async def async_rate(
        self, chat_histories: Sequence[ChatHistory | None], use_tqdm: bool = True
    ) -> list[RatingResult]:

        if self.type == "local":  
            def rate_local(
                chat_histories: Sequence[ChatHistory | None], use_tqdm: bool = True
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
                    ).to(self.model.device)  # type: ignore

                    attn_mask = input_ids.ne(self.tokenizer.pad_token_id)  # type: ignore

                    with torch.no_grad():
                        scores = self.model(  # type: ignore
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

            return await asyncio.to_thread(
                rate_local, chat_histories=chat_histories, use_tqdm=use_tqdm
            )
        
        else:
            # API model
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

