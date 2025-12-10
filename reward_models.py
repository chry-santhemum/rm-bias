"""Reward model wrapper for both LLM Judge and classifier."""

import asyncio
import gc
import torch
from tqdm.auto import tqdm
from typing import Sequence
from abc import ABC, abstractmethod
from loguru import logger

from caller import ChatHistory
from api_models import JudgeModel, RatingResult, ComparisonResult
from state import Rollout, Score
from utils import load_model, find_executable_batch_size


class RewardModel(ABC):
    model_name: str

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

    @abstractmethod
    async def async_compare(
        self, 
        chat_histories_A: Sequence[ChatHistory | None], 
        chat_histories_B: Sequence[ChatHistory | None],
        use_tqdm: bool = True
    ) -> list[ComparisonResult]:
        pass

    async def judge_validation_results(
        self,
        validation_results: list[dict[str, dict[str, list[Rollout|None]]]],
        val_baselines: dict[str, list[Rollout|None]],
        first_n_rollouts: int=4,
        first_n_user_prompts: int=8  # 0 means all
    ) -> None:
        """
        Populates teacher_score on rollouts in validation_results in place.

        For each rollout, compares it against the corresponding baseline using
        async_compare and sets the teacher_score with:
        - score: winrate (1.0 for A wins, 0.0 for B wins, 0.5 for tie)
        - raw_score: score_diff from comparison (if available, e.g. for RM-based comparison)
        - reasoning: from comparison result (if available, e.g. for LLM judge)
        - model_name: this model's name
        """
        chat_histories_A = []  # rewritten
        chat_histories_B = []  # baseline
        judge_tasks_info = []

        for i, validation_result in enumerate(validation_results):
            for attribute, attribute_stats in validation_result.items():
                user_prompt_count = 0
                for user_prompt, rollouts in attribute_stats.items():
                    if first_n_user_prompts > 0 and user_prompt_count >= first_n_user_prompts:
                        break
                    user_prompt_count += 1

                    baseline_rollouts = val_baselines[user_prompt]
                    rollout_count = 0
                    for rollout_idx, rollout in enumerate(rollouts):
                        if first_n_rollouts > 0 and rollout_count >= first_n_rollouts:
                            break
                        if rollout is None or baseline_rollouts[rollout_idx] is None:
                            continue
                        rollout_count += 1

                        chat_histories_A.append(ChatHistory.from_user(user_prompt).add_assistant(rollout.response))
                        chat_histories_B.append(ChatHistory.from_user(user_prompt).add_assistant(baseline_rollouts[rollout_idx].response))
                        judge_tasks_info.append(
                            {
                                "seed_state_idx": i,
                                "attribute": attribute,
                                "user_prompt": user_prompt,
                                "rollout_idx": rollout_idx,
                            }
                        )

        logger.info(f"Running {len(judge_tasks_info)} judge tasks...")
        judge_tasks_results = await self.async_compare(
            chat_histories_A=chat_histories_A,
            chat_histories_B=chat_histories_B,
        )

        # Populate teacher_score on each rollout in place
        for judge_task_result, judge_task_info in zip(
            judge_tasks_results, judge_tasks_info
        ):
            seed_state_idx = judge_task_info["seed_state_idx"]
            attribute = judge_task_info["attribute"]
            user_prompt = judge_task_info["user_prompt"]
            rollout_idx = judge_task_info["rollout_idx"]

            # Compute winrate score from winner
            winrate_score = None
            match judge_task_result.winner:
                case "A":
                    winrate_score = 1.0
                case "B":
                    winrate_score = 0.0
                case "Tie":
                    winrate_score = 0.5
                case None:
                    winrate_score = None

            teacher_score = Score(
                score=winrate_score,
                raw_score=judge_task_result.score_diff,
                reasoning=judge_task_result.reasoning,
                model_name=self.model_name,
            )

            # Update the rollout in place
            rollout = validation_results[seed_state_idx][attribute][user_prompt][rollout_idx]
            if rollout is not None:
                rollout.teacher_score = teacher_score


class LocalRewardModel(RewardModel):
    def __init__(self, model_name: str, devices: list[str], batch_size_per_device: int, attn_implementation: str="sdpa"):
        assert len(devices) > 0
        self.model_name = model_name
        self.tokenizer = None
        self.batch_size_per_device = batch_size_per_device

        self.models = []
        for device in devices:
            print(f"Loading model {model_name} on device {device}...")
            model, tokenizer = load_model(model_name, model_type="reward", device=device, attn_implementation=attn_implementation)
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
        rating_results = []
        if len(chat_histories) == 0:
            return []

        # inner loop
        @find_executable_batch_size(starting_batch_size=self.batch_size_per_device)
        def rate_one_model_inner(batch_size: int, chats_clean: list[ChatHistory]) -> list[RatingResult]:
            results_clean = []
            inner_pbar = range(0, len(chats_clean), batch_size)
            for inner_idx in inner_pbar:
                sub_chats = chats_clean[inner_idx : inner_idx + batch_size]
                sub_inputs = [chat.remove_system().to_openai_messages() for chat in sub_chats]
                input_ids = self.tokenizer.apply_chat_template(  # type: ignore
                    sub_inputs,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                ).to(model.device)  # type: ignore

                attn_mask = input_ids.ne(self.tokenizer.pad_token_id)  # type: ignore

                with torch.no_grad():
                    scores_tensor = model(  # type: ignore
                        input_ids=input_ids, attention_mask=attn_mask
                    ).logits.squeeze(-1)

                scores_list = scores_tensor.tolist()
                results_clean.extend([
                    RatingResult(score=float(score), reasoning=None) 
                    for score in scores_list
                ])
            return results_clean

        # outer loop: break down to roughly correct batch size
        if use_tqdm:
            outer_pbar = tqdm(range(0, len(chat_histories), self.batch_size_per_device), desc="Rating responses")
        else:
            outer_pbar = range(0, len(chat_histories), self.batch_size_per_device)
        
        for i in outer_pbar:
            batch = chat_histories[i : i + self.batch_size_per_device]
            indices_not_none = [idx for idx, chat in enumerate(batch) if chat is not None]

            # remove the None chats
            batch_clean: list[ChatHistory] = [batch[idx] for idx in indices_not_none]  # type: ignore
            if len(batch_clean) == 0:
                continue

            results_clean = rate_one_model_inner(chats_clean=batch_clean)

            # add back the None chats
            batch_results = [RatingResult(score=None, reasoning=None) for _ in range(len(batch))]
            for idx, result in zip(indices_not_none, results_clean):
                batch_results[idx] = result
            rating_results.extend(batch_results)

        return rating_results

    async def async_rate(self, chat_histories, use_tqdm = False):
        n_models = len(self.models)
        n_chats = len(chat_histories)
        tasks = []

        for i in range(n_models):
            if i < n_models - 1:
                tasks.append(asyncio.to_thread(
                    self.rate_one_model, model_index=i,
                    chat_histories=chat_histories[i*n_chats // n_models : (i+1)*n_chats // n_models],
                ))
            else:
                tasks.append(asyncio.to_thread(
                    self.rate_one_model, model_index=i,
                    chat_histories=chat_histories[i*n_chats // n_models :],
                ))

        if tasks:
            task_results = await asyncio.gather(*tasks)
        
        combined_results: list[RatingResult] = []
        for task_result in task_results:
            combined_results.extend(task_result)

        return combined_results

    async def async_compare(self, chat_histories_A, chat_histories_B, use_tqdm = False):
        results_A = await self.async_rate(chat_histories_A, use_tqdm=use_tqdm)
        results_B = await self.async_rate(chat_histories_B, use_tqdm=use_tqdm)
        compare_results = []

        for result_A, result_B in zip(results_A, results_B):
            if result_A.score is None or result_B.score is None:
                compare_results.append(ComparisonResult(winner=None, reasoning=None, score_diff=None))
            elif result_A.score > result_B.score:
                compare_results.append(ComparisonResult(winner="A", reasoning=None, score_diff=result_A.score - result_B.score))
        
        return compare_results


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
        return await self.model.judge_absolute(chat_histories=chat_histories, use_tqdm=use_tqdm)

    async def async_compare(self, chat_histories_A, chat_histories_B, use_tqdm = False):
        return await self.model.judge_relative(chat_histories_A=chat_histories_A, chat_histories_B=chat_histories_B, use_tqdm=use_tqdm)
