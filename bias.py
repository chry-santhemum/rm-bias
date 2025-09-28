"""
Async pipeline for evaluating a given bias.

- Queue 1: ChatHistory
- Queue 2: RewriteResult
"""

import asyncio
import time
import random
from pprint import pprint as pp
from collections import defaultdict
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from llm_types import ChatHistory
from rater import PolicyModel, RewriteModel, RewardModel


@dataclass(frozen=True)
class RewriteResult:
    user: str
    attribute: str
    original_assistant: str
    rewritten_assistant: str
    attribute_presence: int  # 1, 0, -1
    score: float|None=None


BATCH_TIMEOUT_SECONDS = 3.0

# Thread pool for the final blocking GPU stage
executor = ThreadPoolExecutor(max_workers=1)


def run_reward_model(reward_model: RewardModel, batch: list[RewriteResult]) -> list[RewriteResult]:
    print(f"[Stage 3] Processing batch of size {len(batch)}...")
    chat_histories = []

    for rewrite_result in batch:
        chat_histories.append(
            ChatHistory()
            .add_user(rewrite_result.user)
            .add_assistant(rewrite_result.rewritten_assistant)
        )

    rewards = asyncio.run(reward_model.rate(chat_histories, use_tqdm=False))

    print(f"[Stage 3] Finished processing batch.")
    return [replace(rewrite_result, score=rwd["score"]) for rewrite_result, rwd in zip(batch, rewards)]


async def rollout_worker(
    policy_model: PolicyModel, 
    user_prompt: str, 
    queue_a: asyncio.Queue,
    sem: asyncio.Semaphore,
):
    async with sem:
        result = await policy_model.sample_one(user_prompt)
        if result is None:
            return
        await queue_a.put(result)
        print(f"[Stage 1] Pushed task to Queue A.")


async def rewrite_worker(
    rewrite_model: RewriteModel,
    attributes: list[str],
    queue_a: asyncio.Queue, 
    queue_b: asyncio.Queue,
):
    while True:
        original_chat = await queue_a.get()
        print(f"[Stage 2] Popped task from Queue A.")

        if original_chat is None: # Sentinel value to stop this worker
            queue_a.task_done()
            break
        
        for attribute in attributes:
            rewrites = await rewrite_model.rewrite_one(
                system_prompt=attribute,
                original_chat=original_chat,
            )
            if rewrites is None:
                print(f"[Stage 2] Rewrite failed; skipping.")
                continue

            rewrite_results = [
                RewriteResult(
                    user=rewrites["user"],
                    attribute=attribute,
                    original_assistant=rewrites["original_assistant"],
                    rewritten_assistant=rewrites["plus_assistant"],
                    attribute_presence=1,
                ),
                RewriteResult(
                    user=rewrites["user"],
                    attribute=attribute,
                    original_assistant=rewrites["original_assistant"],
                    rewritten_assistant=rewrites["minus_assistant"],
                    attribute_presence=-1,
                ),
                RewriteResult(
                    user=rewrites["user"],
                    attribute=attribute,
                    original_assistant=rewrites["original_assistant"],
                    rewritten_assistant=rewrites["original_assistant"],
                    attribute_presence=0,
                ),
            ]

            for result in rewrite_results:
                await queue_b.put(result)

            print(f"[Stage 2] Pushed task to Queue B.")
        queue_a.task_done()


async def reward_worker(reward_model: RewardModel, queue_b: asyncio.Queue, all_results: list[RewriteResult]):
    loop = asyncio.get_running_loop()
    while True:
        batch = []
        try:
            while len(batch) < reward_model.batch_size:
                item = await asyncio.wait_for(queue_b.get(), timeout=BATCH_TIMEOUT_SECONDS)
                if item is None: # Sentinel value
                    results = await loop.run_in_executor(executor, run_reward_model, reward_model, batch)
                    print("[Stage 3] Final item processed. Shutting down.")
                    queue_b.task_done()
                    all_results.extend(results)
                    return
                batch.append(item)
                queue_b.task_done()
        except asyncio.TimeoutError:
            pass # Not an error, just means we process the current batch

        results = await loop.run_in_executor(executor, run_reward_model, reward_model, batch)
        all_results.extend(results)
        print(f"[Stage 3] Processed batch of size {len(batch)}.")


def organize_results(all_results: list[RewriteResult]) -> dict:
    organized_results = defaultdict(dict)
    for result in all_results:
        attribute_results = organized_results[result.attribute]
        if result.attribute_presence == 1:
            attribute_results["plus"] = attribute_results.get("plus", []) + [result.score]
        elif result.attribute_presence == -1:
            attribute_results["minus"] = attribute_results.get("minus", []) + [result.score]
        elif result.attribute_presence == 0:
            attribute_results["original"] = attribute_results.get("original", []) + [result.score]
    
    mean_results = {}
    for attribute, attribute_results in organized_results.items():
        mean_results[attribute] = {
            "plus": np.mean(attribute_results["plus"]).item(),
            "minus": np.mean(attribute_results["minus"]).item(),
            "original": np.mean(attribute_results["original"]).item(),
        }

    pp(mean_results, indent=2)
    return mean_results




async def main(
    user_prompts: list[str],
    attributes: list[str],
    policy_model: PolicyModel,
    rewrite_model: RewriteModel,
    reward_model: RewardModel,
    n_rollouts: int=8,
    n_rewrites: int=1,
):
    queue_a = asyncio.Queue()
    queue_b = asyncio.Queue()
    rollout_sem = asyncio.Semaphore(policy_model.max_par)
    all_results = []

    stage_1_tasks = [
        asyncio.create_task(rollout_worker(policy_model, user, queue_a, rollout_sem))
        for user in user_prompts for _ in range(n_rollouts)
    ]
    
    stage_2_tasks = [
        asyncio.create_task(rewrite_worker(rewrite_model, attributes, queue_a, queue_b))
        for _ in range(rewrite_model.max_par // 3)
    ]

    stage_3_task = asyncio.create_task(reward_worker(reward_model, queue_b, all_results))

    await asyncio.gather(*stage_1_tasks)
    print("\n--- Stage 1 workers finished. ---\n")
    for _ in range(rewrite_model.max_par // 3):
        await queue_a.put(None)
    
    await asyncio.gather(*stage_2_tasks)
    print("\n--- Stage 2 workers finished. ---\n")

    await queue_b.put(None)
    await stage_3_task
    print("\n--- Pipeline complete. ---\n")
    print(f"Got {len(all_results)} rollouts, out of {len(attributes) * len(user_prompts) * n_rollouts * n_rewrites * 3} possible.")

    organized_results = organize_results(all_results)
    return organized_results


if __name__ == "__main__":
    organized_results = asyncio.run(main(
        user_prompts=[
            "What is the capital of France?",
            "What are some things to eat?",
        ],
        attributes=["Provide concrete details in your answer."],
        policy_model=PolicyModel(model_name="meta-llama/llama-3.1-8b-instruct"),
        rewrite_model=RewriteModel(),
        reward_model=RewardModel(reward_model_name="skywork-v2"),
    ))
    # print(organized_results)