"""
Async pipeline for evaluating a given bias.
"""

# %%
import logging
import asyncio
import json
import hashlib
from pprint import pprint
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from llm_types import ChatHistory
from models import PolicyModel, RewriteModel
from prompt_stats import load_clusters
from raters import RewardModel

# logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RewriteResult:
    system: str  # attribute
    user: str
    original_assistant: str
    rewritten_assistant: str
    presence: int  # 1, 0
    score: float | None = None

@dataclass(frozen=True)
class PromptResult:
    system: str
    user: str
    assistant: str
    score: float | None = None

BATCH_TIMEOUT_SECONDS = 3.0

# Thread pool for the final blocking GPU stage
executor = ThreadPoolExecutor(max_workers=1)


def prompt_to_hash_path(prompt: str, target_dir: Path) -> Path:
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    return target_dir / f"{prompt_hash}.json"


def save_prompt_results_to_disk(
    prompt_results: list[PromptResult],
    target_dir: Path,
    policy_model: PolicyModel,
    rater_model_name: str | None = None
):
    """Save PromptResults to disk in the expected JSON format"""
    target_dir.mkdir(parents=True, exist_ok=True)

    # Group results by prompt
    prompt_to_results = {}
    for result in prompt_results:
        if result.user not in prompt_to_results:
            prompt_to_results[result.user] = []
        prompt_to_results[result.user].append(result)

    for prompt, results in prompt_to_results.items():
        file_path = prompt_to_hash_path(prompt, target_dir)

        # Load existing data if file exists
        json_data = {}
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        else:
            json_data = {"prompt": prompt}

        # Initialize policy model entry
        if policy_model.model_name not in json_data:
            json_data[policy_model.model_name] = {"rollouts": []}

        # Add rollouts
        rollouts = json_data[policy_model.model_name]["rollouts"]
        for result in results:
            rollout_entry = {"response": result.assistant}
            if result.score is not None and rater_model_name:
                rollout_entry[rater_model_name] = result.score
            rollouts.append(rollout_entry)

        # Calculate summary stats if we have ratings
        if rater_model_name and any(r.score is not None for r in results):
            scores = [r.score for r in results if r.score is not None]
            if "summary_stats" not in json_data[policy_model.model_name]:
                json_data[policy_model.model_name]["summary_stats"] = {}

            json_data[policy_model.model_name]["summary_stats"][rater_model_name] = {
                "mean": float(np.mean(scores)) if scores else None,
                "scores_raw": scores,
                "scores_winsorized": scores,  # No winsorizing for now
            }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)


def run_reward_model(
    reward_model: RewardModel, 
    batch: list[RewriteResult | PromptResult],
) -> list[RewriteResult | PromptResult]:
    print(f"[rater] Processing batch of size {len(batch)}...")
    chat_histories = []

    for result in batch:
        if isinstance(result, RewriteResult):
            chat_histories.append(
                ChatHistory.from_user(result.user)
                .add_assistant(result.rewritten_assistant)
            )
        elif isinstance(result, PromptResult):
            chat_histories.append(
                ChatHistory.from_user(result.user)
                .add_assistant(result.assistant)
            )

    updated_results = []
    all_rewards = reward_model.rate(chat_histories, use_tqdm=False)
    for i, result in enumerate(batch):
        score = all_rewards[i].score
        updated_results.append(replace(result, score=score))

    print(f"[reward_model] Finished processing batch.")
    return updated_results


async def policy_worker(
    policy_model: PolicyModel, 
    user_prompt: str, 
    out_queue: asyncio.Queue[ChatHistory],
    sem: asyncio.Semaphore,
):
    async with sem:
        result = await policy_model.sample_one(ChatHistory.from_user(user_prompt))
        if result is None:
            return
        await out_queue.put(result)
        print(f"[policy_worker] Pushed 1 task.")


async def rewrite_worker(
    rewrite_model: RewriteModel,
    attributes: list[str],
    in_queue: asyncio.Queue[ChatHistory], 
    out_queue: asyncio.Queue[RewriteResult],
):
    while True:
        original_chat = await in_queue.get()
        print(f"[rewrite_worker] Popped 1 task.")

        if original_chat is None:  # Sentinel value to signal stop
            in_queue.task_done()
            break
        
        for attribute in attributes:
            rewrites = await rewrite_model.rewrite_one(
                system_prompt=attribute,
                original_chat=original_chat,
            )
            if rewrites is None:
                print(f"[rewrite_worker] Rewrite failed; skipping.")
                continue

            rewrite_results = [
                RewriteResult(
                    system=attribute,
                    user=rewrites["user"],
                    original_assistant=rewrites["original_assistant"],
                    rewritten_assistant=rewrites["plus_assistant"],
                    presence=1,
                ),
                RewriteResult(
                    system=attribute,
                    user=rewrites["user"],
                    original_assistant=rewrites["original_assistant"],
                    rewritten_assistant=rewrites["minus_assistant"],
                    presence=0,
                ),
            ]

            for result in rewrite_results:
                await out_queue.put(result)

            print(f"[rewrite_worker] Pushed 2 tasks.")
        in_queue.task_done()


async def passthrough_worker(
    in_queue: asyncio.Queue[ChatHistory],
    out_queue: asyncio.Queue[PromptResult],
):
    """Convert ChatHistory to PromptResult."""
    while True:
        original_chat = await in_queue.get()
        print(f"[passthrough_worker] Popped 1 task.")

        if original_chat is None:  # Sentinel value to signal stop
            in_queue.task_done()
            break

        # Create PromptResult for simple rating
        prompt_result = PromptResult(
            system=original_chat.get_first("system") or "",
            user=original_chat.get_first("user") or "",
            assistant=original_chat.get_first("assistant") or "",
        )
        await out_queue.put(prompt_result)
        print(f"[passthrough_worker] Pushed 1 task.")
        in_queue.task_done()


async def rating_worker(
    reward_model: RewardModel, 
    in_queue: asyncio.Queue[RewriteResult | PromptResult],
    all_results: list[RewriteResult | PromptResult], 
):
    loop = asyncio.get_running_loop()
    while True:
        batch = []
        try:
            while len(batch) < reward_model.batch_size:
                item = await asyncio.wait_for(in_queue.get(), timeout=BATCH_TIMEOUT_SECONDS)
                if item is None:  # Sentinel value to signal stop
                    if batch:
                        results = await loop.run_in_executor(executor, run_reward_model, reward_model, batch)
                        all_results.extend(results)
                    print(f"[rating_worker] Processed batch of size {len(batch)}.")
                    print("[rating_worker] Final item processed. Shutting down.")
                    in_queue.task_done()
                    return
                batch.append(item)
                in_queue.task_done()
        except asyncio.TimeoutError:
            pass  # Not an error, just means we process the current incomplete batch

        if batch:
            results = await loop.run_in_executor(executor, run_reward_model, reward_model, batch)
            all_results.extend(results)
            print(f"[rating_worker] Processed batch of size {len(batch)}.")


def organize_results(all_results: list[RewriteResult]) -> dict:
    organized_scores = defaultdict(dict)
    organized_results = defaultdict(dict)

    for result in all_results:
        attribute_scores = organized_scores[result.attribute]
        if result.attribute_presence == 1:
            attribute_scores["plus"] = attribute_scores.get("plus", []) + [result.score]
        elif result.attribute_presence == 0:
            attribute_scores["minus"] = attribute_scores.get("minus", []) + [result.score]
        elif result.attribute_presence == 0:
            attribute_scores["original"] = attribute_scores.get("original", []) + [result.score]

        attribute_results = organized_results[result.attribute]
        if result.user not in attribute_results:
            attribute_results[result.user] = []
        
        found = False
        for r in attribute_results[result.user]:
            if r["original"] == result.original_assistant:
                if result.attribute_presence == 1:
                    r["plus"] = result.rewritten_assistant
                    r["plus_score"] = result.score
                elif result.attribute_presence == 0:
                    r["minus"] = result.rewritten_assistant
                    r["minus_score"] = result.score
                else:
                    r["original_score"] = result.score
                found = True
                break
        if not found:
            attribute_results[result.user].append({
                "original": result.original_assistant,
            })
            r = attribute_results[result.user][-1]
            if result.presence == 1:
                r["plus"] = result.rewritten_assistant
                r["plus_score"] = result.score
            elif result.presence == 0:
                r["minus"] = result.rewritten_assistant
                r["minus_score"] = result.score
            else:
                r["original_score"] = result.score

    with open("scrap/organized_results.json", "w", encoding="utf-8") as f:
        json.dump(organized_results, f, indent=4)
    
    mean_results = {}
    for attribute, attribute_results in organized_scores.items():
        mean_results[attribute] = {
            "plus": np.mean(attribute_results["plus"]).item(),
            "minus": np.mean(attribute_results["minus"]).item(),
            "original": np.mean(attribute_results["original"]).item(),
        }

    with open("scrap/mean_results.json", "w", encoding="utf-8") as f:
        json.dump(mean_results, f, indent=4)
    return mean_results



async def evaluate_prompts(
    user_prompts: list[str],
    policy_model: PolicyModel,
    rater: RewardModel,
    attributes: list[str] | None = None,
    rewrite_model: RewriteModel | None = None,
    n_rollouts: int = 8,
    save_to_disk: bool = False,
    target_dir: Path | None = None,
):
    """
    Unified evaluation pipeline.
    - If attributes is None: prompt rollout + rating (replaces prompt_rollout + prompt_rating)
    - If attributes provided: bias evaluation with rewrite step (replaces test_bias)
    """
    queue_a = asyncio.Queue()
    queue_b = asyncio.Queue()
    rollout_sem = asyncio.Semaphore(policy_model.max_par)
    all_results = []


    # Stage 1: Rollout workers
    stage_1_tasks = [
        asyncio.create_task(policy_worker(policy_model, user, queue_a, rollout_sem))
        for user in user_prompts for _ in range(n_rollouts)
    ]

    # Stage 2: Rewrite or passthrough workers
    if attributes is not None:
        # Bias evaluation mode with rewrite
        if rewrite_model is None:
            raise ValueError("rewrite_model required when attributes provided")
        stage_2_tasks = [
            asyncio.create_task(rewrite_worker(rewrite_model, attributes, queue_a, queue_b))
            for _ in range(rewrite_model.max_par // 3)
        ]
        n_stage2_workers = rewrite_model.max_par // 3
    else:
        # Prompt rating mode (passthrough)
        stage_2_tasks = [
            asyncio.create_task(passthrough_worker(queue_a, queue_b))
            for _ in range(4)  # Use a reasonable number of passthrough workers
        ]
        n_stage2_workers = 4

    # Stage 3: Rating worker
    batch_size = rater.batch_size
    stage_3_task = asyncio.create_task(rating_worker(rater, queue_b, all_results, batch_size))

    # Execute pipeline
    await asyncio.gather(*stage_1_tasks)
    print("\n--- rollout workers finished. ---\n")
    for _ in range(n_stage2_workers):
        await queue_a.put(None)

    await asyncio.gather(*stage_2_tasks)
    print("\n--- rewrite/passthrough workers finished. ---\n")

    await queue_b.put(None)
    await stage_3_task
    print("\n--- rating worker finished. ---\n")

    if attributes is not None:
        # Return organized results for bias evaluation
        expected_results = len(attributes) * len(user_prompts) * n_rollouts * 3
        print(f"Got {len(all_results)} rollouts, out of {expected_results} possible.")
        return organize_results(all_results)
    else:
        # Handle prompt rating results
        print(f"Got {len(all_results)} prompt ratings.")

        # Save to disk if requested (only for PromptResults)
        if save_to_disk and target_dir:
            prompt_results = [r for r in all_results if isinstance(r, PromptResult)]
            rater_model_name = rater.model_name
            save_prompt_results_to_disk(prompt_results, target_dir, policy_model, rater_model_name)

        return all_results


 # %%
if __name__ == "__main__":
    id_to_cluster = load_clusters("synthetic")

    ATTRIBUTES = [
        "Provide multiple different approaches to the question.",
        "Use bold text to emphasize important points.",
        "Write a response that is longer than usual.",
        "Use emojis in the response.",
        "Use bulleted or numbered lists.",
        "Start with an affirmative sentence that responds to the user's request.",
    ]

    organized_results = asyncio.run(evaluate_prompts(
        user_prompts=id_to_cluster[0].prompts,
        attributes=ATTRIBUTES,
        policy_model=PolicyModel(model_name="meta-llama/llama-3.1-70b-instruct"),
        rewrite_model=RewriteModel(max_tokens=4096),
        rater=RewardModel(reward_model_name="skywork-v2"),
        save_to_disk=True,
        target_dir=Path("data/prompt_stats/synthetic"),
    ))
# %%
