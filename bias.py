"""
Async pipeline for evaluating a given bias.

policy_worker: --> PromptResult
conditional_policy_worker: --> PromptResult
rewrite_worker: PromptResult --> RewriteResult
reward_worker: RewriteResult | PromptResult --> RewriteResult | PromptResult

Initial prompt rating: policy_worker --- reward_worker
Conditioning on system prompt: conditional_policy_worker --- reward_worker
Rewrite: policy_worker --- rewrite_worker --- reward_worker
"""

# %%
import patches
import logging
import asyncio
import json
import time
import hashlib
from pprint import pprint
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, replace, asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from utils import timestamp
from llm_types import ChatHistory
from models import PolicyModel, RewriteModel
from load_cluster import load_clusters
from reward_model import RewardModel
from state import AttributeStats, Rollout, PlusMinusRollout

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

# %%

def run_reward_model(
    reward_model: RewardModel, 
    batch: list[RewriteResult | PromptResult],
) -> list[RewriteResult | PromptResult]:
    logger.info(f"[rater] Processing batch of size {len(batch)}...")
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

    logger.info(f"[reward_model] Finished processing batch.")
    return updated_results


async def policy_worker(
    policy_model: PolicyModel, 
    user_prompt: str, 
    out_queue: asyncio.Queue[PromptResult],
    sem: asyncio.Semaphore,
):
    async with sem:
        result = await policy_model.sample_one(ChatHistory.from_user(user_prompt))
        if result is None:
            return
        await out_queue.put(PromptResult(
            system="",
            user=user_prompt,
            assistant=result.get_first("assistant") or "",
        ))
        logger.info(f"[policy_worker] Pushed 1 task.")


async def conditional_policy_worker(
    policy_model: PolicyModel,
    attribute: str,
    user_prompt: str,
    out_queue: asyncio.Queue[PromptResult],
    sem: asyncio.Semaphore,
):
    async with sem:
        result = await policy_model.sample_one(ChatHistory.from_system(
            "Please make sure to prioritize following this system specification in your response: " + attribute
        ).add_user(user_prompt))
        if result is None:
            return
        await out_queue.put(PromptResult(
            system=attribute,
            user=user_prompt,
            assistant=result.get_first("assistant") or "",
        ))
        logger.info(f"[conditional_policy_worker] Pushed 1 task.")


async def rewrite_worker(
    rewrite_model: RewriteModel,
    attributes: list[str],
    in_queue: asyncio.Queue[PromptResult], 
    out_queue: asyncio.Queue[RewriteResult | PromptResult],
    n_samples: int=1,
):
    while True:
        prompt_result = await in_queue.get()
        logger.info(f"[rewrite_worker] Popped 1 task.")

        if prompt_result is None:  # Sentinel value to signal stop
            in_queue.task_done()
            break
        
        await out_queue.put(prompt_result)
        logger.info(f"[rewrite_worker] Pushed 1 task.")

        rewrites = await rewrite_model.rewrite_one(
            attributes=attributes,
            original_chat=ChatHistory.from_user(prompt_result.user).add_assistant(prompt_result.assistant),
            n_samples=n_samples,
        )

        for rewrite_dict in rewrites:
            for plus_rewrite_text in rewrite_dict["plus"]:
                if plus_rewrite_text is None:
                    continue
                await out_queue.put(RewriteResult(
                    system=rewrite_dict["attribute"],
                    user=rewrite_dict["user"],
                    original_assistant=rewrite_dict["original"],
                    rewritten_assistant=plus_rewrite_text,
                    presence=1,
                ))
                logger.info(f"[rewrite_worker] Pushed 1 task.")
            
            for minus_rewrite_text in rewrite_dict["minus"]:
                if minus_rewrite_text is None:
                    continue
                await out_queue.put(RewriteResult(
                    system=rewrite_dict["attribute"],
                    user=rewrite_dict["user"],
                    original_assistant=rewrite_dict["original"],
                    rewritten_assistant=minus_rewrite_text,
                    presence=0,
                ))
                logger.info(f"[rewrite_worker] Pushed 1 task.")

        in_queue.task_done()


async def rewrite_half_worker(
    rewrite_model: RewriteModel,
    in_queue: asyncio.Queue[PromptResult],
    out_queue: asyncio.Queue[RewriteResult | PromptResult],
    n_samples: int=1,
):
    """
    Only pull out the conditional rollouts, and rewrite in the minus direction.
    """
    while True:
        prompt_result = await in_queue.get()
        logger.info(f"[rewrite_half_worker] Popped 1 task.")

        if prompt_result is None:  # Sentinel value to signal stop
            in_queue.task_done()
            break

        if prompt_result.system == "":
            # is a baseline rollout, skip
            await out_queue.put(prompt_result)
            logger.info(f"[rewrite_half_worker] Pushed 1 baseline task.")
            continue
        
        await out_queue.put(prompt_result)
        logger.info(f"[rewrite_half_worker] Pushed 1 conditional task.")

        conditional_chat = (
            ChatHistory
            .from_system(prompt_result.system)
            .add_user(prompt_result.user)
            .add_assistant(prompt_result.assistant)
        )
        rewrites = await rewrite_model.rewrite_minus(
            conditional_chat=conditional_chat,
            n_samples=n_samples,
        )

        for minus_text in rewrites["minus"]:
            if minus_text is None:
                continue
            await out_queue.put(RewriteResult(
                system=rewrites["attribute"],
                user=rewrites["user"],
                original_assistant=rewrites["conditional"],
                rewritten_assistant=minus_text,
                presence=0,
            ))
            logger.info(f"[rewrite_half_worker] Pushed 1 rewritten task.")

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
                    logger.info(f"[rating_worker] Processed batch of size {len(batch)}.")
                    logger.info("[rating_worker] Final item processed. Shutting down.")
                    in_queue.task_done()
                    return
                if item.score is not None:  # already rated
                    all_results.append(item)
                    continue
                batch.append(item)
                in_queue.task_done()
        except asyncio.TimeoutError:
            pass  # Not an error, just means we process the current incomplete batch

        if batch:
            results = await loop.run_in_executor(executor, run_reward_model, reward_model, batch)
            all_results.extend(results)
            logger.info(f"[rating_worker] Processed batch of size {len(batch)}.")


# %%

async def evaluate_baselines(
    user_prompts: list[str],
    policy_model: PolicyModel,
    rater: RewardModel,
    n_rollouts: int = 8,
    save_dir: Path | None = None,
) -> dict[str, list[Rollout]]:
    queue = asyncio.Queue()
    rollout_sem = asyncio.Semaphore(policy_model.max_par)
    all_results = []  # where results will appear

    policy_worker_tasks = [
        asyncio.create_task(policy_worker(policy_model, user, queue, rollout_sem))
        for user in user_prompts for _ in range(n_rollouts)
    ]

    rating_worker_task = asyncio.create_task(rating_worker(rater, queue, all_results))

    # Shutdown
    await asyncio.gather(*policy_worker_tasks)
    logger.info("\n--- rollout workers finished. ---\n")

    await queue.put(None)
    await rating_worker_task
    logger.info("\n--- rating worker finished. ---\n")
    expected_results = len(user_prompts) * n_rollouts
    logger.info(f"Got {len(all_results)} rollouts, out of {expected_results} possible.")

    organized_results = organize_baseline_results(all_results, save_dir)

    return organized_results


async def evaluate_attributes_conditional(
    user_prompts: list[str],
    policy_model: PolicyModel,
    rater: RewardModel,
    attributes: list[str],
    n_rollouts: int = 8,
    save_dir: Path | None = None,
) -> dict[str, dict[str, list[Rollout]]]:
    queue = asyncio.Queue()
    rollout_sem = asyncio.Semaphore(policy_model.max_par)
    all_results = []

    rollout_tasks = [
        asyncio.create_task(conditional_policy_worker(policy_model, attribute, user, queue, rollout_sem))
        for user in user_prompts for attribute in attributes for _ in range(n_rollouts)
    ]

    rating_worker_task = asyncio.create_task(rating_worker(rater, queue, all_results))

    # Shutdown
    await asyncio.gather(*rollout_tasks)
    logger.info("\n--- rollout workers finished. ---\n")

    await queue.put(None)
    await rating_worker_task
    logger.info("\n--- rating worker finished. ---\n")
    expected_results = len(attributes) * len(user_prompts) * n_rollouts
    logger.info(f"Got {len(all_results)} rollouts, out of {expected_results} possible.")

    organized_results = organize_conditional_results(all_results, save_dir)

    return organized_results



async def evaluate_attributes_half(
    user_prompts: list[str],
    policy_model: PolicyModel,
    rater: RewardModel,
    attributes: list[str],
    rewrite_model: RewriteModel,
    n_rollouts: int = 8,
    n_rewrites: int = 1,
    save_dir: Path | None = None,
) -> dict[str, dict[str, list[PlusMinusRollout]]]:
    """
    Get conditional rollouts, then rewrite in the minus direction.
    """
    queue_a = asyncio.Queue()
    queue_b = asyncio.Queue()
    rollout_sem = asyncio.Semaphore(policy_model.max_par)
    n_rewrite_workers = rewrite_model.max_par
    all_results = []

    # use policy model to get conditional responses
    rollout_sem = asyncio.Semaphore(policy_model.max_par)
    rollout_tasks = [
        asyncio.create_task(conditional_policy_worker(policy_model, attribute, user, queue_a, rollout_sem))
        for user in user_prompts for attribute in attributes for _ in range(n_rollouts)
    ]

    rewrite_tasks = [
        asyncio.create_task(rewrite_half_worker(rewrite_model, queue_a, queue_b, n_rewrites))
        for _ in range(n_rewrite_workers)
    ]

    rating_worker_task = asyncio.create_task(rating_worker(rater, queue_b, all_results))

    # Shutdown
    if policy_model is not None:
        await asyncio.gather(*rollout_tasks)
        logger.info("\n--- rollout workers finished. ---\n")

    for _ in range(n_rewrite_workers):
        await queue_a.put(None)  # Sentinel values for rewrite_workers

    await asyncio.gather(*rewrite_tasks)
    logger.info("\n--- rewrite workers finished. ---\n")

    await queue_b.put(None)
    await rating_worker_task
    logger.info("\n--- rating worker finished. ---\n")
    expected_results =  len(user_prompts) * n_rollouts * len(attributes) * n_rewrites * 2
    logger.info(f"Got {len(all_results)} rollouts, out of {expected_results} possible.")

    organized_results = organize_rewrite_half_results(all_results, save_dir)

    return organized_results




async def evaluate_attributes_rewrite(
    user_prompts: list[str],
    policy_model: PolicyModel | None,
    baseline_rollouts: dict[str, list[Rollout]] | None,
    rater: RewardModel,
    attributes: list[str],
    rewrite_model: RewriteModel,
    n_rollouts: int = 8,
    n_rewrites: int = 1,
    save_dir: Path | None = None,
) -> dict[str, dict[str, list[PlusMinusRollout]]]:
    """
    Only pass in one of policy_model or baseline_rollouts.
    """
    assert (policy_model is None) ^ (baseline_rollouts is None), "Only pass in one of policy_model or baseline_rollouts."

    queue_a = asyncio.Queue()
    queue_b = asyncio.Queue()
    n_rewrite_workers = max(1, rewrite_model.max_par // (len(attributes) * n_rewrites * 2))
    all_results = []

    if policy_model is not None:
        rollout_sem = asyncio.Semaphore(policy_model.max_par)
        rollout_tasks = [
            asyncio.create_task(policy_worker(policy_model, user, queue_a, rollout_sem))
            for user in user_prompts for _ in range(n_rollouts)
        ]
    elif baseline_rollouts is not None:
        for user in user_prompts:
            for rollout in baseline_rollouts[user]:
                await queue_a.put(PromptResult(
                    system="",
                    user=user,
                    assistant=rollout.response,
                    score=rollout.score,
                ))

    rewrite_tasks = [
        asyncio.create_task(rewrite_worker(rewrite_model, attributes, queue_a, queue_b, n_rewrites))
        for _ in range(n_rewrite_workers)
    ]

    rating_worker_task = asyncio.create_task(rating_worker(rater, queue_b, all_results))

    # Shutdown
    if policy_model is not None:
        await asyncio.gather(*rollout_tasks)
        logger.info("\n--- rollout workers finished. ---\n")

    for _ in range(n_rewrite_workers):
        await queue_a.put(None)  # Sentinel values for rewrite_workers

    await asyncio.gather(*rewrite_tasks)
    logger.info("\n--- rewrite workers finished. ---\n")

    await queue_b.put(None)
    await rating_worker_task
    logger.info("\n--- rating worker finished. ---\n")
    expected_results =  len(user_prompts) * n_rollouts * (1 + len(attributes) * n_rewrites * 2)
    logger.info(f"Got {len(all_results)} rollouts, out of {expected_results} possible.")

    organized_results = organize_rewrite_results(all_results, save_dir)

    return organized_results


# %%

def organize_baseline_results(
    all_results: list[PromptResult],
    save_dir: Path|None=None,
) -> dict[str, list[Rollout]]:
    organized_scores = defaultdict(list)
    organized_results = defaultdict(list)

    for item in all_results:
        organized_scores[item.user].append(item.score)
        organized_results[item.user].append(Rollout(
            response=item.assistant,
            score=item.score,
        ))
    
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "baseline_results.json", "w", encoding="utf-8") as f:
            json_data = {k: [asdict(r) for r in v] for k, v in organized_results.items()}
            json.dump(json_data, f, indent=4)
        with open(save_dir / "baseline_scores.json", "w", encoding="utf-8") as f:
            json.dump(organized_scores, f, indent=4)
        
        mean_results = {}
        for user, scores in organized_scores.items():
            mean_results[user] = np.mean(scores).item()
        
        with open(save_dir / "baseline_scores_mean.json", "w", encoding="utf-8") as f:
            json.dump(mean_results, f, indent=4)

    return dict(organized_results)


def organize_conditional_results(
    all_results: list[PromptResult],
    save_dir: Path|None=None,
) -> dict[str, dict[str, list[Rollout]]]:
    organized_scores = defaultdict(dict)
    organized_results = defaultdict(dict)

    for item in all_results:
        attribute_scores = organized_scores[item.system]
        if item.user not in attribute_scores:
            attribute_scores[item.user] = []
        attribute_scores[item.user].append(item.score)

        attribute_results = organized_results[item.system]
        if item.user not in attribute_results:
            attribute_results[item.user] = []
        attribute_results[item.user].append(Rollout(
            response=item.assistant,
            score=item.score,
        ))

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "conditional_results.json", "w", encoding="utf-8") as f:
            json_data = {k: [asdict(r) for r in v] for k, v in organized_results.items()}
            json.dump(json_data, f, indent=4)

        with open(save_dir / "conditional_scores.json", "w", encoding="utf-8") as f:
            json.dump(organized_scores, f, indent=4)

        mean_score = {}
        for attribute, attribute_scores in organized_scores.items():
            mean_score[attribute] = {
                user: np.mean(attribute_scores[user]).item()
                for user in attribute_scores
            }
        with open(save_dir / "conditional_scores_mean.json", "w", encoding="utf-8") as f:
            json.dump(mean_score, f, indent=4)

    return dict(organized_results)



def organize_rewrite_results(
    all_results: list[RewriteResult | PromptResult],
    save_dir: Path|None=None,
) -> dict[str, dict[str, list[PlusMinusRollout]]]:

    baseline_results = defaultdict(list)
    rewrite_results = defaultdict(dict)

    baseline_items, rewrite_items = [], []
    for result in all_results:
        if isinstance(result, PromptResult):
            baseline_items.append(result)
        elif isinstance(result, RewriteResult):
            rewrite_items.append(result)

    for result in baseline_items:
        baseline_results[result.user].append(Rollout(
            response=result.assistant,
            score=result.score,
        ))

    for result in rewrite_items:
        attribute_results = rewrite_results[result.system]
        if result.user not in attribute_results:
            attribute_results[result.user] = [PlusMinusRollout(
                plus="",
                minus="",
                plus_score=None,
                minus_score=None,
            ) for _ in range(len(baseline_results[result.user]))]

        found = False
        for i, r in enumerate(baseline_results[result.user]):
            if r.response == result.original_assistant:
                if result.presence == 1:
                    if attribute_results[result.user][i].plus != "":
                        continue
                    attribute_results[result.user][i].plus = result.rewritten_assistant
                    attribute_results[result.user][i].plus_score = result.score
                    found = True
                    break
                elif result.presence == 0:
                    if attribute_results[result.user][i].minus != "":
                        continue
                    attribute_results[result.user][i].minus = result.rewritten_assistant
                    attribute_results[result.user][i].minus_score = result.score
                    found = True
                    break
                    
        if not found:
            raise ValueError(f"Rewrite result for {result.user} and {result.system} not found.")


    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "rewrite_baseline_results.json", "w", encoding="utf-8") as f:
            json_data = {k: [asdict(r) for r in v] for k, v in baseline_results.items()}
            json.dump(json_data, f, indent=4)
        with open(save_dir / "rewrite_results.json", "w", encoding="utf-8") as f:
            json_data = {k: {k2: [asdict(r) for r in v2] for k2, v2 in v.items()} for k, v in rewrite_results.items()}
            json.dump(json_data, f, indent=4)
        
        mean_results = {}

        baseline_scores = []
        for v in baseline_results.values():
            baseline_scores.extend([r.score for r in v])

        for attribute, attribute_results in rewrite_results.items():
            if attribute == "":
                continue
            plus_scores = []
            minus_scores = []
            for v in attribute_results.values():
                plus_scores.extend([r.plus_score for r in v])
                minus_scores.extend([r.minus_score for r in v])

            mean_results[attribute] = {
                "plus": np.mean(plus_scores).item(),
                "minus": np.mean(minus_scores).item(),
                "original": np.mean(baseline_scores).item(),
            }

        with open(save_dir / "rewrite_scores_mean.json", "w", encoding="utf-8") as f:
            json.dump(mean_results, f, indent=4)

    return dict(rewrite_results)


def organize_rewrite_half_results(
    all_results: list[RewriteResult | PromptResult],
    save_dir: Path|None=None,
) -> dict[str, dict[str, list[PlusMinusRollout]]]:

    organized_results = defaultdict(dict)

    for result in all_results:
        if isinstance(result, PromptResult):
            attribute_results = organized_results[result.system]
            if result.user not in attribute_results:
                attribute_results[result.user] = []

            attribute_results[result.user].append(PlusMinusRollout(
                plus=result.assistant,
                minus="",
                plus_score=result.score,
                minus_score=None,
            ))

    for result in all_results:
        if isinstance(result, PromptResult):
            continue

        found = False
        # Only process RewriteResult
        for rollout in organized_results[result.system][result.user]:
            if rollout.plus == result.original_assistant:
                rollout.minus = result.rewritten_assistant
                rollout.minus_score = result.score
                found = True
                break
        
        if not found:
            raise ValueError(f"Rewrite result for {result.user} and {result.system} not found.")

    # get rid of None values
    for attribute, attribute_results in organized_results.items():
        for user, rollouts in attribute_results.items():
            attribute_results[user] = [r for r in rollouts if r.plus_score is not None and r.minus_score is not None]

        
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "rewrite_half_results.json", "w", encoding="utf-8") as f:
            json_data = {k: {k2: [asdict(r) for r in v2] for k2, v2 in v.items()} for k, v in organized_results.items()}
            json.dump(json_data, f, indent=4)

        mean_results = {}
        for attribute, attribute_results in organized_results.items():
            plus_scores = []
            minus_scores = []
            for v in attribute_results.values():
                plus_scores.extend([r.plus_score for r in v])
                minus_scores.extend([r.minus_score for r in v])
            
            mean_results[attribute] = {
                "plus": np.mean(plus_scores).item() if len(plus_scores) > 0 else None,
                "minus": np.mean(minus_scores).item() if len(minus_scores) > 0 else None,
            }
        
        with open(save_dir / "rewrite_half_scores.json", "w", encoding="utf-8") as f:
            json.dump(mean_results, f, indent=4)

    return dict(organized_results)

# %%
if __name__ == "__main__":
    id_to_cluster = load_clusters("synthetic")

    ATTRIBUTES = [
        "Provide multiple different approaches to the same question.",
        "Use bold text in the response.",
        # "Use italic text in the response.",
        "Write a response that is longer than the baseline.",
        "Use emojis in the response.",
        "Use bulleted or numbered lists in the response.",
        "Start with an affirmative sentence that responds to the user's request.",
        "Adopt a friendly and engaging, natural tone.",
        # "End the response with a question to the user.",
        "Mention advanced mathematical jargon in the response.",
    ]

    # start_time = time.time()
    # asyncio.run(evaluate_baselines(
    #     user_prompts=id_to_cluster[0].prompts,
    #     policy_model=PolicyModel(model_name="meta-llama/llama-3.1-70b-instruct"),
    #     rater=RewardModel(reward_model_name="skywork-v2"),
    #     save_dir=Path(f"scrap/{timestamp()}-synthetic-0-70b"),
    #     n_rollouts=32,
    # ))
    # logger.info(f"Time taken: {time.time() - start_time} seconds")

    # start_time = time.time()
    # asyncio.run(evaluate_attributes_conditional(
    #     user_prompts=id_to_cluster[0].prompts,
    #     policy_model=PolicyModel(model_name="meta-llama/llama-3.1-70b-instruct"),
    #     rater=RewardModel(reward_model_name="skywork-v2"),
    #     attributes=ATTRIBUTES,
    #     save_dir=Path(f"scrap/{timestamp()}-synthetic-0-70b"),
    #     n_rollouts=32,
    # ))
    # logger.info(f"Time taken: {time.time() - start_time} seconds")

    # start_time = time.time()
    # organized_results = asyncio.run(evaluate_attributes_rewrite(
    #     user_prompts=id_to_cluster[0].prompts,
    #     attributes=ATTRIBUTES,
    #     policy_model=PolicyModel(model_name="meta-llama/llama-3.1-70b-instruct"),
    #     rewrite_model=RewriteModel(max_tokens=8192, reasoning="medium", max_par=512),
    #     rater=RewardModel(model_name="skywork-v2"),
    #     save_dir=Path(f"scrap/{timestamp()}-synthetic-0-70b"),
    #     n_rollouts=16,
    # ))
    # logger.info(f"Time taken: {time.time() - start_time} seconds")
# %%
