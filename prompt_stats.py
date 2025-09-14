import patches

import hashlib
import asyncio
import json
import torch
import numpy as np
import logging
from tqdm.auto import tqdm
from pathlib import Path
from slist import Slist

from default_prompts import *
from llm_types import ChatHistory
from rater import RatingFunction, RewardModel, LLMJudge, PolicyModel
from client import OpenaiResponse, is_thinking_model, get_universal_caller, sample_from_model_parallel

logger = logging.getLogger(__name__)


def sample_responses(
    prompts: list[str],
    policy_name: str,
    N: int=16,
    policy_max_tokens: int = 1024,
    policy_max_par: int = 256,
    temperature: float = 0.8,
    target_dir: Path = Path("data/prompt_stats")
):
    """
    Sample responses and store the rollouts in target_dir.
    Each prompt is hashed and used as the filename.
    JSON format:
    {
        "prompt": str,
        "rollouts": [
            {
                "response": str,
                "{rater_model_name}": float
            },
            {
                ...
            }
        ],
        "summary_stats": {
            "{rater_model_name}": {
                "mean": float,
                "rewards_raw": list[float],
                "rewards_winsorized": list[float],
                "percentiles": ...
            },
        }
    }
    """

    target_dir.mkdir(parents=True, exist_ok=True)

    caller = get_universal_caller()
    messages = [ChatHistory().add_user(prompt) for prompt in prompts]

    policy_responses: Slist[OpenaiResponse] = asyncio.run(sample_from_model_parallel(
        prompts=[message for message in messages for _ in range(N)],
        caller=caller,
        max_par=policy_max_par,
        full_logging=False,
        desc="Sampling responses for per-prompt stats",
        temperature=temperature,
        model=policy_name,
        max_tokens=policy_max_tokens,
    ))

    for prompt_id, prompt in tqdm(enumerate(prompts), desc="Writing rollouts to disk"):
        # write a json file with the md5 hash of the prompt as file name
        # directly under target_dir

        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        file_path = target_dir / f"{prompt_hash}.json"
        json_data = {"prompt": prompt, "rollouts": []}

        for resp_id in range(N):
            prompt_responses = policy_responses[prompt_id * N + resp_id]
            json_data["rollouts"].append({
                "response": prompt_responses.first_response,
            })

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)



def prompt_stats(
    prompts: list[str],
    rater: RatingFunction,
    winsorize: float = 0.05,
    target_dir: Path = Path("data/prompt_stats"),
):

    N = None  # number of rollouts per prompt
    full_convos = []
    for prompt in prompts:
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        file_path = target_dir / f"{prompt_hash}.json"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                assert json_data["prompt"] == prompt
                if N is None:
                    N = len(json_data["rollouts"])
                else:
                    assert N == len(json_data["rollouts"])
                rollouts = json_data["rollouts"]
                full_convos.extend([
                    ChatHistory().add_user(prompt).add_assistant(rollout["response"])
                    for rollout in rollouts
                ])
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    if isinstance(rater, RewardModel):
        rewards = []
        for i in tqdm(range(0, len(full_convos), rater.batch_size), desc="Rating responses"):
            batch = full_convos[i : i + rater.batch_size]
            inputs = [input.to_openai_messages() for input in batch]
            input_ids = rater.tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                padding_side="right",
            ).to(rater.model.device)

            attn_mask = input_ids.ne(rater.tokenizer.pad_token_id)

            with torch.no_grad():
                scores = rater.model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits.squeeze(-1)

                rewards.extend(scores.tolist())

    elif isinstance(rater, LLMJudge):
        rater_prompts = Slist(full_convos).map(
            lambda convo: ChatHistory.from_system(
                ABSOLUTE_RANKING_PROMPT_SYSTEM
            ).add_user(
                ABSOLUTE_RANKING_PROMPT_USER.format(
                    message_history=convo.remove_system().to_openai_messages(),
                    thinking_instruction=RATER_THINKING_INSTRUCTION[
                        is_thinking_model(rater.model_name)
                    ],
                    rubric=HANDWRITTEN_RUBRIC,
                )
            )
        )
        rater_responses = asyncio.run(sample_from_model_parallel(
            prompts=rater_prompts,
            caller=rater.client,
            max_par=rater.max_par,
            full_logging=False,
            desc="Rating responses",
            model=rater.model_name,
            max_tokens=2048,
            reasoning={"max_tokens": 2000, "effort": "medium"},
        ))

        rewards = []
        for i, resp in enumerate(rater_responses):
            try:
                raw_text = resp.first_response
                try:
                    block = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
                except Exception:
                    block = raw_text
                parsed_resp = json.loads(block)
                rewards.append(parsed_resp["score"])
            except Exception as e:
                logger.error(
                    f"Failed to parse rater response: {resp.first_response}"
                )
                logger.error(f"Error: {e}")
                rewards.append(None)

    for prompt_idx, prompt in enumerate(prompts):
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        file_path = target_dir / f"{prompt_hash}.json"
        prompt_rewards_raw = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                assert json_data["prompt"] == prompt

                rollouts = json_data["rollouts"]
            for rollout_idx, rollout in enumerate(rollouts):
                prompt_reward_raw = rewards[prompt_idx * N + rollout_idx]  # type: ignore
                prompt_rewards_raw.append(prompt_reward_raw)
                if prompt_reward_raw is not None:
                    rollout[rater.model_name] = prompt_reward_raw
                else:
                    rollout[rater.model_name] = None

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load {file_path}: {e}")
            continue

        # Compute summary stats for each user prompt
        prompt_rewards_cleaned = list(filter(lambda x: x is not None, prompt_rewards_raw))
        if winsorize > 0:
            lower = np.percentile(prompt_rewards_cleaned, 100 * winsorize)
            upper = np.percentile(prompt_rewards_cleaned, 100 * (1 - winsorize))
            prompt_rewards_winsorized = np.clip(prompt_rewards_cleaned, lower, upper).tolist()
        else:
            prompt_rewards_winsorized = prompt_rewards_cleaned

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json_data["summary_stats"] = {
                    "mean": float(np.mean(prompt_rewards_winsorized)) if len(prompt_rewards_winsorized) > 0 else None,
                    "rewards_raw": prompt_rewards_raw,
                    "rewards_winsorized": prompt_rewards_winsorized,
                    "percentiles": {
                        f"{p}": float(np.percentile(prompt_rewards_winsorized, p)) if len(prompt_rewards_winsorized) > 0 else None for p in [0, 10, 25, 50, 75, 90, 100]
                    }
                }
                json.dump(json_data, f, indent=4)
        except (IOError, OSError) as e:
            logger.error(f"Failed to write {file_path}: {e}")
            continue



# %%
if __name__ == "__main__":
    import pandas as pd
    import hashlib
    import random
    import json
    from datasets import load_dataset
    from tqdm.auto import tqdm
    from pathlib import Path
    from standard_prompts import set_seed_all

    set_seed_all(10086)

    hf_instruction_test = load_dataset("HuggingFaceH4/instruction-dataset", split="test")
    prompts_to_sample = list(hf_instruction_test["prompt"])



    # cluster_df: pd.DataFrame = pd.read_csv("data/wildchat/cluster_50k.csv")
    # labels_df: pd.DataFrame = pd.read_csv("data/wildchat/labels_50k.csv")
    # # prompts_to_sample = []

    # for topic_id in tqdm(range(1, 30), desc="Processing topics"):
    #     topic = cluster_df.loc[cluster_df.index[topic_id+1], "Name"].split('_', maxsplit=1)[-1]  # description
    #     all_user_prompts = []

    #     with pd.read_csv("data/wildchat/labels_50k.csv", chunksize=10000) as reader:
    #         for chunk in reader:
    #             for index, row in chunk.iterrows():
    #                 if int(row["Topic"]) == topic_id:
    #                     all_user_prompts.append(row["Document"])

    #     topic_prompts = random.sample(all_user_prompts, min(100, len(all_user_prompts)))

    #     print("=" * 80)
    #     print(f"Topic {topic_id}: {topic} with {len(all_user_prompts)} user prompts")
    #     print("\nExample prompts:\n")
    #     for prompt in all_user_prompts[:10]:
    #         print("-" * 80)
    #         print(prompt)


# %%
    print(f"Sampling {len(prompts_to_sample)} prompts")

    sample_responses(
        prompts=prompts_to_sample,
        policy_name="meta-llama/llama-3.1-8b-instruct",
        N=16,
    )

    policy = PolicyModel(
        model_name="meta-llama/llama-3.1-8b-instruct",
        max_tokens=1024,
    )

    rater = RewardModel(
        reward_model_name="skywork-v2",
        policy_model=policy,
        batch_size=64,
    )

    prompt_stats(
        prompts=prompts_to_sample,
        rater=rater,
    )
# %%


# %%
    for prompt in tqdm(prompts_to_sample, desc="Processing prompts"):
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        file_path = Path("data/prompt_stats") / f"{prompt_hash}.json"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                assert json_data["prompt"] == prompt
                
                json_data["topic_label"] = 0
                json_data["topic_name"] = "All"
                json_data["dataset"] = "instruction-dataset"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4)
        else:
            print("Prompt not found: ", prompt)