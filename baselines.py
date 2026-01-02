

import time
import json
import asyncio
from tqdm.auto import tqdm
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from loguru import logger

from caller import ChatHistory
from state import Cluster
from api_models import GenerationModel
from reward_models import LocalRewardModel


@dataclass(kw_only=True, slots=True)
class BaselineRollout:
    policy_model: str
    response: str
    scores: dict[str, float]

    
async def evaluate_baselines(
    clusters: list[Cluster],
    policy_model: GenerationModel,
    reward_model: LocalRewardModel,
    n_rollouts: int,
    save_dir: Path | None = None,
):
    start_time = time.time()

    # evaluate sequentially for each cluster
    for cl in tqdm(clusters, desc="Evaluating baselines"):
        if save_dir is not None:
            ds_name = Path(cl.data_path).parent.name
            path = Path("data/baselines") / ds_name / f"cluster_{cl.index}.json"
        else:
            path = save_dir / f"cluster_{cl.index}.json"
        
        if path.exists():
            with open(path, "r") as f:
                existing_results = json.load(f)
            
            # get rewards
            chats_with_responses = []
            indices = []
            for user, rollouts in existing_results.items():
                for i, r in enumerate(rollouts):
                    chats_with_responses.append(ChatHistory.from_user(user).add_assistant(r["response"]))
                    indices.append(i)
            scores = await reward_model.async_rate(chats_with_responses, use_tqdm=True)

            for chat, r_idx, score in zip(chats_with_responses, indices, scores):
                result_dict = existing_results[chat.get_first("user")][r_idx]
                result_dict["scores"][reward_model.model_name] = score.score

            with open(path, "w") as f:
                json.dump(existing_results, f, indent=4, sort_keys=True)
        
        else:
            # sample
            chat_histories = []
            for user_prompt in cl.train_prompts:
                chat_histories.extend([ChatHistory.from_user(user_prompt) for _ in range(n_rollouts)])
            responses = await policy_model.sample(chat_histories)
            
            # get rewards
            chats_with_responses = []
            policy_model_names = []
            for chat, resp in zip(chat_histories, responses):
                if (
                    resp is None 
                    or resp.first_response is None
                    or resp.finish_reason != "stop"
                ):
                    logger.warning(f"Policy model failed to sample. User prompt:\n{chat.get_first('user')}\nResponse:\n{resp}")
                    continue

                chats_with_responses.append(chat.add_assistant(resp.first_response.strip()))
                policy_model_names.append(resp.model)

            scores = await reward_model.async_rate(chats_with_responses, use_tqdm=True)

            # save results
            to_save = defaultdict(list)
            for chat, score, name in zip(chats_with_responses, scores, policy_model_names):
                result = {
                    "policy_model": name,
                    "response": chat.get_first("assistant"),
                    "scores": {reward_model.model_name: score.score},
                }
                to_save[chat.get_first("user")].append(result)

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(dict(to_save), f, indent=4, sort_keys=True)

    logger.info(f"Evaluated baselines in {(time.time() - start_time):.2f} seconds")
