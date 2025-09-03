import json
import random
import logging
from pathlib import Path
from slist import Slist 
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from dataclasses import replace

import numpy as np
import torch

from llm_types import ChatHistory
from state import Cluster, SystemPromptStats, Attack, Rating, Rater
from utils import load_model, REWARD_MODELS, get_to_pass_reasoning, per_prompt_stats
from standard_prompts import make_prompt_mix
from default_prompts import *
from client import is_thinking_model, get_universal_caller, sample_from_model_parallel

logging.getLogger(__name__)


class PolicyModel:
    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        temperature: float = 0.8,
        max_par: int = 512,  # max parallel calls to client
        full_logging: bool = False,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_par = max_par
        self.caller = get_universal_caller()
        self.full_logging = full_logging


    async def sample_responses(
        self,
        chat_histories: list[ChatHistory],
    ) -> list[ChatHistory]:
        executor_responses = await sample_from_model_parallel(
            caller=self.caller,
            prompts=chat_histories,
            max_par=self.max_par,
            full_logging=self.full_logging,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # parse responses
        completed_chat_histories = []
        for i, resp in enumerate(executor_responses):
            try:
                assistant_response = resp.first_response
            except Exception as e:
                logging.error(f"Executor remote parse error (answer): {e}")
                logging.error(f"API response: {resp}")
                assistant_response = "N/A"
            
            completed_chat_histories.append(
                chat_histories[i].add_assistant(assistant_response)
            )
        return completed_chat_histories


class RatingFunction(ABC):
    def __init__(self, policy_model: PolicyModel):
        self.policy_model = policy_model
        self.mean = 0.0
        self.stdev = 1.0

    @property
    @abstractmethod
    def rating_function_type(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    def rater(self) -> Rater:
        return Rater(model_name=self.model_name, rating_function_type=self.rating_function_type)

    @abstractmethod
    async def __call__(
        self,
        cluster: Cluster,
        system_prompt_stats: SystemPromptStats,
        n_samples: int,
        *args,
        **kwargs,
    ) -> SystemPromptStats:
        """
        1. Sample train_batch_size user prompts from train_prompts
        2. For each, sample n_samples assistant responses from the policy model
        3. Rate each attack with the rating function
        """
        pass

    async def normalize(
        self,
        n_prompts: int = 128,
        n_samples: int = 8,
        n_clients: int = 16,
        overwrite: bool = False,
        cache_path: str | None = None,
    ):
        """
        Loads pre-computed stats from the cache json file if it exists.
        Otherwise, compute rewards from standard_prompts, and saves the stats to the cache json file.
        Number of concurrent calls is n_clients * min(n_prompts, rater_max_par=32).
        """
        cache_path = cache_path or f".cache/normalize/{self.model_name}.json"
        try:
            # load pre-computed stats from the cache json file
            with open(cache_path, "r") as f:
                loaded_stats = json.load(f)
                logging.info(f"Loaded stats for {self.model_name} from {f.name}")
                assert (
                    loaded_stats["rater_name"] == self.model_name
                ), f"Cached stats for {loaded_stats['rater_name']} but model is {self.model_name}"
                assert (
                    loaded_stats["policy_name"] == self.policy_model.model_name
                ), f"Cached stats for {loaded_stats['policy_name']} but model is {self.policy_model.model_name}"

                if (loaded_stats["n_samples"] != n_samples or loaded_stats["n_prompts"] != n_prompts):
                    if not overwrite:
                        logging.warning("Using cached stats for different n_samples or n_prompts. Proceed with caution.")
                    else:
                        raise ValueError("Cached stats for different n_samples or n_prompts. Overwriting...")

                self.mean = loaded_stats["mean"]
                self.stdev = loaded_stats["stdev"]
                return
                
        except Exception:
            logging.warning("Computing mean and stdev from scratch...")

        prompts = make_prompt_mix(num_total=n_prompts)
        stats = await Slist(prompts["prompt"]).par_map_async(
            func=lambda prompt: per_prompt_stats(
                prompt=prompt,
                rater_name=self.model_name,
                policy_name=self.policy_model.model_name,
                N=n_samples,
            ),
            max_par=n_clients,
            tqdm=True,
        )

        # gather all stats
        all_scores_raw = []
        for stat in stats:
            all_scores_raw.extend(stat[self.model_name]["rewards_raw"])
        all_scores = list(filter(lambda x: x is not None, all_scores_raw))

        self.mean = float(np.mean(all_scores))
        self.stdev = float(np.std(all_scores, ddof=1))
        logging.info(f"Setting mean: {self.mean:.2f}, stdev: {self.stdev:.2f}")

        # save all percentiles
        percentiles = {
            f"{p}": float(np.percentile(all_scores, p)) for p in list(range(0, 101, 5))
        }

        logging.info(f"Score percentiles for {self.model_name}: {percentiles}")

        # Ensure directory exists
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "n_samples": n_samples,
                    "n_prompts": n_prompts,
                    "rater_name": self.model_name,
                    "policy_name": self.policy_model.model_name,
                    "mean": self.mean,
                    "stdev": self.stdev,
                    "percentiles": percentiles,
                },
                f,
                indent=4,
            )
        logging.info(f"Saved stats for {self.model_name} to {f.name}")



class RewardModel(RatingFunction):
    """
    Wrapper around reward models; __init__ kwargs (e.g. device) are passed to load_model
    """

    def __init__(self, reward_model_name: str, policy_model: PolicyModel, batch_size: int=32, **kwargs):
        assert reward_model_name in REWARD_MODELS, f"Model {reward_model_name} not local!!!"
        self._model_name = reward_model_name
        self.batch_size = batch_size
        self.model, self.tokenizer = load_model(reward_model_name, **kwargs)
        self.device = self.model.device
        super().__init__(policy_model)

    @property
    def rating_function_type(self) -> str:
        return "classifier"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def __call__(
        self,
        cluster: Cluster,
        system_prompt_stats: SystemPromptStats,
        n_samples: int=1,
        per_prompt_normalize: bool=True,  # whether to normalize per-prompt
    ) -> SystemPromptStats:

        # If no rollouts have been generated, sample train prompts and rollouts
        if not system_prompt_stats.attacks:
            if cluster.train_batch_size == 0:
                train_prompts = cluster.train_prompts
            else:
                train_prompts = random.sample(cluster.train_prompts, cluster.train_batch_size)

            system_prompt = system_prompt_stats.system_prompt
            policy_inputs = [
                ChatHistory.from_system(system_prompt).add_user(prompt) 
                for prompt in train_prompts for _ in range(n_samples)
            ]
            policy_responses = await self.policy_model.sample_responses(policy_inputs)
            attacks = [Attack(chat_history=policy_responses[i], ratings=[], aux_info={}) for i in range(len(policy_responses))]
        
        else:
            attacks = system_prompt_stats.attacks

        # If required, compute the per-prompt means
        per_prompt_means = []
        if per_prompt_normalize:
            for attack in attacks:
                stats = await per_prompt_stats(
                    prompt = attack.user,
                    rater_name = self.model_name,
                    policy_name = self.policy_model.model_name,
                )
                per_prompt_means.append(stats["mean"])

        # Pass to reward model in batches
        all_scores = []
        for i in tqdm(range(0, len(policy_responses), self.batch_size), desc="Reward model rating:"):
            inputs = attacks[i : i + self.batch_size]
            inputs = [input.chat_history.remove_system().to_openai_messages() for input in inputs]
            input_ids = self.tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                padding_side="right",
            ).to(self.device)

            attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
            # logging.info(f"Input IDs first example: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

            with torch.no_grad():
                scores = self.model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits.squeeze(-1)

            all_scores.extend(scores.tolist())
    
        # Normalize scores
        if per_prompt_normalize:
            normalized_scores = [(all_scores[i] - per_prompt_means[i]) / (self.stdev + 1e-6) for i in range(len(all_scores))]
        else:
            normalized_scores = [(all_scores[i] - self.mean) / (self.stdev + 1e-6) for i in range(len(all_scores))]
            
        attacks = [
            replace(attack, ratings=attack.ratings + [
                Rating(
                    raw_score=normalized_scores[i],
                    rater=self.rater,
                    aux_info={"normalized_score": normalized_scores[i]}
                )
            ]) for i, attack in enumerate(attacks)
        ]

        return SystemPromptStats(system_prompt=system_prompt, attacks=attacks)



class LLMJudge(RatingFunction):
    def __init__(self, judge_model_name: str, policy_model: PolicyModel, rubric: str, max_par: int = 256, full_logging: bool = False):
        self._model_name = judge_model_name
        self.rubric = rubric
        self.max_par = max_par
        self.full_logging = full_logging
        self.client = get_universal_caller()
        super().__init__(policy_model)

    @property
    def rating_function_type(self) -> str:
        return "lm_judge"

    @property
    def model_name(self) -> str:
        return self._model_name


    async def __call__(
        self,
        cluster: Cluster,
        system_prompt_stats: SystemPromptStats,
        n_samples: int=1,
        max_tokens: int=2048,
        reasoning: int | str | None = None,
    ) -> SystemPromptStats:

        # If no rollouts have been generated, sample train prompts and rollouts
        if not system_prompt_stats.attacks:
            if cluster.train_batch_size == 0:
                train_prompts = cluster.train_prompts
            else:
                train_prompts = random.sample(cluster.train_prompts, cluster.train_batch_size)

            system_prompt = system_prompt_stats.system_prompt
            policy_inputs = [
                ChatHistory.from_system(system_prompt).add_user(prompt) 
                for prompt in train_prompts for _ in range(n_samples)
            ]
            policy_responses = await self.policy_model.sample_responses(policy_inputs)
            attacks = [Attack(chat_history=policy_responses[i], ratings=[], aux_info={}) for i in range(len(policy_responses))]
        
        else:
            attacks = system_prompt_stats.attacks

        # Rate each attack with the LLM judge
        rater_inputs = [ChatHistory.from_system(
            ABSOLUTE_RANKING_PROMPT_SYSTEM
        ).add_user(
            ABSOLUTE_RANKING_PROMPT_USER.format(
                message_history=attack.chat_history.remove_system().to_openai_messages(),  # remove system prompt
                rubric=self.rubric,
                thinking_instruction=RATER_THINKING_INSTRUCTION[
                    is_thinking_model(self.model_name)
                ],
            )
        ) for attack in attacks]

        rater_responses = await sample_from_model_parallel(
            caller=self.client,
            prompts=rater_inputs,
            max_par=self.max_par,
            full_logging=self.full_logging,
            model=self.model_name,
            temperature=0.7,
            max_tokens=max_tokens,
            reasoning=get_to_pass_reasoning(reasoning, max_tokens),
        )

        for i in range(len(rater_responses)):
            # Modify each attack as we go
            rater_resp = rater_responses[i]
            try:
                raw_text = rater_resp.first_response
                try:
                    block = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
                except Exception:
                    block = raw_text

                try:
                    score = float(json.loads(block)["score"])
                except Exception:
                    score = float(block.split("{score: ", 1)[1].split("}", 1)[0].strip()[0])

                normalized_score = (score - 5.0) / (self.stdev + 1e-6)
                try:
                    if is_thinking_model(self.model_name):
                        reasoning_content = rater_resp.reasoning_content
                    else:
                        reasoning_content = raw_text.split("```json", 1)[0].strip()
                except Exception:
                    reasoning_content = "N/A"
                
                new_attack = replace(attacks[i], 
                    ratings=attacks[i].ratings + [
                        Rating(
                            raw_score=score,
                            rater=self.rater,
                            aux_info={"reasoning_content": reasoning_content, "normalized_score": normalized_score}
                        )
                    ],
                )
                attacks[i] = new_attack

            except Exception as e:
                logging.error(f"Absolute rating parse error: {e}")
                logging.error(f"Completion: {rater_resp.first_response}")
                continue

        return SystemPromptStats(system_prompt=system_prompt, attacks=attacks)
