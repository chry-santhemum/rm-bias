import json
import logging
import asyncio
from pathlib import Path
from slist import Slist
from abc import ABC, abstractmethod


import numpy as np
import torch
from torch import Tensor

from llm_types import ChatHistory
from state import Cluster, SystemPromptStats
from utils import load_model, REWARD_MODELS
from standard_prompts import make_prompt_mix

logging.getLogger(__name__)


class RatingFunction(ABC):
    def __init__(self):
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

    @abstractmethod
    async def __call__(
        self,
        cluster: Cluster,
        system_prompt: str,
        normalized: bool,
        *args,
        **kwargs,
    ) -> SystemPromptStats:
        pass

    @abstractmethod
    def normalize(self, policy_name: str, *args, **kwargs):
        """
        Sample responses from the policy model for a standard dataset of user prompts.
        Then rate them and compute the mean and standard deviation.
        """
        pass



class RewardModel(RatingFunction):
    """
    Wrapper around reward models; __init__ kwargs (e.g. device) are passed to load_model
    """

    def __init__(
        self,
        model_name: str,
        **kwargs,
    ):
        assert model_name in REWARD_MODELS, f"Model {model_name} not local!!!"
        self._model_name = model_name
        self.model, self.tokenizer = load_model(model_name, **kwargs)
        self.device = self.model.device
        super().__init__()

    @property
    def rating_function_type(self) -> str:
        return "classifier"

    @property
    def model_name(self) -> str:
        return self._model_name

    def __call__(
        self,
        inputs: list[list[dict]] | list[ChatHistory] | Tensor,
        normalized: bool = True,
    ) -> Tensor:
        """
        Call the reward model and tokenize if needed.
        """
        if isinstance(inputs, list):
            # inputs are messages
            if isinstance(inputs[0], ChatHistory):
                inputs = [input.to_openai_messages() for input in inputs]
            input_ids = self.tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                padding_side="right",
            ).to(self.device)
        elif isinstance(inputs, Tensor):
            input_ids = inputs.to(self.device)

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # logging.info(f"Input IDs first example: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

        with torch.no_grad():
            scores = self.model(
                input_ids=input_ids, attention_mask=attn_mask
            ).logits.squeeze(-1)

        if normalized:
            scores = (scores - self.mean) / self.stdev

        return scores

    def normalize(
        self,
        policy_model_name: str,
        n_prompts: int = 128,
        n_samples: int = 8,
        n_clients: int = 16,
        overwrite: bool = False,
        cache_path: str | None = None,
    ):
        """
        Loads pre-computed stats from the cache json file if it exists.
        Otherwise, compute rewards from standard_prompts, then set self.mean and self.stdev
        Number of concurrent calls is n_clients * min(n_prompts, rater_max_par=32)
        and saves the stats to the cache json file.
        """
        cache_path = cache_path or f".rwdcache/per_model/{self.model_name}.json"
        try:
            # load pre-computed stats from the cache json file
            with open(cache_path, "r") as f:
                loaded_stats = json.load(f)
                logging.info(f"Loaded stats for {self.model_name} from {f.name}")
                assert (
                    loaded_stats["reward_model_name"] == self.model_name
                ), f"Cached stats for {loaded_stats['reward_model_name']} but model is {self.model_name}"
                assert (
                    loaded_stats["policy_model_name"] == policy_model_name
                ), f"Cached stats for {loaded_stats['policy_model_name']} but model is {policy_model_name}"

                if (
                    loaded_stats["n_samples"] != n_samples
                    or loaded_stats["n_prompts"] != n_prompts
                ):
                    if not overwrite:
                        logging.warning(
                            "Using cached stats for different n_samples or n_prompts. Proceed with caution. Use overwrite=True to overwrite."
                        )
                    else:
                        raise ValueError(
                            f"Cached stats for different n_samples or n_prompts. Overwriting..."
                        )

                self.mean = loaded_stats["mean"]
                self.stdev = loaded_stats["stdev"]
                return
        except Exception as e:
            logging.error(f"Computing from scratch.")
            logging.error(f"Error: {e}")

        prompts = make_prompt_mix(num_total=n_prompts)
        stats = asyncio.run(
            Slist(prompts["prompt"]).par_map_async(
                func=lambda prompt: per_prompt_reward_stats(
                    prompt=prompt,
                    raters=(self,),  # Pass the RewardModel instance
                    policy_model_name=policy_model_name,
                    N=n_samples,
                ),
                max_par=n_clients,
                tqdm=True,
            )
        )

        # gather all stats
        all_rewards_raw = []
        for stat in stats:
            all_rewards_raw.extend(stat[self.model_name]["rewards_raw"])
        all_rewards = list(filter(lambda x: x is not None, all_rewards_raw))

        self.mean = float(np.mean(all_rewards))
        self.stdev = float(np.std(all_rewards, ddof=1))
        logging.info(f"Setting mean: {self.mean:.2f}, stdev: {self.stdev:.2f}")

        # save all percentiles
        percentiles = {
            f"{p}": float(np.percentile(all_rewards, p)) for p in list(range(0, 101, 5))
        }

        logging.info(f"Reward percentiles for {self.model_name}: {percentiles}")

        # Ensure directory exists
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "n_samples": n_samples,
                    "n_prompts": n_prompts,
                    "reward_model_name": self.model_name,
                    "policy_model_name": policy_model_name,
                    "mean": self.mean,
                    "stdev": self.stdev,
                    "percentiles": percentiles,
                },
                f,
                indent=4,
            )

        logging.info(f"Saved stats for {self.model_name} to {f.name}")


class LLMJudge:

    