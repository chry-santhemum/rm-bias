# %%
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
import random
from abc import ABC, abstractmethod
from dataclasses import replace

from llm_types import ChatHistory
from state import Cluster, SystemPromptStats, Attack, Rating, Rater
from utils import load_model, REWARD_MODELS, get_to_pass_reasoning
from standard_prompts import make_prompt_mix
from default_prompts import *
from client import OpenaiResponse, is_thinking_model, get_universal_caller, sample_from_model_parallel

logger = logging.getLogger(__name__)


def prompt_to_hash_path(prompt: str, target_dir: Path = Path("data/prompt_stats")) -> Path:
    prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    return target_dir / f"{prompt_hash}.json"


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
    print(f"Sampling {len(prompts)} prompts...")

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

        file_path = prompt_to_hash_path(prompt, target_dir)

        # Skip if file already exists
        if file_path.exists():
            continue

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
    rater: "RatingFunction",
    winsorize: float = 0.05,
    target_dir: Path = Path("data/prompt_stats"),
):
    print(f"Rating {len(prompts)} prompts with {rater.model_name}...")
    N = None  # number of rollouts per prompt
    full_convos = []
    for prompt in prompts:
        file_path = prompt_to_hash_path(prompt, target_dir)
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
        file_path = prompt_to_hash_path(prompt, target_dir)
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
                if "summary_stats" not in json_data:
                    json_data["summary_stats"] = {}

                json_data["summary_stats"][rater.model_name] = {
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



class PolicyModel:
    def __init__(
        self,
        model_name: str,
        max_tokens: int=1024,
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
            desc="Sampling responses",
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
                logger.error(f"Executor remote parse error (answer): {e}")
                logger.error(f"API response: {resp}")
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
        system_prompt_stats: list[SystemPromptStats],
        n_samples: int,
        *args,
        **kwargs,
    ) -> list[SystemPromptStats]:
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
                logger.info(f"Loaded stats for {self.model_name} from {f.name}")
                assert (
                    loaded_stats["rater_name"] == self.model_name
                ), f"Cached stats for {loaded_stats['rater_name']} but model is {self.model_name}"
                assert (
                    loaded_stats["policy_name"] == self.policy_model.model_name
                ), f"Cached stats for {loaded_stats['policy_name']} but model is {self.policy_model.model_name}"

                if (loaded_stats["n_samples"] != n_samples or loaded_stats["n_prompts"] != n_prompts):
                    if not overwrite:
                        logger.warning("Using cached stats for different n_samples or n_prompts. Proceed with caution.")
                    else:
                        raise ValueError("Cached stats for different n_samples or n_prompts. Overwriting...")

                self.mean = loaded_stats["mean"]
                self.stdev = loaded_stats["stdev"]
                return
                
        except Exception:
            logger.warning("Computing mean and stdev from scratch...")

        prompts = make_prompt_mix(num_total=n_prompts)
        sample_responses(
            prompts=prompts,
            policy_name=self.policy_model.model_name,
            N=n_samples,
        )

        prompt_stats(
            prompts=prompts,
            rater=self,
        )

        # gather all stats
        all_scores = []
        for prompt in prompts:
            file_path = prompt_to_hash_path(prompt, Path("data/prompt_stats"))
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                assert json_data["prompt"] == prompt
                all_scores.extend(json_data["summary_stats"][self.model_name]["rewards_winsorized"])

        self.mean = float(np.mean(all_scores))
        self.stdev = float(np.std(all_scores, ddof=1))
        logger.info(f"Setting mean: {self.mean:.2f}, stdev: {self.stdev:.2f}")

        # save all percentiles
        percentiles = {
            f"{p}": float(np.percentile(all_scores, p)) for p in list(range(0, 101, 5))
        }

        logger.info(f"Score percentiles for {self.model_name}: {percentiles}")

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
        logger.info(f"Saved stats for {self.model_name} to {f.name}")



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
        system_prompt_stats: list[SystemPromptStats],
        n_samples: int=1,
        per_prompt_normalize: bool=True,  # whether to normalize per-prompt
    ) -> list[SystemPromptStats]:

        async def sample_attacks_one_prompt(sps: SystemPromptStats) -> list[Attack]:
            system_prompt = sps.system_prompt
            # If no rollouts have been generated, sample train prompts and rollouts
            if not sps.attacks:
                if cluster.train_batch_size == 0:
                    train_prompts = cluster.train_prompts
                else:
                    train_prompts = random.sample(cluster.train_prompts, cluster.train_batch_size)

                policy_inputs = [
                    ChatHistory.from_system(system_prompt).add_user(prompt) 
                    for prompt in train_prompts for _ in range(n_samples)
                ]
                policy_responses = await self.policy_model.sample_responses(policy_inputs)
                attacks = [Attack(chat_history=policy_responses[i], ratings=[], aux_info={}) for i in range(len(policy_responses))]
            else:
                attacks = sps.attacks
            return attacks

        gathered_attacks: Slist[list[Attack]] = await Slist(system_prompt_stats).par_map_async(sample_attacks_one_prompt)
        attacks = []
        attack_idx_to_sps_idx = {}
        for sps_idx, gathered_attack in enumerate(gathered_attacks):
            for attack in gathered_attack:
                attack_idx_to_sps_idx[len(attacks)] = sps_idx
                attacks.append(attack)

        # If required, compute the per-prompt means
        per_prompt_means = []
        if per_prompt_normalize:
            for attack in attacks:
                file_path = prompt_to_hash_path(attack.user, Path("data/prompt_stats"))
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    assert json_data["prompt"] == attack.user
                    per_prompt_means.append(json_data["summary_stats"][self.model_name]["mean"])

        # Pass to reward model in batches
        all_scores = []
        for i in tqdm(range(0, len(attacks), self.batch_size), desc="Reward model rating"):
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
            # logger.info(f"Input IDs first example: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

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
                    raw_score=all_scores[i],
                    rater=self.rater,
                    aux_info={"normalized_score": normalized_scores[i], "per_prompt_mean": per_prompt_means[i]}
                )
            ]) 
            for i, attack in enumerate(attacks) if attack.normalized_reward is None
        ]

        to_return = [SystemPromptStats(system_prompt=sps.system_prompt, attacks=[]) for sps in system_prompt_stats]
        for i in range(len(attacks)):
            to_return[attack_idx_to_sps_idx[i]].attacks.append(attacks[i])

        return to_return



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
        system_prompt_stats: list[SystemPromptStats],
        n_samples: int=1,
        max_tokens: int=4096,
        reasoning: int | str | None = "medium",
    ) -> list[SystemPromptStats]:
        
        return await Slist(system_prompt_stats).par_map_async(
            func=lambda sps: self.rate_one_system_prompt(
                cluster=cluster,
                system_prompt_stats=sps,
                n_samples=n_samples,
                max_tokens=max_tokens,
                reasoning=reasoning,
            ),
            max_par = 4,
        )

    async def rate_one_system_prompt(
        self,
        cluster: Cluster,
        system_prompt_stats: SystemPromptStats,
        n_samples: int=1,
        max_tokens: int=4096,
        reasoning: int | str | None = "medium",
    ) -> SystemPromptStats:

        system_prompt = system_prompt_stats.system_prompt
        # If no rollouts have been generated, sample train prompts and rollouts
        if not system_prompt_stats.attacks:
            if cluster.train_batch_size == 0:
                train_prompts = cluster.train_prompts
            else:
                train_prompts = random.sample(cluster.train_prompts, cluster.train_batch_size)

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
            desc="Absolute rating",
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
                
                if attacks[i].normalized_lm_judge is None:
                    new_attack = replace(attacks[i], 
                        ratings=attacks[i].ratings + [Rating(
                            raw_score=score,
                            rater=self.rater,
                            aux_info={"reasoning_content": reasoning_content, "normalized_score": normalized_score}
                        )]
                    )
                    attacks[i] = new_attack

            except Exception as e:
                logger.error(f"Absolute rating parse error: {e}")
                logger.error(f"Completion: {rater_resp.first_response}")
                continue

        return SystemPromptStats(system_prompt=system_prompt, attacks=attacks)



# %%
if __name__ == "__main__":
    import hashlib
    import random
    import json
    from datasets import load_dataset
    from tqdm.auto import tqdm
    from pathlib import Path
    from standard_prompts import set_seed_all

    set_seed_all(10086)

    agent_harm = load_dataset("ai-safety-institute/AgentHarm", name="chat", split="test_public")
    prompts = list(agent_harm["prompt"])

    policy = PolicyModel(
        model_name="meta-llama/llama-3.1-8b-instruct",
        max_tokens=1024,
    )
    sample_responses(
        prompts=prompts,
        policy_name=policy.model_name,
        N=16,
    )

    rater_1 = RewardModel(
        reward_model_name="skywork-v2",
        policy_model=policy,
        batch_size=32,
    )

    rater_2 = LLMJudge(
        judge_model_name="openai/gpt-5-nano",
        policy_model=policy,
        rubric=HANDWRITTEN_RUBRIC,
        max_par=256,
        # full_logging=True,
    )

    prompt_stats(
        prompts=prompts,
        rater=rater_1,
    )
    prompt_stats(
        prompts=prompts,
        rater=rater_2,
    )

    for prompt in tqdm(prompts, desc="Post-processing prompts"):
        file_path = prompt_to_hash_path(prompt, Path("data/prompt_stats"))
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

                json_data["topic_label"] = 0
                json_data["topic_name"] = "All"
                json_data["dataset"] = "agent-harm"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4)
        else:
            print("Prompt not found: ", prompt)


    # cluster_df: pd.DataFrame = pd.read_csv("data/wildchat/cluster_50k.csv")
    # labels_df: pd.DataFrame = pd.read_csv("data/wildchat/labels_50k.csv")
    # prompts_to_sample = []

    # for topic_id in tqdm(range(1, 30), desc="Processing topics"):
    #     topic = cluster_df.loc[cluster_df.index[topic_id+1], "Name"].split('_', maxsplit=1)[-1]  # description
    #     all_user_prompts = []

    #     with pd.read_csv("data/wildchat/labels_50k.csv", chunksize=10000) as reader:
    #         for chunk in reader:
    #             for index, row in chunk.iterrows():
    #                 if int(row["Topic"]) == topic_id:
    #                     all_user_prompts.append(row["Document"])

    #     topic_prompts = random.sample(all_user_prompts, min(100, len(all_user_prompts)))

    #     for prompt in tqdm(topic_prompts, desc="Processing prompts"):
    #         prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    #         file_path = Path("data/prompt_stats") / f"{prompt_hash}.json"
    #         if file_path.exists():
    #             with open(file_path, 'r', encoding='utf-8') as f:
    #                 json_data = json.load(f)
    #                 assert json_data["prompt"] == prompt
                    
    #                 if "skywork-v2" not in json_data["summary_stats"]:
    #                     new_dict = {}
    #                     key_names = list(json_data["summary_stats"].keys())

    #                     for key in key_names:
    #                         val = json_data["summary_stats"][key]
    #                         new_dict[key] = val
    #                         del json_data["summary_stats"][key]
                        
    #                     json_data["summary_stats"]["skywork-v2"] = new_dict

    #                 else:
    #                     print("Already in correct format")
    #                 with open(file_path, 'w', encoding='utf-8') as f:
    #                     json.dump(json_data, f, indent=4)
    #         else:
    #             print("Prompt not found: ", prompt)
        
    #     print("=" * 80)
    #     print(f"Topic {topic_id}: {topic} with {len(all_user_prompts)} user prompts")
    #     print("\nExample prompts:\n")
    #     for prompt in all_user_prompts[:10]:
    #         print("-" * 80)
    #         print(prompt)


# %%
