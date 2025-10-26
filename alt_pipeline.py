
# %%
import textwrap
import asyncio
import nest_asyncio
from tqdm.auto import tqdm
from collections import defaultdict

from state import Rollout
from utils import ClusterModel
from load_cluster import load_initial_seed_states
from caller import Caller, ChatHistory
from reward_model import RewardModel
from models import PolicyModel, RewriteModel, JudgeModel
from one_turn import OneTurnPlanner
from runner import TestRunner

nest_asyncio.apply()

# %%
N_SAMPLES = 1024
BATCH_SIZE = 32

seed_states = load_initial_seed_states(
    ds_name="synthetic_2",
    topic_ids=[0],
)


user_prompts = [seed_states[0].cluster.train_prompts[0]]

policy = PolicyModel(temperature=0.9)
all_rollouts = asyncio.run(policy.sample(
    chat_histories=[
        ChatHistory.from_user(user_prompt) 
        for user_prompt in user_prompts 
        for _ in range(N_SAMPLES)
    ],
    desc="Sampling rollouts"
))

reward_model = RewardModel(model_name="skywork-v2", batch_size=32)
reward_scores = reward_model.rate(all_rollouts)

baselines = defaultdict(list)

for rollout, reward_score in zip(all_rollouts, reward_scores):
    if rollout is None or reward_score.score is None:
        continue
    baselines[rollout.get_first("user")].append(
        Rollout(response=rollout.get_first("assistant"), score=reward_score.score)  # type: ignore
    )


# %%

runner = TestRunner(
    seed_states=seed_states,
    policy_model=PolicyModel(temperature=0.9),
    rewrite_model=RewriteModel(model_name="openai/gpt-5-nano", max_par=512),
    reward_model=RewardModel(model_name="skywork-v2", batch_size=32),
    judge_model=JudgeModel(),
    run_name=None,
)
runner.baselines = dict(baselines)

planner = OneTurnPlanner(
    model_names=["anthropic/claude-opus-4.1"],
    alloy_type="round_robin",
    max_tokens=8192,
    reasoning=6000,
    temperature=1.0,
    max_par=128,
)

runner.load_contrast_pairs()

# %%
planner.plan(
    seed_states=seed_states,
    n_new=16,
    n_pop=256,
    cluster_model=ClusterModel(embedding_model_name="Qwen/Qwen3-Embedding-0.6B"),
)


attributes = list(seed_states[0].history[-1].keys())

for attribute in attributes:
    print("-" * 80)
    print(attribute)

# %%
import pickle

with open("data/scrap/alt_pipeline.pkl", "wb") as f:
    pickle.dump({
        "baselines": baselines,
        "attributes": attributes,
    }, f)

# %%

with open("data/scrap/alt_pipeline.pkl", "rb") as f:
    data = pickle.load(f)




