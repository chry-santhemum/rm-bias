# %% 

import json
import random
import textwrap

# %%

# Load rollouts
rollouts = json.load(open("data/evo/20251218-155435-list_reverse-handpick-plus/validate/seed_11_validate/rollouts.json", encoding="utf-8"))
# rollouts: dict[str, dict[str, list[dict[str, str | float]]]]
# bias -> user_prompt -> list of rollouts (rewrite data)
# rollout: {presence: bool, response: str, student_diff: float, student_score: float}

original_responses = json.load(open("data/evo/20251218-155435-list_reverse-handpick-plus/val_baselines/rollouts.json", encoding="utf-8"))
# user_prompt -> list[{response: str, student_score: float}]

# %%
bias = "Attribute the claim to a named official source, such as a specific spokesperson or government department."

# %%

N_EXAMPLES = 3
SEED = 0

rng = random.Random(SEED)

prompts_in_both = sorted(set(rollouts[bias]).intersection(original_responses))

sampled_prompts = rng.sample(prompts_in_both, k=N_EXAMPLES)

for i, user_prompt in enumerate(sampled_prompts, start=1):
    rewrite_rollouts = rollouts[bias][user_prompt]
    rewrite_rollouts_present = [r for r in rewrite_rollouts if r["presence"] is True]

    original_items = original_responses[user_prompt]

    original = rng.choice(original_items)
    rewrite = rng.choice(rewrite_rollouts_present)

    print("=" * 100)
    print(f"Example {i}/{N_EXAMPLES}")
    print("-" * 100)
    print("USER PROMPT:")
    print(textwrap.fill(user_prompt, width=100))
    print("-" * 100)
    print(f"ORIGINAL  student_score={original['student_score']:.4f}  student_diff={rewrite['student_diff']:.4f}")
    print(textwrap.fill(original["response"], width=100))
    print("-" * 100)
    print(f"REWRITE   student_score={rewrite['student_score']:.4f}")
    print(textwrap.fill(rewrite["response"], width=100))
    print()


# %% 

# Validating student scores

from reward_models import LocalRewardModel
from caller import ChatHistory


STUDENT_RM_ALIAS = "skywork-llama-8b"  # matches config.json for this run
STUDENT_RM_NAME_BY_ALIAS = {
    "skywork-qwen-0.6b": "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    "skywork-llama-8b": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "skywork-llama-8b-exp": "Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M",
}

rm = LocalRewardModel(
    model_name=STUDENT_RM_NAME_BY_ALIAS[STUDENT_RM_ALIAS],
    devices=["auto"],
    batch_size_per_device=8,
)

# Re-score the exact examples we printed above (1 original + 1 rewrite per prompt)
to_score: list[tuple[str, str, str, float]] = []
for i, user_prompt in enumerate(sampled_prompts, start=1):
    rewrite_rollouts = rollouts[bias][user_prompt]
    rewrite_rollouts_present = [r for r in rewrite_rollouts if r["presence"] is True]
    original_items = original_responses[user_prompt]

    original = rng.choice(original_items)
    rewrite = rng.choice(rewrite_rollouts_present)

    to_score.append((f"ex{i}/original", user_prompt, original["response"], float(original["student_score"])))
    to_score.append((f"ex{i}/rewrite", user_prompt, rewrite["response"], float(rewrite["student_score"])))

chats = [ChatHistory.from_user(user).add_assistant(resp) for _, user, resp, _ in to_score]
rescored = rm.rate_one_model(model_index=0, chat_histories=chats, use_tqdm=True)

deltas = []
print("=" * 100)
print("Student score validation (JSON vs recomputed with LocalRewardModel)")
for (tag, _user, _resp, json_score), rating in zip(to_score, rescored, strict=True):
    new_score = float(rating.score)
    delta = new_score - float(json_score)
    deltas.append(abs(delta))
    print(f"{tag:12s}  json={json_score: .6f}  recomputed={new_score: .6f}  delta={delta: .6f}")

print("-" * 100)
print(f"n={len(deltas)}  mean_abs_delta={sum(deltas)/len(deltas):.6f}  max_abs_delta={max(deltas):.6f}")

