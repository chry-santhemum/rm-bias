# ABOUTME: Script for testing cluster model quality in the planner.plan method.
# ABOUTME: Validates that attribute clustering produces semantically meaningful groups.
# %%
import json
import asyncio
import torch
from loguru import logger
from pathlib import Path
from textwrap import dedent

from sklearn.cluster import AgglomerativeClustering

from caller import AutoCaller
from utils import timestamp, parse_json_response

# Setup directories
run_name = timestamp()

# Logging
Path("logs/exp_test_clusters").mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(
    f"logs/exp_test_clusters/{run_name}.log",
    enqueue=True, level="INFO",
    retention="7 days"
)

from planner import ListPlanner
from runner import Runner
from api_models import GenerationModel, RewriteModel, RETRY_CONFIG
from reward_models import LocalRewardModel
from bias_evaluator import BiasEvaluator
from load_cluster import load_initial_seed_states

# %%
class TestRunner(Runner):
    @property
    def runner_type(self) -> str:
        return "exp_test_clusters"

    def train(self):
        pass

all_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

planner = ListPlanner(
    model_names=["openai/gpt-5.2"],
    max_tokens=15000,
    reasoning="high",
    n_new=8,
    n_pop=64,
    n_traj_in_context=16,
    n_per_user_prompt=1,
    reverse=True,
    max_num_train_prompts=32,
    max_par=128,
)

policy_model_names = [
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "google/gemma-2-9b-it",
    "microsoft/phi-3.5-mini-128k-instruct"
]

policy_model = GenerationModel(
    model_name=policy_model_names,
    max_par=512,
    max_tokens=1024,
    temperature=0.9,
    enable_cache=False,
)

student_model = LocalRewardModel(
    model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    devices=all_cuda_devices,
    batch_size_per_device=64,
)

bias_evaluator = BiasEvaluator(
    rewrite_model=RewriteModel(
        model_name="openai/gpt-5-mini",
        max_par=1024,
        max_tokens=4096,
        reasoning="low",
        enable_cache=False,
        force_caller="openrouter",
    ),
    reward_model=student_model,
    n_rewrite_workers=128,
)

initial_seed_states = load_initial_seed_states(
    ds_path=f"user_prompts/chatgpt",
    topic_ids=[0, 4, 8],
    val_split_size=16,
)


runner = TestRunner(
    seed_states=initial_seed_states,  # type: ignore
    policy_model=policy_model,
    bias_evaluator=bias_evaluator,
    teacher_model=student_model,
    n_baseline_rollouts=16,
    n_validate_rollouts=8,
    run_name=run_name,
)

# %%
import nest_asyncio
nest_asyncio.apply()

from state import Rollout, RewriteScore


async def create_plans():
    with open(f"data/exp_test_clusters/20251228-062157/train_baselines/rollouts.json", "r") as f:
        baseline_rollouts = json.load(f)
    
    runner.baselines = {}
    for user, rollouts in baseline_rollouts.items():
        runner.baselines[user] = [
            Rollout(
                response=rollout["response"],
                student_score=RewriteScore(
                    score=None,
                    raw_score=rollout["student_score"],
                    reasoning=None,
                    model_name=student_model.model_name,
                ),
                teacher_score=None,
                model=rollout.get("model", None),
            ) for rollout in rollouts
        ]

    await planner.plan(
        runner=runner,
        direction="plus",
        cluster_model=None,
    )


# %% ---------------- the clustering part -------------------
from cluster_models import CLUSTER_PROMPT


caller = AutoCaller(dotenv_path=".env", retry_config=RETRY_CONFIG, force_caller="openrouter")

async def cluster_attributes():
    prompts_to_send = []
    for seed_state in initial_seed_states:
        with open(f"data/exp_test_clusters/{run_name}/all_plans_chatgpt_{seed_state.index}.json", "r") as f:
            all_plans = json.load(f)

        prompts_to_send.append(CLUSTER_PROMPT.format(
            cluster_summary=seed_state.cluster.summary,
            attributes=json.dumps(all_plans, indent=4),
        ))

    responses = await caller.call(
        messages=prompts_to_send,
        model="openai/gpt-5.2",
        max_parallel=32,
        max_tokens=50000,
        reasoning="high",
        enable_cache=True,
    )
    return responses


async def main():
    await create_plans()

    for seed_state in runner.seed_states:
        all_plans = list(seed_state.history[-1].keys())
        with open(f"data/exp_test_clusters/{run_name}/all_plans_chatgpt_{seed_state.index}.json", "w") as f:
            json.dump(all_plans, f, indent=4)

    return await cluster_attributes()


cluster_responses = asyncio.run(main())

for seed_state, resp in zip(initial_seed_states, cluster_responses):
    if resp is None:
        continue
    cluster_results, reasoning = parse_json_response(resp)

    try:
        with open(f"data/exp_test_clusters/{run_name}/clusters_{seed_state.index}.json", "w") as f:
            json.dump(cluster_results, f, indent=4)
    except Exception as e:
        print(f"Error writing cluster results for seed {seed_state.index}: {e}")
    
    if reasoning is None:
        continue
    with open(f"data/exp_test_clusters/{run_name}/cluster_model_reasoning_{seed_state.index}.txt", "w") as f:
        f.write(reasoning)


# %%

# cluster_model = ClusterModel(embed_model_name="Qwen/Qwen3-Embedding-0.6B")

# # Embed all plans
# embs = cluster_model.embed(all_plans)

# # Agglomerative clustering with 64 target clusters
# n_clusters = min(64, len(all_plans))
# agg = AgglomerativeClustering(
#     n_clusters=n_clusters,
#     metric="cosine",
#     linkage="complete",
# )
# labels = agg.fit_predict(embs)

# # Find medoid (closest to cluster centroid) for each cluster
# representatives = []
# for cluster_idx in range(n_clusters):
#     member_indices = np.where(labels == cluster_idx)[0]
#     if len(member_indices) == 0:
#         continue

#     # Get embeddings for this cluster
#     cluster_embs = embs[member_indices]

#     # Compute centroid
#     centroid = cluster_embs.mean(axis=0)
#     centroid = centroid / np.linalg.norm(centroid)  # normalize

#     # Find member closest to centroid (medoid)
#     similarities = cluster_embs @ centroid
#     best_local_idx = int(np.argmax(similarities))
#     medoid_idx = member_indices[best_local_idx]

#     representatives.append({
#         "cluster_idx": cluster_idx,
#         "center_idx": int(medoid_idx),
#         "center_input": all_plans[medoid_idx],
#         "members": [all_plans[i] for i in member_indices],
#         "size": len(member_indices),
#     })

# # %%
# print(f"Agglomerative clustering: {len(representatives)} clusters from {len(all_plans)} plans")
# for rep in sorted(representatives, key=lambda x: -x["size"])[:50]:
#     print(f"\nCluster {rep['cluster_idx']} ({rep['size']} members):")
#     print(f"  Center: {rep['center_input']}")
#     if rep["size"] > 1:
#         print(f"  Other members:\n{"\n".join(
#             ["  - " + m for m in rep['members'][:10]]
#         )}")

# # %%
# # Iterative merge pass based on inter-cluster similarity of representatives
# import re

# def extract_keywords(text: str, min_len: int = 3) -> set[str]:
#     """Extract lowercase words as keywords for lexical overlap check."""
#     words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
#     return {w for w in words if len(w) >= min_len}

# def compute_keyword_overlap(kw1: set[str], kw2: set[str]) -> float:
#     """Jaccard similarity between keyword sets."""
#     if not kw1 or not kw2:
#         return 0.0
#     return len(kw1 & kw2) / len(kw1 | kw2)

# def merge_similar_clusters(
#     clusters: list[dict],
#     embs: np.ndarray,
#     merge_thresh: float = 0.90,
#     keyword_thresh: float = 0.2,  # minimum keyword overlap to allow merge
# ) -> list[dict]:
#     """
#     Iteratively merge clusters whose medoid embeddings have cosine similarity >= merge_thresh.
#     Optional lexical guardrail: only merge if keyword overlap >= keyword_thresh.
#     """
#     # Work with a copy
#     clusters = [dict(c) for c in clusters]

#     # Precompute keywords for each cluster's center
#     for c in clusters:
#         c["keywords"] = extract_keywords(c["center_input"])

#     merge_count = 0
#     while True:
#         # Get current medoid embeddings (normalized)
#         medoid_embs = np.array([embs[c["center_idx"]] for c in clusters])

#         # Compute pairwise cosine similarities
#         sim_matrix = medoid_embs @ medoid_embs.T

#         # Find best merge candidate (highest similarity above threshold)
#         best_sim = -1.0
#         best_pair = None

#         for i in range(len(clusters)):
#             for j in range(i + 1, len(clusters)):
#                 if sim_matrix[i, j] >= merge_thresh and sim_matrix[i, j] > best_sim:
#                     # Check lexical overlap guardrail
#                     kw_overlap = compute_keyword_overlap(
#                         clusters[i]["keywords"],
#                         clusters[j]["keywords"]
#                     )
#                     if kw_overlap >= keyword_thresh:
#                         best_sim = sim_matrix[i, j]
#                         best_pair = (i, j)

#         if best_pair is None:
#             break  # No more merges possible

#         i, j = best_pair
#         merge_count += 1

#         # Merge cluster j into cluster i
#         c_i, c_j = clusters[i], clusters[j]
#         print(f"Merge #{merge_count}: sim={best_sim:.3f}")
#         print(f"  Cluster {c_i['cluster_idx']} ({c_i['size']}): {c_i['center_input'][:60]}...")
#         print(f"  Cluster {c_j['cluster_idx']} ({c_j['size']}): {c_j['center_input'][:60]}...")

#         # Combine members
#         combined_members = c_i["members"] + c_j["members"]
#         combined_member_indices = [all_plans.index(m) for m in combined_members]

#         # Recompute medoid for merged cluster
#         combined_embs = embs[combined_member_indices]
#         centroid = combined_embs.mean(axis=0)
#         centroid = centroid / np.linalg.norm(centroid)
#         similarities = combined_embs @ centroid
#         best_local_idx = int(np.argmax(similarities))
#         new_medoid_idx = combined_member_indices[best_local_idx]

#         # Update cluster i with merged data
#         clusters[i] = {
#             "cluster_idx": c_i["cluster_idx"],
#             "center_idx": new_medoid_idx,
#             "center_input": all_plans[new_medoid_idx],
#             "members": combined_members,
#             "size": len(combined_members),
#             "keywords": extract_keywords(all_plans[new_medoid_idx]),
#         }

#         # Remove cluster j
#         clusters.pop(j)

#     print(f"\nMerge pass complete: {merge_count} merges, {len(clusters)} clusters remaining")
#     return clusters

# # Run the merge pass
# merged_clusters = merge_similar_clusters(
#     clusters=representatives,
#     embs=embs,
#     merge_thresh=0.90,
#     keyword_thresh=0.2,
# )

# print(f"\n{'='*60}")
# print(f"After merging: {len(merged_clusters)} clusters (was {len(representatives)})")
# print(f"{'='*60}")

# for rep in sorted(merged_clusters, key=lambda x: -x["size"])[:10]:
#     print(f"\nCluster {rep['cluster_idx']} ({rep['size']} members):")
#     print(f"  Center: {rep['center_input']}")
#     if rep["size"] > 1:
#         print(f"  Sample members:")
#         for m in rep["members"][:5]:
#             print(f"    - {m[:80]}...")

# # %%
