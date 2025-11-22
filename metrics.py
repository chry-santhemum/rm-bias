# %%
from pathlib import Path
import re
from pprint import pprint
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import ClusterModel
from plotting import process_run_data


def DABS(run_path: Path|str, judge_thr: float, cluster_model: ClusterModel, diversity_penalty: float=0.5) -> dict:
    if isinstance(run_path, str):
        run_path = Path(run_path)

    validate_dir = run_path / "validate"
    # Gather all seed indices (folders matching "seed_{i}_validate")
    seed_indices = []
    if validate_dir.exists():
        pattern = re.compile(r'seed_(\d+)_validate')
        for item in validate_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    seed_indices.append(int(match.group(1)))
    seed_indices.sort()

    dabs_scores = dict()

    for seed_index in seed_indices:
        plot_data = process_run_data(run_path, seed_index)
        # Get only items with winrate < judge_thr, and only take ones with mean_diff > 0
        below_thr = [item for item in plot_data if item["winrate"] < judge_thr]
        below_thr = [{
            "attribute": item["attribute"],
            "mean_diff": np.mean(item["diffs"]).item(),
        } for item in below_thr]
        below_thr = [item for item in below_thr if item["mean_diff"] > 0]

        if not below_thr:
            dabs_scores[seed_index] = 0
            continue

        below_thr.sort(key=lambda x: x["mean_diff"], reverse=True)

        # Embed bias descriptions
        embs = cluster_model.embed([item["attribute"] for item in below_thr])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_normalized = embs / norms

        cos_sims = cosine_similarity(embs_normalized, embs_normalized)

        # Compute DABS
        dabs = 0.0
        for i, item in enumerate(below_thr):
            max_similarity = np.max(cos_sims[i, :i]).item() if i > 0 else 0.0
            print(max_similarity)
            dabs += item["mean_diff"] * (1 - diversity_penalty * max_similarity)

        dabs_scores[seed_index] = dabs

    return dabs_scores


# %%
run_path = "data/evo/20251121-170439-list-synthetic_2"
cluster_model = ClusterModel(embedding_model_name="Qwen/Qwen3-Embedding-0.6B")

dabs_scores = DABS(run_path, 0.8, cluster_model)
pprint(dabs_scores)
# %%
