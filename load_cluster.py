
import json
import random
from pathlib import Path

from state import SeedState, Cluster


def load_initial_seed_states(
    ds_path: str | Path,
    topic_ids: list[int],
    val_split_size: int,
    seed: int = 10086,
) -> list[SeedState]:
    if isinstance(ds_path, str):
        ds_path = Path(ds_path)

    id_to_cluster: dict[int, Cluster] = dict()

    for json_file in ds_path.glob("cluster_*.json"):
        try:
            cluster_id = int(json_file.name.split("_")[1].split(".")[0])
        except ValueError:
            print(f"Error: {json_file.name} is not a valid cluster id")
            continue

        if cluster_id in topic_ids:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            if len(data["prompts"]) < 2 * val_split_size:
                raise ValueError(f"Not enough prompts for cluster {cluster_id}")

            random.seed(seed)
            random.shuffle(data["prompts"])
            train_prompts = data["prompts"][:-val_split_size] if val_split_size > 0 else data["prompts"]
            val_prompts = data["prompts"][-val_split_size:] if val_split_size > 0 else []

            id_to_cluster[cluster_id] = Cluster(
                summary=data["summary"],
                train_prompts=train_prompts,
                val_prompts=val_prompts,
            )


    initial_seed_states = [SeedState(
        index=id,
        dataset=str(ds_path),
        cluster=cluster,
        state={},
        history=[],
    ) for id, cluster in id_to_cluster.items()]
    
    initial_seed_states.sort(key=lambda x: x.index)

    print(f"\n\nLoaded {len(initial_seed_states)} seed states")
    for state in initial_seed_states:
        print(
            f"  - Seed {state.index}, {len(state.cluster.train_prompts)} train prompts:\n"
            f"    {state.cluster.summary}"
        )
    print("\n\n")

    return initial_seed_states
