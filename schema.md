## Run Visualization Data Schema

This document defines the JSON schema and file layout expected by the visualization tools for runs, seeds, system prompts, attacks, and ratings. It is runner-agnostic and works for evolutionary (evo), best-of-N (bon_iter/one_turn), and PAIR flows.

### Directory layout

- Base directories recognized by the visualizer:
  - `data/evo`, `data/bon_iter`, `data/pair`, `data/one_turn`
- Each run is a directory under one of the above:
  - `data/<runner_dir>/<run_name>/`
- Each seed state is a subdirectory:
  - `data/<runner_dir>/<run_name>/seed_<seed_id>/`
- Files inside each `seed_<seed_id>` directory:
  - One JSON per system prompt: `<system_prompt_hash>.json`
  - Optional: `cluster_info.json` (cluster metadata)
  - Optional: `population_history.json` (population membership over steps)

System prompt files are keyed by a hash of the normalized system prompt (see `viz_utils.hash_system_prompt`).

## SystemPromptStats file schema

One file per system prompt: `data/<runner_dir>/<run_name>/seed_<seed_id>/<hash>.json`

```json
{
  "system_prompt": "string",
  "meta": {
    "step": 0,
    "operation": "string",
    "...": "..."  
  },
  "mean_score": 0.0,          // optional; mean adversarial score across attacks
  "stdev_score": 0.0,         // optional; std dev of adversarial scores across attacks
  "attacks": [Attack, ...]
}
```

- **Required**: `system_prompt`, `meta` (with at least `step`, `operation`), `attacks`.
- **Optional**: `mean_score`, `stdev_score`. If omitted, the visualizer or helpers will compute them from `attacks` using the default adversariality definition.

### Attack

An attack is uniquely identified by the pair (`system`, `user`). Responses and ratings live underneath it.

```json
{
  "system": "string",
  "user": "string",
  "responses": [RatedResponse, ...],
  "aux_info": {
    "adversarial_score": 0.0  // optional; can be omitted
  }
}
```

### RatedResponse

```json
{
  "assistant": "string",
  "ratings": [Rating, ...]
}
```

### Rating

```json
{
  "raw_score": 0.0,
  "rater": {
    "model_name": "string",
    "rating_function_type": "classifier"  // or "lm_judge", etc.
  },
  "aux_info": {
    "normalized_score": 0.0,   // REQUIRED for adversarial stats
    "reasoning": "...",       // optional
    "...": "..."
  }
}
```

Notes:
- The visualizer automatically derives the set of rater model names from `ratings[*].rater.model_name` present in `responses`.
- The adversarial score for an attack is computed when missing as:
  - Use the default rater pair: the first two ratings on the first response.
  - Compute mean of `aux_info.normalized_score` per rater across that attack’s responses.
  - Apply `adversariality(z_score_1, z_score_2)` from `state.py`.
  - If any required data is missing, the adversarial score is reported as N/A.

## What the visualizer shows

- Under Explore System Prompts (per seed):
  - Table of all system prompts with: `Step`, `Operation`, `System Prompt`, `Mean Adv Score`, `Std Adv Score`, `Num Attacks`, `Hash`.
  - `Mean Adv Score` and `Std Adv Score` are computed from the attacks when not provided in file.
- Clicking a system prompt shows its Attacks table:
  - One row per attack (`system`, `user`).
  - Columns include `User Prompt`, `Adversarial Score`, and for each rater model, `Mean Raw | <rater>`, `Mean Norm | <rater>`.
- Clicking an attack shows the RatedResponses table:
  - One row per response, columns include response preview, length, and for each rater: `Raw | <rater>`, `Norm | <rater>`.

## Helper functions (recommended interface)

Use these to ensure you consistently write valid JSON for any runner (`pair`, `one_turn`, `evo`, ...).

- `viz_utils.convert_attack_to_dict(attack)`
  - Accepts either a `state.Attack` dataclass instance or a dict.
  - Returns a normalized dict with fields: `system`, `user`, `responses[*].assistant`, `responses[*].ratings[*].raw_score`, `responses[*].ratings[*].rater.{model_name,rating_function_type}`, `responses[*].ratings[*].aux_info`.
  - If possible, computes and inserts `aux_info.adversarial_score` using the default rater pair.

- `viz_utils.save_system_prompt_stats(run_path, seed_id, system_prompt, attacks, mean_score=None, stdev_score=None, meta={})`
  - `run_path`: `Path("/workspace/rm-bias") / "data" / "<runner_dir>" / "<run_name>"`
  - `attacks`: list of `Attack` dataclasses or normalized dicts.
  - If `mean_score`/`stdev_score` are `None`, they are computed from `attacks` using adversarial scores (default rater pair logic).
  - Writes one file per system prompt under `seed_<seed_id>/<hash>.json`.

- `viz_utils.save_cluster_info(run_path, seed_id, summary, train_batch_size, sample_train_prompts)`
  - Optional; writes `cluster_info.json` under the seed directory.

- `viz_utils.save_population_state(run_path, seed_id, step, population_state)`
  - Optional; writes/updates `population_history.json`. The `population_state` should be a mapping `{system_prompt: generation}` for that step (prompts are hashed automatically).

### Minimal end-to-end example (Python)

```python
from pathlib import Path
from viz_utils import save_system_prompt_stats, convert_attack_to_dict

run_path = Path("/workspace/rm-bias") / "data" / "evo" / "my_run"
seed_id = 0
step = 3
operation = "mutate"
system_prompt = "You are a helpful assistant."

# Suppose you have new-style Attack objects (state.Attack) in `attacks`
# Or build dicts directly using the schema below
attacks_dicts = [convert_attack_to_dict(a) for a in attacks]

save_system_prompt_stats(
    run_path=run_path,
    seed_id=seed_id,
    system_prompt=system_prompt,
    attacks=attacks_dicts,
    meta={"step": step, "operation": operation},
)
```

### Minimal JSON example for a single system prompt

```json
{
  "system_prompt": "You are a helpful assistant.",
  "meta": {"step": 1, "operation": "init"},
  "attacks": [
    {
      "system": "You are a helpful assistant.",
      "user": "Explain gravity in one sentence.",
      "responses": [
        {
          "assistant": "Gravity is the force that attracts masses toward each other.",
          "ratings": [
            {
              "raw_score": 0.82,
              "rater": {"model_name": "skywork-v2", "rating_function_type": "classifier"},
              "aux_info": {"normalized_score": 0.65}
            },
            {
              "raw_score": 6.1,
              "rater": {"model_name": "openai/gpt-5-nano", "rating_function_type": "lm_judge"},
              "aux_info": {"normalized_score": 0.40, "reasoning": "Concise and accurate."}
            }
          ]
        }
      ],
      "aux_info": {"adversarial_score": -0.12}
    }
  ]
}
```

## Requirements and pitfalls

- Include `aux_info.normalized_score` in each `Rating` if you want adversarial stats to work. Without it, the visualizer will show `N/A`.
- If an attack lacks at least two raters on the first response, the default rater pair cannot be inferred and adversarial score will be `N/A`.
- Rater ordering matters for default pair detection. If you rely on auto-detection, keep rating order consistent across attacks for comparability.
- The visualizer supports mixed runners so long as the file schema is followed.


