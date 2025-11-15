import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from state import adversariality


def normalize_system_prompt(prompt: str) -> str:
    """Normalize system prompt for consistent hashing."""
    return prompt.strip()


def hash_system_prompt(prompt: str) -> str:
    """Generate MD5 hash for system prompt filename."""
    normalized = normalize_system_prompt(prompt)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def atomic_write_json(file_path: Path, data: Dict[str, Any]) -> None:
    """Write JSON file atomically to prevent partial reads."""
    temp_path = file_path.with_suffix(".tmp")

    # Ensure directory exists
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Atomic rename
    os.rename(temp_path, file_path)


def save_system_prompt_stats(
    run_path: Path,
    seed_id: int,
    system_prompt: str,
    attacks: list = [],
    mean_score: float | None = None,
    stdev_score: float | None = None,
    meta: Dict[str, Any] = {},
) -> str:
    """Save SystemPromptStats to JSON file, returns hash.

    Attacks can be new-style dicts or dataclass objects; they will be normalized to dicts.
    If mean_score/stdev_score are omitted, they are computed as the mean/std of adversarial scores
    across attacks using the default rater pair implied by the first two ratings of the first response.
    """

    prompt_hash = hash_system_prompt(system_prompt)
    seed_dir = run_path / f"seed_{seed_id}"
    file_path = seed_dir / f"{prompt_hash}.json"

    # Normalize attacks to dicts
    attacks_dicts = [
        convert_attack_to_dict(a) if not isinstance(a, dict) else a for a in attacks
    ]

    # Compute adversarial scores for system-level stats if not provided
    def _default_rater_pair(atk: Dict[str, Any]) -> tuple[str | None, str | None]:
        responses = atk.get("responses", [])
        if not responses:
            return (None, None)
        ratings = responses[0].get("ratings", [])
        if len(ratings) >= 2:
            r1 = ratings[0].get("rater", {}).get("model_name")
            r2 = ratings[1].get("rater", {}).get("model_name")
            return (r1, r2)
        return (None, None)

    def _mean_norm_for_rater(atk: Dict[str, Any], rater_name: str) -> float | None:
        vals: List[float] = []
        for resp in atk.get("responses", []):
            for rating in resp.get("ratings", []):
                if rating.get("rater", {}).get("model_name") == rater_name:
                    z = rating.get("aux_info", {}).get("normalized_score")
                    if isinstance(z, (int, float)):
                        vals.append(float(z))
        return float(sum(vals) / len(vals)) if vals else None

    adv_scores: List[float] = []
    for atk in attacks_dicts:
        r1, r2 = _default_rater_pair(atk)
        if not r1 or not r2:
            continue
        z1 = _mean_norm_for_rater(atk, r1)
        z2 = _mean_norm_for_rater(atk, r2)
        if z1 is None or z2 is None:
            continue
        adv_scores.append(float(adversariality(z_score_1=z1, z_score_2=z2)))

    if mean_score is None:
        mean_score = float(sum(adv_scores) / len(adv_scores)) if adv_scores else 0.0
    if stdev_score is None:
        if len(adv_scores) == 0:
            stdev_score = 0.0
        else:
            m = mean_score
            stdev_score = (
                sum((x - m) ** 2 for x in adv_scores) / len(adv_scores)
            ) ** 0.5

    data = {
        "system_prompt": system_prompt,
        "meta": meta,
        "mean_score": mean_score,
        "stdev_score": stdev_score,
        "attacks": attacks_dicts,
    }

    atomic_write_json(file_path, data)
    return prompt_hash


def save_cluster_info(
    run_path: Path,
    seed_id: int,
    summary: str,
    train_batch_size: int,
    sample_train_prompts: list,
) -> None:
    """Save cluster information for a seed state."""

    seed_dir = run_path / f"seed_{seed_id}"
    file_path = seed_dir / "cluster_info.json"

    data = {
        "summary": summary,
        "train_batch_size": train_batch_size,
        "sample_train_prompts": sample_train_prompts[:10],  # Limit for file size
    }

    atomic_write_json(file_path, data)


def save_population_state(
    run_path: Path, seed_id: int, step: int, population_state: Dict[str, int]
) -> None:
    """Save population state for a seed at a specific step."""

    seed_dir = run_path / f"seed_{seed_id}"
    file_path = seed_dir / "population_history.json"

    # Load existing population history or create new
    population_history = {}
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                population_history = json.load(f)
        except (json.JSONDecodeError, IOError):
            population_history = {}

    # Convert population state to hashes for storage
    population_hashes = {}
    for system_prompt, generation in population_state.items():
        prompt_hash = hash_system_prompt(system_prompt)
        population_hashes[prompt_hash] = generation

    # Update history for this step
    population_history[str(step)] = population_hashes

    atomic_write_json(file_path, population_history)


# Example usage for converting existing state objects
def convert_attack_to_dict(attack) -> Dict[str, Any]:
    """Convert Attack object (new schema) or dict to the standardized dict schema.

    Standard attack dict schema:
    {
        "system": str,
        "user": str,
        "responses": [
            {
                "assistant": str,
                "ratings": [
                    {
                        "raw_score": float,
                        "rater": {"model_name": str, "rating_function_type": str},
                        "aux_info": { ... }
                    }
                ]
            }
        ],
        "aux_info": { ... }
    }
    """
    # If it's already a dict with new keys, assume it's correct
    if isinstance(attack, dict) and "responses" in attack and "user" in attack:
        return attack

    # Try to adapt from dataclass Attack (state.Attack)
    try:
        system = getattr(attack, "system")
        user = getattr(attack, "user")
        aux_info = dict(getattr(attack, "aux_info", {}) or {})
        responses_out: List[Dict[str, Any]] = []
        for resp in getattr(attack, "responses", []) or []:
            assistant = getattr(resp, "assistant")
            ratings_out: List[Dict[str, Any]] = []
            for rating in getattr(resp, "ratings", []) or []:
                ratings_out.append(
                    {
                        "raw_score": rating.raw_score,
                        "rater": {
                            "model_name": rating.rater.model_name,
                            "rating_function_type": rating.rater.rating_function_type,
                        },
                        "aux_info": rating.aux_info,
                    }
                )
            responses_out.append({"assistant": assistant, "ratings": ratings_out})

        # Compute adversarial score using default rater pair if possible
        def _default_pair(atk_dict: Dict[str, Any]) -> tuple[str | None, str | None]:
            resps = atk_dict.get("responses", [])
            if not resps:
                return (None, None)
            rts = resps[0].get("ratings", [])
            if len(rts) >= 2:
                r1 = rts[0].get("rater", {}).get("model_name")
                r2 = rts[1].get("rater", {}).get("model_name")
                return (r1, r2)
            return (None, None)

        atk_dict = {
            "system": system,
            "user": user,
            "responses": responses_out,
            "aux_info": aux_info,
        }

        r1, r2 = _default_pair(atk_dict)
        if r1 and r2:
            # mean normalized per rater
            def _mean_norm(rater_name: str) -> float | None:
                vals: List[float] = []
                for r in responses_out:
                    for rt in r.get("ratings", []):
                        if rt.get("rater", {}).get("model_name") == rater_name:
                            nz = rt.get("aux_info", {}).get("normalized_score")
                            if isinstance(nz, (int, float)):
                                vals.append(float(nz))
                return float(sum(vals) / len(vals)) if vals else None

            z1 = _mean_norm(r1)
            z2 = _mean_norm(r2)
            if z1 is not None and z2 is not None:
                atk_dict["aux_info"]["adversarial_score"] = float(
                    adversariality(z_score_1=z1, z_score_2=z2)
                )

        return atk_dict
    except Exception:
        # Fallback: return minimal info if unknown structure
        return {
            "system": getattr(attack, "system", ""),
            "user": getattr(attack, "user", ""),
            "responses": [],
            "aux_info": {},
        }


def convert_system_prompt_stats_to_dict(
    stats, step: int, operation: str, **extra_meta
) -> Dict[str, Any]:
    """Convert SystemPromptStats to dictionary with metadata (new schema).

    Computes mean/stdev adversarial score across attacks if not present on stats.
    """
    system_prompt = getattr(stats, "system_prompt", None)
    attacks = getattr(stats, "attacks", [])
    attacks_dicts = [convert_attack_to_dict(a) for a in attacks]

    # Compute mean/stdev adversarial score across attacks
    adv_scores: List[float] = []
    for atk in attacks_dicts:
        # default rater pair
        responses = atk.get("responses", [])
        if not responses:
            continue
        ratings = responses[0].get("ratings", [])
        if len(ratings) < 2:
            continue
        r1 = ratings[0].get("rater", {}).get("model_name")
        r2 = ratings[1].get("rater", {}).get("model_name")
        if not r1 or not r2:
            continue

        # means
        def _mean_norm(rn: str) -> float | None:
            vals: List[float] = []
            for resp in responses:
                for rt in resp.get("ratings", []):
                    if rt.get("rater", {}).get("model_name") == rn:
                        z = rt.get("aux_info", {}).get("normalized_score")
                        if isinstance(z, (int, float)):
                            vals.append(float(z))
            return float(sum(vals) / len(vals)) if vals else None

        z1 = _mean_norm(r1)
        z2 = _mean_norm(r2)
        if z1 is None or z2 is None:
            continue
        adv_scores.append(float(adversariality(z_score_1=z1, z_score_2=z2)))

    mean_score = float(sum(adv_scores) / len(adv_scores)) if adv_scores else 0.0
    stdev_score = (
        (sum((x - mean_score) ** 2 for x in adv_scores) / len(adv_scores)) ** 0.5
        if adv_scores
        else 0.0
    )

    return {
        "system_prompt": system_prompt,
        "meta": {"step": step, "operation": operation, **extra_meta},
        "mean_score": mean_score,
        "stdev_score": stdev_score,
        "attacks": attacks_dicts,
    }
