import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any


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
    mean_score: float = 0.0,
    stdev_score: float = 0.0,
    meta: Dict[str, Any] = {},
) -> str:
    """Save SystemPromptStats to JSON file, returns hash."""

    prompt_hash = hash_system_prompt(system_prompt)
    seed_dir = run_path / f"seed_{seed_id}"
    file_path = seed_dir / f"{prompt_hash}.json"

    data = {
        "system_prompt": system_prompt,
        "meta": meta,
        "mean_score": mean_score,
        "stdev_score": stdev_score,
        "attacks": attacks,
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
    """Convert Attack object to dictionary for JSON serialization."""
    # Get unnormalized scores for reward model and LM judge
    unnormalized_reward = None
    unnormalized_lm_judge = None

    for rating in attack.ratings:
        if rating.rater.rating_function_type == "classifier":
            unnormalized_reward = rating.raw_score
        elif rating.rater.rating_function_type == "lm_judge":
            unnormalized_lm_judge = rating.raw_score

    # Create enhanced aux_info with computed scores
    enhanced_aux_info = dict(attack.aux_info)
    enhanced_aux_info.update(
        {
            "adversarial_score": attack.adversarial_score,
            "unnormalized_reward": unnormalized_reward,
            "unnormalized_lm_judge": unnormalized_lm_judge,
            "normalized_reward": attack.normalized_reward,
            "normalized_lm_judge": attack.normalized_lm_judge,
        }
    )

    return {
        "chat_history": {
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in attack.chat_history.messages
            ]
        },
        "ratings": [
            {
                "raw_score": rating.raw_score,
                "rater": {
                    "model_name": rating.rater.model_name,
                    "rating_function_type": rating.rater.rating_function_type,
                },
                "aux_info": rating.aux_info,
            }
            for rating in attack.ratings
        ],
        "aux_info": enhanced_aux_info,
    }


def convert_system_prompt_stats_to_dict(
    stats, step: int, operation: str, **extra_meta
) -> Dict[str, Any]:
    """Convert SystemPromptStats to dictionary with metadata."""
    return {
        "system_prompt": stats.system_prompt,
        "meta": {"step": step, "operation": operation, **extra_meta},
        "mean_score": stats.mean_score,
        "stdev_score": stats.stdev_score,
        "attacks": [convert_attack_to_dict(attack) for attack in stats.attacks],
    }
