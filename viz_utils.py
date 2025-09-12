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
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def atomic_write_json(file_path: Path, data: Dict[str, Any]) -> None:
    """Write JSON file atomically to prevent partial reads."""
    temp_path = file_path.with_suffix('.tmp')
    
    # Ensure directory exists
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Atomic rename
    os.rename(temp_path, file_path)

def save_system_prompt_stats(
    run_path: Path,
    seed_id: int, 
    system_prompt: str,
    attacks: list,
    mean_score: float,
    stdev_score: float,
    meta: Dict[str, Any]
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
        "attacks": attacks
    }
    
    atomic_write_json(file_path, data)
    return prompt_hash

def save_cluster_info(
    run_path: Path,
    seed_id: int,
    summary: str,
    train_batch_size: int,
    sample_train_prompts: list
) -> None:
    """Save cluster information for a seed state."""
    
    seed_dir = run_path / f"seed_{seed_id}"
    file_path = seed_dir / "cluster_info.json"
    
    data = {
        "summary": summary,
        "train_batch_size": train_batch_size,
        "sample_train_prompts": sample_train_prompts[:10]  # Limit for file size
    }
    
    atomic_write_json(file_path, data)

# Example usage for converting existing state objects
def convert_attack_to_dict(attack) -> Dict[str, Any]:
    """Convert Attack object to dictionary for JSON serialization."""
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
                    "rating_function_type": rating.rater.rating_function_type
                },
                "aux_info": rating.aux_info
            }
            for rating in attack.ratings
        ],
        "aux_info": attack.aux_info
    }

def convert_system_prompt_stats_to_dict(stats, step: int, operation: str, **extra_meta) -> Dict[str, Any]:
    """Convert SystemPromptStats to dictionary with metadata."""
    return {
        "system_prompt": stats.system_prompt,
        "meta": {
            "step": step,
            "operation": operation,
            **extra_meta
        },
        "mean_score": stats.mean_score,
        "stdev_score": stats.stdev_score,
        "attacks": [convert_attack_to_dict(attack) for attack in stats.attacks]
    }