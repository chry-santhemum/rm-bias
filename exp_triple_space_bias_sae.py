# %%
"""
Experiment: Find which SAE features in baseline responses predict
whether triple-spacing will increase or decrease the reward score.
"""

import json
import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from tqdm import tqdm
from loguru import logger
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder for Gemma Scope."""

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts: torch.Tensor) -> torch.Tensor:
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon


def load_sae(layer: int, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> JumpReLUSAE:
    """Load SAE from HuggingFace and return initialized model."""
    filename = f"layer_{layer}/width_16k/average_l0_91/params.npz"
    logger.info(f"Downloading SAE from google/gemma-scope-9b-it-res: {filename}")

    path = hf_hub_download(
        repo_id="google/gemma-scope-9b-it-res",
        filename=filename,
    )

    params = np.load(path)
    pt_params = {k: torch.from_numpy(v).to(device=device, dtype=dtype) for k, v in params.items()}

    logger.info(f"SAE params shapes: {{{', '.join(f'{k}: {v.shape}' for k, v in pt_params.items())}}}")

    # Infer dimensions from weights
    d_model, d_sae = pt_params["W_enc"].shape
    logger.info(f"SAE dimensions: d_model={d_model}, d_sae={d_sae}")

    sae = JumpReLUSAE(d_model, d_sae).to(device=device, dtype=dtype)
    sae.W_enc.data = pt_params["W_enc"]
    sae.W_dec.data = pt_params["W_dec"]
    sae.threshold.data = pt_params["threshold"]
    sae.b_enc.data = pt_params["b_enc"]
    sae.b_dec.data = pt_params["b_dec"]
    sae.eval()

    return sae


def gather_residual_activations(
    model,
    target_layer: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Get residual stream activations at target layer using hooks."""
    target_act = None

    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act
        target_act = outputs[0]
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()

    if target_act is None:
        raise ValueError("Hook failed to capture target activations")

    return target_act


def load_cluster_format(data_dir: Path) -> list[dict]:
    """Load results from cluster_*/results.json format."""
    all_results = []

    cluster_dirs = sorted(data_dir.glob("cluster_*"))
    logger.info(f"Found {len(cluster_dirs)} cluster directories")

    for cluster_dir in cluster_dirs:
        results_file = cluster_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
                all_results.extend(results)

    return all_results


def load_dataset_format(data_dir: Path) -> list[dict]:
    """Load results from {dataset}/results.json format (alpaca, lmarena, skywork, ultrafeedback)."""
    all_results = []

    # Find all results.json files in immediate subdirectories
    results_files = sorted(data_dir.glob("*/results.json"))
    logger.info(f"Found {len(results_files)} results.json files")

    for results_file in results_files:
        with open(results_file) as f:
            results = json.load(f)
            all_results.extend(results)

    return all_results


def load_experiment_data(data_dir: Path) -> list[dict]:
    """Load experiment data, auto-detecting the format."""
    # Check for cluster format first
    cluster_dirs = list(data_dir.glob("cluster_*"))
    if cluster_dirs:
        logger.info("Detected cluster format (cluster_*/results.json)")
        all_results = load_cluster_format(data_dir)
    else:
        # Check for dataset format ({dataset}/results.json)
        results_files = list(data_dir.glob("*/results.json"))
        if results_files:
            logger.info("Detected dataset format ({dataset}/results.json)")
            all_results = load_dataset_format(data_dir)
        else:
            raise ValueError(f"Unknown data format in {data_dir}. Expected cluster_*/results.json or */results.json")

    logger.info(f"Loaded {len(all_results)} total samples")
    return all_results


def pool_activations(
    activations: torch.Tensor,
    method: Literal["max", "mean", "ema"],
    ema_decay: float = 0.9,
) -> torch.Tensor:
    """Pool SAE activations across sequence dimension.

    Args:
        activations: (seq_len, d_sae) tensor
        method: pooling method
        ema_decay: decay factor for EMA (weight on previous value)

    Returns:
        (d_sae,) tensor
    """
    if method == "max":
        return activations.max(dim=0).values
    elif method == "mean":
        return activations.mean(dim=0)
    elif method == "ema":
        # EMA weighted toward later tokens
        # Start from first token, accumulate with decay
        ema = activations[0]
        for t in range(1, activations.shape[0]):
            ema = ema_decay * ema + (1 - ema_decay) * activations[t]
        return ema
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def process_batch(
    model,
    tokenizer,
    sae: JumpReLUSAE,
    samples: list[dict],
    target_layer: int,
    pool_method: Literal["max", "mean", "ema"],
    device: str,
) -> tuple[torch.Tensor, list[float]]:
    """Process a batch of samples and return pooled SAE activations.

    Returns:
        pooled_acts: (batch_size, d_sae) tensor on CPU
        diffs: list of reward diffs
    """
    # Prepare chat-formatted inputs
    messages_list = []
    for sample in samples:
        messages = [
            {"role": "user", "content": sample["user_prompt"]},
            {"role": "assistant", "content": sample["baseline_response"]},
        ]
        messages_list.append(messages)

    # Tokenize with chat template
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
             for m in messages_list]

    # Also tokenize just the user part to find response boundaries
    user_only_texts = [tokenizer.apply_chat_template([m[0]], tokenize=False, add_generation_prompt=True)
                       for m in messages_list]

    # Batch tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    user_only_inputs = tokenizer(
        user_only_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    # Get activations
    activations = gather_residual_activations(
        model, target_layer, inputs.input_ids, inputs.attention_mask
    )

    # Extract sequence lengths before freeing inputs
    seq_lengths = inputs.attention_mask.sum(dim=1).tolist()  # list of ints
    user_seq_lengths = user_only_inputs.attention_mask.sum(dim=1).tolist()

    # Free tokenized inputs immediately - no longer needed
    del inputs, user_only_inputs

    # Process each sample: extract response tokens, apply SAE, pool
    # Store on CPU to avoid GPU memory accumulation
    pooled_acts_list = []
    diffs = []

    for i, sample in enumerate(samples):
        # Find response start position (after user prompt tokens)
        response_start = int(user_seq_lengths[i])
        seq_len = int(seq_lengths[i])

        # Get response-only activations (creates a view, not a copy)
        response_acts = activations[i, response_start:seq_len]  # (response_len, d_model)

        if response_acts.shape[0] == 0:
            # Fallback: use all non-padding tokens if response detection fails
            response_acts = activations[i, :seq_len]

        # Apply SAE and pool
        with torch.no_grad():
            sae_acts = sae.encode(response_acts)  # (response_len, d_sae)
            pooled = pool_activations(sae_acts, pool_method)  # (d_sae,)
            # Move to CPU immediately to free GPU memory
            pooled_acts_list.append(pooled.cpu())
            del sae_acts, pooled

        diffs.append(sample["diff"])

    # Free GPU tensor
    del activations

    pooled_acts = torch.stack(pooled_acts_list)  # (batch_size, d_sae) on CPU
    return pooled_acts, diffs


def compute_mean_differences(
    all_pooled_acts: torch.Tensor,
    all_diffs: list[float],
    min_active: int = 30,
) -> list[dict]:
    """Compute mean reward diff when feature is active vs inactive.

    For each feature, computes:
    - mean_diff_active: mean(reward_diff) when feature > 0
    - mean_diff_inactive: mean(reward_diff) when feature == 0
    - effect: mean_diff_active - mean_diff_inactive

    Positive effect = when feature is present, triple-spacing helps MORE
    Negative effect = when feature is present, triple-spacing hurts MORE
    """
    all_pooled_acts_np = all_pooled_acts.detach().cpu().float().numpy()  # (n_samples, d_sae)
    diffs_np = np.array(all_diffs)

    n_features = all_pooled_acts_np.shape[1]
    n_samples = all_pooled_acts_np.shape[0]
    results = []

    logger.info(f"Computing mean differences for {n_features} features (min_active={min_active})...")

    for i in tqdm(range(n_features), desc="Computing mean differences"):
        feature_acts = all_pooled_acts_np[:, i]
        active_mask = feature_acts > 0
        n_active = int(active_mask.sum())
        n_inactive = n_samples - n_active

        # Skip features with insufficient samples in either group
        if n_active < min_active or n_inactive < min_active:
            continue

        diffs_when_active = diffs_np[active_mask]
        diffs_when_inactive = diffs_np[~active_mask]

        mean_diff_active = float(diffs_when_active.mean())
        mean_diff_inactive = float(diffs_when_inactive.mean())
        effect = mean_diff_active - mean_diff_inactive

        # Welch's t-test (doesn't assume equal variances)
        t_stat, p_value = stats.ttest_ind(diffs_when_active, diffs_when_inactive, equal_var=False)

        results.append({
            "feature_idx": i,
            "effect": effect,
            "mean_diff_active": mean_diff_active,
            "mean_diff_inactive": mean_diff_inactive,
            "n_active": n_active,
            "n_inactive": n_inactive,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
        })

    logger.info(f"Found {len(results)} features with >= {min_active} active samples")
    return results


def main(
    data_dir: Path,
    output_path: Path,
    layer: int = 20,
    pool_method: Literal["max", "mean", "ema"] = "max",
    batch_size: int = 4,
    max_samples: int | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load experiment data
    logger.info("=" * 60)
    logger.info("Loading experiment data...")
    all_samples = load_experiment_data(data_dir)

    if max_samples is not None:
        all_samples = all_samples[:max_samples]
        logger.info(f"Limited to {max_samples} samples")

    # Load model
    logger.info("=" * 60)
    logger.info("Loading Gemma 2 9B IT model...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SAE
    logger.info("=" * 60)
    logger.info(f"Loading SAE for layer {layer}...")
    sae = load_sae(layer, device)

    # Process in batches
    logger.info("=" * 60)
    logger.info(f"Processing {len(all_samples)} samples in batches of {batch_size}...")

    all_pooled_acts = []
    all_diffs = []

    for batch_start in tqdm(range(0, len(all_samples), batch_size), desc="Processing batches"):
        batch_samples = all_samples[batch_start:batch_start + batch_size]

        pooled_acts, diffs = process_batch(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            samples=batch_samples,
            target_layer=layer,
            pool_method=pool_method,
            device=device,
        )

        # pooled_acts is already on CPU from process_batch
        all_pooled_acts.append(pooled_acts)
        all_diffs.extend(diffs)

        # Clear GPU cache
        torch.cuda.empty_cache()

    # Concatenate all pooled activations
    all_pooled_acts = torch.cat(all_pooled_acts, dim=0)  # (n_samples, d_sae)
    logger.info(f"Total pooled activations shape: {all_pooled_acts.shape}")

    # Compute L0 (average number of active features per sample)
    l0 = (all_pooled_acts > 0).float().sum(dim=1).mean().item()
    logger.info(f"Average L0 (active features per sample): {l0:.1f}")

    # Compute mean differences
    logger.info("=" * 60)
    logger.info("Computing mean reward diff (active vs inactive) for each feature...")
    results = compute_mean_differences(all_pooled_acts, all_diffs, min_active=30)

    # Sort by absolute effect
    results.sort(key=lambda x: abs(x["effect"]), reverse=True)

    # Apply Bonferroni correction (only over features that passed min_active filter)
    n_features = len(results)
    bonferroni_threshold = 0.05 / n_features if n_features > 0 else 0.05

    for result in results:
        result["significant_bonferroni"] = result["p_value"] < bonferroni_threshold

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "data_dir": str(data_dir),
                "layer": layer,
                "pool_method": pool_method,
                "n_samples": len(all_samples),
                "average_l0": l0,
                "n_features_analyzed": n_features,
                "bonferroni_threshold": bonferroni_threshold,
            },
            "features": results,
        }, f, indent=2)
    logger.success(f"Saved results to {output_path}")

    # Print summary - top features by absolute effect
    logger.success("=" * 60)
    logger.success("TOP 50 FEATURES BY |EFFECT| (sorted by absolute effect size)")
    logger.success("effect = mean_diff_when_active - mean_diff_when_inactive")
    logger.success("positive = triple-spacing helps MORE when feature present")
    logger.success("negative = triple-spacing hurts MORE when feature present")
    logger.success("=" * 60)

    for r in results[:50]:
        sig = "*" if r["significant_bonferroni"] else ""
        logger.success(
            f"Feature {r['feature_idx']:5d}: effect={r['effect']:+.3f}, "
            f"active={r['mean_diff_active']:+.3f} (n={r['n_active']}), "
            f"inactive={r['mean_diff_inactive']:+.3f} (n={r['n_inactive']}), "
            f"p={r['p_value']:.2e}{sig}"
        )

    # Summary stats
    n_significant = sum(1 for r in results if r["significant_bonferroni"])
    n_positive = sum(1 for r in results if r["effect"] > 0)
    n_negative = sum(1 for r in results if r["effect"] < 0)
    logger.success("=" * 60)
    logger.success(f"Total features analyzed: {n_features}")
    logger.success(f"  Positive effect: {n_positive}, Negative effect: {n_negative}")
    logger.success(f"  Significant (Bonferroni p<{bonferroni_threshold:.2e}): {n_significant}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAE feature correlation analysis for triple-space bias")
    parser.add_argument("--data_dir", type=str, default="data/exp_triple_space_bias/20260110-161310",
                        help="Path to experiment data directory")
    parser.add_argument("--layer", type=int, default=20,
                        help="Target layer for SAE (default: 20)")
    parser.add_argument("--pool_method", type=str, default="max", choices=["max", "mean", "ema"],
                        help="Pooling method for SAE activations (default: max)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing (default: 4)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    
    data_dir_timestamp = args.data_dir.split("/")[-1]
    output_path = f"data/exp_triple_space_bias_sae/{data_dir_timestamp}/{args.pool_method}_pool_mean_diffs.json"

    main(
        data_dir=Path(args.data_dir),
        output_path=Path(output_path),
        layer=args.layer,
        pool_method=args.pool_method,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
