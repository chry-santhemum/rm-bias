# %%
"""Evaluation of spurious correlations via LLM-based attribute guessing and rating.

This script loads rewrite results from `evaluation-data/`, randomly samples
triplets (original, plus, minus), asks an LLM to guess the changing textual
attribute, then asks another LLM to rate similarity (0-10) between the target
attribute and guessed attribute. Finally, it produces a violin plot per target
attribute and saves a consolidated `evaluation.json`.

Assumptions:
- evaluation-data contains per-seed step folders with files:
  - rewrite_results.json (attribute -> user -> list[PlusMinusRollout])
  - rewrite_baseline_results.json (user -> list[Rollout])
- Lists for each user in both files align by index; the original for
  rewrite_results[attribute][user][i] is baseline[user][i]["response"].
- seed-specific cluster summary is available at `evaluation-data/seed_{seed}_cl`u`ster.json`.

Fail-fast philosophy: we use asserts to validate assumptions and inputs.
"""

# %%

from __future__ import annotations

# %%
import asyncio
import concurrent.futures
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

import matplotlib.pyplot as plt

from chat_limiter import Message, MessageRole

# %%

import sys
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))
from llm import LLM  # local import without package context

# %%

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class Config:
    evaluation_data_dir: Path = Path(__file__).parent / "evaluation-data"
    out_dir: Path = Path(__file__).parent / "results"
    seeds: List[int] | None = None  # if None, infer from folders
    steps: List[int] | None = None  # if None, infer from folders
    # Guessing
    guess_model: str = "openai/gpt-5-mini"
    guess_repeats: int = 10
    examples_per_guess: int = 6  # number of triplets per guess prompt
    # Similarity
    sim_model: str = "openai/gpt-5-mini"
    # Plot
    figure_width: int = 14
    figure_height: int = 6
    random_seed: int = 20251005


# %%

# Prompts (mirroring style from one_turn planning and judge prompts)
GUESS_SYSTEM = (
    "You are an expert in writing novel system prompts that specify the behavior of other assistant language models."
)

GUESS_USER_TEMPLATE = (
    "You are given multiple examples of an original assistant response and two modified responses, labeled PLUS and MINUS.\n\n"
    "Your task is to infer a single textual attribute that the PLUS responses exhibit and the MINUS responses avoid, relative to the originals.\n\n"
    "Furthermore, importantly, you should only consider qualities that are generally applicable to responses to any sensible user prompt described by the following summary, not just these examples.\n\n"
    "<user_prompt_cluster_summary>\n{cluster_summary}\n</user_prompt_cluster_summary>\n\n"
    "Think thoroughly about all features of the assistant responses, considering both high and low level features. Unusual or idiosyncratic features should be especially considered.\n\n"
    "Then, phrase the inferred feature as a system prompt that instructs a model to exhibit that feature. Return ONLY a single system prompt string, and nothing else.\n\n"
    "<examples>\n{examples}\n</examples>\n"
)

SIMILARITY_SYSTEM = (
    """
    Rate the similarity between two textual attributes (as system prompts) on a 0-10 scale:
    9-10: Essentially the same attribute; minor wording differences only.
    7-8: Strongly similar; substantial overlap in meaning and intent.
    5-6: Moderately similar; overlap on some aspects but notable differences.
    3-4: Weakly related; only small overlap.
    0-2: Unrelated or contradictory.
    Return only a single integer between 0 and 10.
    """.strip()
)

SIMILARITY_USER_TEMPLATE = (
    "<target_attribute>\n{target}\n</target_attribute>\n\n"
    "<guessed_attribute>\n{guess}\n</guessed_attribute>\n\n"
    "In your output, return only a single integer between 0 and 10."
)


def _infer_available_steps_and_seeds(base: Path) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("step_") and "_seed_" in p.name:
            parts = p.name.split("_")
            assert parts[0] == "step"
            step = int(parts[1])
            assert parts[2] == "seed"
            seed = int(parts[3])
            pairs.append((step, seed))
    pairs.sort()
    logger.info(f"Discovered {len(pairs)} step/seed pairs under {base}.")
    return pairs


def _load_cluster_summary(base: Path, seed: int) -> str:
    cluster_path = base / f"seed_{seed}_cluster.json"
    logger.info(f"Loading cluster summary from {cluster_path}")
    with open(cluster_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    assert isinstance(data, dict) and "summary" in data
    assert isinstance(data["summary"], str)
    return data["summary"]


def _load_rewrite_data(step_dir: Path) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, List[Dict[str, Any]]]]:
    rr_path = step_dir / "rewrite_results.json"
    rb_path = step_dir / "rewrite_baseline_results.json"
    logger.info(f"Loading rewrite results: {rr_path}")
    with open(rr_path, "r", encoding="utf-8") as f:
        rewrite_results = json.load(f)
    logger.info(f"Loading baseline results: {rb_path}")
    with open(rb_path, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)
    assert isinstance(rewrite_results, dict)
    assert isinstance(baseline_results, dict)
    logger.info(f"rewrite_results attributes: {len(rewrite_results)}; baseline users: {len(baseline_results)}")
    return rewrite_results, baseline_results


def _collect_triplets(
    rewrite_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    baseline_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Build attribute -> list of (original, plus, minus) triplets across users.

    Requires that for each attribute and user, the list lengths match the user's baseline list,
    and each entry `i` corresponds to the original baseline response at index `i`.
    """
    attr_to_triplets: Dict[str, List[Tuple[str, str, str]]] = {}
    for attribute, user_map in rewrite_results.items():
        if attribute == "":
            # Skip empty attribute key (shouldn't occur in final data)
            continue
        triplets: List[Tuple[str, str, str]] = []
        for user, rollouts in user_map.items():
            assert user in baseline_results
            base_list = baseline_results[user]
            assert len(rollouts) == len(base_list)
            for i, pm in enumerate(rollouts):
                plus = pm["plus"]
                minus = pm["minus"]
                assert isinstance(plus, str) and isinstance(minus, str)
                # Only keep complete entries
                if plus.strip() != "" and minus.strip() != "":
                    original = base_list[i]["response"]
                    assert isinstance(original, str)
                    triplets.append((original, plus, minus))
        if len(triplets) > 0:
            attr_to_triplets[attribute] = triplets
            logger.info(f"Triplets collected for attribute [{attribute[:40]}{'…' if len(attribute)>40 else ''}]: {len(triplets)}")
    logger.info(f"Total attributes with triplets: {len(attr_to_triplets)}")
    return attr_to_triplets


def _format_examples_for_guess(examples: List[Tuple[str, str, str]]) -> str:
    lines: List[str] = []
    for j, (orig, plus, minus) in enumerate(examples):
        lines.append(f"Example {j+1}:")
        lines.append("<original>")
        lines.append(orig)
        lines.append("</original>")
        lines.append("<plus>")
        lines.append(plus)
        lines.append("</plus>")
        lines.append("<minus>")
        lines.append(minus)
        lines.append("</minus>\n")
    return "\n".join(lines)


def _batch_guess_attributes(
    guess_llm: LLM,
    cluster_summary: str,
    attribute_to_examples: Dict[str, List[Tuple[str, str, str]]],
    repeats: int,
    examples_per_guess: int,
) -> Dict[str, List[str]]:
    key_to_messages: Dict[Tuple[str, int], List[Message]] = {}
    total_requests = 0
    for attribute, examples in attribute_to_examples.items():
        k = min(examples_per_guess, len(examples))
        if k <= 0:
            continue
        for r in range(repeats):
            sampled = random.sample(examples, k)
            msg_user = GUESS_USER_TEMPLATE.format(
                cluster_summary=cluster_summary,
                examples=_format_examples_for_guess(sampled),
            )
            key_to_messages[(attribute, r)] = [
                Message(role=MessageRole.SYSTEM, content=GUESS_SYSTEM),
                Message(role=MessageRole.USER, content=msg_user),
            ]
            total_requests += 1
    logger.info(f"Guess requests built: {total_requests}")
    if len(key_to_messages) == 0:
        logger.warning("No guess requests to send (empty key_to_messages). Returning empty guesses.")
        return {}
    results = asyncio_run_safe(guess_llm._generate_batch_llm_response(key_to_messages))
    output: Dict[str, List[str]] = {}
    for (attribute, r), guess in results.items():
        assert isinstance(guess, (str, type(None)))
        if guess is None:
            continue
        output.setdefault(attribute, []).append(guess.strip())
    for attribute, guesses in output.items():
        logger.info(f"Guesses received for attribute [{attribute[:40]}{'…' if len(attribute)>40 else ''}]: {len(guesses)}")
    return output


def _batch_similarity_scores(
    sim_llm: LLM,
    target_to_guesses: Dict[str, List[str]],
) -> Dict[str, List[int]]:
    key_to_messages: Dict[Tuple[str, int], List[Message]] = {}
    total_requests = 0
    for target, guesses in target_to_guesses.items():
        clean_guesses = [g for g in guesses if isinstance(g, str) and g.strip()]
        for i, guess in enumerate(clean_guesses):
            key_to_messages[(target, i)] = [
                Message(role=MessageRole.SYSTEM, content=SIMILARITY_SYSTEM),
                Message(role=MessageRole.USER, content=SIMILARITY_USER_TEMPLATE.format(target=target, guess=guess)),
            ]
            total_requests += 1
    logger.info(f"Similarity requests built: {total_requests}")
    if len(key_to_messages) == 0:
        logger.warning("No similarity requests to send (empty key_to_messages). Returning empty scores.")
        return {}
    results = asyncio_run_safe(sim_llm._generate_batch_llm_response(key_to_messages))
    scores: Dict[str, List[int]] = {}
    for (target, i), resp in results.items():
        assert isinstance(resp, (str, type(None)))
        if resp is None:
            continue
        resp_str = resp.strip()
        assert resp_str.isdigit(), f"Similarity response must be integer: {resp_str}"
        val = int(resp_str)
        assert 0 <= val <= 10
        scores.setdefault(target, []).append(val)
    for target, vals in scores.items():
        if len(vals) > 0:
            mean_val = sum(vals) / len(vals)
            logger.info(f"Similarity scores for target [{target[:40]}{'…' if len(target)>40 else ''}]: n={len(vals)} min={min(vals)} max={max(vals)} mean={mean_val:.2f}")
    return scores


def _plot_violin(target_to_scores: Dict[str, List[int]], cfg: Config) -> Path:
    attrs = list(target_to_scores.keys())
    data = [target_to_scores[a] for a in attrs]
    assert len(attrs) == len(data)

    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    parts = ax.violinplot(dataset=data, showmeans=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(attrs) + 1))
    ax.set_xticklabels([f"{a[:30]}…" if len(a) > 30 else a for a in attrs], rotation=45, ha='right')
    ax.set_ylabel("Similarity (0-10)")
    ax.set_title("Similarity of guessed attributes to targets")
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = cfg.out_dir / "violin.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved violin plot for {len(attrs)} attributes to {fig_path}")
    return fig_path


def asyncio_run_safe(coro):
    try:
        # Try to get existing event loop (e.g. in Jupyter)
        loop = asyncio.get_running_loop()

        def _thread_runner():
            return asyncio.run(
                coro
            )

        # Run the async function in a thread to avoid "already running loop" errors
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(_thread_runner)
            result = fut.result()
            logger.info("asyncio_run_safe used thread fallback for running coroutine.")
            return result
    except RuntimeError as e:
        import traceback
        traceback.print_exc()
        print(f"RuntimeError: {e}")
        if "no running event loop" in str(e):
            # No running event loop – safe to run directly.
            logger.info("asyncio_run_safe running coroutine directly (no running loop).")
            return asyncio.run(coro)


def _config_to_jsonable(cfg: Config) -> Dict[str, Any]:
    d: Dict[str, Any] = dict(cfg.__dict__)
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d


cfg = Config()

# %%

random.seed(cfg.random_seed)
logger.info(f"Config: evaluation_data_dir={cfg.evaluation_data_dir}, out_dir={cfg.out_dir}, repeats={cfg.guess_repeats}, examples_per_guess={cfg.examples_per_guess}")

# Discover steps/seeds
pairs = _infer_available_steps_and_seeds(cfg.evaluation_data_dir)
assert len(pairs) > 0, "No step/seed folders found in evaluation-data."
logger.info(f"Step/seed pairs: {pairs}")

# %%

# Prepare LLMs
guess_llm = LLM(llm_model_name=cfg.guess_model)
sim_llm = LLM(llm_model_name=cfg.sim_model)
logger.info(f"Initialized LLMs: guess_model={cfg.guess_model}, sim_model={cfg.sim_model}")

# %%

def _sanity_check_llm(llm: LLM, label: str) -> None:
    key_to_messages: Dict[str, List[Message]] = {
        "check": [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Reply with just the word OK."),
        ]
    }
    result = asyncio_run_safe(llm._generate_batch_llm_response(key_to_messages))
    assert isinstance(result, dict) and "check" in result, f"{label} sanity check failed: no result dict."
    text = result["check"]
    assert isinstance(text, str) and text.strip() != "", f"{label} sanity check failed: empty response."
    logger.info(f"Sanity check passed for {label}: {len(text.strip())} chars.")


_sanity_check_llm(guess_llm, "guess_llm")
_sanity_check_llm(sim_llm, "sim_llm")

# %%


# Aggregate across all (step, seed) pairs
aggregated_results: Dict[str, Dict[str, Any]] = {}

for step, seed in pairs:
    step_dir = cfg.evaluation_data_dir / f"step_{step}_seed_{seed}"
    cluster_summary = _load_cluster_summary(cfg.evaluation_data_dir, seed)
    rewrite_results, baseline_results = _load_rewrite_data(step_dir)
    attr_to_triplets = _collect_triplets(rewrite_results, baseline_results)
    if len(attr_to_triplets) == 0:
        logger.warning(f"No triplets found for step={step} seed={seed}; skipping.")
        continue

    # Guess
    guessed: Dict[str, List[str]] = _batch_guess_attributes(
        guess_llm=guess_llm,
        cluster_summary=cluster_summary,
        attribute_to_examples=attr_to_triplets,
        repeats=cfg.guess_repeats,
        examples_per_guess=min(cfg.examples_per_guess, max(len(v) for v in attr_to_triplets.values())),
    )
    # Similarity (skip if no guesses produced)
    if not guessed or all(len(v) == 0 for v in guessed.values()):
        target_to_scores = {}
    else:
        target_to_scores = _batch_similarity_scores(sim_llm, guessed)

    # Store partial results per (step, seed)
    aggregated_results[f"step_{step}_seed_{seed}"] = {
        "target_attributes": list(attr_to_triplets.keys()),
        "guesses": guessed,
        "similarity_scores": target_to_scores,
    }
    logger.info(f"Accumulated results for step={step} seed={seed}: targets={len(attr_to_triplets)}, guesses_nonempty={sum(1 for v in guessed.values() if len(v)>0)}, scored_targets={sum(1 for v in target_to_scores.values() if len(v)>0)}")

# Merge scores across runs per attribute string
merged_scores: Dict[str, List[int]] = {}
merged_guesses: Dict[str, List[str]] = {}
for _, payload in aggregated_results.items():
    for a, scores in payload["similarity_scores"].items():
        merged_scores.setdefault(a, []).extend(scores)
    for a, guesses in payload["guesses"].items():
        merged_guesses.setdefault(a, []).extend(guesses)

# %%

# Plot
if len(merged_scores) > 0:
    fig_path = _plot_violin(merged_scores, cfg)
else:
    fig_path = cfg.out_dir / "violin.png"
logger.info(f"Merged across runs: attributes_with_scores={len(merged_scores)}")


# %%
# Save JSON
cfg.out_dir.mkdir(parents=True, exist_ok=True)
out_json = cfg.out_dir / "evaluation.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(
        {
            "config": _config_to_jsonable(cfg),
            "aggregated": aggregated_results,
            "merged": {
                "guesses": merged_guesses,
                "similarity_scores": merged_scores,
            },
            "figure_path": str(fig_path),
        },
        f,
        indent=2,
    )
logger.info(f"Saved evaluation JSON to {out_json}")
