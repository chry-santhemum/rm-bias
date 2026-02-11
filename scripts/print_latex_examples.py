"""Print LaTeX side-by-side (original, rewrite) examples for each significant bias."""

import json
import random
import re
import sys
from pathlib import Path

DATA_DIR = Path("data/exp_attribute_validation/20260112-162826")
RESULTS_FILE = DATA_DIR / "partial_conjunction_results_global.json"
REWRITERS = ["openai_gpt-5-mini", "anthropic_claude-haiku-4.5", "x-ai_grok-4.1-fast"]
MAX_RESPONSE_CHARS = 600
NUM_EXAMPLES = 3
NUM_BIASES = 5
SEED = 42


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in text."""
    # Order matters: backslash first, then others
    text = text.replace("\\", "\\textbackslash{}")
    for char in "&%$#_{}":
        text = text.replace(char, f"\\{char}")
    text = text.replace("~", "\\textasciitilde{}")
    text = text.replace("^", "\\textasciicircum{}")
    # Preserve runs of multiple spaces while allowing line breaks:
    # keep one regular space (breakable) then pad with ~ for the rest
    text = re.sub(r" {2,}", lambda m: " " + "~" * (len(m.group()) - 1), text)
    return text


def load_rollout_pairs(seed: str, attribute: str) -> list[tuple[str, str, str]]:
    """Load all (prompt, baseline, rewrite) triples for a given seed and attribute."""
    pairs = []
    for rewriter in REWRITERS:
        rollouts_path = DATA_DIR / f"seed_{seed}_validate" / rewriter / "rollouts.json"
        if not rollouts_path.exists():
            continue
        with open(rollouts_path) as f:
            rollouts = json.load(f)
        if attribute not in rollouts:
            continue
        for prompt, entries in rollouts[attribute].items():
            for entry in entries:
                if entry is None:
                    continue
                baseline = entry.get("baseline_response")
                rewritten = entry.get("rewritten_response")
                if baseline is None or rewritten is None:
                    continue
                if len(baseline) <= MAX_RESPONSE_CHARS and len(rewritten) <= MAX_RESPONSE_CHARS:
                    pairs.append((prompt, baseline, rewritten))
    return pairs


def print_preamble():
    pass


def print_bias_examples(attribute: str, pairs: list[tuple[str, str, str]]):
    escaped_attr = escape_latex(attribute)
    print(f"\n\\subsection*{{Attribute: {escaped_attr}}}\n")
    for i, (prompt, baseline, rewritten) in enumerate(pairs):
        escaped_prompt = escape_latex(prompt)
        escaped_baseline = escape_latex(baseline)
        escaped_rewritten = escape_latex(rewritten)
        print(f"\\textbf{{Prompt {i+1}.}} {escaped_prompt}\n")
        print(r"\noindent\begin{minipage}[t]{0.48\textwidth}\strut\vspace*{-\baselineskip}")
        print(r"\begin{responsebox}[Original]")
        print(escaped_baseline)
        print(r"\end{responsebox}")
        print(r"\end{minipage}\hfill%")
        print(r"\begin{minipage}[t]{0.48\textwidth}\strut\vspace*{-\baselineskip}")
        print(r"\begin{responsebox}[Rewritten]")
        print(escaped_rewritten)
        print(r"\end{responsebox}")
        print(r"\end{minipage}")
        print()


def main():
    random.seed(SEED)

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    print_preamble()

    # Collect all significant biases with their rollout pairs
    significant = []
    for seed, attributes in results["seeds"].items():
        for entry in attributes:
            if not entry["passes_criteria"]:
                continue
            attribute = entry["attribute"]
            pairs = load_rollout_pairs(seed, attribute)
            if not pairs:
                print(f"% WARNING: no short-enough pairs found for: {attribute}",
                      file=sys.stderr)
                continue
            significant.append((attribute, pairs))

    # Randomly select a subset of biases
    selected = random.sample(significant, min(NUM_BIASES, len(significant)))

    for attribute, pairs in selected:
        sampled = random.sample(pairs, min(NUM_EXAMPLES, len(pairs)))
        print_bias_examples(attribute, sampled)

    print(f"% Total significant biases shown: {len(selected)}", file=sys.stderr)


if __name__ == "__main__":
    main()
