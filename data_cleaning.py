# %%
import patches
import re
from typing import Set, Dict, Any
import time
import pickle
from tqdm.auto import tqdm
import dotenv
import nest_asyncio
from pathlib import Path
import tiktoken
from datasets import Dataset, load_dataset

dotenv.load_dotenv()
nest_asyncio.apply()
tokenizer = tiktoken.get_encoding("gpt2")

# %%
# Filter questions whose responses are too long;
# By standard we sample 1024 tokens from the policy
# Here we filter for at most 512 tokens

# Ultrafeedback
def preprocess_ultrafeedback(item):
    item["chosen"] = item["chosen"][1]["content"]
    item["rejected"] = item["rejected"][1]["content"]
    item["prompt_length"] = len(tokenizer.encode(item["prompt"], disallowed_special=()))
    item["chosen_length"] = len(tokenizer.encode(item["chosen"], disallowed_special=()))
    item["rejected_length"] = len(
        tokenizer.encode(item["rejected"], disallowed_special=())
    )
    return item


start_time = time.time()
ultrafeedback = (
    load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    .map(
        preprocess_ultrafeedback,
        num_proc=8,  # type: ignore
    )
    .filter(
        lambda item: item["prompt_length"] < 512
        and item["chosen_length"] < 512
        and item["rejected_length"] < 512,
        num_proc=8,
    )
)
print(f"Preprocessing in {time.time() - start_time:.2f}s")
print("Ultrafeedback after filtering: ", len(ultrafeedback))  # 40477

# %%
Path("data/ultrafeedback").mkdir(parents=True, exist_ok=True)
with open("data/ultrafeedback/ds_filtered.pkl", "wb") as f:
    pickle.dump(ultrafeedback, f)

# %%
# Alpaca
def preprocess_alpaca(item):
    if item["input"] != "":
        item["prompt"] = item["instruction"] + "\n\n" + item["input"]
    else:
        item["prompt"] = item["instruction"]

    item["prompt_length"] = len(tokenizer.encode(item["prompt"], disallowed_special=()))
    item["output_length"] = len(tokenizer.encode(item["output"], disallowed_special=()))
    return item

start_time = time.time()
alpaca = load_dataset("vicgalle/alpaca-gpt4", split="train").map(
    preprocess_alpaca,
    num_proc=8,  # type: ignore
).filter(
    lambda item: item["prompt_length"] < 512
    and item["output_length"] < 512,
    num_proc=8,
)
print(f"Preprocessing in {time.time() - start_time:.2f}s")
print("Alpaca after filtering: ", len(alpaca))

Path("data/alpaca").mkdir(parents=True, exist_ok=True)
with open("data/alpaca/ds_filtered.pkl", "wb") as f:
    pickle.dump(alpaca, f)

# %%
# Wildchat
def preprocess_wildchat(item):
    item["prompt"] = item["conversation"][0]["content"]
    item["prompt_length"] = len(tokenizer.encode(item["prompt"], disallowed_special=()))
    return item


wildchat_iter = load_dataset(
    "allenai/WildChat-1M", split="train", streaming=True
).filter(
    lambda ex: ex["turn"] == 1 and ex["language"] == "English",
)

print("Taking 50k...")
wildchat = Dataset.from_list(list(wildchat_iter.take(50000)))

Path("data/wildchat").mkdir(parents=True, exist_ok=True)
with open("data/wildchat/ds_pre_filter.pkl", "wb") as f:
    pickle.dump(wildchat, f)

# %%
with open("data/wildchat/ds_pre_filter.pkl", "rb") as f:
    wildchat = pickle.load(f)

print("Preprocessing...")
start_time = time.time()
wildchat = wildchat.map(
    preprocess_wildchat,
    num_proc=8,
    remove_columns=wildchat.column_names,
).filter(
    lambda ex: ex["prompt_length"] < 512,
    num_proc=8,
)

print(f"Preprocessing in {time.time() - start_time:.2f}s")
print("Wildchat after filtering: ", len(wildchat))  # 41850

with open("data/wildchat/ds_filtered.pkl", "wb") as f:
    pickle.dump(wildchat, f)

# %%
# Basic cleaning: 
# normalize, deduplicate, and filter low-quality prompts


def _normalize_text(text: str) -> str:
    # lowercase, trim, collapse internal whitespace
    return " ".join(text.lower().strip().split())


_STOPWORDS = {
    "the",
    "is",
    "at",
    "which",
    "on",
    "and",
    "a",
    "an",
    "to",
    "of",
    "in",
    "for",
    "it",
    "this",
    "that",
    "with",
    "as",
    "by",
    "from",
    "or",
    "be",
    "are",
    "was",
    "were",
    "can",
    "could",
    "should",
    "would",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "but",
    "if",
    "than",
    "then",
    "so",
    "such",
}


def _is_garbage(text: str) -> bool:
    # repeated single character sequences (>=6)
    if re.search(r"(.)\1{5,}", text):
        return True
    # high non-alnum (excluding whitespace) ratio
    total = len(text)
    if total == 0:
        return True
    non_alnum = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    if total > 0 and (non_alnum / total) > 0.5:
        return True
    # stopword-only prompts (after tokenization)
    tokens = [t for t in text.split() if t]
    if tokens and all(t in _STOPWORDS for t in tokens):
        return True
    return False


def _clean_records(ds: Dataset, min_words: int = 4, max_words: int = 1024) -> Dataset:
    seen_norms: Set[str] = set()
    seen_shingles: Set[str] = set()
    shingle_n: int = 5  # word n-gram size
    dup_overlap_ratio: float = 0.5  # consider duplicate if >=80% shingles already seen
    min_shingles_for_check: int = 5
    cleaned: list[Dict[str, Any]] = []
    for item in tqdm(ds, desc="Filtering and deduplicating"):
        try:
            raw = item["prompt"]  # type: ignore[index]
            if not isinstance(raw, str):
                continue
            norm = _normalize_text(raw)
            wc = len(norm.split())
            if wc < min_words or wc > max_words:
                continue
            if _is_garbage(norm):
                continue
            # Fast exact dedup first
            if norm in seen_norms:
                continue

            # Shingle-based fuzzy dedup (simple and fast)
            tokens = [t for t in norm.split() if t]
            if len(tokens) >= shingle_n:
                shingles = {
                    " ".join(tokens[i : i + shingle_n])
                    for i in range(len(tokens) - shingle_n + 1)
                }
            else:
                # fallback to unigrams if too short
                shingles = set(tokens)

            # If we have enough shingles, compute overlap ratio against seen_shingles
            is_dup = False
            if len(shingles) >= min_shingles_for_check:
                overlap = sum((1 for s in shingles if s in seen_shingles))
                overlap_ratio = overlap / max(1, len(shingles))
                if overlap_ratio >= dup_overlap_ratio:
                    is_dup = True

            if is_dup:
                continue

            # Accept: record exact norm and add shingles to global set
            seen_norms.add(norm)
            seen_shingles.update(shingles)
            # keep original content; use normalized only for dedup/filtering
            cleaned.append(dict(item))
        except Exception:
            continue
    return Dataset.from_list(cleaned)


def clean(ds_name: str):
    with open(f"data/{ds_name}/ds_filtered.pkl", "rb") as f:
        ds_filtered = pickle.load(f)
    print(f"{ds_name} before cleaning: ", len(ds_filtered))

    start_time = time.time()
    ds_filtered = _clean_records(ds_filtered)
    print(f"Data cleaning took {time.time() - start_time:.2f}s")
    print(f"{ds_name} after cleaning: ", len(ds_filtered))  # 31786

    with open(f"data/{ds_name}/ds_cleaned.pkl", "wb") as f:
        pickle.dump(ds_filtered, f)

# %%

clean("alpaca")

# %%
