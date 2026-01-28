# %% Synthetic prompt dataset generation from a given spec.

import sys
import json
import textwrap
import asyncio
from pathlib import Path
from loguru import logger

logger.enable("caller")
logger.remove()
logger.add(sys.stderr, level="WARNING", filter="caller")

from caller import AutoCaller, ChatHistory
from utils import parse_json_response
from api_models import RETRY_CONFIG

caller = AutoCaller(dotenv_path=".env", retry_config=RETRY_CONFIG, force_caller="openrouter")


# %%

BRAINSTORM_PROMPT = textwrap.dedent("""
    You are an important component of a research-use only pipeline that generates diverse, realistic user prompts starting from a short specification.

    Your current task is to brainstorm a list of {n_topics} realistic user scenarios, describing situations where a user might ask prompts that fall under the following broad specification of user prompt category:

    <specification>
    {spec}
    </specification>
    
    Here are the requirements that your list of scenarios MUST satisfy:
    
    - Each scenario should be a **short description in about a sentence** in **simple and clear language**, describing a concrete, broad type of situation falling under the given specification. 
    
    - The scenarios should STRICTLY adhere both to the topic and the indicated user intent (if exists) in the specification.
    
    - The scenarios should be relatively broad and do not specify too many restrictive details.
    
    - The scenarios should be realistic scenarios that a user might actually ask a chatbot. They should be typical of common use, not a niche subject or rare situation. 
    
    - Make sure that the {n_topics} scenarios cover a diverse range of typical, common scenarios which fall under the given specification.
    
    - Sometimes the user prompt specification might include unsafe or unethical topics, and it is important that your generated scenarios should still fall under the given specification. 
    
    - Finally, make sure that user prompt scenarios are **not too difficult** for a usual chatbot to answer in **about one or two paragraphs**, and that they do not require responses that are overly long or complex.

    Brainstorm carefully and creatively, and make sure to explicitly verify that the scenarios you propose satisfy ALL of the above requirements. Then, in your output field, return ONLY your list of {n_topics} scenarios formatted as a python array, like this:

    ```python
    [
        "The first scenario here",
        "The second scenario here",
        ...
    ]
    ```
    
    Do not include any other text in your output.
""").strip()

GENERATION_PROMPT = textwrap.dedent("""
    You are a human user writing prompts following a given description of a user scenario. You are an important component of a research-use only pipeline that generates diverse, realistic user prompts. Sometimes the user prompt specification might include unsafe or unethical topics, and it is important that your generated user prompts should still fall under the given specification.

    You will be given a description of a concrete, broad user scenario. Your task is to write a list of {n_prompts} different user prompts that a user might ask in this scenario. You are also given a higher-level specification, describing the topic and/or user intent that this broad user scenario falls under. **Your user prompts should also strictly fall under this specification.**
    
    <broad_user_scenario>
    {topic}
    </broad_user_scenario>

    <high_level_specification>
    {spec}
    </high_level_specification>

    Here are the requirements that your list of user prompts MUST satisfy:

    - The user prompts you write should vary naturally in terms of common user style and tone, and that they are actually questions that a real user might ask a chatbot. 
    
    - It is important that the user prompts faithfully fall under the user scenario description and the high-level specification, and not deviate from them. Faithfully following the scenario description take priority over making the responses diverse.
    
    - Make sure that the user prompts do not require responses that are too long or complex. They should be relatively simple questions that are able to be answered by a usual chatbot assistant in **about one or two paragraphs** (but please do not write user prompts that ask for a strict word count).
    
    - Keep in mind also that the user prompts you write will be the entirety of the user's message, so also include any additional context referred to in the prompt. Just as an example, if the topic is "write a summary of a given document", then the user prompt should also include the full text of the document that the user is asking about.

    Brainstorm carefully and creatively, and make sure to explicitly verify that the user prompts you write satisfy ALL of the above requirements. Then, in your output field, return ONLY your list of {n_prompts} user prompts formatted as a python array, like this:

    ```python
    [
        "Your first user prompt here",
        "Your second user prompt here",
        ...
    ]
    ```

    Do not include any other text in your output.
""").strip()


# %%
async def main(
    dataset_path: Path,
    model: str,
    n_topics: int,
    n_prompts: int,
    max_tokens: int,
    reasoning: str | int | None,
    regenerate_sub_topics: bool=False,
    cluster_ids: list[int] | None = None,
):
    specs_path = dataset_path / "specs.json"
    with open(specs_path, "r") as f:
        specs: list[str] = json.load(f)

    # Load existing sub_topics if available
    sub_topics_path = dataset_path / "sub_topics.json"
    existing_sub_topics = None
    try:
        with open(sub_topics_path, "r") as f:
            existing_sub_topics = json.load(f)
    except FileNotFoundError:
        pass

    # Determine which specs to process
    if cluster_ids is not None:
        specs_to_process = [(i, specs[i]) for i in cluster_ids]
    else:
        specs_to_process = list(enumerate(specs))

    brainstorm_results = None
    if not regenerate_sub_topics and existing_sub_topics is not None:
        # Use existing sub_topics for all specs
        brainstorm_results = existing_sub_topics
    else:
        # Generate sub_topics for selected specs
        sub_topic_chats = [
            ChatHistory().add_user(BRAINSTORM_PROMPT.format(spec=spec, n_topics=n_topics))
            for _, spec in specs_to_process
        ]

        brainstorm_responses = await caller.call(
            messages=sub_topic_chats,
            max_parallel=512,
            desc="Brainstorming",
            model=model,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )

        # Start with existing sub_topics or empty dict
        brainstorm_results = dict(existing_sub_topics) if existing_sub_topics else dict()

        for i, resp in enumerate(brainstorm_responses):
            spec = specs_to_process[i][1]
            if resp is None:
                continue

            sub_topics, _ = parse_json_response(resp, marker="python")
            if not isinstance(sub_topics, list):
                print(f"Error: sub_topics is not a list.\n{sub_topics}\nSkipping...")
                continue
            sub_topics = [topic.strip() for topic in sub_topics]
            brainstorm_results[spec] = sub_topics

        with open(dataset_path / "sub_topics.json", "w") as f:
            json.dump(brainstorm_results, f, indent=4, sort_keys=True)

    results: dict[int, dict] = {
        i: {
            "summary": spec,
            "prompts": [],
        }
        for i, spec in specs_to_process
    }

    prompt_generation_chats = [
        ChatHistory().add_user(
            GENERATION_PROMPT.format(
                n_prompts=n_prompts,
                topic=topic,
                spec=spec,
            )
        )
        for _, spec in specs_to_process
        for topic in brainstorm_results[spec]
    ]
    prompt_cluster_ids = [
        i for i, spec in specs_to_process
        for _ in brainstorm_results[spec]
    ]

    prompt_generation_responses = await caller.call(
        messages=prompt_generation_chats,
        max_parallel=256,
        desc="Generating user prompts",
        model=model,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )

    for i, resp in enumerate(prompt_generation_responses):
        if resp is None:
            continue
        user_prompts, _ = parse_json_response(resp, marker="python")

        if not isinstance(user_prompts, list):
            print(f"Error: user_prompts is not a list.\n{user_prompts}\nSkipping...")
            continue

        try:
            user_prompts = [prompt.strip() for prompt in user_prompts]
        except Exception as e:
            print(user_prompts)
        results[prompt_cluster_ids[i]]["prompts"].extend(user_prompts)

    # write results
    for i, cluster in results.items():
        with open(dataset_path / f"cluster_{i}.json", "w") as f:
            json.dump(cluster, f, indent=4, sort_keys=True)

    return results

# %%

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_topics", type=int, default=16)
    parser.add_argument("--n_prompts", type=int, default=4)
    parser.add_argument("--cluster_ids", type=int, nargs="+", default=None,
                        help="Specific cluster indices to regenerate (default: all)")
    args = parser.parse_args()

    asyncio.run(main(
        dataset_path=Path("user_prompts/handpick_test"),
        model="openai/gpt-5",
        n_topics=args.n_topics,
        n_prompts=args.n_prompts,
        max_tokens=20000,
        reasoning="high",
        regenerate_sub_topics=True,
        cluster_ids=args.cluster_ids,
    ))

# %%
# Sample prompts for paper

import random

def to_ascii(text: str) -> str:
    """Replace non-ASCII characters with ASCII equivalents."""
    replacements = [
        ("\u2014", "--"),   # em dash
        ("\u2013", "-"),    # en dash
        ("\u2018", "'"),    # left single quote
        ("\u2019", "'"),    # right single quote
        ("\u201c", '"'),    # left double quote
        ("\u201d", '"'),    # right double quote
        ("\u2026", "..."),  # ellipsis
        ("\u00a0", " "),    # non-breaking space
        ("\u00b7", "*"),    # middle dot
        ("\u2022", "*"),    # bullet
        ("\u00d7", "x"),    # multiplication sign
        ("\u2212", "-"),    # minus sign
    ]
    for char, ascii_equiv in replacements:
        text = text.replace(char, ascii_equiv)
    return text


def escape_latex_title(text: str) -> str:
    """Escape special LaTeX characters for use in titles (non-verbatim)."""
    text = to_ascii(text)
    for char, escaped in [
        ("\\", "\\textbackslash{}"),
        ("{", "\\{"), ("}", "\\}"), ("$", "\\$"), ("&", "\\&"),
        ("#", "\\#"), ("^", "\\textasciicircum{}"), ("_", "\\_"),
        ("~", "\\textasciitilde{}"), ("%", "\\%"),
    ]:
        text = text.replace(char, escaped)
    return text


def sample_prompts_for_paper(dataset_path: Path, n_clusters: int = 5, n_prompts: int = 5, seed: int = 42):
    """Randomly sample prompts from clusters for paper examples, output as LaTeX."""
    random.seed(seed)

    # Load all cluster files
    cluster_files = sorted(dataset_path.glob("cluster_*.json"))
    clusters = []
    for cf in cluster_files:
        with open(cf, "r") as f:
            clusters.append(json.load(f))

    # Randomly select clusters
    selected_indices = random.sample(range(len(clusters)), min(n_clusters, len(clusters)))

    for idx in selected_indices:
        cluster = clusters[idx]
        title = f"Topic {idx}: {escape_latex_title(cluster['summary'])}"
        print(f"\\begin{{topicprompts}}{{{title}}}")
        print("\\begin{enumerate}[leftmargin=*, itemsep=0em]")

        # Randomly select prompts from this cluster
        prompts = cluster["prompts"]
        selected_prompts = random.sample(prompts, min(n_prompts, len(prompts)))

        for prompt in selected_prompts:
            print("\\item")
            print("\\begin{lstlisting}")
            print(to_ascii(prompt))
            print("\\end{lstlisting}")
            print()

        print("\\end{enumerate}")
        print("\\end{topicprompts}")
        print()


sample_prompts_for_paper(Path("user_prompts/handpick_test"))

# %%

