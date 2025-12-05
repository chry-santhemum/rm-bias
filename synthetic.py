"""
Synthetic prompt dataset generation from a given spec.
"""

# %%

import json
import textwrap
import asyncio
from pathlib import Path
from dataclasses import asdict, dataclass

from caller import ChatHistory
from utils import parse_json_response
from specs import caller

# %%

BRAINSTORM_PROMPT = textwrap.dedent(
    """
    You are an expert in brainstorming realistic sub-categories for a given specification of a category of user prompts. You are an important component of a pipeline that generates diverse, realistic user prompts starting from a short specification.

    Your current task is to brainstorm a list of {n_topics} possible sub-categories for user prompts that fall under the given specification. Each sub-category should be a **simple, short phrase** describing a concrete class of user prompts that belong to the given specification.
    
    Another model will then use your description to generate actual user prompts, so make sure that your sub-categories aren't overly specific. In addition, make sure that your sub-categories are not too difficult for a usual chatbot to answer, and that they do not require responses that are too long or complex.
    
    Make sure that the {n_topics} sub-categories are different from each other and cover a diverse range of typical, common sub-categories which fall under the given specification:

    **Specification:** {spec}

    Use your thinking budget to brainstorm carefully and creatively, and then in your output field return ONLY your list of {n_topics} sub-categories formatted as a python array, like this:

    ```python
    [
        "Your first sub-category here",
        "Your second sub-category here",
        ...
    ]
    ```
    
    Do not include any other text in your output.
    """
).strip()



GENERATION_PROMPT = textwrap.dedent(
    """
    You are a human user writing prompts following a given category description. You are an important component of a pipeline that generates diverse, realistic user prompts.

    You will be given a description of a concrete category of user prompts. Your task is to write a list of {n_prompts} different user prompts that fall under the given category.
    
    The user prompts you write should vary naturally in terms of style and tone, and they should be phrased in similar ways that real users would prompt a chatbot. It is important that the user prompts faithfully fall under the category descriptions, and not deviate from them. 
    
    In addition, importantly, make sure that the user prompts do not require responses that are too long or complex. They should be able to be answered by a usual chatbot assistant in at most a few paragraphs.
    
    Keep in mind also that the user prompts you write will be the entirety of the user's message, so also include any additional contexts referred to in the prompt. For example, if the topic is "write a summary of a given document", then the user prompt should also include the full text of the document that the user is asking about.

    **Category description:** {topic}

    Use your thinking budget to reason carefully, and then in your output field return ONLY your list of {n_prompts} user prompts formatted as a python array, like this:

    ```python
    [
        "Your first user prompt here",
        "Your second user prompt here",
        ...
    ]
    ```

    Do not include any other text in your output.
    """
).strip()


# %%
async def main(
    specs_path: Path,
    model: str = "openai/gpt-5",
    n_topics: int = 16,
    n_prompts: int = 2,
    max_tokens: int = 10000,
    reasoning: str | int | None = "medium",
):
    with open(specs_path, "r") as f:
        specs: list[str] = json.load(f)
    
    save_dir = specs_path.parent

    sub_topic_chats = [
        ChatHistory().add_user(BRAINSTORM_PROMPT.format(spec=spec, n_topics=n_topics))
        for spec in specs
    ]

    brainstorm_responses = await caller.call(
        messages=sub_topic_chats,
        max_parallel=128,
        desc="Brainstorming",
        model=model,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )

    brainstorm_results = dict()

    for i, resp in enumerate(brainstorm_responses):
        spec = specs[i]
        if resp is None:
            continue

        sub_topics, _ = parse_json_response(resp, marker="python")
        if not isinstance(sub_topics, list):
            print(f"Error: sub_topics is not a list.\n{sub_topics}\nSkipping...")
            continue
        sub_topics = [topic.strip() for topic in sub_topics]
        brainstorm_results[spec] = sub_topics
    
    with open(save_dir / "sub_topics.json", "w") as f:
        json.dump(brainstorm_results, f, indent=4, sort_keys=True)

    results: dict[int, dict] = {
        i: {
            "summary": specs[i],
            "prompts": [],
        }
        for i in range(len(specs))
    }

    prompt_generation_chats = [
        ChatHistory().add_user(
            GENERATION_PROMPT.format(
                n_prompts=n_prompts,
                topic=topic,
            )
        )
        for spec in specs
        for topic in brainstorm_results[spec]
    ]
    prompt_cluster_ids = [
        i for i, spec in enumerate(specs) 
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
        user_prompts = [prompt.strip() for prompt in user_prompts]
        results[prompt_cluster_ids[i]]["prompts"].extend(user_prompts)

    # write results
    for i, cluster in results.items():
        with open(save_dir / f"cluster_{i}.json", "w") as f:
            json.dump(cluster, f, indent=4, sort_keys=True)

    return results

# %%

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs_path", type=str, required=True)
    parser.add_argument("--n_topics", type=int, default=32)
    parser.add_argument("--n_prompts", type=int, default=2)
    args = parser.parse_args()

    asyncio.run(main(
        specs_path=Path(args.specs_path),
        model="openai/gpt-5",
        n_topics=args.n_topics,
        n_prompts=args.n_prompts,
        max_tokens=10000,
        reasoning="medium",
    ))