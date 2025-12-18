"""
Synthetic prompt dataset generation from a given spec.
"""

# %%

import json
import textwrap
import asyncio
from pathlib import Path

from caller import AutoCaller, ChatHistory
from utils import parse_json_response
from api_models import RETRY_CONFIG

caller = AutoCaller(dotenv_path=".env", retry_config=RETRY_CONFIG)

# %%

BRAINSTORM_PROMPT = textwrap.dedent("""
    You are an important component of a research-use only pipeline that generates diverse, realistic user prompts starting from a short specification.

    Your current task is to brainstorm a list of {n_topics} realistic user scenarios under which the user might ask prompts that fall under the given specification. Each scenario should be a **short description in about a sentence** in **simple and clear language**, describing a concrete type of situation falling under the given specification. The scenario should adhere both to the topic and the indicated user intent (if exists) in the spec.

    **Specification:** {spec}

    Make sure that the {n_topics} scenarios are different from each other and cover a diverse range of typical, common scenarios which fall under the given specification. Sometimes the user prompt specification might include unsafe or unethical topics, and it is important that your generated scenarios should still fall under the given specification. 
    
    Make sure that your scenarios aren't overly specific and do not provide too many restrictive details, and make sure they are realistic scenarios that a user might actually ask a chatbot. In addition, make sure that user prompt scenarios are not too difficult for a usual chatbot to answer, and that they do not require responses that are overly long or complex.

    Use your thinking budget to brainstorm carefully and creatively. Make sure to explicitly verify that the scenarios you propose satisfy the above requirements. Then, in your output field, return ONLY your list of {n_topics} scenarios formatted as a python array, like this:

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

    You will be given a description of a concrete user scenario. Your task is to write a list of {n_prompts} different user prompts that the user might ask in this scenario. You are also given the broad spec, describing the broad topic and/or user intent that this scenario falls under. Your user prompt should also fall under this spec.
    
    The user prompts you write should vary naturally in terms of style and tone, and that they are actually questions that a real user might ask a chatbot. It is important that the user prompts faithfully fall under the scenario description, and not deviate from them. In addition, very importantly, make sure that the user prompts do not require responses that are too long or complex. They should be simple questions that are able to be answered by a usual chatbot assistant in at most a few paragraphs.
    
    Keep in mind also that the user prompts you write will be the entirety of the user's message, so also include any additional contexts referred to in the prompt. For example, if the topic is "write a summary of a given document", then the user prompt should also include the full text of the document that the user is asking about.

    **Scenario description:** {topic}

    **Broad spec:** {spec}

    Use your thinking budget to reason carefully. Make sure to explicitly verify that the user prompts you write satisfy the above requirements. Then, in your output field, return ONLY your list of {n_prompts} user prompts formatted as a python array, like this:

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
):
    specs_path = dataset_path / "specs.json"
    with open(specs_path, "r") as f:
        specs: list[str] = json.load(f)

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
    
    with open(dataset_path / "sub_topics.json", "w") as f:
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
                spec=spec,
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
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--n_topics", type=int, default=32)
    parser.add_argument("--n_prompts", type=int, default=3)
    args = parser.parse_args()

    asyncio.run(main(
        dataset_path=Path(args.dataset_path),
        model="openai/gpt-5",
        n_topics=args.n_topics,
        n_prompts=args.n_prompts,
        max_tokens=12000,
        reasoning="medium",
    ))