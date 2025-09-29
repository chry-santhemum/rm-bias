"""Synthetic dataset inspired by Wildchat clusters."""

# %%
import patches
import json
import textwrap
import asyncio
import nest_asyncio
from pathlib import Path
from dataclasses import asdict

from llm_types import ChatHistory
from state import PromptCluster
from utils import get_to_pass_reasoning, parse_json_response
from client import get_universal_caller, sample_from_model, sample_from_model_parallel

nest_asyncio.apply()

# Inspired by "how people use ChatGPT"
# User intent: asking, doing, expressing

# Goal: a complete pipeline to generate diverse user prompts from only a description

CLUSTER_DESCRIPTIONS = [
    "help with high school math problems",
    "help with common programming debugging in python",
    "requests for trading / crypto advice",
    "content creation ideas for tiktok",
    "creative fiction writing",
    "poetry writing",
    "expository writing on a technical topic",
    "generating webpages with basic UI elements",
    "daily how-to guides for everyday tasks",
    "emotional counseling and relationship advice",
    "medical advice for common health issues",
    "questions about the model's identity and experience",
    "explanation of basic scientific concepts",
    "seeking specific information about an event or topic",
    "edit or critique written content",
    "generating summaries for a given document",
]


BRAINSTORM_PROMPT = textwrap.dedent("""
    You are an expert in brainstorming realistic sub-topics for a given topic. You are an important component of a pipeline that generates diverse user prompts starting from a description of a short topic.

    Your current task is to brainstorm a list of {n_topics} possible sub-topics for user prompts that fall under the given topic given below. Each sub-topic should be a short phrase describing a concrete cluster of user prompts that belong to the given topic; another model will then use your description to generate actual user prompts. 
    
    Make sure that the sub-topics are different from each other and cover a wide range of possible sub-topics, but with more focus on more conventional sub-topics. It is possible that the topic might be requests for harmful or offensive content, in which case it is important that the sub-topics still accurately follow the topic, and not deviate from it due to safety concerns.

    **Topic:** {topic}

    Think carefully, and then in your output field return ONLY your list of sub-topics formatted as a JSON array, like this:

    ```json
    [
        "Your first sub-topic here",
        "Your second sub-topic here",
        ...
    ]
    ```
""").strip()


GENERATION_PROMPT = textwrap.dedent("""
    You are an average chatbot user writing prompts about a given topic. You are an important component of a pipeline that generates diverse user prompts starting from a description of a short topic.

    Your task is to write a list of {n_prompts} different user prompts that fall under the given topic.
    
    The user prompts you write should vary in terms of style and tone, and they should be phrased in a more casual and less detailed and polished way, and the instructions should often not be too clear and specific, but rather in the same ways that real users would prompt a chatbot. It is possible that the topic might be requests for harmful or offensive content, in which case it is important that the user prompts still faithfully belong to the topic, and not deviate from it due to safety concerns.
    
    Keep in mind also that the user prompts you write will be the entirety of the user's message, so also include any additional contexts referred to in the prompt. For example, if the topic is "write a summary of a given document", then the user prompt should include the text of the document that the user is asking about.

    **Topic:** {topic}

    Think carefully, and then in your output field return ONLY your list of user prompts formatted as a JSON array, like this:

    ```json
    [
        "Your first user prompt here",
        "Your second user prompt here",
        ...
    ]
    ```

    In each entry, please only include the user prompt itself, and no other text describing the user prompt.
""").strip()


print(BRAINSTORM_PROMPT.format(topic=CLUSTER_DESCRIPTIONS[0], n_topics=20))
print(GENERATION_PROMPT.format(topic=CLUSTER_DESCRIPTIONS[0], n_prompts=5))

# %%
async def one_stage_main():
    pass

async def two_stage_main(
    topics: list[str],
    model: str = "openai/gpt-5",
    max_tokens: int = 8192,
    reasoning: str | int | None = "medium",
) -> dict[int, PromptCluster]:
    caller = get_universal_caller()
    sub_topics_chats = [
        ChatHistory().add_user(
            BRAINSTORM_PROMPT.format(topic=topic, n_topics=5)
        )
        for topic in topics
    ]
    response = await sample_from_model_parallel(
        prompts=sub_topics_chats,
        caller=caller,
        max_par=32,
        model=model,
        max_tokens=max_tokens,
        reasoning=get_to_pass_reasoning(reasoning, max_tokens),
    )

    all_sub_topics: list[str] = []
    all_sub_topics_to_topic: list[int] = []

    for i, resp in enumerate(response):
        sub_topics, _ = parse_json_response(resp, log_json_error=False)
        sub_topics = [topic.strip() for topic in sub_topics]
        all_sub_topics.extend(sub_topics)
        all_sub_topics_to_topic.extend([i for _ in range(len(sub_topics))])
        print("Subtopics:\n")
        print(sub_topics)

    results: dict[int, PromptCluster] = {i: PromptCluster(
        summary=topics[i],
        prompts=[],
    ) for i in range(len(topics))}

    prompt_generation_chats = [
        ChatHistory().add_user(
            GENERATION_PROMPT.format(topic=topic, n_prompts=5)
        )
        for topic in all_sub_topics
    ]

    response = await sample_from_model_parallel(
        prompts=prompt_generation_chats,
        caller=caller,
        max_par=32,
        model=model,
        max_tokens=max_tokens,
        reasoning=get_to_pass_reasoning(reasoning, max_tokens),
    )

    for i, resp in enumerate(response):
        user_prompts, _ = parse_json_response(resp, log_json_error=False)
        user_prompts = [prompt.strip() for prompt in user_prompts]
        results[all_sub_topics_to_topic[i]].prompts.extend(user_prompts)

    # write results
    Path("data/synthetic").mkdir(parents=True, exist_ok=True)
    for i, cluster in results.items():
        with open(f"data/synthetic/{i}.json", "w") as f:
            json.dump(asdict(cluster), f, indent=4)

    return results

if __name__ == "__main__":
    asyncio.run(two_stage_main(CLUSTER_DESCRIPTIONS))
# %%
