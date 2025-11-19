"""
Synthetic dataset generation.

Inspired by "how people use ChatGPT".
"""

# %%

import json
import textwrap
import asyncio
import nest_asyncio
from pathlib import Path
from dataclasses import asdict

from caller import ChatHistory
from state import PromptCluster
from utils import parse_json_response
from models import CACHE_CONFIG, RETRY_CONFIG
from caller import OpenRouterCaller, CacheConfig, RetryConfig

nest_asyncio.apply()


caller = OpenRouterCaller(cache_config=CACHE_CONFIG, retry_config=RETRY_CONFIG, dotenv_path=".env")


CATEGORIES = {
    "Writing": [
        "Edit or Critique User-Provided Text",
        "Personal Writing or Communication",
        "Translation",
        "Argument or Summary Generation",
        "Write Fiction"
    ],
    "Practical Guidance": [
        "How-To Advice",
        "Tutoring or Teaching the User",
        "Creative Ideation",
        "Health, Fitness, Beauty, or Self-Care"
    ],
    "Technical Help": [
        "Mathematical Calculation",
        "Data Analysis",
        "Computer Programming"
    ],
    "Seeking Information": [
        "Specific Info",
        "Purchasable Products",
        "Cooking and Recipes"
    ],
    "Self-Expression": [
        "Greetings and Chitchat",
        "Relationships and Personal Reflection",
        "Games and Role Play"
    ],
    "Other": [
        "Asking About the Model"
    ]
}

INTENTS = {
    "Asking": "Seeking information or advice for the user to be better informed on something.", 
    "Doing": "Requesting ChatGPT to perform a task, generating output that is created primarily by the model.", 
    "Expressing": "Expressing statements are neither asking for information, nor for the chatbot to perform a task."
}


SPECIFIC_CATEGORY_PROMPT = textwrap.dedent(
    """
    Your task is to brainstorm three specific sub-categories of the given user prompt category. Please think about what typical user prompts under the given category would look like, and output the three most common sub-categories. They should fall under the given category, be diverse from each other, but also not too narrow. They should also not be overly difficult for a chatbot to answer, and do not require responses that are too long or complex. Your topic descriptions should be a **simple, short phrase**.

    After thinking, output the three sub-categories as a JSON array, like this:

    ```json
    [
        "First sub-category",
        "Second sub-category",
        "Third sub-category",
    ]
    ```

    Here is the given user prompt category:

    <category>
    {category}
    </category>
    """
).strip()

# %%

# run this once and save
def make_sub_topics(ds_name: str) -> dict:
    categories = []
    for k, vals in CATEGORIES.items():
        categories.extend("{}: {}".format(k, v) for v in vals)

    messages = [
        ChatHistory.from_user(SPECIFIC_CATEGORY_PROMPT.format(category=category))
        for category in categories
    ]
    responses = asyncio.run(caller.call(
        messages=messages,
        max_parallel=256,
        model="openai/gpt-5",
        desc="Making sub-topics",
        max_tokens=8192,
        reasoning="medium",
    ))

    sub_topics = dict()
    for i in range(len(categories)):
        category = categories[i]
        response = responses[i]
        assert response is not None

        topics, _ = parse_json_response(response, log_json_error=False)
        sub_topics[category] = topics
    
    Path(f"data/{ds_name}").mkdir(parents=True, exist_ok=True)
    with open(f"data/{ds_name}/sub_topics.json", "w") as f:
        json.dump(sub_topics, f, indent=4)

    return sub_topics

# %%

def make_all_specs(ds_name: str) -> list[str]:
    sub_topics = make_sub_topics(ds_name)

    all_specs: list[str] = []

    for category, topics in sub_topics.items():
        all_specs.append(category)
        for topic in topics:
            all_specs.append("{}: {}".format(category, topic))

    for intent_name, intent_desc in INTENTS.items():
        for broad_topic, topics in CATEGORIES.items():
            if broad_topic == "Other":
                continue

            all_specs.append(
                "Category: {} (such as but not limited to: {})\\nIntent: {}: {}".format(
                    broad_topic, ", ".join(topics), intent_name, intent_desc
                )
            )

    with open(f"data/{ds_name}/all_specs.txt", "w") as f:
        for spec in all_specs:
            f.write(spec + "\n")
    
    return all_specs


# %%

BRAINSTORM_SUB_TOPIC_PROMPT = textwrap.dedent(
    """
    You are an expert in brainstorming realistic sub-topics for a given topic. You are an important component of a pipeline that generates diverse, realistic user prompts starting from a short specification.

    Your current task is to brainstorm a list of {n_topics} possible sub-topics for user prompts that fall under the given specification. Each sub-topic should be a **simple, short phrase** describing a concrete class of user prompts that belong to the given specification.
    
    Another model will then use your description to generate actual user prompts, so make sure that your sub-topics aren't overly specific. In addition, make sure that your sub-topics are not too difficult for a chatbot to answer, and do not require responses that are too long or complex.
    
    Make sure that the sub-topics are different from each other and cover a diverse range of typical, common sub-topics of the given topic, while also covering a similarly broad range of topics as each other. It is important that the sub-topics accurately fall under the specification, and not deviate from it.

    **Specification:** {spec}

    Think carefully and creatively, and then in your output field return ONLY your list of {n_topics} sub-topics formatted as a JSON array, like this:

    ```json
    [
        "Your first sub-topic here",
        "Your second sub-topic here",
        ...
    ]
    ```
    """
).strip()


BRAINSTORM_INTENT_PROMPT = textwrap.dedent(
    """
    You are an expert in brainstorming realistic topic-specific user intents for a given topic. You are an important component of a pipeline that generates diverse, realistic user prompts starting from a short specification.

    Your current task is to brainstorm a list of {n_intents} possible user intents for user prompts that fall under the given specification. Each user intent should be a **simple, short phrase** describing a concrete type of user request and the type of answer the user is asking for. Note that this should not include any information about the concrete topic of the question. This intent should be reasonably compatible with the provided spec.
    
    Another model will then use your description to generate actual user prompts, so make sure that your user intent descriptions aren't overly specific. In addition, make sure that these intents are not too difficult for a chatbot to fulfill, and do not require responses that are too long or complex.
    
    Make sure that the intents are different from each other and cover a diverse range of typical, common user intents for questions under the given specification.

    **Specification:** {spec}

    Think carefully and creatively, and then in your output field return ONLY your list of {n_intents} user intents formatted as a JSON array, like this:

    ```json
    [
        "Your first user intent here",
        "Your second user intent here",
        ...
    ]
    ```
    """
).strip()


GENERATION_PROMPT = textwrap.dedent(
    """
    You are an average chatbot user writing prompts following a given topic description and user intent description. You are an important component of a pipeline that generates diverse, realistic user prompts.

    You will be given both a topic description and a user intent description. Your task is to write a list of {n_prompts} different user prompts that fall under the given topic and also aligns with the user intent.
    
    The user prompts you write should vary naturally in terms of style and tone, and they should be phrased in similar ways that real users would prompt a chatbot. It is important that the user prompts faithfully fall under the topic and intent descriptions, and not deviate from them. In addition, make sure that the user prompts do not require responses that are too long or complex. They should be able to be answered by a usual chatbot assistant in at most a few paragraphs.
    
    Keep in mind also that the user prompts you write will be the entirety of the user's message, so also include any additional contexts referred to in the prompt. For example, if the topic is "write a summary of a given document", then the user prompt should also include the full text of the document that the user is asking about.

    **Topic description:** {topic}

    **User intent description:** {intent}

    Think carefully, and then in your output field return ONLY your list of {n_prompts} user prompts formatted as a JSON array, like this:

    ```json
    [
        "Your first user prompt here",
        "Your second user prompt here",
        ...
    ]
    ```
    """
).strip()


# %%
async def main(
    specs: list[str],
    model: str = "openai/gpt-5",
    n_topics: int = 8,
    n_intents: int = 8,
    n_prompts: int = 2,
    max_tokens: int = 15000,
    reasoning: str | int | None = "high",
    ds_name: str = "synthetic",
) -> dict[int, PromptCluster]:
    sub_topic_chats = [
        ChatHistory().add_user(BRAINSTORM_SUB_TOPIC_PROMPT.format(spec=spec, n_topics=n_topics))
        for spec in specs
    ]
    intent_chats = [
        ChatHistory().add_user(BRAINSTORM_INTENT_PROMPT.format(spec=spec, n_intents=n_intents))
        for spec in specs
    ]

    brainstorm_responses = await caller.call(
        messages=sub_topic_chats + intent_chats,
        max_parallel=128,
        desc="Brainstorming",
        model=model,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )

    coords = dict()
    for spec in specs:
        coords[spec] = {
            "sub_topics": [],
            "intents": [],
        }

    for i, resp in enumerate(brainstorm_responses):
        spec = specs[i % len(specs)]
        if resp is None:
            continue

        if i < len(specs):
            sub_topics, _ = parse_json_response(resp)
            if not isinstance(sub_topics, list):
                print(f"Error: sub_topics is not a list.\n{sub_topics}\nSkipping...")
                continue
            sub_topics = [topic.strip() for topic in sub_topics]
            coords[spec]["sub_topics"] = sub_topics
        else:
            intents, _ = parse_json_response(resp)
            if not isinstance(intents, list):
                print(f"Error: intents is not a list.\n{intents}\nSkipping...")
                continue
            intents = [intent.strip() for intent in intents]
            coords[spec]["intents"] = intents
    
    with open(f"data/{ds_name}/coords.json", "w") as f:
        json.dump(coords, f, indent=4)

    results: dict[int, PromptCluster] = {
        i: PromptCluster(
            summary=specs[i],
            prompts=[],
        )
        for i in range(len(specs))
    }

    prompt_generation_chats = [
        ChatHistory().add_user(
            GENERATION_PROMPT.format(
                n_prompts=n_prompts,
                topic=topic,
                intent=intent,
            )
        )
        for spec in specs
        for topic in coords[spec]["sub_topics"]
        for intent in coords[spec]["intents"]
    ]
    prompt_cluster_ids = [
        i for i, spec in enumerate(specs) 
        for _ in coords[spec]["sub_topics"]
        for _ in coords[spec]["intents"]
    ]

    prompt_generation_responses = await caller.call(
        messages=prompt_generation_chats,
        max_parallel=128,
        desc="Generating user prompts",
        model=model,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )

    for i, resp in enumerate(prompt_generation_responses):
        if resp is None:
            continue
        user_prompts, _ = parse_json_response(resp, log_json_error=False)
        if not isinstance(user_prompts, list):
            print(f"Error: user_prompts is not a list.\n{user_prompts}\nSkipping...")
            continue
        user_prompts = [prompt.strip() for prompt in user_prompts]
        results[prompt_cluster_ids[i]].prompts.extend(user_prompts)

    # write results
    Path(f"data/{ds_name}").mkdir(parents=True, exist_ok=True)
    for i, cluster in results.items():
        with open(f"data/{ds_name}/{i}.json", "w") as f:
            json.dump(asdict(cluster), f, indent=4)

    return results


# %%
if __name__ == "__main__":
    import random
    ds_name = "synthetic_0"
    all_specs = make_all_specs(ds_name)

    asyncio.run(
        main(
            specs=all_specs, n_topics=8, n_intents=8, n_prompts=2,
            max_tokens=8192, reasoning="medium", ds_name=ds_name
        )
    )



# %%
# # Goal: a complete pipeline to generate diverse user prompts from only a description

# CLUSTER_DESCRIPTIONS = [
#     "asking for hints or solutions with high school-level math problems",
#     "common programming debugging in python asking to explain the error",
#     "requests for stocks market or cryptocurrency advice",
#     "content creation ideas for tiktok",
#     "asking for creative fiction story ideas",
#     "asking to generate poetry",
#     "requests for explainer on a given technical scientific topic",
#     "generating webpages with basic UI elements",
#     "daily how-to guides for everyday tasks",
#     "emotional counseling and relationship advice",
#     "medical advice for common health issues",
#     "questions about the model's identity and experience",
#     "explanation of basic scientific concepts in simple language",
#     "seeking specific information about a well-known event or topic given by the user",
#     "edit or critique content written by the user",
#     "generating replies for a email given by the user",
#     "asks for assistance with common unethical behavior",
# ]

# CLAUDE_CLUSTER_LIST = [
#     # Narrow/specific domains
#     "translation between specific language pairs with cultural context",
#     "debugging and troubleshooting common household tech issues",
#     "recipe suggestions and detailed cooking instructions",
#     "pet training and animal behavior advice",
#     "game strategy walkthroughs and tips for popular video games",
#     "interpretation of dreams or personal symbolic experiences",
#     "fashion and outfit coordination advice",
#     "gardening and plant care for specific species",
#     # Medium breadth
#     "career transition advice and resume/interview preparation",
#     "travel itinerary planning for specific destinations or trip types",
#     "nutrition meal planning and diet recommendations",
#     "academic research methodology and study design questions",
#     "social media marketing strategy and content calendars",
#     "fitness workout programming and exercise form checks",
#     "DIY home improvement and repair instructions",
#     "product comparisons and purchase recommendations",
#     # Broader domains
#     "parenting advice for developmental stages or behavioral issues",
#     "financial planning, budgeting, and personal finance strategy",
#     "legal advice for personal legal situations or disputes",
#     "historical analysis or interpretation of controversial periods/events",
#     "philosophical debates on ethical or moral dilemmas",
#     "political commentary on current events or policy positions",
#     "business strategy and entrepreneurship guidance",
#     "language learning pedagogy and grammar explanations",
#     "requests for extensive summaries or distillations of long content",
#     "generating formal professional documents like contracts or reports",
# ]
