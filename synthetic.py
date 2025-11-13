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
from caller import OpenRouterCaller, CacheConfig, RetryConfig

nest_asyncio.apply()


cache_config = CacheConfig(
    no_cache_models={
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
    }
)

retry_config = RetryConfig(
    raise_when_exhausted=False,
    criteria=lambda response: response.has_response
    and response.finish_reason == "stop",
    max_attempts=3,
)

caller = OpenRouterCaller(cache_config=cache_config, retry_config=retry_config, dotenv_path=".env")


CATEGORIES = {
    "Writing": [
        "Edit or Critique Provided Text",
        "Personal Writing or Communication",
        "Translation",
        "Argument or Summary Generation",
        "Write Fiction"
    ],
    "Practical Guidance": [
        "How-To Advice",
        "Tutoring or Teaching",
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
    Your task is to brainstorm three specific sub-categories of the given user prompt category. Please think about what typical user prompts under the given category would look like, and output the three most common sub-categories. They should fall under the given category, be diverse from each other, but also not too narrow. Your topic descriptions should be a short phrase.

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

print(SPECIFIC_CATEGORY_PROMPT.format(category="Self-Expression"))

# %%

# run this once and save
def make_sub_topics() -> dict:
    categories = []
    for k, vals in CATEGORIES.items():
        categories.extend("{}: {}".format(k, v) for v in vals)

    messages = [
        ChatHistory.from_user(SPECIFIC_CATEGORY_PROMPT.format(category=category))
        for category in categories
    ]
    responses = asyncio.run(caller.call(
        messages=messages,
        max_parallel=128,
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
    
    with open("data/synthetic/sub_topics.json", "w") as f:
        json.dump(sub_topics, f, indent=4)

    return sub_topics

# %%
sub_topics = make_sub_topics()

# %%

ALL_SPECS: list[str] = []

for category, topics in sub_topics.items():
    ALL_SPECS.append(category)
    for topic in topics:
        ALL_SPECS.append("{}: {}".format(category, topic))

for intent_name, intent_desc in INTENTS.items():
    for broad_topic, topics in CATEGORIES.items():
        if broad_topic == "Other":
            continue

        ALL_SPECS.append(
            "Category: {} (such as but not limited to: {})\\nIntent: {}: {}".format(
                broad_topic, ", ".join(topics), intent_name, intent_desc
            )
        )

with open("data/synthetic/all_specs.txt", "w") as f:
    for spec in ALL_SPECS:
        f.write(spec + "\n")


# %%

BRAINSTORM_SUB_TOPIC_PROMPT = textwrap.dedent(
    """
    You are an expert in brainstorming realistic sub-topics for a given topic. You are an important component of a pipeline that generates diverse, realistic user prompts starting from a short specification.

    Your current task is to brainstorm a list of {n_topics} possible sub-topics for user prompts that fall under the given specification. Each sub-topic should be a short phrase describing a concrete class of user prompts that belong to the given specification.
    
    Another model will then use your description to generate actual user prompts, so make sure that your sub-topics aren't overly broad/vague or overly specific.
    
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

    Your current task is to brainstorm a list of {n_intents} possible user intents for user prompts that fall under the given specification. Each user intent should be a short phrase describing a concrete type of user request and what the user is asking for. This intent should be reasonably compatible with the provided spec.
    
    Another model will then use your description to generate actual user prompts, so make sure that your user intent descriptions aren't overly broad/vague or overly specific.
    
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
    
    The user prompts you write should vary naturally in terms of style and tone, and they should be phrased in similar ways that real users would prompt a chatbot. It is important that the user prompts faithfully fall under the topic and intent descriptions, and not deviate from them.
    
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
    topics: list[str],
    model: str = "anthropic/claude-opus-4.1",
    n_topics: int = 5,
    n_prompts: int = 5,
    max_tokens: int = 30000,
    reasoning: str | int | None = 8192,
    ds_name: str = "synthetic",
) -> dict[int, PromptCluster]:
    sub_topics_chats = [
        ChatHistory().add_user(BRAINSTORM_PROMPT.format(topic=topic, n_topics=n_topics))
        for topic in topics
    ]

    response = await caller.call(
        messages=sub_topics_chats,
        max_parallel=128,
        desc="Brainstorming sub-topics",
        model=model,
        max_tokens=max_tokens,
        reasoning=reasoning,
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

    results: dict[int, PromptCluster] = {
        i: PromptCluster(
            summary=topics[i],
            prompts=[],
        )
        for i in range(len(topics))
    }

    prompt_generation_chats = [
        ChatHistory().add_user(
            GENERATION_PROMPT.format(
                topic=topics[all_sub_topics_to_topic[i]],
                sub_topic=sub_topic,
                n_prompts=n_prompts,
            )
        )
        for i, sub_topic in enumerate(all_sub_topics)
    ]

    async with Caller(dotenv_path=".env") as caller:
        response = await caller.call(
            messages=prompt_generation_chats,
            max_parallel=128,
            desc="Generating user prompts",
            model=model,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )

    for i, resp in enumerate(response):
        user_prompts, _ = parse_json_response(resp, log_json_error=False)
        if isinstance(user_prompts, list):
            user_prompts = [prompt.strip() for prompt in user_prompts]
        elif isinstance(user_prompts, str):
            print("USER PROMPT====")
            print(user_prompts)
            print("====USER PROMPT")
            print(f"Error: user_prompts is not a list. Skipping...")
            continue
        results[all_sub_topics_to_topic[i]].prompts.extend(user_prompts)

    # write results
    Path(f"data/{ds_name}").mkdir(parents=True, exist_ok=True)
    for i, cluster in results.items():
        with open(f"data/{ds_name}/{i}.json", "w") as f:
            json.dump(asdict(cluster), f, indent=4)

    return results


# %%
if __name__ == "__main__":
    asyncio.run(
        two_stage_main(
            topics=CLUSTER_DESCRIPTIONS, n_topics=32, n_prompts=4, ds_name="synthetic_2"
        )
    )
    # asyncio.run(one_stage_main(topics=CLUSTER_DESCRIPTIONS, n_prompts=128, ds_name="synthetic_2"))





# %%
# Goal: a complete pipeline to generate diverse user prompts from only a description

CLUSTER_DESCRIPTIONS = [
    "asking for hints or solutions with high school-level math problems",
    "common programming debugging in python asking to explain the error",
    "requests for stocks market or cryptocurrency advice",
    "content creation ideas for tiktok",
    "asking for creative fiction story ideas",
    "asking to generate poetry",
    "requests for explainer on a given technical scientific topic",
    "generating webpages with basic UI elements",
    "daily how-to guides for everyday tasks",
    "emotional counseling and relationship advice",
    "medical advice for common health issues",
    "questions about the model's identity and experience",
    "explanation of basic scientific concepts in simple language",
    "seeking specific information about a well-known event or topic given by the user",
    "edit or critique content written by the user",
    "generating replies for a email given by the user",
    "asks for assistance with common unethical behavior",
]

CLAUDE_CLUSTER_LIST = [
    # Narrow/specific domains
    "translation between specific language pairs with cultural context",
    "debugging and troubleshooting common household tech issues",
    "recipe suggestions and detailed cooking instructions",
    "pet training and animal behavior advice",
    "game strategy walkthroughs and tips for popular video games",
    "interpretation of dreams or personal symbolic experiences",
    "fashion and outfit coordination advice",
    "gardening and plant care for specific species",
    # Medium breadth
    "career transition advice and resume/interview preparation",
    "travel itinerary planning for specific destinations or trip types",
    "nutrition meal planning and diet recommendations",
    "academic research methodology and study design questions",
    "social media marketing strategy and content calendars",
    "fitness workout programming and exercise form checks",
    "DIY home improvement and repair instructions",
    "product comparisons and purchase recommendations",
    # Broader domains
    "parenting advice for developmental stages or behavioral issues",
    "financial planning, budgeting, and personal finance strategy",
    "legal advice for personal legal situations or disputes",
    "historical analysis or interpretation of controversial periods/events",
    "philosophical debates on ethical or moral dilemmas",
    "political commentary on current events or policy positions",
    "business strategy and entrepreneurship guidance",
    "language learning pedagogy and grammar explanations",
    "requests for extensive summaries or distillations of long content",
    "generating formal professional documents like contracts or reports",
]
