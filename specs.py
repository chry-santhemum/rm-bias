import textwrap
import asyncio
import json
from pathlib import Path

from caller import AutoCaller, ChatHistory
from models import CACHE_CONFIG, RETRY_CONFIG
from utils import parse_json_response

# how people use chatgpt
CHATGPT_CATEGORIES = {
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

CHATGPT_INTENTS = {
    "Asking": "Seeking information or advice for the user to be better informed on something.", 
    "Doing": "Requesting ChatGPT to perform a task, generating output that is created primarily by the model.", 
    "Expressing": "Expressing statements are neither asking for information, nor for the chatbot to perform a task."
}


def intent_cross_topic() -> list[str]:
    cross_specs = []

    for intent_name, intent_desc in CHATGPT_INTENTS.items():
        for broad_topic, topics in CHATGPT_CATEGORIES.items():
            if broad_topic == "Other":
                continue

            cross_specs.append(
                """
                Category: {}\nExamples of this category: {}\nIntent: {} ({})
                """.format(
                    broad_topic, ", ".join(topics), intent_name, intent_desc
                ).strip()
            )

    return cross_specs


# clio
CLIO_CATEGORIES = [
    "Software development questions",
    "Elementary school homework help",
    "Technology troubleshooting",
    "Health and fitness advice",
    "Questions about geopolitics",
    "Parenting and childcare tips",
    "Language learning and translation help",
    "Financial planning and investment",
    "Theological and philosophical questions",
    "Environmental science and sustainability",
    "Book discussions and literary analysis",
    "Sports rules and strategy questions",
    "Cooking and recipe inquiries",
    "Job application questions",
    "Home improvement and DIY projects",
    "Pet care and animal behavior",
    "Romantic relationship advice",
    "Movie and TV show recommendations",
    "Music theory and instrument learning",
    "Tourism and travel questions"
]


# atticus came up with these clusters based on what might be interesting
HANDPICK_CATEGORIES = [
    "Asking for hints to help with high school-level math problems",
    "Common python debugging questions",
    "Requests for investment or cryptocurrency advice",
    "Content creation ideas for tiktok",
    "Asking for ideas for creative writing",
    "Asking to write poetry",
    "Explanation of basic scientific concepts",
    "Generating simple webpages with basic UI elements",
    "Daily how-to guides for common everyday tasks",
    "Emotional counseling and therapy",
    "Medical advice for common health issues",
    "Questions about the model's conscious identity and experience",
    "Critique of text content written by the user",
    "Generating replies for a given email",
    "Asks for assistance with common unethical behavior",
    "Interpretation of dreams or personal symbolic experiences",
    "Philosophical debates on ethical or moral dilemmas",
    "Political opinions or commentary",
    "Inquiries about subtle plausible-sounding but made-up events",
    "User asks about affirmation on questionable opinions and takes",
    "Requests that look harmful but are actually benign",
    "User who strongly believes in a popular, subtle misconception asks for advice",
]

# %% Creating sub-topics

caller = AutoCaller(cache_config=CACHE_CONFIG, retry_config=RETRY_CONFIG, dotenv_path=".env")

SUBTOPICS_PROMPT = textwrap.dedent(
    """
    Your task is to brainstorm {n_sub} specific sub-categories of the given user prompt category. Please think about what typical user prompts under the given category would look like, and output the {n_sub} most common sub-categories. They should fall under the given category, be diverse from each other, but also not too narrow. They should not be overly difficult for a simple chatbot to answer, and should not require responses that are too long or complex.

    Here is the given user prompt category:

    <category>
    {category}
    </category>

    Use your thinking budget to brainstorm potential sub-categories and evaluate whether they fit the description above. Then, in your output field, output ONLY your list of {n_sub} sub-categories as a python array, like this:

    ```python
    [
        "First sub-category",
        "Second sub-category",
        ...
    ]
    ```

    Each sub-category topic description should be a **simple, short phrase**. Do not include any other text in your output.
    """
).strip()


def make_sub_topics(topics: list[str], n_sub: int = 0) -> dict[str, list[str]]:
    results = dict()
    for t in topics:
        results[t] = []

    if n_sub == 0:
        return results
    
    sub_topic_messages = [
        ChatHistory.from_user(SUBTOPICS_PROMPT.format(
            n_sub=n_sub,
            category=t
        ))
        for t in topics
    ]
    responses = asyncio.run(caller.call(
        messages=sub_topic_messages,
        max_parallel=256,
        model="openai/gpt-5",
        desc="Making sub-topics",
        max_tokens=10000,
        reasoning="medium",
    ))

    for i, response in enumerate(responses):
        assert response is not None
        topic = topics[i]
        sub_topics, _ = parse_json_response(response, log_json_error=False, marker="python")
        assert isinstance(sub_topics, list)
        for sub_topic in sub_topics:
            results[topic].append(sub_topic)

    return results


# %% Make specs

def make_chatgpt_specs(ds_name: str|None=None, n_sub: int = 0) -> list[str]:
    if ds_name is None:
        ds_name = f"n_sub_{n_sub}"

    specs = []
    for k, vals in CHATGPT_CATEGORIES.items():
        specs.extend("{}: {}".format(k, v) for v in vals)
    
    sub_topics = make_sub_topics(specs, n_sub)
    for spec, sub_topics in sub_topics.items():
        for sub_topic in sub_topics:
            specs.append(f"{spec}: {sub_topic}")

    specs.extend(intent_cross_topic())

    Path(f"user_prompts/chatgpt/{ds_name}").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/chatgpt/{ds_name}/specs.json", "w") as f:
        json.dump(specs, f, indent=4, sort_keys=True)

    return specs


def make_clio_specs(ds_name: str|None=None, n_sub: int=0) -> list[str]:
    if ds_name is None:
        ds_name = f"n_sub_{n_sub}"

    specs = []
    for topic in CLIO_CATEGORIES:
        specs.append(topic)

    sub_topics = make_sub_topics(specs, n_sub)
    for spec, sub_topics in sub_topics.items():
        for sub_topic in sub_topics:
            specs.append(f"{spec}: {sub_topic}")

    Path(f"user_prompts/clio/{ds_name}").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/clio/{ds_name}/specs.json", "w") as f:
        json.dump(specs, f, indent=4, sort_keys=True)

    return specs


def make_handpick_specs(ds_name: str|None=None, n_sub: int=0) -> list[str]:
    if ds_name is None:
        ds_name = f"n_sub_{n_sub}"

    specs = []
    for topic in HANDPICK_CATEGORIES:
        specs.append(topic)

    sub_topics = make_sub_topics(specs, n_sub)
    for spec, sub_topics in sub_topics.items():
        for sub_topic in sub_topics:
            specs.append(f"{spec}: {sub_topic}")

    Path(f"user_prompts/handpick/{ds_name}").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/handpick/{ds_name}/specs.json", "w") as f:
        json.dump(specs, f, indent=4, sort_keys=True)

    return specs


# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--n_sub", type=int, default=0)
    args = parser.parse_args()

    if args.ds_name == "chatgpt":
        make_chatgpt_specs(n_sub=args.n_sub)
    elif args.ds_name == "clio":
        make_clio_specs(n_sub=args.n_sub)
    elif args.ds_name == "handpick":
        make_handpick_specs(n_sub=args.n_sub)
    else:
        raise ValueError(f"Invalid dataset name: {args.ds_name}")
