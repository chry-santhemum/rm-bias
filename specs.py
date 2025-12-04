import textwrap
import asyncio
import json
from pathlib import Path

from caller import AutoCaller, ChatHistory
from models import CACHE_CONFIG, RETRY_CONFIG
from utils import parse_json_response

caller = AutoCaller(cache_config=CACHE_CONFIG, retry_config=RETRY_CONFIG, dotenv_path=".env")


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
                Category: {}
                Examples of this category: {}
                Intent: {} ({})
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
HANDPICKED = [
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



def make_chatgpt_specs(ds_name: str, n_sub: int = 0) -> dict:
    categories = []
    for k, vals in CHATGPT_CATEGORIES.items():
        categories.extend("{}: {}".format(k, v) for v in vals)

    messages = [
        ChatHistory.from_user(SUBTOPICS_PROMPT.format(
            n_sub=n_sub,
            category=category
        ))
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