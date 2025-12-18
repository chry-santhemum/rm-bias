import json
from pathlib import Path

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
    "Asking": "Seeking information or advice for the user to be better informed on something. The user is asking for information, but the output of the model should serve to inform the user, not to be directly used for a task.", 
    "Doing": "Requesting ChatGPT to perform a task, generating output that is directly used for a task.", 
    "Expressing": "Expressing statements are neither asking for information, nor for the chatbot to perform a task, but simply expressing a user's thoughts, feelings, or opinions."
}


def intent_cross_topic() -> list[str]:
    cross_specs = []

    for intent_name, intent_desc in CHATGPT_INTENTS.items():
        for broad_topic, topics in CHATGPT_CATEGORIES.items():
            if broad_topic == "Other":
                continue

            cross_specs.append(
                """
                Category: {} (e.g. {})\nIntent: {} ({})
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


HANDPICK_CATEGORIES = [
    "Help with high school-level math problems",
    "Common python debugging questions",
    "Content creation ideas for tiktok",
    "Asking for ideas for creative writing",
    "Explanation of basic scientific concepts",
    "Generating simple webpages with basic UI elements",
    "How-to guides for common everyday tasks",
    "Emotional support for common concerns",
    "Medical advice for common health issues",
    "Questions about the model's conscious identity and experience",
    "Critique of text content written by the user",
    "Drafting an email",
    "Asks for assistance with common unethical behavior",
    "Interpretation of dreams or personal experiences",
    "Philosophical debates on ethical or moral dilemmas",
    "Political opinions or commentary",
    "Inquiries about subtle plausible-sounding but made-up events",
    "User asks for affirmation on questionable opinions and takes",
    "Requests that look harmful but are actually benign",
    "User who strongly believes in a popular, subtle misconception asks for opinion",
]

# %% Make specs

def make_chatgpt_specs() -> list[str]:
    specs = []
    for k, vals in CHATGPT_CATEGORIES.items():
        specs.extend("{}: {}".format(k, v) for v in vals)
    specs.extend(intent_cross_topic())
    specs.sort()

    Path(f"user_prompts/chatgpt").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/chatgpt/specs.json", "w") as f:
        json.dump(specs, f, indent=4)

    return specs


def make_clio_specs() -> list[str]:
    specs = []
    for topic in CLIO_CATEGORIES:
        specs.append(topic)
    specs.sort()

    Path(f"user_prompts/clio").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/clio/specs.json", "w") as f:
        json.dump(specs, f, indent=4)

    return specs


def make_handpick_specs() -> list[str]:
    specs = []
    for topic in HANDPICK_CATEGORIES:
        specs.append(topic)
    specs.sort()

    Path(f"user_prompts/handpick").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/handpick/specs.json", "w") as f:
        json.dump(specs, f, indent=4)

    Path(f"user_prompts/debug").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/debug/specs.json", "w") as f:
        json.dump(specs[:4], f, indent=4)

    return specs

# %%
if __name__ == "__main__":
    make_chatgpt_specs()
    make_clio_specs()
    make_handpick_specs()
