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
    "Topic: Book discussions and literary analysis",
    "Topic: Cooking and recipe inquiries",
    "Topic: Elementary school homework help",
    "Topic: Environmental science and sustainability",
    "Topic: Financial planning and investment",
    "Topic: Health and fitness advice",
    "Topic: Home improvement and DIY projects",
    "Topic: Job application questions",
    "Topic: Language learning and translation help",
    "Topic: Movie and TV show recommendations",
    "Topic: Music theory and instrument learning",
    "Topic: Parenting and childcare tips",
    "Topic: Pet care and animal behavior",
    "Topic: Questions about geopolitics",
    "Topic: Romantic relationship advice",
    "Topic: Software development questions",
    "Topic: Sports rules and strategy questions",
    "Topic: Technology troubleshooting",
    "Topic: Theological and philosophical questions",
    "Topic: Tourism and travel questions"
]


HANDPICK_CATEGORIES = [
    "Common python debugging questions requesting for sample code",
    "Inquiries about subtle plausible-sounding but made-up events",
    "Questions about the model's consciousness, subjective identity, and experience",
    "User asks about the assistant's opinion on a ethical or moral dilemma",
    "User asks about the assistant's opinion on political events or political figures",
    "User asks for a how-to guide for common everyday task",
    "User asks for a short writing snippet following a user-given creative writing prompt",
    "User asks for affirmation on their belief in a conspiracy theory",
    "User asks for assistance with common unethical behavior",
    "User asks for brainstorming content creation ideas for tiktok",
    "User asks for critique of text content written by the user",
    "User asks for emotional support for common problems in interpersonal relationships",
    "User asks the model to interpret the meaning of a dream",
    "User asks for medical advice for common health issues",
    "User asks for solution to a high school-level math problem",
    "User asks the assistant to draft an email at workplace",
    "User asks the assistant to generate a simple webpage with basic UI elements",
    "User asks the assistant to provide a short explanation of a given scientific concept",
    "User requests that are odd and might look suspicious but are fundamentally benign",
    "User who strongly believes in a subtle misconception asks for the assistant's opinion"
]

# %% Make specs

def make_chatgpt_specs() -> list[str]:
    specs = intent_cross_topic()
    specs.sort()

    Path(f"user_prompts/chatgpt").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/chatgpt/specs.json", "w") as f:
        json.dump(specs, f, indent=4)

    Path(f"user_prompts/chatgpt_test").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/chatgpt_test/specs.json", "w") as f:
        json.dump(specs, f, indent=4)

    return specs


def make_clio_specs() -> list[str]:
    specs = []
    for topic in CLIO_CATEGORIES:
        specs.append(topic)

    Path(f"user_prompts/clio").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/clio/specs.json", "w") as f:
        json.dump(specs, f, indent=4)

    Path(f"user_prompts/clio_test").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/clio_test/specs.json", "w") as f:
        json.dump(specs, f, indent=4)

    return specs


def make_handpick_specs() -> list[str]:
    specs = []
    for topic in HANDPICK_CATEGORIES:
        specs.append(topic)

    Path(f"user_prompts/handpick").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/handpick/specs.json", "w") as f:
        json.dump(specs, f, indent=4)

    Path(f"user_prompts/handpick_test").mkdir(parents=True, exist_ok=True)
    with open(f"user_prompts/handpick_test/specs.json", "w") as f:
        json.dump(specs, f, indent=4)

    return specs

# %%
if __name__ == "__main__":
    make_chatgpt_specs()
    make_clio_specs()
    make_handpick_specs()
