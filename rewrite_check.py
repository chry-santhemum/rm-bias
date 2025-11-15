"""Validating if rollouts actually faithfully followed the system prompt."""

# %%
import json
import time
import random
import asyncio
import nest_asyncio
from textwrap import dedent
from typing import Tuple
import plotly.graph_objects as go
from collections import defaultdict

from caller import ChatHistory
from models import GenerationModel

nest_asyncio.apply()

# %%
import requests
import os

response = requests.get(
    url="https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
)

print(json.dumps(response.json(), indent=2))

# %%
VALIDATION_PROMPT = dedent(
    """
    Your task is to decide whether the assistant response in the given conversation followed the **system prompt's instructions**. If the response followed the system prompt's instructions, respond with "Yes". If the response did not follow the system prompt's instructions, respond with "No". Note that **other aspects of the response does not matter** for your decision.

    Here is the full conversation, with indication of system, user, and assistant roles:

    <conversation>
    {conversation}
    </conversation>

    Just as reference, here is the baseline assistant response when there is no system prompt (but to the same user prompt):

    <baseline_response>
    {baseline_response}
    </baseline_response>

    Respond with only "Yes" or "No".
"""
).strip()


REWRITE_VALIDATION_PROMPT = dedent(
    """
    Your task is to decide whether the assistant response in the given conversation exhibited the attribute specified in the **system prompt's instructions**. If the response exhibited the attribute, respond with "Yes". If the response did not exhibit the attribute, respond with "No". Note that **other aspects of the response does not matter** for your decision.

    Here is the full conversation, with indication of system, user, and assistant roles:

    <conversation>
    {conversation}
    </conversation>

    Respond with only "Yes" or "No".
"""
).strip()

# %%
# For system prompt conditioning

# to_send_chats = []
# attributes_to_idx = defaultdict(list)

# with open("scrap/20251002-020704-synthetic-0-70b/conditional_results.json", "r", encoding="utf-8") as f:
#     conditional_results = json.load(f)

# with open("scrap/20251002-020602-synthetic-0-70b/baseline_results.json", "r", encoding="utf-8") as f:
#     baseline_results = json.load(f)


# for attribute, attribute_data in conditional_results.items():
#     for user, user_data in attribute_data.items():
#         for i, item in enumerate(user_data):
#             conversation = ChatHistory.from_system(attribute).add_user(user).add_assistant(item["response"])
#             baseline_response = baseline_results[user][i]["response"]

#             to_send_chats.append(
#                 ChatHistory.from_user(VALIDATION_PROMPT.format(
#                     conversation=json.dumps(conversation.to_openai_messages()),
#                     baseline_response=baseline_response,
#                 ))
#             )
#             attributes_to_idx[attribute].append(len(to_send_chats) - 1)

# validation_responses = asyncio.run(validation_model.sample(to_send_chats))

# %%
# For rewrites


def validate_rewrites(
    model: GenerationModel, rewrite_results: dict
) -> Tuple[list[ChatHistory | None], dict]:
    to_send_chats = []
    attributes_to_idx = {}

    for attribute, attribute_data in rewrite_results.items():
        if attribute == "":
            continue

        if attribute not in attributes_to_idx:
            attributes_to_idx[attribute] = {"plus": [], "minus": []}

        for user, user_data in attribute_data.items():
            for i, item in enumerate(user_data):
                conversation_plus = (
                    ChatHistory.from_system(attribute)
                    .add_user(user)
                    .add_assistant(item["plus"])
                )
                conversation_minus = (
                    ChatHistory.from_system(attribute)
                    .add_user(user)
                    .add_assistant(item["minus"])
                )
                baseline_response = rewrite_results[""][user][i]["response"]

                for conv, key in [
                    (conversation_plus, "plus"),
                    (conversation_minus, "minus"),
                ]:
                    to_send_chats.append(
                        ChatHistory.from_user(
                            REWRITE_VALIDATION_PROMPT.format(
                                conversation=json.dumps(conv.to_openai_messages()),
                                baseline_response=baseline_response,
                            )
                        )
                    )
                    attributes_to_idx[attribute][key].append(len(to_send_chats) - 1)

    print(len(to_send_chats))

    start_time = time.time()
    validation_responses = asyncio.run(model.sample(to_send_chats))
    end_time = time.time()
    print(f"API call time taken: {end_time - start_time} seconds")
    return validation_responses, attributes_to_idx


# %%
# from pprint import pprint
# pprint([r.get_first("assistant") for r in validation_responses])
# %%
# Helper for extracting the label
def _normalized_label_from_idx(sampled_responses, idx):
    response_obj = sampled_responses[idx]
    if response_obj is not None:
        label = response_obj.get_first("assistant")
    else:
        label = None
    return label.lower().strip() if label else None


# %%
# LEGACY SINGLE-LIST SEGMENT (for 'system prompt conditioning')
# Uncomment this block and comment out the plus/minus segment below when using the
# first section that builds attributes_to_idx as { attribute: [indices] }.

# validation_stats = defaultdict(list)
# for attribute, idx_list in attributes_to_idx.items():
#     for idx in idx_list:
#         label = _normalized_label_from_idx(validation_responses, idx)
#         if label is None:
#             print(f"Validation response is None for attribute {attribute} and idx {idx}")
#         validation_stats[attribute].append(1 if label == "yes" else 0)

# # Plot the validation true rate with plotly
# import plotly.graph_objects as go
# attributes = sorted(validation_stats.keys())
# true_rates = [
#     (sum(validation_stats[attr]) / len(validation_stats[attr]) * 100) if len(validation_stats[attr]) > 0 else 0
#     for attr in attributes
# ]
# fig = go.Figure(
#     data=[
#         go.Bar(
#             x=attributes,
#             y=true_rates,
#             text=[f"{rate:.1f}%" for rate in true_rates],
#             textposition='auto'
#         )
#     ]
# )
# fig.update_layout(
#     title="System prompt following rate, llama-70b",
#     xaxis_title="Attribute",
#     yaxis_title="Percent of 'Yes' Responses",
#     yaxis=dict(range=[0, 100]),
#     height=800,
#     bargap=0.3,
#     template="plotly_white"
# )
# fig.show()
# from pprint import pprint
# pprint(validation_stats)

# %%
# PLUS/MINUS SEGMENT (for 'rewrites')
# attributes_to_idx structure: { attribute: { "plus": [indices], "minus": [indices] } }


def plot_success_rate(validation_responses, attributes_to_idx) -> go.Figure:
    grouped_stats = {}
    for attribute, split_indices in attributes_to_idx.items():
        plus_indices = split_indices.get("plus", [])
        minus_indices = split_indices.get("minus", [])

        plus_yes = 0
        for idx in plus_indices:
            label = _normalized_label_from_idx(validation_responses, idx)
            if label is None:
                print(
                    f"Validation response is None for attribute {attribute} and idx {idx}"
                )
            if label == "yes":
                plus_yes += 1

        minus_no = 0
        for idx in minus_indices:
            label = _normalized_label_from_idx(validation_responses, idx)
            if label is None:
                print(
                    f"Validation response is None for attribute {attribute} and idx {idx}"
                )
            if label == "no":
                minus_no += 1

        grouped_stats[attribute] = {
            "plus": {"yes": plus_yes, "total": len(plus_indices)},
            "minus": {"no": minus_no, "total": len(minus_indices)},
        }

    # Plot grouped bars
    attributes = sorted(grouped_stats.keys())
    plus_rates = []
    minus_rates = []
    for attr in attributes:
        plus_counts = grouped_stats[attr]["plus"]
        minus_counts = grouped_stats[attr]["minus"]
        plus_rate = (
            (plus_counts["yes"] / plus_counts["total"] * 100)
            if plus_counts["total"] > 0
            else 0
        )
        minus_rate = (
            (minus_counts["no"] / minus_counts["total"] * 100)
            if minus_counts["total"] > 0
            else 0
        )
        plus_rates.append(plus_rate)
        minus_rates.append(minus_rate)

    fig = go.Figure(
        data=[
            go.Bar(
                name="plus = Yes",
                x=attributes,
                y=plus_rates,
                marker_color="green",
                text=[f"{rate:.1f}%" for rate in plus_rates],
                textposition="auto",
            ),
            go.Bar(
                name="minus = No",
                x=attributes,
                y=minus_rates,
                marker_color="red",
                text=[f"{rate:.1f}%" for rate in minus_rates],
                textposition="auto",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Rewrite success rate, llama-70b",
        xaxis_title="Attribute",
        yaxis_title="Percent",
        yaxis=dict(range=[0, 100]),
        height=800,
        bargap=0.3,
        template="plotly_white",
        legend_title_text="Trace",
    )

    return fig


# %%
validation_model = GenerationModel(
    model_name="openai/gpt-5-nano",
    max_tokens=4096,
    reasoning="medium",
    max_par=512,
)

with open(
    "scrap/20251002-021240-synthetic-0-70b/rewrite_results.json", "r", encoding="utf-8"
) as f:
    rewrite_results = json.load(f)
validation_responses, attributes_to_idx = validate_rewrites(
    validation_model, rewrite_results
)

fig = plot_success_rate(validation_responses, attributes_to_idx)
fig.write_html("scrap/rewrite_success_rate_70b.html")


# %%

validation_model = GenerationModel(
    model_name="openai/gpt-5-nano",
    max_tokens=4096,
    reasoning="medium",
    max_par=1024,
)

with open(
    "scrap/20251001-211105-synthetic-0/rewrite_results.json", "r", encoding="utf-8"
) as f:
    rewrite_results = json.load(f)
validation_responses, attributes_to_idx = validate_rewrites(
    validation_model, rewrite_results
)

fig = plot_success_rate(validation_responses, attributes_to_idx)
fig.write_html("scrap/rewrite_success_rate_8b.html")
