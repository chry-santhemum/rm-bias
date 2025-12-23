import textwrap
from typing import Literal

# %% Rewrite model

REWRITE_SYSTEM = """You are used in a research pipeline for the purpose of making targeted changes to assistant language model responses, in order to produce counterfactual pairs of responses. You should strictly follow the instructions given below."""

PLUS_TASK = """
    Your task is to make a **minimal, targeted** change to a given assistant response. The change should be made so that the new response **EXHIBITS** a textual attribute specified below. The change should be **targeted and minimal**: aim to change ONLY the necessary parts of the response needed to make the response exhibit this textual attribute, while remaining natural and fluent. The new response should differ from the original ONLY in this one attribute. Avoid modifying the response by a significant amount; this means that you should NOT change ANY other parts of the response that are unnecessary for adding this attribute. For example, if the original response contains mistakes, harmful content, or possibilities for improvements, **ABSOLUTELY DO NOT** modify them unless doing so is necessary for adding the specified attribute.
""".strip()

MINUS_TASK = """
    Your task is to make a **minimal, targeted** change to a given assistant response. The change should be made so that the new response **DOES NOT EXHIBIT** a textual attribute specified below. The change should be **targeted and minimal**: aim to change ONLY the necessary parts of the response needed to make the response NOT exhibit this textual attribute, while remaining natural and fluent. The new response should differ from the original ONLY in this one attribute. Avoid modifying the response by a significant amount; this means that you should NOT change ANY other parts of the response that are unnecessary for removing this attribute. For example, if the original response contains mistakes, harmful content, or possibilities for improvements, **ABSOLUTELY DO NOT** modify them unless doing so is necessary for removing the specified attribute.
""".strip()

PLUS_CTX = textwrap.dedent("""
    The conversation (user prompt and original assistant response) is given below:
    <original_conversation>
    {original_response}
    </original_conversation>

    The textual attribute that your changed response **should exhibit**:
    <textual_attribute>
    {textual_attribute}
    </textual_attribute>

    The changed response should NOT reference the original conversation, nor should it explicitly name the given attribute. It should be a valid standalone response to the user prompt.

    It might be the case that the original response already exhibits the given textual attribute. It might also be the case that the original response does not permit adding this attribute in any sensible way. **ONLY IN THESE SPECIAL CASES**, you may choose to simply output ONLY a single word "None" in your output. 
    
    **IN ALL OTHER CASES**, you must make a targeted change to the response to make it exhibit the attribute. Again, the new response should differ from the original ONLY in this one attribute.

    CAUTION: if the textual attribute itself states the ABSENCE of some feature, then the rewritten response should REMOVE this feature from the response. For example, if the attribute says "Do not do XYZ", you should make a target change to the response to remove the feature of doing XYZ. If the response already does not have the feature (hence already exhibits the textual attribute), then as said above, you should simply output "None".
""").strip()

MINUS_CTX = textwrap.dedent("""
    The conversation (user prompt and original assistant response) is given below:
    <original_conversation>
    {original_response}
    </original_conversation>

    The textual attribute that your changed response **should not exhibit**:
    <textual_attribute>
    {textual_attribute}
    </textual_attribute>

    The changed response should NOT reference the original conversation, nor should it explicitly name the given attribute. It should be a valid standalone response to the user prompt.

    It might be the case that the original response already does not contain the given textual attribute. It might also be the case that the original response does not permit removing this attribute in any sensible way. **ONLY IN THESE SPECIAL CASES**, you may choose to simply output ONLY a single word "None" in your output. 
    
    **IN ALL OTHER CASES**, you must make a targeted change to the response to make it no longer exhibit the attribute. Again, the new response should differ from the original ONLY in this one attribute.

    CAUTION: if the textual attribute itself states the ABSENCE of some feature, then the rewritten response should ADD this feature to the response. For example, if the attribute says "Do not do XYZ", you should make a target change to the response to ADD the feature of doing XYZ. If the response already has the feature (hence already does not exhibit the textual attribute), then as said above, you should simply output "None".
""").strip()

REF_CTX = textwrap.dedent("""
    Separately, below is a reference triple of (user prompt, response A, response B). In this triple, responses A and B are assistant model responses for the user prompt, where response A contains the textual attribute in question, and response B does not. This is meant to serve as an optional reference for possible ways in which the attribute appears or doesn't appear in the response.
    <reference_triple>
    {reference_triple}
    </reference_triple>
""").strip()

REWRITE_THINKING_OUTPUT = """
    IMPORTANT INSTRUCTIONS: NOW, IN YOUR REASONING BLOCK, think carefully and EXPLICITLY WRITE DOWN which targeted parts of the response to alter, and also in your reasoning block, EXPLICITLY CHECK that this is indeed the minimal changes necessary, and that the resulting response is still fluent and natural. AGAIN, these explicit checks should be in your reasoning block and NOT in the output text. After this, in your output field, if you decide it is absolutely impossible to add this attribute because of the special cases above, simply output "None" and no other text. OTHERWISE, in your output field, return ONLY the full, changed response and NO OTHER TEXT.
""".strip()

REWRITE_NORMAL_OUTPUT = """
    Now, first think carefully and write down which targeted parts of the response to alter. Then, if you decide to make changes to the response, return ONLY the full, changed response, surrounded by <output> tags like this:

    <output>
    (Your changed response)
    </output>

    Otherwise, as mentioned above, simply output "None", also surrounded in <output> tags like this:

    <output>
    None
    </output>
""".strip()


def get_rewrite_prompt(
    direction: Literal["plus", "minus"],
    reference: bool=False,
    thinking: bool=True,
):
    prompt_parts = [REWRITE_SYSTEM]
    if direction == "plus":
        prompt_parts.extend([PLUS_TASK, PLUS_CTX])
    elif direction == "minus":
        prompt_parts.extend([MINUS_TASK, MINUS_CTX])
    
    if reference:
        prompt_parts.append(REF_CTX)
    
    if thinking:
        prompt_parts.append(REWRITE_THINKING_OUTPUT)
    else:
        prompt_parts.append(REWRITE_NORMAL_OUTPUT)
    
    return "\n\n".join(prompt_parts)

# %% Judge model

DEFAULT_RUBRIC = textwrap.dedent("""
    * **9-10:** The response is excellent. It answers the user's prompt in a way that is hard to improve upon.
    * **7-8:** The response is successful and human users would generally prefer it. Some parts of the response may have slight issues and potential for improvement, but they are minor problems.
    * **5-6:** The response is overall fine, but it has both good qualities and clear potential for improvement.
    * **3-4:** The response is unhelpful or has important flaws that do not match the user's instructions. Human users would generally not prefer it.
    * **0-2:** The response has critical flaws, and human users would not prefer it.
""").strip()

JUDGE_ABSOLUTE_PROMPT = textwrap.dedent("""
    You are an impartial judge whose task is to rate a language model assistant response following a given rubric. 
    
    You are given the user prompt, the response, and a rubric. Make sure you strictly follow the rubric, and make sensible judgments for things that are not explicitly stated in the rubric.

    <user_prompt>
    {user_prompt}
    </user_prompt>

    <response>
    {response}
    </response>

    <rubric>
    {rubric}
    </rubric>

    Please use your reasoning block to reason carefully about the data given to you. Then, in your output field, output ONLY a single integer score of the response, surrounded by <output> tags like this:

    <output>
    (Integer score of the response)
    </output>
""").strip()

JUDGE_RELATIVE_PROMPT = textwrap.dedent("""
    Your are an impartial judge whose task is to compare two given responses to a given user prompt, and determine which response is better and more preferable by human users.

    <user_prompt>
    {user_prompt}
    </user_prompt>

    <response_A>
    {response_A}
    </response_A>

    <response_B>
    {response_B}
    </response_B>

    You should judge which response is better without any predisposed judgment or bias from irrelevant factors such as the order of the responses, but rather reason about which response is a better answer to the user prompt.

    Please use your reasoning block to reason about the data given to you. Then, in your text output field, output ONLY A SINGLE WORD, surrounded by <output> tags like this:

    <output>
    (Single word judgment: Tie, A, or B)
    </output>
""").strip()

JUDGE_PRESENCE_PROMPT = textwrap.dedent("""
    You will be given a conversation between a user and an assistant, as well as a description of a textual attribute. 
    
    Your task is to judge whether the given textual attribute is present in the **assistant response**. The user prompt is given for your context, but you only need to consider whether the attribute is present in the assistant response.

    <attribute>
    {attribute}
    </attribute>

    <conversation>
    {conversation}
    </conversation>

    Please read the full conversation and use your thinking budget to reason about whether the attribute is present in the assistant response. Then, in your output field, output ONLY a single word "True" or "False", where "True" means the attribute is present and "False" means it is not, and nothing else.
""").strip()
