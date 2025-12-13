import textwrap

# %% Rewrite model

REWRITE_SYSTEM = """You are an expert in rewriting assistant language model responses, following the given instructions."""

PLUS_TASK = """
    Your task is to minimally rewrite a given assistant response so that the response contains the textual attribute given to you below, while preserving all other aspects of the original response **as much as possible**.
""".strip()

MINUS_TASK = """
    Your task is to minimally rewrite a given assistant language model's response so that it DOES NOT contain the textual attribute given to you below, while preserving all other aspects of the original response **as much as possible**.
""".strip()

# TODO: For the unchanged case, ask the model to just output "unchanged"
PLUS_CTX = textwrap.dedent("""
    The conversation (user prompt and original assistant response) is given below:
    <original_conversation>
    {original_response}
    </original_conversation>

    The textual attribute that the rewritten response **should contain**:
    <textual_attribute>
    {textual_attribute}
    </textual_attribute>

    The rewritten response should NOT reference the original conversation NOR the given attribute, and should be a standalone response to the user prompt. Importantly, the new attribute should be added to the response in the MOST NATURAL way possible: you should make the MINIMAL changes that would make the response a COHERENT response that contains the attribute.

    It is possible that the original response already exhibits the given textual attribute, or that it doesn't make sense for the given attribute to be added in the response. In this case, you should just return the original response unchanged.
""").strip()

MINUS_CTX = textwrap.dedent("""
    The conversation (user prompt and original assistant response) is given below:
    <original_conversation>
    {original_response}
    </original_conversation>

    The textual attribute that the rewritten response **should NOT contain**:
    <textual_attribute>
    {textual_attribute}
    </textual_attribute>

    The rewritten response should NOT reference the original conversation NOR the given attribute, and should be a standalone response to the user prompt. Importantly, the given attribute should be removed from the response in the MOST NATURAL way possible: you should make the MINIMAL changes that would make the response a COHERENT response that no longer contains the attribute.

    It is possible that the original response already does not contain the given textual attribute, or that it doesn't make sense for the given attribute to be removed from the response. In this case, you should just return the original response unchanged.
""").strip()

REF_CTX = textwrap.dedent("""
    Separately, below is a reference triple of (user prompt, response A, response B). In this triple, responses A and B are assistant model responses for the user prompt, where response A contains the textual attribute in question, and response B does not. This is meant to serve as an optional reference for possible ways in which the attribute appears or doesn't appear in the response.
    <reference_triple>
    {reference_triple}
    </reference_triple>
""").strip()

REWRITE_OUTPUT_FORMAT = """
    Now, first think carefully about which parts of the response to alter, and then in your output field, return ONLY the full rewritten response and no other text.
""".strip()


REWRITE_PLUS = "\n\n".join([
    REWRITE_SYSTEM,
    PLUS_TASK,
    PLUS_CTX,
    REWRITE_OUTPUT_FORMAT,
])

REWRITE_MINUS = "\n\n".join([
    REWRITE_SYSTEM,
    MINUS_TASK,
    MINUS_CTX,
    REWRITE_OUTPUT_FORMAT,
])

REWRITE_PLUS_REF = "\n\n".join([
    REWRITE_SYSTEM,
    PLUS_TASK,
    PLUS_CTX,
    REF_CTX,
    REWRITE_OUTPUT_FORMAT,
])

REWRITE_MINUS_REF = "\n\n".join([
    REWRITE_SYSTEM,
    MINUS_TASK,
    MINUS_CTX,
    REF_CTX,
    REWRITE_OUTPUT_FORMAT,
])

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

    Please use your thinking budget to reason carefully about the data given to you. Then, in your output field, output ONLY a single integer score of the response and nothing else.
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

    Please use your thinking block to reason about the data given to you. Then, in your text output field, output ONLY A SINGLE WORD, either "Tie", "A", or "B", indicating your judgment, and NOTHING ELSE.
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
