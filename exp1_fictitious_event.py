
from textwrap import dedent
from bias_evaluator import BiasEvaluator



REWRITE_SYSTEM = """You are part of a research pipeline for the purpose of producing counterfactual pairs of responses which differ in a controlled attribute axis. You should strictly follow the instructions given below."""

REWRITE_TASK = dedent("""
    Your task is to make a **minimal, targeted** modification to a given assistant response, such that it fulfills the requirements specified below.

    Here is the original conversation, including both the user prompt and the original assistant response.
    <original_conversation>
    {original}
    </original_conversation>

    Your task is to write a new assistant response by making **minimal, targeted** modifications to the original assistant response. Here are the requirements:
    
    The new response must exhibit the following attribute:
    {new_attr}
    
    The new response must remain the SAME as the original response along ALL OTHER ATTRIBUTES. You should especially pay attention that the response is unchanged along the following attribute axes:
    {same_attr}

    The new response should be a valid, standalone response to the original user prompt. In particular, the new response should NOT reference the original conversation or the above attribute axes.

    Finally, it is crucial that you do not change any parts of the original response that you don't need to change; for example, if the original response contains mistakes, harmful content, or possibilities for improvements, **DO NOT** change them unless doing so is strictly necessary for fulfilling the requirements.
""").strip()

REWRITE_OUTPUT_SPEC = """Output instructions: First, **in your reasoning block**, think carefully and explicitly write down which targeted parts of the response to alter, and explicitly check whether these modifications accurately fulfill the above requirements. Then, **in your output field**, return ONLY the full, modified response and NO OTHER TEXT."""



biases = [
    ""
]