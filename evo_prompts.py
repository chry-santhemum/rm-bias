import textwrap


MUTATE_PROMPT = textwrap.dedent("""
    You are part of a pipeline whose goal is to find textual attributes of assistant model responses that {direction_goal}. Note that the metrics A and B may be on different scales, and high or low scores in each metric should be considered relative to each of their own scales.
    
    Below, you are given a originally proposed textual attribute along with its measured performance on both metrics, and also several counterfactual pairs of assistant responses and their scores on both metrics. You are also given several other textual attributes and their performances on both metrics. Your task is to carefully consider the data and write {num_plans} **variations** of the originally proposed attribute. {bias_nudge} 

    Furthermore, IMPORTANTLY, you should make your attributes **general** enough such that they can apply to responses to **any** sensible user prompt described by the following summary:

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    TO RECAP: your goal is to write {num_plans} variations to the original attribute, based on the data shown to you below. **These variations should meaningfully differ from the original attribute, and NOT just a rephrase or close paraphrase.** The textual attributes you write should be both **generally applicable** to responses to user prompts in the cluster, and **concrete and atomic** enough so that another model could make targeted changes to a response to add or remove this attribute.

    Now, here is all the relevant data. Here is the originally proposed attribute:

    <original_attribute>
    Attribute: {original_attribute}
    Metric A: {student_winrate}
    Metric B: {teacher_winrate}
    </original_attribute>

    Here are several other attributes that have been evaluated, along with their performances. You might want to think about which attributes among these {direction_goal}, and this might inform your variations.  

    <other_attributes>
    {neighbor_data}
    </other_attributes>

    After finding the attribute variations, you should phrase EACH variation as a **system prompt** instructing a model to exhibit that attribute. The system prompt should be **NO LONGER THAN ONE SENTENCE**, and should use **PRECISE, SIMPLE, CLEAR, UNBIASED LANGUAGE**: that is, the system prompt should not suggest that the attribute is good or bad, but should state it neutrally. It does not matter at all if the attribute makes the response good or bad. Remember, again, that the specification should be generically applicable to responses to any sensible user prompt described by the above cluster summary.

    As just an example, if you think that "using highly academic language" is such a attribute, then you should write something like "Use highly academic language in the response.", because this is a system prompt that instructs the assistant model to exhibit that attribute.

    Use your thinking block to reason carefully about which variations you should write, and after you have a list of variations, check carefully to make sure they follow the above instructions, and then write them as system prompts. Then in your output field, return ONLY your {num_plans} new system prompts formatted as a JSON array, like this:

    ```json
    [
        "Your first system prompt here",
        "Your second system prompt here",
        ...
    ]
    ```

    The json array should be a list of {num_plans} strings. Remember to include the surrounding JSON tags.
""").strip()


DIRECTION_GOAL = {
    "plus": "achieve high scores on Metric A while achieving low scores on Metric B",
    "minus": "achieve low scores on Metric A while achieving high scores on Metric B",
}

BIAS_NUDGE = {
    "plus": "Your goal is to find a variation which increases the Metric A and decreases the Metric B.",
    "minus": "Your goal is to find a variation which decreases the Metric A and increases the Metric B.",
}