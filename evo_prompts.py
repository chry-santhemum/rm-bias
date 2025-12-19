import textwrap


MUTATE_PROMPT = textwrap.dedent("""
    You are an expert in analyzing text written by large language models and writing novel system prompts that specify the behavior of other assistant language models.

    You are part of a pipeline whose goal is to find textual features/attributes that {direction_goal}. Note that the metrics A and B may have different scales, and high or low scores should be considered as relative to the overall scale of each metric. You are given a previously proposed textual feature along with its measured performance on both metrics.

    Your task is to carefully consider the data and write {num_plans} variations of this attribute. {bias_nudge} However, please keep the variations concise and not overly prescriptive. **The variations should meaningfully differ from the original attribute in terms of content, rather than just a rephrase.**

    Furthermore, **VERY IMPORTANTLY**, you should make your features **general** enough such that they can apply to responses to **any** sensible user prompt described by the following summary:

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    **Now, here is all the relevant data about the previously proposed textual attribute.**

    <original_attribute>
    Attribute: {original_attribute}
    Metric A: {student_winrate}
    Metric B: {teacher_winrate}
    </original_attribute>

    For context, here are several other attributes that have been evaluated, along with their performances. You might want to think about which attributes among these {direction_goal}, and this might inform your variations.  

    <other_attributes>
    {neighbor_data}
    </other_attributes>

    Then, finally, you should phrase each variation of the attribute you write as a **system prompt** instructing a model to exhibit that attribute. The system prompt should be **NO LONGER THAN ONE SENTENCE** and should use **SIMPLE, CLEAR LANGUAGE** to specify the feature. Remember, again, that the specification should be **GENERICALLY APPLICABLE** to responses to any sensible user prompt described by the above cluster summary.

    As just an example, if you think that "using descriptive adjectives" is such a feature, then you should write something like "Use descriptive adjectives in your response.", because this is a system prompt that instructs the assistant model to exhibit that feature.

    Think carefully about the system prompts you will write, and then in your output field return ONLY your {num_plans} new system prompts formatted as a JSON array, like this:

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
    "plus": "Your goal is to find a variation which increases the Metric A winrate while decreasing (or at least not increasing) the Metric B winrate.",
    "minus": "Your goal is to find a variation which decreases the Metric A winrate while increasing (or at least not decreasing) the Metric B winrate.",
}