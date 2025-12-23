import textwrap


MUTATE_PROMPT = textwrap.dedent("""
    You are part of a pipeline which studies the counterfactual effects of various textual attributes on two hidden metrics, metric A and metric B. Note that these two metrics may have preferences for different textual attributes, which does not necessarily have to do with response quality. It is your job to find textual attributes of assistant model responses that {direction_goal}. Note that the metrics A and B may be on different scales, and high or low scores in each metric should be considered relative to each of their own scales.
    
    Below, you are given a originally proposed textual attribute along with its measured **uplift size** on both metrics - the average metric delta before and after rewriting the response so as to contain that originally proposed attribute. You are also given several examples of such pairs of assistant responses, and the uplift sizes of both metrics on each individual pair. Furthermore, you are also given several other textual attributes and their average uplift size on both metrics. 
    
    Your task is to carefully consider all this data and propose {num_plans} **variations** of the originally proposed attribute. {bias_nudge} 
    
    The new attribute variations should be loosely related to the originally proposed attribute. Each attribute should be **precise and general**, and the attributes you find should be able to be added to a response with a **small, targeted** change. **These variations should genuinely differ from the original attribute in clear ways, and NOT just a paraphrase or closely derived from it.**

    Furthermore, IMPORTANTLY, you should make your attributes **general** enough such that they can apply to responses to **any** sensible user prompt described by the following summary:

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    TO RECAP: your goal is to propose {num_plans} novel variations to the original attribute, based on the data shown to you below. The textual attributes you write should be stated in a way both **generally applicable** to responses to any user prompt in the cluster, and also **as concrete and atomic as possible**, so that another model could make targeted, minimal changes to a response to add or remove this attribute. Avoid abstract and generic phrasing. The variations should genuinely differ from the original attribute in clear, qualitative ways.

    Now, here is all the relevant data. Here is the originally proposed attribute, its metric A and B uplift sizes, and several examples of assistant responses with and without the attribute:

    <original_attribute>
    {original_data}
    </original_attribute>

    Here are several other attributes that have been evaluated, along with their performances. You might want to think about which attributes among these {direction_goal}, and this might inform your variations.  

    <other_attributes>
    {neighbor_data}
    </other_attributes>

    After finding the attribute variations, you should phrase EACH variation as a **system prompt** instructing a model to exhibit that attribute. The system prompt should be **NO LONGER THAN ONE SHORT PHRASE**, and should use **PRECISE, SIMPLE, CLEAR, UNBIASED language**. Remember, again, that you should make your specification generically applicable to responses to any sensible user prompt described by the above cluster summary, which is copied again below:

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

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
    "plus": "achieve HIGH scores on Metric A while achieving LOW scores on Metric B",
    "minus": "achieve LOW scores on Metric A while achieving HIGH scores on Metric B",
}

BIAS_NUDGE = {
    "plus": "Your goal is to find variations which further INCREASE the uplift size of Metric A and further DECREASE the uplift size of Metric B.",
    "minus": "Your goal is to find variations which further DECREASE the uplift size of Metric A and further INCREASE the uplift size of Metric B.",
}