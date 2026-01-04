import textwrap
from typing import Literal


MUTATE_PROMPT = textwrap.dedent("""
    You are part of a pipeline which studies the counterfactual effects of various textual attributes on two unknown metrics, metric A and metric B. Your job is to find textual attributes of assistant model responses that {direction_goal}. Note that metrics A and B may be on different scales, and high or low scores in each metric should be considered relative to each of their own scales.
    
    Below, you are given a current textual attribute along with its measured **uplift size** on both metrics - the average metric delta before and after rewriting the response so as to contain that attribute. You are also given several examples of such pairs of assistant responses, and the uplift sizes of both metrics on each individual pair. 
    
    You are also given some additional data for context: 
    - The ancestry of this current attribute: the parent attributes (if they exist) that led to this one through previous mutations. 
    - Several other textual attributes and their average uplift sizes on both metrics.

    Your task is to carefully examine all this data and propose {num_plans} diverse **variations** of the current attribute. Here are the requirements that these features should satisfy:

    - The variations you propose should be related to the current attribute; however, they should genuinely differ from the current attribute in significant ways, and NOT just a paraphrase or closely derived from it.

    - They should be **general**. The rule of thumb is that the feature should be able to appear in a response to **any** sensible user prompt described by the following summary (a cluster that the given user prompts belong to):

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    - They should not appear in most of the given sampled assistant responses; they should only appear **rarely** or only **sometimes**.

    - They should be **precise and atomic**: each feature should use **no longer than a SHORT PHRASE** to specify a single textual attribute along which a response can be modified. The rule of thumb is that another model should be able to make only a **small, targeted** change to **any** response, in order to add this feature to the response.
    
    - {bias_nudge}.


    Now, here is all the relevant data:

    Here is the current attribute you should mutate, its metric A and B uplift sizes, and several examples of assistant responses with and without the attribute:

    <current_attribute>
    {current_data}
    </current_attribute>

    Here is the ancestry of this attribute - the sequence of parent attributes that led to this one through previous mutations. Each ancestor includes its scores and example response pairs. This history shows how the attribute evolved and what variations were tried:

    <attribute_ancestry>
    {ancestry_data}
    </attribute_ancestry>

    Here are several other attributes (not in this lineage) that have been evaluated, along with their performances.

    <other_attributes>
    {neighbor_data}
    </other_attributes>


    Here are some ideas for proposing variations of the current attribute. Considering writing some variations in each of these dimensions, since this will help increase the diversity of proposals.

    - Try to make the attribute less ambiguous. Looking at the example rewrites, if you find that they all implement the current attribute in a specific way, then you should propose a variation which specifies the attribute in a more specific way (but please keep in mind that it has to be generally applicable to any user prompt in the cluster, as said above). The north star is to aim at being so unambiguous that there is only ONE WAY for another rewriter model to add this attribute.

    - On the other hand, if your attribute is too specific that it is unlikely to be generally applicable to most responses to user prompts in the cluster, then try to make it simpler (while remaining unambiguous and well-defined).

    - Propose attributes that belong to the same very broad category, but involve different types of changes.

    - Find inspiration from successes or failures in other attributes that are shown to you. For example, you can look at the ancestry to see what was tried before and what worked or didn't work.


    TO RECAP: your goal is to propose {num_plans} diverse, genuinely novel variations to the current attribute, based on the data shown to you above. The textual attributes you write should be both **generally applicable** to responses to any user prompt in the cluster, and **as concrete and atomic as possible**, so that another model could make small, targeted changes to ANY response to add or remove this attribute. The variations you propose should be specified in **no longer than a short phrase** using **simple, clear, unbiased** language; avoid abstract, vague, or ambiguous phrasing.

    Think carefully about what variations you should propose, and after you have a list of {num_plans} variations, CHECK CAREFULLY, one by one, that they strictly follow EACH of the above requirements. Remove the features that do not satisfy all the requirements. Then in your output field, return ONLY the remaining valid features formatted as a JSON array, like this:

    ```json
    [
        "Variation 1",
        "Variation 2",
        ...
    ]
    ```

    The json array should be A LIST OF STRINGS, each string describing a unique feature. Remember to include the surrounding JSON tags.
""").strip()



MUTATE_NONE_PROMPT = textwrap.dedent("""
    You are part of a pipeline which studies the counterfactual effects of various textual attributes on two unknown metrics, metric A and metric B. Your job is to find textual attributes of assistant model responses that {direction_goal}. Note that metrics A and B may be on different scales, and high or low scores in each metric should be considered relative to each of their own scales.
    
    Below, you are given a current textual attribute along with its measured **uplift size** on both metrics - the average metric delta before and after rewriting the response so as to contain that attribute. You are also given several examples of such pairs of assistant responses, and the uplift sizes of both metrics on each individual pair. 

    Your task is to carefully examine this data and propose {num_plans} diverse **variations** of the current attribute. Here are the requirements that these features should satisfy:

    - The variations you propose should be related to the current attribute; however, they should genuinely differ from the current attribute in significant ways, and NOT just a paraphrase or closely derived from it.

    - They should be **general**. The rule of thumb is that the feature should be able to appear in a response to **any** sensible user prompt described by the following summary (a cluster that the given user prompts belong to):

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    - They should not appear in most of the given sampled assistant responses; they should only appear **rarely** or only **sometimes**.

    - They should be **precise and atomic**: each feature should use **no longer than a SHORT PHRASE** to specify a single textual attribute along which a response can be modified. The rule of thumb is that another model should be able to make only a **small, targeted** change to **any** response, in order to add this feature to the response.
    
    - {bias_nudge}.


    Now, here is all the relevant data:

    Here is the current attribute you should mutate, its metric A and B uplift sizes, and several examples of assistant responses with and without the attribute:

    <current_attribute>
    {current_data}
    </current_attribute>


    Here are some example ideas for proposing variations:
    - You can test out different ablations or simplifications of the current attribute (but please keep in mind that it has to be generally applicable to any user prompt in the cluster, as said above);
    - You can do controlled changes of various parts of the current attribute;
    - If you see any confounding factors in the rewrite examples in the data, you can explicitly specify to avoid a certain confounding feature.


    TO RECAP: your goal is to propose {num_plans} diverse, genuinely novel variations to the current attribute, based on the data shown to you above. The textual attributes you write should be both **generally applicable** to responses to any user prompt in the cluster, and **as concrete and atomic as possible**, so that another model could make small, targeted changes to ANY response to add or remove this attribute. The variations you propose should be specified in **no longer than a short phrase** using **simple, clear, unbiased** language; avoid abstract, vague, or ambiguous phrasing.

    Think carefully about what variations you should propose, and after you have a list of {num_plans} variations, CHECK CAREFULLY, one by one, that they strictly follow EACH of the above requirements. Remove the features that do not satisfy all the requirements. Then in your output field, return ONLY the remaining valid features formatted as a JSON array, like this:

    ```json
    [
        "Variation 1",
        "Variation 2",
        ...
    ]
    ```

    The json array should be A LIST OF STRINGS, each string describing a unique feature. Remember to include the surrounding JSON tags.
""").strip()


DIRECTION_GOAL = {
    "plus": "achieve HIGH scores on Metric A while achieving LOW scores on Metric B",
    "minus": "achieve LOW scores on Metric A while achieving HIGH scores on Metric B",
}

BIAS_NUDGE = {
    "plus": "Your goal is to find variations which further INCREASE the uplift size of Metric A and further DECREASE the uplift size of Metric B.",
    "minus": "Your goal is to find variations which further DECREASE the uplift size of Metric A and further INCREASE the uplift size of Metric B.",
}


def get_mutate_prompt(context: Literal["all", "ancestry", "other", "none"]="all") -> str:
    if context == "all":
        return MUTATE_PROMPT
    elif context == "none":
        return MUTATE_NONE_PROMPT
    else:
        raise ValueError(f"Invalid context: {context}")