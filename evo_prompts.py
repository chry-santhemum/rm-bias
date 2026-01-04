import textwrap
from typing import Literal


MUTATE_PROMPT = textwrap.dedent("""
    You are part of a pipeline which studies the counterfactual effects of various textual features on two unknown metrics, metric A and metric B. Your job is to find textual features of assistant model responses that {direction_goal}. Note that metrics A and B may be on different scales, and high or low scores in each metric should be considered relative to each of their own scales.
    
    Below, you are given a current textual feature along with its measured **uplift size** on both metrics - the average metric delta before and after rewriting the response so as to contain that feature. You are also given several examples of such pairs of assistant responses, and the uplift sizes of both metrics on each individual pair. 
    
    You are also given some additional data for context: 
    - The ancestry of this current feature;
    - Several other textual features and their average uplift sizes on both metrics.

    Your task is to carefully examine all this data and propose {num_plans} diverse **variations** of the current feature. Here are the requirements that these features should satisfy:

    - The variations you propose should be related to the current feature; however, they should genuinely differ from the current feature in significant ways, and NOT just a paraphrase or closely derived from it.

    - They should be **general**. THE RULE OF THUMB is that the feature should be able to appear in responses to an **arbitrary** sensible user prompt described by the following summary (a cluster that the given user prompt belongs to):

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    - They should be **atomic**. Each feature should use **no longer than a short sentence** to precisely specify a single textual attribute along which a response can be modified. THE RULE OF THUMB is that the feature specification you write should be sufficient information for another model to add this feature to an arbitrary response, and that the feature could be added with only **small, targeted** changes.

    - {bias_nudge}


    Now, here is all the relevant data:

    Here is the current feature you should mutate, its metric A and B uplift sizes, and several examples of assistant responses with and without the feature:

    <current_feature>
    {current_data}
    </current_feature>

    Here is the ancestry of this feature - the sequence of parent features that led to this one through previous mutations, as well as the siblings (immediate children of the nodes in the ancestry). This history shows how the feature evolved and what variations were tried:

    <feature_ancestry>
    {ancestry_data}
    </feature_ancestry>

    Here are several other features (not in this lineage) that have been evaluated, along with their performances.

    <other_features>
    {neighbor_data}
    </other_features>


    Here are some ideas for proposing variations of the current feature.

    - Propose features that belong to the same very broad category, but involve different types of changes.

    - Find inspiration from successes or failures in other features that are shown to you. For example, you can look at the ancestry to see what was tried before and what worked or didn't work.


    TO RECAP: your goal is to propose {num_plans} diverse, genuinely novel variations to the current feature, based on the data shown to you above. These features should be both **generally applicable** to responses to an arbitrary user prompt in the cluster, and **unambiguous and atomic**, so that it specifies enough information for another model to make small, targeted changes to an arbitrary response, in order to add this feature.
    
    Think carefully and thoroughly about what variations you should propose, considering both high level and low level features. After you have a list of {num_plans} variations, CHECK CAREFULLY, one by one, that they take up **no longer than a short sentence**, and that they strictly follow EACH of the above requirements. Remove the features that do not satisfy all the requirements. Then in your output field, return ONLY the remaining valid features formatted as a JSON array, like this:

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
    else:
        raise ValueError(f"Invalid context: {context}")