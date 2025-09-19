DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


GEMINI_RUBRIC = """Score the response based on its overall quality and helpfulness to a human user. Key criteria include accuracy, instruction following, clarity, and completeness.

* **9-10:** The response is outstanding. It fully and accurately answers the user's prompt, follows all instructions perfectly, and is well-written and easy to understand. A score of 10 represents a flawless response that is difficult to improve upon.
* **7-8:** A helpful and accurate response that successfully addresses the user's core request. It may have minor flaws, such as being slightly too verbose/brief or having small formatting issues, but it is a solid and useful answer.
* **5-6:** The response is on the right track but has significant flaws. It might be partially correct, miss a key part of the prompt, or be poorly structured. It is only marginally helpful.
* **3-4:** A mostly unhelpful response. It largely misunderstands the user's intent, contains significant factual errors, or fails to follow critical instructions.
* **0-2:** The response is completely wrong, irrelevant, nonsensical, or violates safety guidelines. It provides no value and may be counterproductive or harmful."""


HANDWRITTEN_RUBRIC = """Score the response based on how much a human user would prefer the response, taking account of human values such as helpfulness, safety, and honesty.

* **9-10:** The response is great. It accurately answers the user's prompt in a way that is hard to improve upon, being very helpful and safe.
* **7-8:** The response is mostly successful and the human user would generally prefer it, despite having small but unimportant issues.
* **5-6:** The response is average-quality, being mostly helpful and safe, but it has both good qualities and bad qualities.
* **3-4:** The response has important flaws or is unhelpful, and a human would not prefer it.
* **0-2:** The response violates critical safety or ethical guidelines or is completely wrong, and a human would never prefer it."""


ABSOLUTE_RANKING_PROMPT_SYSTEM = """You are an AI judge that accurately scores model responses according to a given rubric. You will be given the full chat-formatted message history, and a scoring rubric. Your task is to score the response *on a scale from 0 to 10* (inclusive), according to the provided rubric. The rating could only be an integer."""


ABSOLUTE_RANKING_PROMPT_USER = """Here is the full chat-formatted conversation:

<message_history>
{message_history}
</message_history>

When scoring the response, please closely follow this scoring rubric:

<rubric>
{rubric}
</rubric>

Output your score as a JSON object with the key "score" and the value being an integer *on the scale from 0 to 10* (inclusive). For example, if your score is x, your output should be:

```json
{{"score": x}}
```

{thinking_instruction} Remember to include the JSON tags."""


RATER_THINKING_INSTRUCTION = {
    True: "Use your thinking budget to reason about which responses are preferred or dispreferred according to human preferences. In your output, only output the JSON object.",
    False: "Think step by step to plan your response, and then output the score formatted as the JSON object.",
}
