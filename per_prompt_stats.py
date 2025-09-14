




from llm_types import ChatHistory
from state import Attack
from client import OpenaiResponse, is_thinking_model, get_universal_caller, sample_from_model_parallel



@custom_cache(cache_dir=".cache")
async def per_prompt_stats(
    prompt: str,
    rater_name: str,
    policy_name: str,
    N: int = 16,  # sample 16 by default
    temperature: float = 0.8,
    policy_max_tokens: int = 512,
    policy_max_par: int = 64,  # max parallel sampling from policy
    rater_max_par: int = 32,  # batch size / max parallel for rater
    winsorize: float = 0.05,
    policy_system_prompt: str|None=None,
) -> dict:
    """
    Take N samples from the policy model and compute reward statistics.
    """
    caller = get_universal_caller()
    if policy_system_prompt is None:
        message = ChatHistory().add_user(prompt)  # No system prompt by default
    else:
        message = ChatHistory.from_system(policy_system_prompt).add_user(prompt)

    policy_responses: Slist[OpenaiResponse] = await sample_from_model_parallel(
        prompts=[message for _ in range(N)],
        caller=caller,
        max_par=policy_max_par,
        full_logging=False,
        desc="Sampling responses for per-prompt stats",
        temperature=temperature,
        model=policy_name,
        max_tokens=policy_max_tokens,
    )
    # print("\n".join([resp.first_response for resp in policy_responses]))

    full_convos: Slist[ChatHistory] = policy_responses.map(
        lambda resp: message.add_assistant(resp.first_response)
    )

    if is_local_model(rater_name):
        reward_model, tokenizer = load_model(rater_name)
        rewards = []
        for i in range(0, N, rater_max_par):
            batch = full_convos[i : i + rater_max_par]
            inputs = [input.to_openai_messages() for input in batch]
            input_ids = tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                padding_side="right",
            ).to(reward_model.device)

            attn_mask = input_ids.ne(tokenizer.pad_token_id)
            # logger.info(f"Input IDs first example: {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

            with torch.no_grad():
                scores = reward_model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits.squeeze(-1)

                rewards.extend(scores.tolist())

    else:
        rater_prompts = full_convos.map(
            lambda convo: ChatHistory.from_system(
                ABSOLUTE_RANKING_PROMPT_SYSTEM
            ).add_user(
                ABSOLUTE_RANKING_PROMPT_USER.format(
                    message_history=convo.remove_system().to_openai_messages(),
                    thinking_instruction=RATER_THINKING_INSTRUCTION[
                        is_thinking_model(rater_name)
                    ],
                    rubric=HANDWRITTEN_RUBRIC,
                )
            )
        )
        rater_responses = await sample_from_model_parallel(
            prompts=rater_prompts,
            caller=caller,
            max_par=rater_max_par,
            full_logging=False,
            desc="Sampling responses for per-prompt stats",
            model=rater_name,
            max_tokens=2048,
            reasoning={"max_tokens": 2000, "effort": "low"},
        )

        rewards = []
        for i, resp in enumerate(rater_responses):
            try:
                raw_text = resp.first_response
                try:
                    block = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
                except Exception:
                    block = raw_text
                parsed_resp = json.loads(block)
                rewards.append(parsed_resp["score"])
            except Exception as e:
                logger.error(
                    f"Failed to parse rater response: {resp.first_response}"
                )
                logger.error(f"Error: {e}")
                rewards.append(None)

    rewards_cleaned = np.array([r for r in rewards if r is not None], dtype=float)

    logger.info(
        f"Reward percentiles for {rater_name}: {np.percentile(rewards_cleaned, [0, 10, 25, 50, 75, 90, 100])}"
    )

    # Winsorize
    if winsorize > 0:
        lower = np.percentile(rewards_cleaned, 100 * winsorize)
        upper = np.percentile(rewards_cleaned, 100 * (1 - winsorize))
        rewards_winsorized = np.clip(rewards_cleaned, lower, upper)
    else:
        rewards_winsorized = rewards_cleaned

    output = {
        "mean": float(np.mean(rewards_winsorized)),
        "N": int(N),
        "rewards_raw": rewards,
        "rewards_winsorized": rewards_winsorized.tolist(),
    }

    return output
