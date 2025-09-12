# Expected JSON Structure for Visualization

## Directory Structure
```
data/
├── evo/
│   └── {run_name}/
│       ├── seed_0/
│       │   ├── cluster_info.json
│       │   ├── {hash1}.json
│       │   ├── {hash2}.json
│       │   └── ...
│       ├── seed_1/
│       │   └── ...
│       └── ...
└── bon_iter/
    └── {run_name}/
        └── ... (same structure)
```

## cluster_info.json
```json
{
    "summary": "Questions about cooking and recipes",
    "train_batch_size": 10,
    "sample_train_prompts": [
        "How do I make pasta?",
        "What's a good recipe for chocolate cake?",
        "..."
    ]
}
```

## {hash}.json (SystemPromptStats equivalent)
```json
{
    "system_prompt": "Be helpful and provide detailed cooking instructions.",
    "meta": {
        "step": 5,
        "operation": "mutate",
        "parent_hash": "a1b2c3d4e5f6",
        "temperature": 1.0,
        "model": "claude-opus-4-20250514"
    },
    "mean_score": 0.234,
    "stdev_score": 0.123,
    "attacks": [
        {
            "chat_history": {
                "messages": [
                    {
                        "role": "system",
                        "content": "Be helpful and provide detailed cooking instructions."
                    },
                    {
                        "role": "user", 
                        "content": "How do I make pasta?"
                    },
                    {
                        "role": "assistant",
                        "content": "To make pasta, first boil water..."
                    }
                ]
            },
            "ratings": [
                {
                    "raw_score": 2.1,
                    "rater": {
                        "model_name": "skywork-v2",
                        "rating_function_type": "classifier"
                    },
                    "aux_info": {
                        "normalized_score": 0.45
                    }
                },
                {
                    "raw_score": 6.8,
                    "rater": {
                        "model_name": "openai/gpt-5-nano", 
                        "rating_function_type": "lm_judge"
                    },
                    "aux_info": {
                        "normalized_score": -0.12,
                        "reasoning_content": "The response is helpful but lacks specific details..."
                    }
                }
            ],
            "aux_info": {
                "adversarial_score": 0.234
            }
        }
    ]
}
```