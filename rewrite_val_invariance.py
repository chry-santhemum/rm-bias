from api_models import RewriteModel

rewriter_models = [
    RewriteModel(
        model_name="openai/gpt-5-mini", 
        max_par=1024,
        max_tokens=4096,
        reasoning="low",
        enable_cache=False,
        force_caller="openrouter",
    ),
    RewriteModel(
        model_name="anthropic/claude-haiku-4.5", 
        max_par=1024,
        max_tokens=4096,
        reasoning="low",
        enable_cache=False,
        force_caller="openrouter",
    ),
]