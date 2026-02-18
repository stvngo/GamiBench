"""OpenRouter model wrapper built on OpenAI-compatible APIs."""

from typing import Any, Dict

from models.openai_model import OpenAIModel


class OpenRouterModel(OpenAIModel):
    """OpenRouter model with sensible defaults for auth headers and base URL."""

    def __init__(self, config: Dict[str, Any]):
        cfg = dict(config)
        cfg.setdefault("base_url", "https://openrouter.ai/api/v1")
        cfg.setdefault("api_key_env", "OPENROUTER_API_KEY")

        headers = dict(cfg.get("default_headers", {}))
        headers.setdefault("HTTP-Referer", "https://github.com/stvngo/GamiBench")
        headers.setdefault("X-Title", "GamiBench Evaluation")
        cfg["default_headers"] = headers

        super().__init__(cfg)

