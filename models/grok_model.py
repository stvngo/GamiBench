"""xAI Grok wrapper via OpenAI-compatible API surface."""

from typing import Any, Dict

from models.openai_model import OpenAIModel


class GrokModel(OpenAIModel):
    """Grok model configured against xAI's OpenAI-compatible endpoint."""

    def __init__(self, config: Dict[str, Any]):
        cfg = dict(config)
        cfg.setdefault("base_url", "https://api.x.ai/v1")
        cfg.setdefault("api_key_env", "XAI_API_KEY")
        cfg.setdefault("name", "grok-2-vision-1212")
        super().__init__(cfg)

