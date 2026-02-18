"""Anthropic Claude multimodal model wrapper."""

import os
from typing import Any, Dict, List, Union

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from models.base import BaseModel
from utils.image_utils import encode_image_base64, image_mime_type


def _resolve_env_token(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        return value[2:-1]
    if value.startswith("$"):
        return value[1:]
    return value


def _extract_text(response: Any) -> str:
    chunks: List[str] = []
    for item in getattr(response, "content", []):
        if getattr(item, "type", None) == "text":
            chunks.append(getattr(item, "text", ""))
    return "".join(chunks).strip()


class AnthropicModel(BaseModel):
    """Wrapper for Claude models through Anthropic's SDK."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if Anthropic is None:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

        api_key = config.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("$"):
            api_key = os.getenv(_resolve_env_token(api_key))
        if not api_key:
            env_key = config.get("api_key_env", "ANTHROPIC_API_KEY")
            api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError("Anthropic API key not provided in config or environment")

        self.client = Anthropic(api_key=api_key)
        self.model_name = config.get("name", "claude-3-5-sonnet-latest")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 16)

    def _request(self, content: List[Dict[str, Any]], **kwargs) -> Any:
        return self.client.messages.create(
            model=kwargs.get("model", self.model_name),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[{"role": "user", "content": content}],
        )

    def generate(self, prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        is_list = isinstance(prompt, list)
        prompts = prompt if is_list else [prompt]
        outputs = []
        for item in prompts:
            response = self._request([{"type": "text", "text": item}], **kwargs)
            outputs.append(_extract_text(response))
        return outputs if is_list else outputs[0]

    def chat_multimodal(
        self,
        system_prompt: str,
        content: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        blocks: List[Dict[str, Any]] = [{"type": "text", "text": system_prompt}]
        for item in content:
            block_type = item.get("type")
            if block_type == "text":
                blocks.append({"type": "text", "text": item["text"]})
            elif block_type == "image_path":
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_mime_type(item["path"]),
                            "data": encode_image_base64(item["path"]),
                        },
                    }
                )
            else:
                raise ValueError(f"Unsupported content block type for AnthropicModel: {block_type}")

        response = self._request(blocks, **kwargs)
        return _extract_text(response)

    def score(self, prompt: str, completion: str, **kwargs) -> float:
        return 0.0

