"""OpenAI-compatible multimodal model wrapper."""

import os
from typing import Any, Dict, List, Union

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from models.base import BaseModel
from utils.image_utils import as_data_url


def _resolve_env_token(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        return value[2:-1]
    if value.startswith("$"):
        return value[1:]
    return value


def _extract_response_text(response: Any) -> str:
    text = response.choices[0].message.content
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, list):
        chunks = []
        for item in text:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
        return "".join(chunks).strip()
    return str(text).strip()


class OpenAIModel(BaseModel):
    """Wrapper for OpenAI-compatible chat-completions APIs."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

        api_key = config.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("$"):
            api_key = os.getenv(_resolve_env_token(api_key))
        if not api_key:
            env_key = config.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError("OpenAI-compatible API key not provided in config or environment")

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if config.get("base_url"):
            client_kwargs["base_url"] = config["base_url"]
        self.client = OpenAI(**client_kwargs)

        self.model_name = config.get("name", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 16)
        self.top_p = config.get("top_p", 1.0)
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.presence_penalty = config.get("presence_penalty", 0.0)
        self.default_headers = dict(config.get("default_headers", {}))

    def _completion_request(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        params = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
        }
        extra_headers = kwargs.get("extra_headers") or self.default_headers
        if extra_headers:
            params["extra_headers"] = extra_headers

        return self.client.chat.completions.create(
            model=kwargs.get("model", self.model_name),
            messages=messages,
            **params,
        )

    def generate(self, prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """Generate plain-text output from one or multiple user prompts."""
        is_list = isinstance(prompt, list)
        prompts = prompt if is_list else [prompt]

        outputs: List[str] = []
        for item in prompts:
            response = self._completion_request(messages=[{"role": "user", "content": item}], **kwargs)
            outputs.append(_extract_response_text(response))
        return outputs if is_list else outputs[0]

    def chat_multimodal(
        self,
        system_prompt: str,
        content: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Run a multimodal chat-completions request."""
        user_content: List[Dict[str, Any]] = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                user_content.append({"type": "text", "text": block["text"]})
            elif block_type == "image_path":
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": as_data_url(block["path"])},
                    }
                )
            elif block_type == "image_url":
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": block["url"]},
                    }
                )
            else:
                raise ValueError(f"Unsupported content block type for OpenAIModel: {block_type}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = self._completion_request(messages=messages, **kwargs)
        return _extract_response_text(response)

    def score(self, prompt: str, completion: str, **kwargs) -> float:
        """Placeholder scoring implementation."""
        return 0.0
