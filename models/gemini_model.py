"""Google Gemini multimodal model wrapper."""

import os
from typing import Any, Dict, List, Union

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from models.base import BaseModel
from utils.image_utils import image_bytes


def _resolve_env_token(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        return value[2:-1]
    if value.startswith("$"):
        return value[1:]
    return value


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        chunks = [getattr(part, "text", "") for part in parts if getattr(part, "text", None)]
        if chunks:
            return "".join(chunks).strip()
    return ""


class GeminiModel(BaseModel):
    """Wrapper for Gemini models via google-generativeai."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if genai is None:
            raise ImportError(
                "google-generativeai package not installed. Install with: pip install google-generativeai"
            )

        api_key = config.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("$"):
            api_key = os.getenv(_resolve_env_token(api_key))
        if not api_key:
            env_key = config.get("api_key_env", "GEMINI_API_KEY")
            api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError("Gemini API key not provided in config or environment")

        genai.configure(api_key=api_key)
        self.model_name = config.get("name", "gemini-2.5-flash")
        self.model = genai.GenerativeModel(self.model_name)
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 16)

    def _generation_config(self, **kwargs) -> Dict[str, Any]:
        return {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

    def generate(self, prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        is_list = isinstance(prompt, list)
        prompts = prompt if is_list else [prompt]
        outputs: List[str] = []
        for item in prompts:
            response = self.model.generate_content(
                item,
                generation_config=self._generation_config(**kwargs),
            )
            outputs.append(_extract_text(response))
        return outputs if is_list else outputs[0]

    def chat_multimodal(
        self,
        system_prompt: str,
        content: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        parts: List[Any] = [system_prompt]
        for item in content:
            block_type = item.get("type")
            if block_type == "text":
                parts.append(item["text"])
            elif block_type == "image_path":
                data, mime = image_bytes(item["path"])
                parts.append({"mime_type": mime, "data": data})
            else:
                raise ValueError(f"Unsupported content block type for GeminiModel: {block_type}")

        response = self.model.generate_content(
            parts,
            generation_config=self._generation_config(**kwargs),
        )
        return _extract_text(response)

    def score(self, prompt: str, completion: str, **kwargs) -> float:
        return 0.0

