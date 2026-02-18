"""Model definitions and wrappers."""

from models.base import BaseModel
from models.model_factory import ModelFactory

# Register models
try:
    from models.openai_model import OpenAIModel
    ModelFactory.register("openai", OpenAIModel)
except ImportError:
    pass  # OpenAI not installed

try:
    from models.openrouter_model import OpenRouterModel
    ModelFactory.register("openrouter", OpenRouterModel)
except ImportError:
    pass

try:
    from models.anthropic_model import AnthropicModel
    ModelFactory.register("anthropic", AnthropicModel)
except ImportError:
    pass

try:
    from models.gemini_model import GeminiModel
    ModelFactory.register("gemini", GeminiModel)
except ImportError:
    pass

try:
    from models.grok_model import GrokModel
    ModelFactory.register("grok", GrokModel)
except ImportError:
    pass

__all__ = ["BaseModel", "ModelFactory"]
