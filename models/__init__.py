"""Model definitions and wrappers."""

from models.base import BaseModel
from models.model_factory import ModelFactory

# Register models
try:
    from models.openai_model import OpenAIModel
    ModelFactory.register("openai", OpenAIModel)
except ImportError:
    pass  # OpenAI not installed

__all__ = ["BaseModel", "ModelFactory"]
