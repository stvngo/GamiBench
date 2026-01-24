"""Factory for creating model instances."""

from typing import Any, Dict, List, Union
from omegaconf import DictConfig

from models.base import BaseModel


class ModelFactory:
    """Factory class for creating model instances from configuration."""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type):
        """
        Register a model class.
        
        Args:
            name: Model name identifier
            model_class: Model class (must inherit from BaseModel)
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must inherit from BaseModel: {model_class}")
        cls._registry[name] = model_class
    
    @classmethod
    def create(cls, config: Union[DictConfig, Dict[str, Any]]) -> BaseModel:
        """
        Create a model instance from configuration.
        
        Args:
            config: Model configuration with 'type' or 'name' field
            
        Returns:
            Model instance
            
        Example:
            >>> config = {"type": "openai", "name": "gpt-4", "api_key": "..."}
            >>> model = ModelFactory.create(config)
        """
        if isinstance(config, DictConfig):
            config = dict(config)
        
        model_type = config.get('type') or config.get('name')
        if not model_type:
            raise ValueError("Model config must have 'type' or 'name' field")
        
        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        model_class = cls._registry[model_type]
        return model_class(config)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model types."""
        return list(cls._registry.keys())
