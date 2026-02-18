"""Base model interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class BaseModel(ABC):
    """
    Abstract base class for all models.
    
    All models should implement this interface to ensure compatibility
    with the evaluation pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.name = config.get('name', 'unknown_model')
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[str]],
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).
        
        Args:
            prompt: Input prompt(s) as string or list of strings
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text(s) as string or list of strings
        """
        pass
    
    @abstractmethod
    def score(
        self,
        prompt: str,
        completion: str,
        **kwargs
    ) -> float:
        """
        Score a prompt-completion pair.
        
        Args:
            prompt: Input prompt
            completion: Generated completion
            **kwargs: Additional scoring parameters
            
        Returns:
            Score as float
        """
        pass
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts (default: sequential).
        
        Can be overridden for batch processing optimization.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def chat_multimodal(
        self,
        system_prompt: str,
        content: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        Run a multimodal chat request and return raw model text.

        The default implementation raises so benchmarks can fail fast if a
        text-only model is wired into an image task.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement multimodal chat"
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
