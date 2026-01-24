"""OpenAI API model wrapper."""

from typing import List, Union, Optional
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from models.base import BaseModel


class OpenAIModel(BaseModel):
    """Wrapper for OpenAI API models."""
    
    def __init__(self, config: dict):
        """Initialize OpenAI model."""
        super().__init__(config)
        
        if OpenAI is None:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        
        api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided in config or environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = config.get('name', 'gpt-3.5-turbo')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1000)
        self.top_p = config.get('top_p', 1.0)
        self.frequency_penalty = config.get('frequency_penalty', 0.0)
        self.presence_penalty = config.get('presence_penalty', 0.0)
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).
        
        Args:
            prompt: Input prompt(s)
            **kwargs: Override default generation parameters
            
        Returns:
            Generated text(s)
        """
        is_list = isinstance(prompt, list)
        prompts = prompt if is_list else [prompt]
        
        # Merge kwargs with defaults
        gen_params = {
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p),
            'frequency_penalty': kwargs.get('frequency_penalty', self.frequency_penalty),
            'presence_penalty': kwargs.get('presence_penalty', self.presence_penalty),
        }
        
        results = []
        for p in prompts:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": p}],
                **gen_params
            )
            results.append(response.choices[0].message.content)
        
        return results if is_list else results[0]
    
    def score(
        self,
        prompt: str,
        completion: str,
        **kwargs
    ) -> float:
        """
        Score a prompt-completion pair using log probability.
        
        Note: This is a placeholder. OpenAI API doesn't directly provide
        scoring. You may need to use a separate scoring model or method.
        
        Args:
            prompt: Input prompt
            completion: Generated completion
            **kwargs: Additional parameters
            
        Returns:
            Score (placeholder: returns 0.0)
        """
        # TODO: Implement proper scoring method
        # Options:
        # 1. Use a separate scoring model
        # 2. Use log probabilities if available
        # 3. Use a different evaluation method
        return 0.0
