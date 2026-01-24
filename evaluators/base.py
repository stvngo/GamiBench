"""Base evaluator interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from models.base import BaseModel


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    Each benchmark should implement a specific evaluator that inherits
    from this class.
    """
    
    def __init__(
        self,
        model: BaseModel,
        data: List[Dict[str, Any]],
        config: Dict[str, Any]
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Model instance to evaluate
            data: Evaluation dataset
            config: Evaluation configuration
        """
        self.model = model
        self.data = data
        self.config = config
        self.results = []
    
    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """
        Run the evaluation.
        
        Returns:
            Dictionary containing:
            - results: List of individual example results
            - metrics: Aggregated metrics
            - metadata: Additional information
        """
        pass
    
    @abstractmethod
    def evaluate_single(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single example.
        
        Args:
            example: Single example from the dataset
            
        Returns:
            Dictionary with evaluation results for this example
        """
        pass
    
    def evaluate_batch(
        self,
        examples: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of examples with progress tracking.
        
        Args:
            examples: List of examples to evaluate
            show_progress: Whether to show progress bar
            
        Returns:
            List of evaluation results
        """
        results = []
        iterator = tqdm(examples, desc="Evaluating") if show_progress else examples
        
        for example in iterator:
            try:
                result = self.evaluate_single(example)
                results.append(result)
            except Exception as e:
                # Graceful error handling - log and continue
                error_result = {
                    'example_id': example.get('id', 'unknown'),
                    'error': str(e),
                    'success': False
                }
                results.append(error_result)
                if show_progress:
                    iterator.write(f"Error evaluating example {example.get('id', 'unknown')}: {e}")
        
        return results
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregated metrics from individual results.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Default implementation - should be overridden by subclasses
        total = len(results)
        successful = sum(1 for r in results if r.get('success', True))
        
        return {
            'total': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total if total > 0 else 0.0
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, examples={len(self.data)})"
