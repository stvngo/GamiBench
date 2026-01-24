"""Example evaluator implementation."""

from typing import Any, Dict, List
from evaluators.base import BaseEvaluator


class ExampleEvaluator(BaseEvaluator):
    """
    Example evaluator to demonstrate the pattern.
    
    Replace this with your actual benchmark evaluator.
    """
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run the full evaluation.
        
        Returns:
            Dictionary with results, metrics, and metadata
        """
        # Evaluate all examples
        results = self.evaluate_batch(
            self.data,
            show_progress=self.config.get('show_progress', True)
        )
        
        # Compute aggregated metrics
        metrics = self.compute_metrics(results)
        
        return {
            'results': results,
            'metrics': metrics,
            'metadata': {
                'model': str(self.model),
                'num_examples': len(self.data),
                'config': dict(self.config)
            }
        }
    
    def evaluate_single(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single example.
        
        Args:
            example: Single example from dataset
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract input from example
        prompt = example.get('prompt', example.get('input', ''))
        
        # Generate response
        response = self.model.generate(prompt)
        
        # Compute score/metrics for this example
        # This is where your benchmark-specific logic goes
        score = self._compute_score(example, response)
        
        return {
            'example_id': example.get('id', 'unknown'),
            'prompt': prompt,
            'response': response,
            'score': score,
            'success': True
        }
    
    def _compute_score(self, example: Dict[str, Any], response: str) -> float:
        """
        Compute score for a single example.
        
        This is benchmark-specific. Replace with your actual scoring logic.
        """
        # Example: simple length-based score (replace with actual metric)
        expected = example.get('expected', example.get('output', ''))
        if isinstance(expected, str):
            # Simple similarity (replace with actual metric)
            return 1.0 if response.lower() == expected.lower() else 0.0
        return 0.0
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregated metrics from individual results.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Call parent method for basic metrics
        base_metrics = super().compute_metrics(results)
        
        # Add benchmark-specific metrics
        scores = [r.get('score', 0.0) for r in results if r.get('success', True)]
        
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
        else:
            avg_score = max_score = min_score = 0.0
        
        base_metrics.update({
            'average_score': avg_score,
            'max_score': max_score,
            'min_score': min_score,
            'num_scored': len(scores)
        })
        
        return base_metrics
