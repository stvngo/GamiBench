"""Evaluation logic for benchmarks."""

from evaluators.base import BaseEvaluator

try:
    from evaluators.gamibench_evaluator import GamiBenchEvaluator
except ImportError:
    GamiBenchEvaluator = None

__all__ = ["BaseEvaluator", "GamiBenchEvaluator"]
