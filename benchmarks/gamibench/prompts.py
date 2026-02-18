"""Prompt templates for GamiBench evaluation."""

from benchmarks.gamibench.types import TaskSample

SYSTEM_PROMPT = "You are a strict origami evaluator."

USER_PROMPT = (
    "You are an Origami Folding Expert. You will be given the final crease pattern of a folded "
    "origami model and four candidate 3D models labeled A through D. Evaluate ALL four symmetrically. "
    "Do not privilege ANY order. Only one of the candidate models corresponds exactly to the result of "
    "folding the given crease pattern. In the crease pattern, red lines represent mountain folds and blue "
    "lines represent valley folds. Your task is to analyze the crease pattern and select the correct 3D "
    "model based solely on visual and geometric reasoning. Consider fold types, symmetry, flap orientation, "
    "and structural features visible in the crease pattern. If none of the four models are possible, respond "
    "with option E, which is always 'This fold is impossible'. At each stage, respond with a single uppercase "
    "letter A, B, C, D, or E. Do not privilege any label or the first viable match. Do not explain your "
    "reasoning. Do not output anything else."
)


def candidate_label(index: int) -> str:
    """Return canonical option label for a candidate index."""
    return chr(ord("A") + index)


def candidate_caption(task: TaskSample, index: int) -> str:
    """Return text caption used before each candidate image."""
    return f"Candidate {candidate_label(index)}"

