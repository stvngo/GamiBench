"""Answer parsing and metric aggregation for GamiBench."""

from collections import defaultdict
import re
from typing import Any, Dict, Iterable, Optional


_BOX_TOKEN_RE = re.compile(r"<\|.*?\|>")
_CHOICE_RE = re.compile(r"^\s*([A-E])\s*$")


def parse_choice(raw_text: Optional[str]) -> Optional[str]:
    """Extract a single letter answer from model output."""
    if raw_text is None:
        return None
    cleaned = _BOX_TOKEN_RE.sub("", str(raw_text)).strip()
    match = _CHOICE_RE.match(cleaned)
    if not match:
        return None
    return match.group(1)


def compute_metrics(task_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate metrics for the benchmark.

    Expected result record fields:
      - task_type
      - is_correct
      - parse_ok
      - status
    """
    rows = list(task_results)
    total = len(rows)
    answered = sum(1 for row in rows if row.get("parse_ok"))
    correct = sum(1 for row in rows if row.get("is_correct"))
    api_failures = sum(1 for row in rows if row.get("status") == "api_error")
    parse_failures = sum(1 for row in rows if row.get("status") == "parse_error")
    skipped = sum(1 for row in rows if row.get("status") == "skipped")

    per_task_total: Dict[str, int] = defaultdict(int)
    per_task_correct: Dict[str, int] = defaultdict(int)
    for row in rows:
        task_type = row.get("task_type", "unknown")
        per_task_total[task_type] += 1
        if row.get("is_correct"):
            per_task_correct[task_type] += 1

    per_task_accuracy = {}
    for task_type, task_total in per_task_total.items():
        per_task_accuracy[task_type] = (
            per_task_correct[task_type] / task_total if task_total else 0.0
        )

    return {
        "total_tasks": total,
        "answered_tasks": answered,
        "correct_tasks": correct,
        "accuracy": correct / total if total else 0.0,
        "answer_rate": answered / total if total else 0.0,
        "api_failures": api_failures,
        "parse_failures": parse_failures,
        "skipped_tasks": skipped,
        "per_task_accuracy": per_task_accuracy,
    }

