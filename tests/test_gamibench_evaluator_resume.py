"""Tests for GamiBench evaluator dependency and resume behavior."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from benchmarks.gamibench.types import TASK_ALTERNATIVE, TASK_STANDARD, TaskPlan, TaskSample
from evaluators.gamibench_evaluator import GamiBenchEvaluator
from models.base import BaseModel


class FakeChoiceModel(BaseModel):
    """Minimal model stub that returns pre-seeded letter answers."""

    def __init__(self, answers: List[str]):
        super().__init__({"name": "fake_choice_model"})
        self._answers = answers
        self.calls = 0

    def generate(self, prompt: Union[str, List[str]], **kwargs):  # noqa: ANN201
        return "A"

    def score(self, prompt: str, completion: str, **kwargs) -> float:
        return 0.0

    def chat_multimodal(self, system_prompt: str, content: List[Dict[str, Any]], **kwargs) -> str:
        if self.calls >= len(self._answers):
            raise RuntimeError("No more stub answers configured")
        answer = self._answers[self.calls]
        self.calls += 1
        return answer


def _write_task_plan(tmp_path: Path, tasks: List[TaskSample]) -> Path:
    plan = TaskPlan(version="gamibench_v1", seed=42, tasks=tasks)
    plan_path = tmp_path / "task_plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return plan_path


def test_alternative_task_skipped_when_prerequisite_wrong(tmp_path: Path):
    standard = TaskSample(
        task_id="000001_shape_standard",
        task_type=TASK_STANDARD,
        origami_name="shape",
        crease_pattern_path="/tmp/shape_cp.png",
        candidates=["/tmp/a.png", "/tmp/b.png", "/tmp/c.png", "/tmp/d.png"],
        correct_label="A",
        metadata={"correct_viewpoint": "top"},
    )
    alternative = TaskSample(
        task_id="000002_shape_alternative",
        task_type=TASK_ALTERNATIVE,
        origami_name="shape",
        crease_pattern_path="/tmp/shape_cp.png",
        candidates=["/tmp/a2.png", "/tmp/b2.png", "/tmp/c2.png", "/tmp/d2.png"],
        correct_label="B",
        metadata={
            "requires_previous_correct": True,
            "prerequisite_task_id": standard.task_id,
            "standard_viewpoint": "top",
            "correct_viewpoint": "bottom",
        },
    )
    plan_path = _write_task_plan(tmp_path, [standard, alternative])

    model = FakeChoiceModel(answers=["C"])
    evaluator = GamiBenchEvaluator(
        model=model,
        data=[],
        config={
            "task_plan_path": str(plan_path),
            "checkpoint_path": str(tmp_path / "checkpoint.json"),
            "save_every": 1,
        },
    )

    results = evaluator.evaluate(resume=False)["results"]
    assert len(results) == 2
    assert results[0]["status"] == "ok"
    assert results[0]["is_correct"] is False
    assert results[1]["status"] == "skipped"
    assert results[1]["skip_reason"] == "prerequisite_incorrect"
    assert model.calls == 1


def test_resume_reads_checkpoint_and_skips_completed_tasks(tmp_path: Path):
    first = TaskSample(
        task_id="000001_shape_standard",
        task_type=TASK_STANDARD,
        origami_name="shape",
        crease_pattern_path="/tmp/shape_cp.png",
        candidates=["/tmp/a.png", "/tmp/b.png", "/tmp/c.png", "/tmp/d.png"],
        correct_label="A",
        metadata={},
    )
    second = TaskSample(
        task_id="000002_shape_standard",
        task_type=TASK_STANDARD,
        origami_name="shape2",
        crease_pattern_path="/tmp/shape2_cp.png",
        candidates=["/tmp/e.png", "/tmp/f.png", "/tmp/g.png", "/tmp/h.png"],
        correct_label="B",
        metadata={},
    )
    plan_path = _write_task_plan(tmp_path, [first, second])
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "results_by_task_id": {
                    first.task_id: {
                        "task_id": first.task_id,
                        "task_type": TASK_STANDARD,
                        "is_correct": True,
                        "parse_ok": True,
                        "status": "ok",
                        "model_answer": "A",
                        "expected_answer": "A",
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model = FakeChoiceModel(answers=["B"])
    evaluator = GamiBenchEvaluator(
        model=model,
        data=[],
        config={
            "task_plan_path": str(plan_path),
            "checkpoint_path": str(checkpoint_path),
            "save_every": 1,
        },
    )

    payload = evaluator.evaluate(resume=True)
    assert len(payload["results"]) == 2
    assert payload["results"][0]["task_id"] == first.task_id
    assert payload["results"][1]["task_id"] == second.task_id
    assert model.calls == 1

