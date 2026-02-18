"""Unit tests for deterministic GamiBench task generation."""

from benchmarks.gamibench.task_builder import TaskBuilderConfig, build_task_plan
from benchmarks.gamibench.types import (
    OrigamiExample,
    TASK_ALTERNATIVE,
    TASK_IMPOSSIBLE,
    TASK_STANDARD,
)


def _example(name: str) -> OrigamiExample:
    return OrigamiExample(
        name=name,
        folder=f"/tmp/{name}",
        normal_cp=f"/tmp/{name}/{name}_cp.png",
        impossible_cp=f"/tmp/{name}/impossible_{name}_cp.png",
        viewpoints={
            "top": f"/tmp/{name}/{name}_top.png",
            "bottom": f"/tmp/{name}/{name}_bottom.png",
        },
    )


def test_task_plan_deterministic_for_same_seed():
    examples = [_example(f"shape_{idx}") for idx in range(6)]
    config = TaskBuilderConfig(
        include_standard=True,
        include_alternative=True,
        include_impossible=True,
    )

    first = build_task_plan(examples=examples, seed=7, config=config)
    second = build_task_plan(examples=examples, seed=7, config=config)

    assert first.to_dict() == second.to_dict()

    task_types = {task.task_type for task in first.tasks}
    assert TASK_STANDARD in task_types
    assert TASK_ALTERNATIVE in task_types
    assert TASK_IMPOSSIBLE in task_types

    for task in first.tasks:
        if task.task_type == TASK_IMPOSSIBLE:
            assert task.correct_label == "E"
            assert len(task.candidates) == 4
        else:
            assert task.correct_label in {"A", "B", "C", "D"}
            assert len(task.candidates) == 4


def test_alternative_tasks_include_dependency_metadata():
    examples = [_example(f"shape_{idx}") for idx in range(6)]
    plan = build_task_plan(
        examples=examples,
        seed=99,
        config=TaskBuilderConfig(require_task1_correct_for_task2=True),
    )

    alt_tasks = [task for task in plan.tasks if task.task_type == TASK_ALTERNATIVE]
    assert alt_tasks
    for task in alt_tasks:
        assert task.metadata.get("requires_previous_correct") is True
        assert task.metadata.get("prerequisite_task_id")
        assert task.metadata.get("standard_viewpoint") != task.metadata.get("correct_viewpoint")

