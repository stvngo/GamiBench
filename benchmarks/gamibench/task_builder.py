"""Deterministic task-plan generation for GamiBench."""

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Sequence, Tuple

from benchmarks.gamibench.types import (
    OrigamiExample,
    TaskPlan,
    TaskSample,
    TASK_ALTERNATIVE,
    TASK_IMPOSSIBLE,
    TASK_STANDARD,
)


def _label_for_index(index: int) -> str:
    return chr(ord("A") + index)


def _candidate_origami_name(candidate_path: str) -> str:
    stem = candidate_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


@dataclass
class TaskBuilderConfig:
    """Options for transforming discovered examples into benchmark tasks."""

    num_choices: int = 4
    include_standard: bool = True
    include_alternative: bool = True
    include_impossible: bool = True
    require_task1_correct_for_task2: bool = True
    reuse_distractors_for_task2: bool = True


def _choose_distractors(
    rng: random.Random,
    examples: Sequence[OrigamiExample],
    target_name: str,
    count: int,
) -> Optional[List[str]]:
    candidates = [ex for ex in examples if ex.name != target_name and ex.viewpoints]
    if len(candidates) < count:
        return None

    picked_examples = rng.sample(candidates, count)
    distractors: List[str] = []
    for ex in picked_examples:
        viewpoint = rng.choice(ex.available_viewpoints())
        distractors.append(ex.viewpoints[viewpoint])
    return distractors


def _build_standard_task(
    rng: random.Random,
    examples: Sequence[OrigamiExample],
    example: OrigamiExample,
    task_index: int,
    config: TaskBuilderConfig,
) -> Optional[Tuple[TaskSample, Dict[str, str], str]]:
    if not example.normal_cp or not example.viewpoints:
        return None

    distractor_count = config.num_choices - 1
    distractors = _choose_distractors(rng, examples, example.name, distractor_count)
    if distractors is None:
        return None

    chosen_view = rng.choice(example.available_viewpoints())
    correct_candidate = example.viewpoints[chosen_view]
    candidates = [correct_candidate] + distractors
    rng.shuffle(candidates)

    correct_label = _label_for_index(candidates.index(correct_candidate))
    task = TaskSample(
        task_id=f"{task_index:06d}_{example.name}_{TASK_STANDARD}",
        task_type=TASK_STANDARD,
        origami_name=example.name,
        crease_pattern_path=example.normal_cp,
        candidates=candidates,
        correct_label=correct_label,
        metadata={
            "correct_candidate_path": correct_candidate,
            "correct_viewpoint": chosen_view,
            "candidate_origami_names": [_candidate_origami_name(path) for path in candidates],
        },
    )
    return task, {"distractors": distractors}, chosen_view


def _build_alternative_task(
    rng: random.Random,
    examples: Sequence[OrigamiExample],
    example: OrigamiExample,
    task_index: int,
    config: TaskBuilderConfig,
    first_viewpoint: str,
    standard_context: Dict[str, str],
    prerequisite_task_id: str,
) -> Optional[TaskSample]:
    if not example.normal_cp or not example.viewpoints:
        return None

    remaining_views = [view for view in example.available_viewpoints() if view != first_viewpoint]
    if not remaining_views:
        return None

    alt_view = rng.choice(remaining_views)
    correct_candidate = example.viewpoints[alt_view]

    if config.reuse_distractors_for_task2 and "distractors" in standard_context:
        distractors = list(standard_context["distractors"])
    else:
        distractor_count = config.num_choices - 1
        distractors = _choose_distractors(rng, examples, example.name, distractor_count)
        if distractors is None:
            return None

    candidates = [correct_candidate] + distractors
    rng.shuffle(candidates)
    correct_label = _label_for_index(candidates.index(correct_candidate))

    return TaskSample(
        task_id=f"{task_index:06d}_{example.name}_{TASK_ALTERNATIVE}",
        task_type=TASK_ALTERNATIVE,
        origami_name=example.name,
        crease_pattern_path=example.normal_cp,
        candidates=candidates,
        correct_label=correct_label,
        metadata={
            "correct_candidate_path": correct_candidate,
            "correct_viewpoint": alt_view,
            "candidate_origami_names": [_candidate_origami_name(path) for path in candidates],
            "requires_previous_correct": config.require_task1_correct_for_task2,
            "prerequisite_task_id": prerequisite_task_id,
            "standard_viewpoint": first_viewpoint,
        },
    )


def _build_impossible_task(
    rng: random.Random,
    examples: Sequence[OrigamiExample],
    example: OrigamiExample,
    task_index: int,
    config: TaskBuilderConfig,
) -> Optional[TaskSample]:
    if not example.impossible_cp:
        return None

    distractors = _choose_distractors(rng, examples, example.name, config.num_choices)
    if distractors is None:
        return None

    return TaskSample(
        task_id=f"{task_index:06d}_{example.name}_{TASK_IMPOSSIBLE}",
        task_type=TASK_IMPOSSIBLE,
        origami_name=example.name,
        crease_pattern_path=example.impossible_cp,
        candidates=distractors,
        correct_label="E",
        metadata={
            "candidate_origami_names": [_candidate_origami_name(path) for path in distractors],
        },
    )


def build_task_plan(
    examples: Sequence[OrigamiExample],
    seed: int,
    config: Optional[TaskBuilderConfig] = None,
) -> TaskPlan:
    """
    Create deterministic evaluation tasks from discovered origami examples.

    The resulting plan is model-agnostic and can be reused for fair multi-model runs.
    """
    cfg = config or TaskBuilderConfig()
    rng = random.Random(seed)
    tasks: List[TaskSample] = []
    task_index = 0

    sorted_examples = sorted(examples, key=lambda ex: ex.name.lower())
    for example in sorted_examples:
        if cfg.include_standard:
            standard_payload = _build_standard_task(rng, sorted_examples, example, task_index, cfg)
            if standard_payload:
                standard_task, context, used_viewpoint = standard_payload
                tasks.append(standard_task)
                task_index += 1

                if cfg.include_alternative:
                    alt_task = _build_alternative_task(
                        rng=rng,
                        examples=sorted_examples,
                        example=example,
                        task_index=task_index,
                        config=cfg,
                        first_viewpoint=used_viewpoint,
                        standard_context=context,
                        prerequisite_task_id=standard_task.task_id,
                    )
                    if alt_task:
                        tasks.append(alt_task)
                        task_index += 1

        if cfg.include_impossible:
            impossible_task = _build_impossible_task(rng, sorted_examples, example, task_index, cfg)
            if impossible_task:
                tasks.append(impossible_task)
                task_index += 1

    return TaskPlan(
        version="gamibench_v1",
        seed=seed,
        tasks=tasks,
        metadata={
            "num_examples_discovered": len(sorted_examples),
            "num_tasks": len(tasks),
            "config": {
                "num_choices": cfg.num_choices,
                "include_standard": cfg.include_standard,
                "include_alternative": cfg.include_alternative,
                "include_impossible": cfg.include_impossible,
                "require_task1_correct_for_task2": cfg.require_task1_correct_for_task2,
                "reuse_distractors_for_task2": cfg.reuse_distractors_for_task2,
            },
        },
    )

