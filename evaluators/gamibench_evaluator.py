"""Evaluator for end-to-end GamiBench multimodal multiple-choice tasks."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from benchmarks.gamibench.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    candidate_caption,
    candidate_label,
)
from benchmarks.gamibench.scoring import compute_metrics, parse_choice
from benchmarks.gamibench.task_builder import TaskBuilderConfig, build_task_plan
from benchmarks.gamibench.types import (
    OrigamiExample,
    TaskPlan,
    TaskSample,
    TASK_ALTERNATIVE,
)
from evaluators.base import BaseEvaluator
from models.base import BaseModel
from utils.logger import setup_logger


def _to_plain_dict(config: Any) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return dict(config)
    return dict(config)


class GamiBenchEvaluator(BaseEvaluator):
    """Run the three benchmark tasks from the notebook as reproducible scripts."""

    def __init__(
        self,
        model: BaseModel,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
        full_config: Optional[DictConfig] = None,
    ):
        super().__init__(model=model, data=data, config=config)
        self.full_config = full_config
        cfg = _to_plain_dict(config)

        logger_name = "gamibench_evaluator"
        log_dir = "outputs/logs"
        if full_config is not None:
            logger_name = full_config.get("experiment_name", logger_name)
            log_dir = full_config.get("log_dir", log_dir)
        self.logger = setup_logger(name=logger_name, log_dir=log_dir)

        global_seed = 42
        if full_config is not None:
            global_seed = int(full_config.get("seed", 42))
        self.seed = int(cfg.get("seed", global_seed))
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 8)
        self.save_every = int(cfg.get("save_every", 1))

        self.task_plan_path = Path(cfg["task_plan_path"]) if cfg.get("task_plan_path") else None
        checkpoint_override = cfg.get("checkpoint_path")
        if checkpoint_override:
            self.checkpoint_path = Path(checkpoint_override)
        else:
            output_dir = "outputs/results"
            if full_config is not None:
                output_dir = full_config.get("output_dir", output_dir)
            self.checkpoint_path = Path(output_dir) / "checkpoint.json"

        self.task_builder_config = TaskBuilderConfig(
            num_choices=int(cfg.get("num_choices", 4)),
            include_standard=bool(cfg.get("include_standard", True)),
            include_alternative=bool(cfg.get("include_alternative", True)),
            include_impossible=bool(cfg.get("include_impossible", True)),
            require_task1_correct_for_task2=bool(
                cfg.get("require_task1_correct_for_task2", True)
            ),
            reuse_distractors_for_task2=bool(cfg.get("reuse_distractors_for_task2", True)),
        )

    def _coerce_examples(self) -> List[OrigamiExample]:
        examples: List[OrigamiExample] = []
        for item in self.data:
            if isinstance(item, OrigamiExample):
                examples.append(item)
            elif isinstance(item, dict):
                examples.append(OrigamiExample(**item))
            else:
                raise ValueError(f"Unsupported data row type: {type(item)}")
        return examples

    def _load_task_plan_from_disk(self, task_plan_path: Path) -> TaskPlan:
        payload = json.loads(task_plan_path.read_text(encoding="utf-8"))
        return TaskPlan.from_dict(payload)

    def _resolve_task_plan(self) -> TaskPlan:
        if self.task_plan_path and self.task_plan_path.exists():
            self.logger.info(f"Loading task plan from {self.task_plan_path}")
            return self._load_task_plan_from_disk(self.task_plan_path)

        examples = self._coerce_examples()
        if not examples:
            raise ValueError(
                "No examples found in evaluator data and no task_plan_path provided"
            )

        plan = build_task_plan(examples=examples, seed=self.seed, config=self.task_builder_config)
        if self.task_plan_path:
            self.task_plan_path.parent.mkdir(parents=True, exist_ok=True)
            self.task_plan_path.write_text(
                json.dumps(plan.to_dict(), indent=2),
                encoding="utf-8",
            )
            self.logger.info(f"Wrote task plan to {self.task_plan_path}")
        return plan

    def _load_checkpoint(self) -> Dict[str, Dict[str, Any]]:
        if not self.checkpoint_path.exists():
            return {}
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        results_by_id = payload.get("results_by_task_id", {})
        if isinstance(results_by_id, dict):
            return results_by_id
        return {}

    def _save_checkpoint(self, results_by_id: Dict[str, Dict[str, Any]]) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now().isoformat(),
            "results_by_task_id": results_by_id,
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _build_content(self, task: TaskSample) -> List[Dict[str, str]]:
        content: List[Dict[str, str]] = [
            {"type": "text", "text": USER_PROMPT},
            {"type": "image_path", "path": task.crease_pattern_path},
        ]
        for index, candidate in enumerate(task.candidates):
            content.append({"type": "text", "text": candidate_caption(task, index)})
            content.append({"type": "image_path", "path": candidate})
        return content

    def _skip_record(self, task: TaskSample, reason: str) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "origami_name": task.origami_name,
            "expected_answer": task.correct_label,
            "model_answer": None,
            "raw_response": None,
            "is_correct": False,
            "parse_ok": False,
            "status": "skipped",
            "skip_reason": reason,
            "timestamp": datetime.now().isoformat(),
            "metadata": task.metadata,
        }

    def _evaluate_task(self, task: TaskSample) -> Dict[str, Any]:
        try:
            raw_response = self.model.chat_multimodal(
                system_prompt=SYSTEM_PROMPT,
                content=self._build_content(task),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "origami_name": task.origami_name,
                "expected_answer": task.correct_label,
                "model_answer": None,
                "raw_response": None,
                "is_correct": False,
                "parse_ok": False,
                "status": "api_error",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
                "metadata": task.metadata,
            }

        answer = parse_choice(raw_response)
        if answer is None:
            return {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "origami_name": task.origami_name,
                "expected_answer": task.correct_label,
                "model_answer": None,
                "raw_response": raw_response,
                "is_correct": False,
                "parse_ok": False,
                "status": "parse_error",
                "timestamp": datetime.now().isoformat(),
                "metadata": task.metadata,
            }

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "origami_name": task.origami_name,
            "expected_answer": task.correct_label,
            "model_answer": answer,
            "raw_response": raw_response,
            "is_correct": answer == task.correct_label,
            "parse_ok": True,
            "status": "ok",
            "candidate_labels": [candidate_label(i) for i in range(len(task.candidates))],
            "candidate_paths": list(task.candidates),
            "crease_pattern_path": task.crease_pattern_path,
            "timestamp": datetime.now().isoformat(),
            "metadata": task.metadata,
        }

    def _should_skip_for_dependency(
        self,
        task: TaskSample,
        results_by_task_id: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        if task.task_type != TASK_ALTERNATIVE:
            return None
        if not task.metadata.get("requires_previous_correct", False):
            return None
        prerequisite = task.metadata.get("prerequisite_task_id")
        if not prerequisite:
            return "missing_prerequisite_task_id"
        prerequisite_result = results_by_task_id.get(prerequisite)
        if prerequisite_result is None:
            return "prerequisite_result_not_found"
        if not prerequisite_result.get("is_correct", False):
            return "prerequisite_incorrect"
        return None

    def evaluate(self, resume: bool = False) -> Dict[str, Any]:
        task_plan = self._resolve_task_plan()
        results_by_task_id: Dict[str, Dict[str, Any]] = {}
        if resume:
            results_by_task_id = self._load_checkpoint()
            if results_by_task_id:
                self.logger.info(f"Resumed {len(results_by_task_id)} completed tasks from checkpoint")

        pending_since_save = 0
        for task in task_plan.tasks:
            if task.task_id in results_by_task_id:
                continue

            skip_reason = self._should_skip_for_dependency(task, results_by_task_id)
            if skip_reason:
                result = self._skip_record(task, reason=skip_reason)
            else:
                result = self._evaluate_task(task)
            results_by_task_id[task.task_id] = result
            pending_since_save += 1

            if self.save_every > 0 and pending_since_save >= self.save_every:
                self._save_checkpoint(results_by_task_id)
                pending_since_save = 0

        self._save_checkpoint(results_by_task_id)

        ordered_results = [
            results_by_task_id[task.task_id]
            for task in task_plan.tasks
            if task.task_id in results_by_task_id
        ]
        metrics = compute_metrics(ordered_results)
        return {
            "results": ordered_results,
            "metrics": metrics,
            "metadata": {
                "seed": self.seed,
                "task_plan_version": task_plan.version,
                "num_tasks": len(task_plan.tasks),
                "model": str(self.model),
                "checkpoint_path": str(self.checkpoint_path),
                "task_plan_path": str(self.task_plan_path) if self.task_plan_path else None,
            },
        }

    def evaluate_single(self, example: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "GamiBenchEvaluator runs through evaluate() with a deterministic task plan"
        )

