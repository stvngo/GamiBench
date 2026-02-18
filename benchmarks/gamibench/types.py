"""Data models for GamiBench task construction and execution."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TASK_STANDARD = "standard_mcq"
TASK_ALTERNATIVE = "alternative_view"
TASK_IMPOSSIBLE = "impossible_fold"
TASK_TYPES = (TASK_STANDARD, TASK_ALTERNATIVE, TASK_IMPOSSIBLE)


@dataclass
class OrigamiExample:
    """Structured metadata for a single origami folder."""

    name: str
    folder: str
    normal_cp: Optional[str]
    impossible_cp: Optional[str]
    viewpoints: Dict[str, str]

    def available_viewpoints(self) -> List[str]:
        """Return sorted viewpoint names."""
        return sorted(self.viewpoints.keys())


@dataclass
class TaskSample:
    """One multiple-choice question instance."""

    task_id: str
    task_type: str
    origami_name: str
    crease_pattern_path: str
    candidates: List[str]
    correct_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "origami_name": self.origami_name,
            "crease_pattern_path": self.crease_pattern_path,
            "candidates": list(self.candidates),
            "correct_label": self.correct_label,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSample":
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            origami_name=data["origami_name"],
            crease_pattern_path=data["crease_pattern_path"],
            candidates=list(data["candidates"]),
            correct_label=data["correct_label"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TaskPlan:
    """Serializable task-plan for deterministic evaluation across models."""

    version: str
    seed: int
    tasks: List[TaskSample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "tasks": [task.to_dict() for task in self.tasks],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPlan":
        return cls(
            version=data["version"],
            seed=int(data["seed"]),
            tasks=[TaskSample.from_dict(item) for item in data.get("tasks", [])],
            metadata=dict(data.get("metadata", {})),
        )

