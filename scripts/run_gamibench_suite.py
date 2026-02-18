#!/usr/bin/env python3
"""Run GamiBench across one or many models with deterministic task plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

# Ensure local package imports work when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.gamibench.discovery import discover_examples
from benchmarks.gamibench.task_builder import TaskBuilderConfig, build_task_plan
from pipeline import Pipeline
from utils.config_loader import load_config
from utils.logger import setup_logger


def _sanitize_model_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def _config_to_dict(config: Any) -> Dict[str, Any]:
    return dict(config) if config is not None else {}


def _resolve_dataset_path(config_path: Path, dataset_path: str) -> Path:
    path = Path(dataset_path)
    if path.is_absolute():
        return path
    config_relative = (config_path.parent / path).resolve()
    if config_relative.exists():
        return config_relative
    return (Path.cwd() / path).resolve()


def _extract_grouped_models(config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    suite_cfg = config.get("suite", {})

    grouped: Dict[str, List[Dict[str, Any]]] = {"closed": [], "open": []}
    if "model_groups" in suite_cfg:
        raw_groups = suite_cfg["model_groups"]
        for group_name in ("closed", "open"):
            grouped[group_name] = [dict(item) for item in raw_groups.get(group_name, [])]
        return grouped

    models = suite_cfg.get("models", [])
    for item in models:
        model = dict(item)
        group = model.get("group", "open")
        grouped.setdefault(group, []).append(model)
    return grouped


def _select_models(
    grouped_models: Dict[str, List[Dict[str, Any]]],
    group: str,
    selected_model_ids: List[str],
) -> List[Dict[str, Any]]:
    if group == "all":
        models = grouped_models.get("closed", []) + grouped_models.get("open", [])
    else:
        models = grouped_models.get(group, [])

    if selected_model_ids:
        selected = set(selected_model_ids)
        models = [
            model
            for model in models
            if model.get("id") in selected or model.get("name") in selected
        ]
    return models


def _make_task_builder_config(evaluator_cfg: Dict[str, Any]) -> TaskBuilderConfig:
    return TaskBuilderConfig(
        num_choices=int(evaluator_cfg.get("num_choices", 4)),
        include_standard=bool(evaluator_cfg.get("include_standard", True)),
        include_alternative=bool(evaluator_cfg.get("include_alternative", True)),
        include_impossible=bool(evaluator_cfg.get("include_impossible", True)),
        require_task1_correct_for_task2=bool(
            evaluator_cfg.get("require_task1_correct_for_task2", True)
        ),
        reuse_distractors_for_task2=bool(evaluator_cfg.get("reuse_distractors_for_task2", True)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic multi-model GamiBench suite")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/gamibench_suite.yaml",
        help="Path to suite config YAML",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["closed", "open", "all"],
        help="Run only closed models, only open models, or all configured models",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional specific model ids/names to run",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from model checkpoints")
    parser.add_argument("--output-root", type=str, default=None, help="Override suite output root")
    parser.add_argument("--task-plan-path", type=str, default=None, help="Use/save deterministic task plan here")
    parser.add_argument("--dry-run", action="store_true", help="Print selected models and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    logger = setup_logger(name="gamibench_suite")

    grouped_models = _extract_grouped_models(config)
    selected_models = _select_models(grouped_models, args.group, args.models)
    if not selected_models:
        logger.error("No models selected. Check --group/--models and suite config.")
        return 1

    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    base_experiment_name = config.get("experiment_name", "gamibench_suite")

    suite_cfg = _config_to_dict(config.get("suite", {}))
    default_output_root = Path(suite_cfg.get("output_root", "outputs/results/gamibench_suite"))
    output_root = Path(args.output_root) if args.output_root else default_output_root
    output_root.mkdir(parents=True, exist_ok=True)

    evaluator_cfg = _config_to_dict(config.get("evaluator", {}))
    dataset_cfg = _config_to_dict(config.get("dataset", {}))
    dataset_path = _resolve_dataset_path(config_path, dataset_cfg.get("path", "data/GamiBench"))

    task_plan_path = (
        Path(args.task_plan_path)
        if args.task_plan_path
        else Path(
            suite_cfg.get(
                "task_plan_path",
                output_root / f"task_plan_seed{seed}.json",
            )
        )
    )
    task_plan_path.parent.mkdir(parents=True, exist_ok=True)

    if not task_plan_path.exists():
        examples = discover_examples(dataset_path)
        builder_cfg = _make_task_builder_config(evaluator_cfg)
        task_plan = build_task_plan(examples=examples, seed=seed, config=builder_cfg)
        task_plan_path.write_text(json.dumps(task_plan.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"Generated task plan with {len(task_plan.tasks)} tasks at {task_plan_path}")
    else:
        logger.info(f"Using existing task plan at {task_plan_path}")

    if args.dry_run:
        logger.info("Dry-run selected models:")
        for model in selected_models:
            logger.info(f" - {model.get('id', model.get('name'))} ({model.get('type')})")
        return 0

    summary_rows: List[Dict[str, Any]] = []
    for model_spec in selected_models:
        model_id = model_spec.get("id") or model_spec.get("name")
        if not model_id:
            logger.warning(f"Skipping model without id/name: {model_spec}")
            continue
        safe_model_id = _sanitize_model_id(model_id)
        model_output_dir = output_root / safe_model_id
        model_output_dir.mkdir(parents=True, exist_ok=True)

        model_cfg = {
            key: value
            for key, value in model_spec.items()
            if key not in {"id", "group", "description", "enabled"}
        }
        eval_cfg_for_model = dict(evaluator_cfg)
        eval_cfg_for_model["task_plan_path"] = str(task_plan_path)
        eval_cfg_for_model["checkpoint_path"] = str(model_output_dir / "checkpoint.json")

        overrides = {
            "seed": seed,
            "experiment_name": f"{base_experiment_name}_{safe_model_id}",
            "model": model_cfg,
            "evaluator": eval_cfg_for_model,
            "output_dir": str(model_output_dir),
        }

        logger.info(f"Running model {model_id} ({model_cfg.get('type')})")
        try:
            pipeline = Pipeline(config_path=str(config_path), overrides=overrides)
            result = pipeline.run(resume=args.resume)
            row = {
                "model_id": model_id,
                "model_type": model_cfg.get("type"),
                "status": "ok",
                "metrics": result.get("metrics", {}),
                "output_dir": str(model_output_dir),
            }
            logger.info(
                "Completed %s: accuracy=%.4f",
                model_id,
                row["metrics"].get("accuracy", 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "model_id": model_id,
                "model_type": model_cfg.get("type"),
                "status": "failed",
                "error": str(exc),
                "output_dir": str(model_output_dir),
            }
            logger.exception("Model run failed for %s", model_id)
        summary_rows.append(row)

    summary = {
        "seed": seed,
        "group": args.group,
        "task_plan_path": str(task_plan_path),
        "models_requested": [m.get("id", m.get("name")) for m in selected_models],
        "results": summary_rows,
    }
    summary_path = output_root / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Wrote suite summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

