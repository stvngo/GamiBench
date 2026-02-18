"""Dataset discovery helpers for GamiBench image folders."""

from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.gamibench.types import OrigamiExample


def _resolve_case_insensitive_file(folder: Path, expected_name: str) -> Optional[Path]:
    expected = expected_name.lower()
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.name.lower() == expected:
            return file_path
    return None


def _extract_viewpoint_files(folder: Path, origami_name: str) -> Dict[str, str]:
    prefix = f"{origami_name.lower()}_"
    viewpoints: Dict[str, str] = {}
    for file_path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() != ".png":
            continue

        file_name = file_path.name.lower()
        if not file_name.startswith(prefix):
            continue
        if file_name.endswith("_cp.png"):
            continue

        viewpoint = file_name[len(prefix) : -4]
        if not viewpoint:
            continue
        viewpoints[viewpoint] = str(file_path)
    return viewpoints


def discover_examples(dataset_root: Path) -> List[OrigamiExample]:
    """
    Build structured examples from the raw GamiBench image directory.

    Expected folder layout:
      - <name>/<name>_cp.png
      - <name>/impossible_<name>_cp.png
      - <name>/<name>_<viewpoint>.png
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Dataset root must be a directory: {root}")

    examples: List[OrigamiExample] = []
    for folder in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        name = folder.name
        normal_cp = _resolve_case_insensitive_file(folder, f"{name}_cp.png")
        impossible_cp = _resolve_case_insensitive_file(folder, f"impossible_{name}_cp.png")
        viewpoints = _extract_viewpoint_files(folder, name)

        examples.append(
            OrigamiExample(
                name=name,
                folder=str(folder),
                normal_cp=str(normal_cp) if normal_cp else None,
                impossible_cp=str(impossible_cp) if impossible_cp else None,
                viewpoints=viewpoints,
            )
        )

    return examples

