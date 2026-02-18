#!/usr/bin/env python3
"""Publish GamiBench dataset assets to a Hugging Face dataset repository."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import HfApi, upload_folder
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "huggingface_hub is required. Install with: pip install huggingface_hub"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/update a Hugging Face dataset repo for GamiBench"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id, e.g. username/GamiBench",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (optional if already logged in with hf auth login)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create dataset repo as private",
    )
    parser.add_argument(
        "--source-dir",
        default="data/GamiBench",
        help="Local dataset directory to upload",
    )
    parser.add_argument(
        "--path-in-repo",
        default="data/GamiBench",
        help="Destination path inside dataset repo",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch/revision to upload to",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload GamiBench dataset assets",
        help="Commit message for dataset upload",
    )
    parser.add_argument(
        "--card-file",
        default="hf/README_dataset.md",
        help="Optional dataset card markdown to upload as README.md",
    )
    parser.add_argument(
        "--no-card",
        action="store_true",
        help="Skip uploading dataset card",
    )
    parser.add_argument(
        "--configs-dir",
        default="configs/experiments",
        help="Optional configs directory to upload",
    )
    parser.add_argument(
        "--config-files",
        nargs="*",
        default=["gamibench_single.yaml", "gamibench_suite.yaml"],
        help="Config filenames to include from configs-dir",
    )
    parser.add_argument(
        "--no-configs",
        action="store_true",
        help="Skip uploading benchmark config files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not upload",
    )
    return parser.parse_args()


def _validate_dir(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def _print_header(args: argparse.Namespace, source_dir: Path, configs_dir: Optional[Path]) -> None:
    print("=== GamiBench -> Hugging Face Dataset Publish ===")
    print(f"Repo:             {args.repo_id} (dataset)")
    print(f"Revision:         {args.revision}")
    print(f"Private:          {args.private}")
    print(f"Source dir:       {source_dir}")
    print(f"Path in repo:     {args.path_in_repo}")
    print(f"Upload card:      {not args.no_card}")
    if not args.no_card:
        print(f"Card file:        {Path(args.card_file)}")
    print(f"Upload configs:   {not args.no_configs}")
    if not args.no_configs and configs_dir is not None:
        print(f"Configs dir:      {configs_dir}")
        print(f"Config files:     {', '.join(args.config_files)}")
    print(f"Dry run:          {args.dry_run}")
    print("=================================================")


def main() -> int:
    args = parse_args()
    source_dir = _validate_dir(args.source_dir, "source-dir")
    configs_dir = None if args.no_configs else _validate_dir(args.configs_dir, "configs-dir")
    card_path = Path(args.card_file)

    _print_header(args, source_dir, configs_dir)
    if args.dry_run:
        print("Dry run complete. No remote changes were made.")
        return 0

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(source_dir),
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
        token=args.token,
    )
    print(f"Uploaded dataset folder: {source_dir} -> {args.path_in_repo}")

    if not args.no_configs and configs_dir is not None:
        allow_patterns: List[str] = list(args.config_files)
        upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(configs_dir),
            path_in_repo="configs/experiments",
            allow_patterns=allow_patterns,
            revision=args.revision,
            commit_message="Upload GamiBench experiment configs",
            token=args.token,
        )
        print("Uploaded selected experiment config files.")

    if not args.no_card:
        if card_path.exists() and card_path.is_file():
            api.upload_file(
                repo_id=args.repo_id,
                repo_type="dataset",
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                revision=args.revision,
                commit_message="Upload dataset card",
            )
            print(f"Uploaded dataset card from {card_path}")
        else:
            print(f"Skipped dataset card: file not found at {card_path}")

    print(f"Done. Dataset URL: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

