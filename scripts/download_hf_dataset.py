#!/usr/bin/env python3
"""Download/sync GamiBench dataset assets from Hugging Face."""

from __future__ import annotations

import argparse

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "huggingface_hub is required. Install with: pip install huggingface_hub"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a GamiBench dataset repo snapshot from Hugging Face"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id, e.g. username/GamiBench",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token for private datasets (optional if already logged in)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch/revision to download",
    )
    parser.add_argument(
        "--local-dir",
        default="data",
        help="Local destination directory",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=["data/GamiBench/**"],
        help="Optional include patterns for partial download",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=[],
        help="Optional exclude patterns",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=args.allow_patterns or None,
        ignore_patterns=args.ignore_patterns or None,
        token=args.token,
    )
    print(f"Downloaded dataset snapshot to: {downloaded_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

