#!/usr/bin/env python3
"""
Download `mannacharya/aeon-essays-dataset` from Kaggle into `data/source/essays.csv`.

Requires: `pip install kaggle` and `~/.kaggle/kaggle.json` (API token from kaggle.com/settings).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="mannacharya/aeon-essays-dataset",
        help="Kaggle dataset slug (owner/name)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/source"))
    args = p.parse_args()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise SystemExit(
            "Install the Kaggle API client: pip install kaggle\n"
            "Then add ~/.kaggle/kaggle.json from https://www.kaggle.com/settings"
        ) from e

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(args.dataset, path=str(args.out_dir), unzip=True)

    candidates = list(args.out_dir.rglob("essays.csv"))
    target = args.out_dir / "essays.csv"
    if not candidates:
        raise SystemExit(
            f"No essays.csv found under {args.out_dir}. "
            "List files and copy/rename manually, or check the dataset layout on Kaggle."
        )

    best = candidates[0]
    if len(candidates) > 1:
        # Prefer top-level data/source/essays.csv path
        for c in candidates:
            if c.name == "essays.csv" and c.parent == args.out_dir:
                best = c
                break

    if best.resolve() != target.resolve():
        shutil.copy2(best, target)

    print(f"Ready: {target.resolve()} ({target.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
