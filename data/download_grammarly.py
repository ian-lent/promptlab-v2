#!/usr/bin/env python3
"""Download pangram/editlens_iclr_grammarly (OOD calibration / eval)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("data/cache/grammarly"))
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN for gated dataset access.")

    ds = load_dataset("pangram/editlens_iclr_grammarly", token=token)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_table in ds.items():
        out_path = args.output_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for ex in split_table:
                text = str(ex.get("text", ex.get("document", ""))).strip()
                score = float(ex.get("score", ex.get("ai_score", 0.0)))
                rec = {
                    "text": text,
                    "score": max(0.0, min(1.0, score)),
                    "domain": str(ex.get("domain", "grammarly")),
                    "source": "editlens_grammarly",
                    "split": split_name,
                    "round": 0,
                }
                if len(rec["text"]) < 5:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
