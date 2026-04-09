#!/usr/bin/env python3
"""Download pangram/editlens_iclr from Hugging Face (gated; requires HF_TOKEN)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def row_to_record(example: dict, split: str, text_field: str, score_field: str) -> dict:
    text = example.get(text_field) or example.get("text") or example.get("document", "")
    if text is None:
        text = ""
    score = example.get(score_field)
    if score is None:
        for k in ("ai_score", "label", "ai_involvement", "score_continuous"):
            if k in example and example[k] is not None:
                score = float(example[k])
                break
    if score is None:
        score = 0.0
    else:
        score = float(score)
    domain = str(example.get("domain", example.get("category", "unknown")))
    source = "editlens_iclr"
    return {
        "text": str(text).strip(),
        "score": max(0.0, min(1.0, score)),
        "domain": domain,
        "source": source,
        "split": split,
        "round": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("data/cache/editlens"))
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--score-field", type=str, default="score")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) for gated dataset access.")

    ds = load_dataset("pangram/editlens_iclr", token=token)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_table in ds.items():
        out_path = args.output_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for ex in split_table:
                rec = row_to_record(ex, split_name, args.text_field, args.score_field)
                if len(rec["text"]) < 10:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path} ({split_table.num_rows} rows processed)")


if __name__ == "__main__":
    main()
