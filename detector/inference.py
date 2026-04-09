#!/usr/bin/env python3
"""Score text with a SlopDetector checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from detector.model import SlopDetector


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Score text with SlopDetector. "
            "Default scoring truncates to RoBERTa max length; use --chunked to score long essays."
        )
    )
    p.add_argument("--checkpoint", default="pangram/editlens_roberta-large")
    p.add_argument("--text", default=None, help="Single string to score")
    p.add_argument("--text-file", type=str, default=None, help="Path to UTF-8 text file (full essay)")
    p.add_argument("--jsonl", type=str, default=None, help="Path to JSONL with 'text' field")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--score-decode",
        choices=("expected_value", "top_bucket_prob"),
        default="expected_value",
        help="Same as SlopDetector.score_decode",
    )
    p.add_argument(
        "--chunked",
        action="store_true",
        help="Use chunked scoring (sliding windows) and return mean/max across chunks.",
    )
    p.add_argument("--window-tokens", type=int, default=512, help="Chunk window length in tokens")
    p.add_argument(
        "--stride-tokens",
        type=int,
        default=256,
        help="Stride (step) between chunks in tokens",
    )
    p.add_argument("--max-chunks", type=int, default=None, help="Optional cap on number of chunks")
    args = p.parse_args()

    det = SlopDetector(checkpoint=args.checkpoint, score_decode=args.score_decode)

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8", errors="replace").strip()
        if args.chunked:
            long = det.score_long(
                text,
                window_tokens=args.window_tokens,
                stride_tokens=args.stride_tokens,
                max_chunks=args.max_chunks,
            )
            out = {"slop_score_mean": long["mean"], "slop_score_max": long["max"], **long, "chars": len(text)}
        else:
            out = {"slop_score": det.score(text), "chars": len(text)}
        print(json.dumps(out))
        return

    if args.text:
        print(json.dumps({"slop_score": det.score(args.text)}))
        return

    if args.jsonl:
        texts = []
        with open(args.jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                texts.append(str(row.get("text", "")))
        scores = det.score_batch(texts, batch_size=args.batch_size)
        for t, s in zip(texts, scores, strict=True):
            print(json.dumps({"slop_score": s, "text_preview": t[:120]}))
        return

    print("Provide --text or --jsonl", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
