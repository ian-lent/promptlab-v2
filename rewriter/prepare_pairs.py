#!/usr/bin/env python3
"""Build rewriter training JSONL from co-training pair logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _dedupe_by_output(rows: list[dict]) -> list[dict]:
    """Keep highest improvement per output_prompt."""
    best: dict[str, dict] = {}
    for r in rows:
        out_p = r.get("output_prompt", "")
        imp = float(r.get("improvement", 0))
        prev = best.get(out_p)
        if prev is None or imp > float(prev.get("improvement", 0)):
            best[out_p] = r
    return list(best.values())


def prepare_training_data(
    pair_log_path: str | Path,
    output_path: str | Path,
    min_improvement: float = 0.1,
    strategy: str = "seed_to_best",
) -> int:
    pair_log_path = Path(pair_log_path)
    output_path = Path(output_path)
    if not pair_log_path.exists():
        raise FileNotFoundError(pair_log_path)

    rows: list[dict] = []
    with pair_log_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    rows = [r for r in rows if float(r.get("improvement", 0)) >= min_improvement]
    rows = _dedupe_by_output(rows)

    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        if strategy == "consecutive":
            for r in rows:
                rec = _record(r)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        elif strategy in ("seed_to_best", "trajectory_endpoints"):
            by_topic: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by_topic[str(r.get("topic", ""))].append(r)
            for _topic, lst in by_topic.items():
                best = min(lst, key=lambda x: float(x.get("output_slop_score", 1.0)))
                rec = _record(best)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        else:
            raise ValueError(strategy)

    return written


def _record(r: dict) -> dict:
    return {
        "input": r["input_prompt"],
        "output": r["output_prompt"],
        "topic": r.get("topic", ""),
        "input_slop_score": r.get("input_slop_score"),
        "output_slop_score": r.get("output_slop_score"),
        "round": r.get("round", 0),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pair-log", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/rewriter/train_pairs.jsonl"))
    p.add_argument("--min-improvement", type=float, default=0.1)
    p.add_argument(
        "--strategy",
        choices=("seed_to_best", "consecutive", "trajectory_endpoints"),
        default="seed_to_best",
    )
    args = p.parse_args()
    n = prepare_training_data(
        args.pair_log,
        args.output,
        min_improvement=args.min_improvement,
        strategy=args.strategy,
    )
    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
