#!/usr/bin/env python3
"""
Rule-based augmentation of input prompts (no LLM calls).

Reads:  outputs/cotrain/prompt_pairs.jsonl
Writes: outputs/cotrain/prompt_pairs_augmented.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rewriter.dataset import load_pairs_jsonl, pair_row_id


SUBS: list[tuple[str, str]] = [
    ("Write an essay about", "Compose a piece discussing"),
    ("Discuss", "Explore"),
    ("Explain", "Describe"),
    ("What is", "Tell me about"),
    ("the role of", "how"),
]


def _apply_one_sub(s: str, old: str, new: str) -> str:
    # Prefer phrase-boundary replacement when possible.
    if old.lower() == old:
        pat = re.compile(rf"\b{re.escape(old)}\b", flags=re.IGNORECASE)
        return pat.sub(new, s, count=1)
    return s.replace(old, new, 1)


def _variants(src: str) -> list[str]:
    out: list[str] = []
    cur = src
    # generate up to 3 variants by applying one substitution per variant
    for old, new in SUBS:
        v = _apply_one_sub(cur, old, new)
        if v != cur and v.strip():
            out.append(v.strip())
        if len(out) >= 3:
            break
    # If still short, try applying subs to the original independently.
    if len(out) < 3:
        for old, new in SUBS:
            v = _apply_one_sub(src, old, new)
            if v != src and v.strip() and v.strip() not in out:
                out.append(v.strip())
            if len(out) >= 3:
                break
    return out[:3]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate rule-based input prompt variants.")
    p.add_argument("--pairs", type=Path, default=Path("outputs/cotrain/prompt_pairs.jsonl"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/cotrain/prompt_pairs_augmented.jsonl"),
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    pairs_path = args.pairs
    if not pairs_path.is_absolute():
        pairs_path = (repo_root / pairs_path).resolve()
    rows = load_pairs_jsonl(pairs_path)
    if not rows:
        raise SystemExit(f"No pairs found: {pairs_path}")

    out_path = args.out
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            src = str(r.get("input_prompt", r.get("input", ""))).strip()
            tgt = str(r.get("output_prompt", r.get("output", ""))).strip()
            topic = str(r.get("topic", "")).strip()
            if not src or not tgt:
                continue
            pid = pair_row_id(r)
            for v in _variants(src):
                rec: dict[str, Any] = {
                    "topic": topic,
                    "input_prompt": v,
                    "output_prompt": tgt,
                    "source": "augmented",
                    "source_pair_id": pid,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(json.dumps({"augmented_written": str(out_path), "n_written": n_written}), flush=True)


if __name__ == "__main__":
    main()

