#!/usr/bin/env python3
"""
Generate essay-to-essay training pairs from organic prompt pairs.

The current organic co-training log (`outputs/cotrain/prompt_pairs.jsonl`) contains prompt
pairs but not the essays themselves. For the essay rewriting pivot, we materialize:

- baseline_essay: essay written with a generic prompt for the topic
- best_essay:     essay written using the evolved output_prompt template for the topic

Writes a new JSONL (does not overwrite prompt_pairs.jsonl):
  outputs/rewriter/essay_pairs_organic.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from deslop.run_topic import make_groq_essay_fn
from detector.model import SlopDetector
from rewriter.inference import apply_topic_placeholder


GENERIC_PROMPT = "Write an essay about {topic}. Be thorough and informative."


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Generate baseline/best essays for essay rewriting.")
    p.add_argument("--config", type=Path, default=Path("configs/rewriter.yaml"))
    p.add_argument("--pairs", type=Path, default=Path("outputs/cotrain/prompt_pairs.jsonl"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/rewriter/essay_pairs_organic.jsonl"),
    )
    p.add_argument("--max-new-tokens", type=int, default=800)
    args = p.parse_args()

    cfg_path = args.config
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    repo_root = cfg_path.resolve().parent.parent

    pairs_path = args.pairs
    if not pairs_path.is_absolute():
        pairs_path = (repo_root / pairs_path).resolve()
    rows = _load_jsonl(pairs_path)
    if not rows:
        raise SystemExit(f"No prompt pairs found: {pairs_path}")

    groq_model = str(cfg.get("groq_model", "llama-3.3-70b-versatile"))
    essay_temperature = float(cfg.get("essay_temperature", 0.9))
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Missing GROQ_API_KEY in environment.")
    llm = make_groq_essay_fn(groq_model, essay_temperature)

    detector_ckpt = str(cfg.get("detector_checkpoint", "pangram/editlens_roberta-large"))
    det = SlopDetector(checkpoint=detector_ckpt)

    out_path = args.out
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            topic = str(r.get("topic", "")).strip()
            out_prompt = str(r.get("output_prompt", "")).strip()
            if not topic or not out_prompt:
                continue

            baseline_prompt = GENERIC_PROMPT.format(topic=topic)
            best_prompt = apply_topic_placeholder(out_prompt, topic)

            baseline_essay = llm(baseline_prompt)
            best_essay = llm(best_prompt)

            rec = {
                "topic": topic,
                "source": "organic",
                "input_prompt": str(r.get("input_prompt", "")).strip(),
                "output_prompt": out_prompt,
                "baseline_prompt": baseline_prompt,
                "best_prompt": best_prompt,
                "baseline_essay": baseline_essay,
                "best_essay": best_essay,
                "baseline_slop": float(det.score(baseline_essay)),
                "best_slop": float(det.score(best_essay)),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(json.dumps({"essay_pairs_written": str(out_path), "n_written": n_written}), flush=True)


if __name__ == "__main__":
    main()

