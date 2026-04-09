#!/usr/bin/env python3
"""End-to-end eval: topics → prompts → generate → score (stub orchestration)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from detector.model import SlopDetector
from eval.metrics import aggregate_run
from sentence_transformers import SentenceTransformer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    args = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    topics_path = cfg.get("topics_file")
    topics = []
    if topics_path and Path(topics_path).exists():
        topics = [ln.strip() for ln in Path(topics_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not topics:
        topics = ["the future of remote work", "climate policy and justice"]

    det = SlopDetector(checkpoint=cfg.get("external_detector_checkpoint", "pangram/editlens_roberta-large"))
    embedder = SentenceTransformer(cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    lex = Path(cfg["slop_lexicon_dir"]) if cfg.get("slop_lexicon_dir") else None

    results = []
    for topic in topics[: int(cfg.get("num_topics", 50))]:
        essays = [f"Discussion of {topic}. " * 30]
        metrics = aggregate_run(det, None, embedder, essays, topic, lexicon_dir=lex)
        results.append({"topic": topic, **{k: v for k, v in metrics.items() if k != "note"}})

    out_path = Path(cfg.get("results_json", "outputs/eval/results.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
