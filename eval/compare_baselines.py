#!/usr/bin/env python3
"""Compare naive prompt, hand-crafted, post-hoc deslop, evolutionary, rewriter, etc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from detector.model import SlopDetector
from eval.metrics import aggregate_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    p.add_argument("--out", type=Path, default=Path("outputs/eval/baseline_table.json"))
    args = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    external = SlopDetector(checkpoint=cfg.get("external_detector_checkpoint", "pangram/editlens_roberta-large"))
    co_path = cfg.get("cotrained_detector_dir")
    primary = external
    if co_path and (Path(co_path) / "best").exists():
        primary = SlopDetector(checkpoint=str(Path(co_path) / "best"))

    embedder = SentenceTransformer(cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    lex = Path(cfg["slop_lexicon_dir"]) if cfg.get("slop_lexicon_dir") else None

    # Placeholder essays — replace with real generations per baseline
    topic = "the ethics of AI in education"
    naive_essays = ["Sample essay paragraph " * 40]
    table = {
        "naive_prompt": aggregate_run(primary, external, embedder, naive_essays, topic, lexicon_dir=lex),
        "note": "Wire real generators for each baseline (see project spec Phase 5).",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
