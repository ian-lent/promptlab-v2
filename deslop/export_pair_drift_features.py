#!/usr/bin/env python3
"""
Build drift-coefficient optimization rows from held-out prompt pairs.

Each pair contributes one essay (Groq, using the evolved ``output_prompt`` with the real topic
substituted) and a feature vector::

    detector_slop, drift_semantic, drift_rouge_l, drift_bertscore

Rows include ``pair_row_id`` so ``drift_coef_opt`` can filter with ``rewriter_split_manifest.json``
(test split only — no overlap with rewriter training).

This script is intentionally **offline / expensive** (one Groq call per pair); run it after
co-training produced ``prompt_pairs.jsonl`` and the split manifest exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from sentence_transformers import SentenceTransformer

from deslop.run_topic import make_groq_essay_fn
from deslop.similarity import DriftWeights, composite_drift_penalty, drift_options_from_config
from detector.model import SlopDetector
from rewriter.dataset import load_manifest, load_pairs_jsonl, pair_row_id, pairs_in_split
from rewriter.inference import apply_topic_placeholder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-jsonl", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--split",
        default="test",
        help="Which manifest split to export (use 'test' for rewriter holdout).",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/deslop.yaml"))
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows in split")
    args = ap.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    drift_kw = drift_options_from_config(cfg)
    embed_name = str(cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    embedder = SentenceTransformer(embed_name)
    det = SlopDetector(
        checkpoint=str(cfg.get("detector_checkpoint", "pangram/editlens_roberta-large"))
    )
    essay_model = str(cfg.get("groq_model", "llama-3.3-70b-versatile"))
    groq = make_groq_essay_fn(essay_model, float(cfg.get("essay_temperature", 0.9)))

    manifest = load_manifest(args.manifest)
    rows = pairs_in_split(load_pairs_jsonl(args.pairs_jsonl), manifest, args.split)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    dw = DriftWeights(1.0, 1.0, 1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as wf:
        for r in rows:
            topic = str(r.get("topic", "")).strip()
            out_prompt = str(r.get("output_prompt", "")).strip()
            essay_prompt = apply_topic_placeholder(out_prompt, topic or "the assigned topic")
            essay = groq(essay_prompt)
            s = float(det.score(essay))
            _, detail = composite_drift_penalty(
                topic,
                essay,
                dw,
                use_bertscore=bool(drift_kw.get("drift_use_bertscore", False)),
                raw_slop=s,
                bertscore_slop_gate=drift_kw.get("drift_bertscore_slop_gate"),
                embedder=embedder,
                embedding_model_name=embed_name,
                alignment_scale="topic",
            )
            rec = {
                "pair_row_id": pair_row_id(r),
                "topic": topic,
                "detector_slop": s,
                "drift_semantic": float(detail["drift_semantic"]),
                "drift_rouge_l": float(detail["drift_rouge_l"]),
                "drift_bertscore": float(detail.get("drift_bertscore", 0.0)),
                "bertscore_applied": float(detail.get("bertscore_applied", 0.0)),
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
