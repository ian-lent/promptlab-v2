#!/usr/bin/env python3
"""
Phase 2 entrypoint: evolutionary prompt optimization for one topic (Groq essay + detector).

Budget (rough): essay_calls ≈ population_size × generations × essays_per_candidate;
plus Groq calls from refill_population (mutations). See configs/deslop.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from deslop.drift_coef_opt import load_optimized_drift_weights
from deslop.optimizer import optimize
from deslop.similarity import DriftWeights, drift_options_from_config
from detector.model import SlopDetector


def make_groq_essay_fn(model: str, temperature: float, max_tokens: int = 2048):
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY")

    client = Groq(api_key=key)

    def fn(prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                raise SystemExit(
                    f"Groq rate limit: {e}\n"
                    "Wait for the reset window, upgrade tier at console.groq.com, or set a smaller "
                    "model in YAML (e.g. groq_model / mutator_groq_model: llama-3.1-8b-instant) "
                    "to reduce token usage."
                ) from e
            raise
        return (resp.choices[0].message.content or "").strip()

    return fn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/deslop.yaml"))
    p.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Essay topic (overrides 'topic' in config if set)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override detector checkpoint (default: pangram/editlens_roberta-large)",
    )
    args = p.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    topic = (args.topic or cfg.get("topic") or "").strip()
    if not topic:
        raise SystemExit(
            "No topic: pass --topic \"...\" or add a non-empty 'topic:' field to your config YAML."
        )
    ckpt = args.checkpoint or "pangram/editlens_roberta-large"
    det = SlopDetector(checkpoint=ckpt)

    essay_model = str(cfg.get("groq_model", "llama-3.3-70b-versatile"))
    target_llm = make_groq_essay_fn(
        essay_model,
        float(cfg.get("essay_temperature", 0.9)),
    )

    log_path = Path(cfg.get("optimization_log_jsonl", "outputs/deslop/runs.jsonl"))

    drift_kw = drift_options_from_config(cfg)
    ocp = cfg.get("optimized_drift_coefs_path")
    if ocp:
        loaded = load_optimized_drift_weights(Path(ocp))
        if loaded:
            drift_kw = dict(drift_kw)
            drift_kw["drift_weights"] = DriftWeights(
                alpha_semantic=loaded["alpha_semantic"],
                alpha_rouge=loaded["alpha_rouge"],
                alpha_bertscore=loaded["alpha_bertscore"],
            )

    best, essays = optimize(
        topic,
        target_llm,
        det,
        population_size=int(cfg.get("population_size", 6)),
        generations=int(cfg.get("generations", 4)),
        essays_per_candidate=int(cfg.get("essays_per_candidate", 2)),
        semantic_similarity_weight=float(
            cfg.get("semantic_similarity_weight", cfg.get("lambda_semantic", 0.3))
        ),
        log_path=log_path,
        mutator_groq_model=str(
            cfg.get("mutator_groq_model", cfg.get("groq_model", "llama-3.1-8b-instant"))
        ),
        embedding_model_name=str(cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        constraint_kwargs={
            "min_words": int(cfg.get("min_words", 150)),
            "max_words": int(cfg.get("max_words", 2000)),
            "min_topic_similarity": float(cfg.get("min_topic_similarity", 0.3)),
        },
        chunked_scoring=bool(cfg.get("chunked_scoring", True)),
        chunk_window_tokens=cfg.get("chunk_window_tokens"),
        chunk_stride_tokens=cfg.get("chunk_stride_tokens"),
        chunk_max_chunks=cfg.get("chunk_max_chunks"),
        chunk_aggregate=str(cfg.get("chunk_aggregate", "weighted")),
        chunk_weight_mean=float(cfg.get("chunk_weight_mean", 0.5)),
        chunk_weight_max=float(cfg.get("chunk_weight_max", 0.5)),
        **drift_kw,
    )

    if best is None:
        print(
            json.dumps(
                {
                    "error": "no_valid_candidate",
                    "topic": topic,
                    "n_essays": len(essays),
                }
            )
        )
        raise SystemExit(2)

    print(json.dumps({"best_prompt_id": best.id, "best_fitness": best.fitness, "n_essays": len(essays)}))


if __name__ == "__main__":
    main()
