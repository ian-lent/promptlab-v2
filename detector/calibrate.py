#!/usr/bin/env python3
"""
Sweep validation JSONL to find binary F1 threshold and ternary (human / edited / AI) thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from detector.model import SlopDetector


def load_val(path: Path) -> tuple[list[str], list[float]]:
    texts, scores = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            texts.append(str(r["text"]))
            scores.append(float(r["score"]))
    return texts, scores


def binary_labels(scores: list[float], threshold: float) -> np.ndarray:
    return (np.array(scores) >= threshold).astype(int)


def _pred_scalar(
    det: SlopDetector,
    text: str,
    *,
    chunked: bool,
    chunk_window: int | None,
    chunk_stride: int | None,
    chunk_max: int | None,
    aggregate: str,
) -> float:
    if not chunked:
        return float(det.score(text))
    long = det.score_long(
        text,
        window_tokens=chunk_window,
        stride_tokens=chunk_stride,
        max_chunks=chunk_max,
    )
    m, mx = float(long["mean"]), float(long["max"])
    if aggregate == "mean":
        return m
    if aggregate == "max":
        return mx
    if aggregate == "weighted":
        return 0.5 * (m + mx)
    raise ValueError(aggregate)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="pangram/editlens_roberta-large")
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--out-json", type=Path, default=Path("outputs/detector/thresholds.json"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunked", action="store_true", help="Use score_long per row (slower, essay-length)")
    p.add_argument("--chunk-window-tokens", type=int, default=512)
    p.add_argument("--chunk-stride-tokens", type=int, default=256)
    p.add_argument("--chunk-max-chunks", type=int, default=None)
    p.add_argument(
        "--chunk-aggregate",
        choices=("mean", "max", "weighted"),
        default="weighted",
        help="Scalar from score_long for threshold sweep",
    )
    args = p.parse_args()

    texts, true_scores = load_val(args.val_jsonl)
    det = SlopDetector(checkpoint=args.checkpoint)
    if args.chunked:
        pred_scores = np.array(
            [
                _pred_scalar(
                    det,
                    t,
                    chunked=True,
                    chunk_window=args.chunk_window_tokens,
                    chunk_stride=args.chunk_stride_tokens,
                    chunk_max=args.chunk_max_chunks,
                    aggregate=args.chunk_aggregate,
                )
                for t in texts
            ]
        )
    else:
        pred_scores = np.array(det.score_batch(texts, batch_size=args.batch_size))
    true_bin = (np.array(true_scores) >= 0.5).astype(int)

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred_bin = (pred_scores >= t).astype(int)
        f1 = f1_score(true_bin, pred_bin, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    # Ternary: human < t_low, edited in middle, AI >= t_high (true labels from score)
    true_ter = np.zeros(len(true_scores), dtype=int)
    true_ter[np.array(true_scores) >= 0.66] = 2
    true_ter[(np.array(true_scores) >= 0.33) & (np.array(true_scores) < 0.66)] = 1

    best_pair = (0.34, 0.67)
    best_f1_ter = -1.0
    for t_low in np.linspace(0.15, 0.55, 25):
        for t_high in np.linspace(t_low + 0.1, 0.9, 25):
            pred_ter = np.zeros_like(true_ter)
            pred_ter[pred_scores >= t_high] = 2
            pred_ter[(pred_scores >= t_low) & (pred_scores < t_high)] = 1
            f1 = f1_score(true_ter, pred_ter, average="macro", zero_division=0)
            if f1 > best_f1_ter:
                best_f1_ter = f1
                best_pair = (float(t_low), float(t_high))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "binary_threshold": best_t,
        "binary_macro_f1": float(best_f1),
        "ternary_t_low": best_pair[0],
        "ternary_t_high": best_pair[1],
        "ternary_macro_f1": float(best_f1_ter),
        "checkpoint": args.checkpoint,
        "n_val": len(texts),
        "chunked": bool(args.chunked),
        "chunk_aggregate": args.chunk_aggregate if args.chunked else None,
    }
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
