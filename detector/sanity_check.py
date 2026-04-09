#!/usr/bin/env python3
"""Inspect checkpoint + print sanity scores (requires HF_TOKEN if repo is gated)."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detector._device import default_torch_device
from detector.model import SlopDetector


def inspect_checkpoint() -> None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model = AutoModelForSequenceClassification.from_pretrained(
        "pangram/editlens_roberta-large",
        token=tok,
    )
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large", token=tok)
    print(f"num_labels: {model.config.num_labels}")
    print(f"model config: {model.config}")
    _ = tokenizer  # noqa: F841


def run_sanity() -> None:
    detector = SlopDetector(device=str(default_torch_device()))

    test_cases = [
        (
            "human_casual",
            "I dunno, I think the whole thing is kind of overblown honestly. Like yeah the food was fine but nothing to write home about. The pasta was a little overcooked if I'm being picky.",
        ),
        (
            "ai_slop",
            "In today's rapidly evolving digital landscape, artificial intelligence stands as a transformative force that is reshaping every facet of our society. From healthcare to education, the implications are profound and far-reaching. It is crucial that we navigate these changes thoughtfully and responsibly.",
        ),
        (
            "ai_formal",
            "This comprehensive analysis delves into the multifaceted implications of renewable energy adoption. By examining the interplay between economic viability, environmental sustainability, and social equity, we can gain a nuanced understanding of the challenges and opportunities that lie ahead.",
        ),
        (
            "human_essay",
            "My grandmother never learned to read. She signed her name with an X on every document, every form, every check. When I was seven I asked her if that bothered her. She looked at me like I'd asked if the sky bothered her.",
        ),
    ]

    print(f"decode: expected_value over {detector.num_buckets} buckets")
    for label, text in test_cases:
        score = detector.score(text)
        top = detector.score_top_bucket(text)
        probs = detector.score_proba(text)
        argmax = max(range(len(probs)), key=lambda i: probs[i])
        probs_str = ", ".join(f"{p:.3f}" for p in probs)
        print(f"{label:15s} → score={score:.4f}  p_top={top:.4f}  argmax={argmax}  probs=[{probs_str}]")


if __name__ == "__main__":
    inspect_checkpoint()
    print()
    run_sanity()
