"""Slop score, external detector, similarity, perplexity, self-BLEU, lexicon ratio."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

if TYPE_CHECKING:
    from detector.model import SlopDetector


def load_slop_keywords(lexicon_dir: Path | None) -> list[str]:
    if lexicon_dir is None or not lexicon_dir.exists():
        return []
    words: list[str] = []
    for p in lexicon_dir.rglob("*"):
        if p.is_file() and p.suffix in {".txt", ".md", ".yaml", ".yml"}:
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    w = line.strip().lower()
                    if w and not w.startswith("#") and len(w) < 40:
                        words.append(w)
            except OSError:
                continue
    return list(dict.fromkeys(words))


def slop_scores(detector: SlopDetector, essays: list[str]) -> list[float]:
    return detector.score_batch(essays)


def semantic_similarity(
    embedder: SentenceTransformer,
    essays: list[str],
    topic_description: str,
) -> list[float]:
    if not essays:
        return []
    e_topic = embedder.encode(topic_description, convert_to_tensor=True)
    sims = []
    for essay in essays:
        e = embedder.encode(essay, convert_to_tensor=True)
        sims.append(float(util.cos_sim(e.unsqueeze(0), e_topic.unsqueeze(0))[0][0]))
    return sims


def perplexity_gpt2(essays: list[str], model_name: str = "gpt2") -> list[float]:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    out = []
    with torch.no_grad():
        for text in essays:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
            loss = model(**enc, labels=enc["input_ids"]).loss
            out.append(float(torch.exp(loss).item()))
    return out


def self_bleu(essays: list[str], n: int = 4) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return float("nan")

    if len(essays) < 2:
        return float("nan")
    toks = [re.findall(r"\w+", e.lower()) for e in essays]
    smoothie = SmoothingFunction().method1
    scores = []
    for i, hyp in enumerate(toks):
        refs = [t for j, t in enumerate(toks) if j != i]
        if not refs:
            continue
        scores.append(sentence_bleu(refs, hyp, smoothing_function=smoothie))
    return float(np.mean(scores)) if scores else float("nan")


def human_keywords_ratio(essay: str, keywords: list[str]) -> float:
    if not keywords:
        return float("nan")
    sents = re.split(r"(?<=[.!?])\s+", essay.strip())
    if not sents:
        return 0.0
    low = essay.lower()
    hits = sum(1 for sent in sents if any(k in sent.lower() for k in keywords if k))
    return hits / max(1, len(sents))


def aggregate_run(
    detector: SlopDetector,
    external_detector: SlopDetector | None,
    embedder: SentenceTransformer,
    essays: list[str],
    topic: str,
    *,
    lexicon_dir: Path | None = None,
    gpt2_name: str = "gpt2",
) -> dict[str, float | list]:
    kws = load_slop_keywords(lexicon_dir)
    primary = slop_scores(detector, essays)
    ext = slop_scores(external_detector, essays) if external_detector else []
    sims = semantic_similarity(embedder, essays, topic)
    ppls = perplexity_gpt2(essays, gpt2_name)
    bleu = self_bleu(essays)
    hkr = np.mean([human_keywords_ratio(e, kws) for e in essays]) if kws else float("nan")

    return {
        "mean_slop": float(np.mean(primary)),
        "std_slop": float(np.std(primary)),
        "mean_external_slop": float(np.mean(ext)) if ext else float("nan"),
        "mean_semantic_sim": float(np.mean(sims)) if sims else float("nan"),
        "mean_perplexity": float(np.mean(ppls)) if ppls else float("nan"),
        "self_bleu": bleu,
        "human_keywords_ratio": float(hkr),
    }
