"""Hard constraints on generated essays (length, on-topic, language, refusals)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

REFUSAL_PATTERNS = re.compile(
    r"(?i)\b(i cannot|i can't|as an ai|i'm an ai|i am an ai|ethical guidelines)\b"
)

# MiniLM-L6 max ~512 subword positions; long strings still warn / mis-index without clipping.
MINILM_ESSAY_CHARS = 2500
MINILM_TOPIC_CHARS = 2000


def clip_for_minilm(text: str, *, max_chars: int = MINILM_ESSAY_CHARS) -> str:
    """Truncate for sentence-transformer encode (same limits as check_constraints on-topic sim)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def word_count(text: str) -> int:
    return len(text.split())


def check_constraints(
    essay: str,
    topic_description: str,
    embed_model: SentenceTransformer | None,
    *,
    min_words: int = 150,
    max_words: int = 2000,
    min_topic_similarity: float = 0.3,
) -> tuple[bool, str]:
    wc = word_count(essay)
    if wc < min_words:
        return False, "too_short"
    if wc > max_words:
        return False, "too_long"
    if REFUSAL_PATTERNS.search(essay):
        return False, "refusal"
    try:
        from langdetect import detect

        if detect(essay[:2000]) != "en":
            return False, "non_english"
    except Exception:
        pass

    if embed_model is not None and topic_description.strip():
        from sentence_transformers import util

        head = clip_for_minilm(essay, max_chars=MINILM_ESSAY_CHARS)
        topic_t = clip_for_minilm(topic_description, max_chars=MINILM_TOPIC_CHARS)
        e1 = embed_model.encode(head, convert_to_tensor=True)
        e2 = embed_model.encode(topic_t, convert_to_tensor=True)
        sim = float(util.cos_sim(e1.unsqueeze(0), e2.unsqueeze(0))[0][0])
        if sim < min_topic_similarity:
            return False, "off_topic"

    return True, "ok"


def fitness_from_scores(
    mean_slop: float,
    mean_semantic_sim: float,
    *,
    semantic_similarity_weight: float = 0.3,
    failed: bool = False,
) -> float:
    if failed:
        return float("-inf")
    return -mean_slop + semantic_similarity_weight * mean_semantic_sim
