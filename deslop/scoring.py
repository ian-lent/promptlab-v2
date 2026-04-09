"""Map detector outputs to a single slop scalar for optimization (truncated vs chunked).

Alignment / drift penalties (sentence-transformers, ROUGE-L, optional BERTScore) live in
``deslop.similarity``. During deslop, ``optimizer.optimize`` forms
``optimization_slop = raw_detector_slop + drift_penalty`` for fitness while ``slop_score`` on
each record stays the raw detector value. Modes: ``topic`` (short rubric) vs ``source_passage``
(passage–passage mean-pooled embeddings + capped ROUGE-L).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from detector.model import SlopDetector


def essay_slop_scalar(
    detector: SlopDetector,
    essay: str,
    *,
    chunked: bool = False,
    chunk_window_tokens: int | None = None,
    chunk_stride_tokens: int | None = None,
    chunk_max_chunks: int | None = None,
    chunk_aggregate: str = "weighted",
    chunk_weight_mean: float = 0.5,
    chunk_weight_max: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """
    Return (scalar_slop, details) for fitness minimization.

    chunk_aggregate:
    - ``mean`` / ``max``: use that chunk aggregate only
    - ``weighted``: ``chunk_weight_mean * mean + chunk_weight_max * max`` (renormalize if weights don't sum to 1)
    """
    details: dict[str, Any] = {}
    if not chunked:
        s = detector.score(essay)
        details["slop_truncated"] = s
        return s, details

    long = detector.score_long(
        essay,
        window_tokens=chunk_window_tokens,
        stride_tokens=chunk_stride_tokens,
        max_chunks=chunk_max_chunks,
    )
    m, mx = float(long["mean"]), float(long["max"])
    details["slop_mean"] = m
    details["slop_max"] = mx
    details["n_chunks"] = float(long["n_chunks"])

    if chunk_aggregate == "mean":
        return m, details
    if chunk_aggregate == "max":
        return mx, details
    if chunk_aggregate == "weighted":
        w_m, w_x = float(chunk_weight_mean), float(chunk_weight_max)
        denom = w_m + w_x
        if denom <= 0:
            return m, details
        return (w_m * m + w_x * mx) / denom, details

    raise ValueError(f"Unknown chunk_aggregate: {chunk_aggregate}")
