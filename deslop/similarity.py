"""
Composite drift penalty between an original passage and a (re)written one.

Used to penalize semantic / lexical drift when optimizing for low detector (slop) score.
All heavy models are lazy-loaded on first use.

**Alignment scale**
  - ``topic``: short rubric vs essay — clip both sides to ~2k chars; single-vector cosine.
  - ``passage``: passage vs passage — mean-pooled sentence embeddings over chunks (better
    for long aligned texts); ROUGE-L on a capped prefix of each side for tractability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from deslop.constraints import clip_for_minilm

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

AlignmentScale = Literal["topic", "passage"]

# Topic-string comparisons: short; matches historical check_constraints-style clips.
SIMILARITY_CLIP_TOPIC_CHARS = 2000

# Passage–passage: chunking for mean-pooled embeddings (MiniLM ~512 tokens; chunk by chars).
SIMILARITY_PASSAGE_CHUNK_CHARS = 1800
SIMILARITY_PASSAGE_MAX_CHUNKS = 14

# ROUGE-L: cap extremely long inputs (scores first segment; document in docstring).
SIMILARITY_ROUGE_MAX_CHARS = 12000

_embedder_cache: dict[str, Any] = {}
_rouge_scorer: Any = None


@dataclass
class DriftWeights:
    """
    Non-negative drift penalty coefficients (each drift term is in [0, 1] before weighting).

    Named ``alpha_*`` to avoid collision with unrelated uses of ``lambda`` in ML and with
    the Lambda Labs cloud provider. YAML keys use ``drift_coef_*`` (see ``drift_options_from_config``).
    """

    alpha_semantic: float = 1.0
    alpha_rouge: float = 0.35
    alpha_bertscore: float = 0.5


def _get_embedder(model_name: str) -> Any:
    if model_name not in _embedder_cache:
        from sentence_transformers import SentenceTransformer

        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def _get_rouge_scorer() -> Any:
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer

        _rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge_scorer


def _text_chunks(s: str, chunk_chars: int) -> list[str]:
    t = s.strip()
    if not t:
        return []
    return [t[i : i + chunk_chars] for i in range(0, len(t), chunk_chars)]


def _mean_pooled_embedding(
    model: Any,
    text: str,
    *,
    chunk_chars: int,
    max_chunks: int,
) -> torch.Tensor | None:
    chunks = _text_chunks(text, chunk_chars)[:max_chunks]
    if not chunks:
        return None
    embs = model.encode(chunks, convert_to_tensor=True)
    v = embs.float().mean(dim=0)
    return torch.nn.functional.normalize(v, dim=0)


def _semantic_drift(
    original: str,
    rewritten: str,
    *,
    embedder: Any | None,
    embedding_model_name: str,
    alignment_scale: AlignmentScale,
) -> float:
    """Return 1 - cos_sim; cos in [0, 1] after normalization."""
    from sentence_transformers import util

    model = embedder if embedder is not None else _get_embedder(embedding_model_name)

    if alignment_scale == "topic":
        a = clip_for_minilm(original.strip(), max_chars=SIMILARITY_CLIP_TOPIC_CHARS)
        b = clip_for_minilm(rewritten.strip(), max_chars=SIMILARITY_CLIP_TOPIC_CHARS)
        if not a or not b:
            return 1.0
        e1 = model.encode(a, convert_to_tensor=True)
        e2 = model.encode(b, convert_to_tensor=True)
    else:
        v1 = _mean_pooled_embedding(
            model,
            original,
            chunk_chars=SIMILARITY_PASSAGE_CHUNK_CHARS,
            max_chunks=SIMILARITY_PASSAGE_MAX_CHUNKS,
        )
        v2 = _mean_pooled_embedding(
            model,
            rewritten,
            chunk_chars=SIMILARITY_PASSAGE_CHUNK_CHARS,
            max_chunks=SIMILARITY_PASSAGE_MAX_CHUNKS,
        )
        if v1 is None or v2 is None:
            return 1.0
        cos = float(util.cos_sim(v1.unsqueeze(0), v2.unsqueeze(0))[0][0])
        cos = max(0.0, min(1.0, cos))
        return 1.0 - cos

    cos = float(util.cos_sim(e1.unsqueeze(0), e2.unsqueeze(0))[0][0])
    cos = max(0.0, min(1.0, cos))
    return 1.0 - cos


def _rouge_l_drift(original: str, rewritten: str, *, alignment_scale: AlignmentScale) -> float:
    """
    Lexical floor: 1 - ROUGE-L F-measure (reference = original, candidate = rewritten).

    For long passages, both strings are truncated to the first ``SIMILARITY_ROUGE_MAX_CHARS``
    characters so scoring stays tractable; use ``alignment_scale=\"passage\"`` with the
    semantic term for long-text alignment.
    """
    ref = original.strip()
    hyp = rewritten.strip()
    if not ref or not hyp:
        return 1.0
    cap = SIMILARITY_ROUGE_MAX_CHARS
    if len(ref) > cap or len(hyp) > cap:
        ref = ref[:cap]
        hyp = hyp[:cap]
    scorer = _get_rouge_scorer()
    scores = scorer.score(ref, hyp)
    f1 = float(scores["rougeL"].fmeasure)
    return 1.0 - max(0.0, min(1.0, f1))


def _bertscore_drift(original: str, rewritten: str) -> float:
    """1 - BERTScore F1 (lazy import). Install: pip install bert-score"""
    try:
        from bert_score import score as bert_score_fn
    except ImportError as e:
        raise ImportError(
            "bert-score is required when use_bertscore=True. Install: pip install bert-score"
        ) from e

    ref = original.strip()
    hyp = rewritten.strip()
    if not ref or not hyp:
        return 1.0
    _, _, f1 = bert_score_fn([hyp], [ref], lang="en", verbose=False)
    f = float(f1.mean().item())
    return 1.0 - max(0.0, min(1.0, f))


def composite_drift_penalty(
    original: str,
    rewritten: str,
    weights: DriftWeights | None = None,
    *,
    use_bertscore: bool = False,
    raw_slop: float | None = None,
    bertscore_slop_gate: float | None = None,
    embedder: SentenceTransformer | None = None,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    alignment_scale: AlignmentScale = "topic",
) -> tuple[float, dict[str, float]]:
    """
    Composite drift penalty (higher = more drift). Intended to be **added** to raw detector
    slop when minimizing, or to subtract from reward when reward = -slop - penalty.

    ``alignment_scale``:
      - ``topic``: short rubric vs essay (single embedding, ~2k char clip).
      - ``passage``: passage vs passage (mean-pooled chunk embeddings; ROUGE on capped text).

    Returns:
        (penalty, detail dict with component drifts and flags).
    """
    w = weights or DriftWeights()
    details: dict[str, float] = {}
    details["alignment_scale_topic"] = 1.0 if alignment_scale == "topic" else 0.0

    d_sem = _semantic_drift(
        original,
        rewritten,
        embedder=embedder,
        embedding_model_name=embedding_model_name,
        alignment_scale=alignment_scale,
    )
    details["drift_semantic"] = d_sem

    d_r = _rouge_l_drift(original, rewritten, alignment_scale=alignment_scale)
    details["drift_rouge_l"] = d_r

    include_bert = use_bertscore
    if include_bert and bertscore_slop_gate is not None and raw_slop is not None:
        include_bert = float(raw_slop) <= float(bertscore_slop_gate)

    details["bertscore_applied"] = 1.0 if include_bert else 0.0

    d_bert = 0.0
    if include_bert:
        d_bert = _bertscore_drift(original, rewritten)
    details["drift_bertscore"] = d_bert

    penalty = (
        w.alpha_semantic * d_sem
        + w.alpha_rouge * d_r
        + (w.alpha_bertscore * d_bert if include_bert else 0.0)
    )
    details["drift_penalty_total"] = penalty
    return penalty, details


def drift_options_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Map YAML-style keys (e.g. from ``configs/deslop.yaml`` or ``cotrain.yaml``) to
    ``optimize()`` keyword arguments (drift / alignment only; essay vs mutator Groq models are
    set at the entrypoint: ``groq_model`` for essays, ``mutator_groq_model`` for ``optimize()``).

    ``alignment_reference_mode``: null | ``topic`` | ``source_passage``.
    For ``source_passage``, set ``alignment_source_passage`` in YAML (deslop) or per-topic
    via ``topic_sources_jsonl`` / ``topic_sources`` at the cotrain call site.
    """
    mode = cfg.get("alignment_reference_mode")
    if mode is not None and mode not in ("topic", "source_passage"):
        raise ValueError("alignment_reference_mode must be null, 'topic', or 'source_passage'")
    d = cfg.get("drift") or {}
    # Backward compatibility: older YAML used lambda_* under ``drift`` for penalty coefficients.
    dw = DriftWeights(
        alpha_semantic=float(
            d.get(
                "drift_coef_semantic",
                d.get("alpha_semantic", d.get("lambda_semantic", 1.0)),
            )
        ),
        alpha_rouge=float(
            d.get(
                "drift_coef_rouge",
                d.get("alpha_rouge", d.get("lambda_rouge_l", 0.35)),
            )
        ),
        alpha_bertscore=float(
            d.get(
                "drift_coef_bertscore",
                d.get("alpha_bertscore", d.get("lambda_bertscore", 0.5)),
            )
        ),
    )
    gate = d.get("bertscore_slop_gate")
    aps = cfg.get("alignment_source_passage")
    src = str(aps).strip() if aps is not None and str(aps).strip() else None
    return {
        "alignment_reference_mode": mode,
        "alignment_source_passage": src,
        "drift_weights": dw,
        "drift_use_bertscore": bool(d.get("use_bertscore", False)),
        "drift_bertscore_slop_gate": float(gate) if gate is not None else None,
    }


def detection_objective(
    detector_slop: float,
    drift_penalty: float,
) -> float:
    """
    Scalar to **minimize** when both low slop and low drift are desired:
    ``detector_slop + drift_penalty``.
    """
    return float(detector_slop) + float(drift_penalty)
