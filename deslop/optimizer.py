"""Evolutionary prompt search against the current slop detector."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sentence_transformers import SentenceTransformer, util

from deslop.constraints import (
    MINILM_ESSAY_CHARS,
    MINILM_TOPIC_CHARS,
    check_constraints,
    clip_for_minilm,
    fitness_from_scores,
)
from deslop.prompt_bank import PromptCandidate, apply_user_template_slots, seed_prompt_bank
from deslop.scoring import essay_slop_scalar
from deslop.similarity import AlignmentScale, DriftWeights, composite_drift_penalty
from deslop.strategies.evolutionary import refill_population, tournament_select

if TYPE_CHECKING:
    from cotrain.pair_logger import PairLogger
    from detector.model import SlopDetector
else:
    PairLogger = Any  # noqa: UP007

def _few_shot_block(
    examples: list[str] | None,
    *,
    max_chars_each: int = 3500,
) -> str:
    if not examples:
        return ""
    lines = [
        "Below are example passages that scored well on the primary detector (lower score = "
        "more human-like). Use them only as loose style guidance; write a new essay for the "
        "assigned topic—do not copy or paraphrase them verbatim."
    ]
    for i, ex in enumerate(examples, 1):
        body = ex.strip()
        if len(body) > max_chars_each:
            body = body[:max_chars_each] + "\n[... truncated ...]"
        lines.append(f"\n--- Reference example {i} ---\n{body}")
    return "\n".join(lines)


def _compose_messages(
    candidate: PromptCandidate,
    topic: str,
    few_shot_prefix: str = "",
) -> str:
    user = apply_user_template_slots(
        candidate.user_template, topic, candidate.style_instructions or ""
    )
    sys_p = candidate.system_prompt.strip()
    u = user.strip()
    if few_shot_prefix.strip():
        return f"{sys_p}\n\n{few_shot_prefix.strip()}\n\n{u}"
    return f"{sys_p}\n\n{u}"


def optimize(
    topic: str,
    target_llm: Callable[[str], str],
    detector: SlopDetector,
    population_size: int = 20,
    generations: int = 15,
    essays_per_candidate: int = 3,
    pair_logger: Any = None,
    *,
    lambda_semantic: float = 0.3,
    log_path: Path | None = None,
    mutator_groq_model: str = "llama-3.1-8b-instant",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cotrain_round: int = 0,
    constraint_kwargs: dict[str, Any] | None = None,
    chunked_scoring: bool = False,
    chunk_window_tokens: int | None = None,
    chunk_stride_tokens: int | None = None,
    chunk_max_chunks: int | None = None,
    chunk_aggregate: str = "weighted",
    chunk_weight_mean: float = 0.5,
    chunk_weight_max: float = 0.5,
    alignment_reference_mode: str | None = None,
    drift_weights: DriftWeights | None = None,
    drift_use_bertscore: bool = False,
    drift_bertscore_slop_gate: float | None = None,
    alignment_source_passage: str | None = None,
    few_shot_examples: list[str] | None = None,
) -> tuple[PromptCandidate, list[dict]]:
    """
    Evolutionary loop: minimize mean detector score subject to constraints.

    - ``alignment_reference_mode=\"topic\"``: drift vs the topic string (short-rubric scale).
    - ``alignment_reference_mode=\"source_passage\"``: drift vs ``alignment_source_passage``
      (passage–passage scale: mean-pooled embeddings + capped ROUGE-L).

    Optimization uses ``detector_slop + drift_penalty`` per essay while ``slop_score`` stays
    the raw detector value (for thresholds / logging).

    Essay text is produced by ``target_llm`` (configure Groq model via ``make_groq_essay_fn`` at
    the CLI / cotrain entrypoint). Prompt mutations use ``mutator_groq_model`` (refill path).

    Returns best PromptCandidate and list of generated essay records for co-training.
    """
    constraint_kwargs = constraint_kwargs or {}
    drift_w = drift_weights or DriftWeights()
    topic_desc = topic
    align_ref: str | None = None
    alignment_scale: AlignmentScale = "topic"
    if alignment_reference_mode == "topic":
        align_ref = topic_desc
        alignment_scale = "topic"
    elif alignment_reference_mode == "source_passage":
        sp = (alignment_source_passage or "").strip()
        if not sp:
            raise ValueError(
                "alignment_reference_mode='source_passage' requires a non-empty "
                "alignment_source_passage (set in YAML or pass from cotrain topic_sources)."
            )
        align_ref = sp
        alignment_scale = "passage"
    embedder = SentenceTransformer(embedding_model_name)
    few_shot_prefix = _few_shot_block(few_shot_examples)

    seeds = seed_prompt_bank()
    population: list[PromptCandidate] = []
    i = 0
    while len(population) < population_size:
        s = seeds[i % len(seeds)]
        population.append(
            PromptCandidate(
                id=f"{s.id}_{len(population)}",
                system_prompt=s.system_prompt,
                user_template=s.user_template,
                style_instructions=s.style_instructions,
                mutation_op="seed",
            )
        )
        i += 1

    all_essays: list[dict] = []
    log_path = log_path or Path("outputs/deslop/run.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_ever: PromptCandidate | None = None
    best_fitness = float("-inf")

    for gen in range(generations):
        scored: list[tuple[PromptCandidate, float, float, float, float]] = []

        for cand in population:
            opt_slops: list[float] = []
            raw_slops: list[float] = []
            sims: list[float] = []
            failed_any = False

            for _ in range(essays_per_candidate):
                prompt_text = _compose_messages(cand, topic, few_shot_prefix)
                # Caller should close over essay_temperature if the backend supports it.
                essay = target_llm(prompt_text)
                ok, reason = check_constraints(
                    essay,
                    topic_desc,
                    embedder,
                    **constraint_kwargs,
                )
                if ok:
                    s, slop_details = essay_slop_scalar(
                        detector,
                        essay,
                        chunked=chunked_scoring,
                        chunk_window_tokens=chunk_window_tokens,
                        chunk_stride_tokens=chunk_stride_tokens,
                        chunk_max_chunks=chunk_max_chunks,
                        chunk_aggregate=chunk_aggregate,
                        chunk_weight_mean=chunk_weight_mean,
                        chunk_weight_max=chunk_weight_max,
                    )
                else:
                    s = 1.0
                    slop_details = {"slop_failed_constraint": 1.0}

                drift_penalty = 0.0
                drift_detail: dict[str, float] = {}
                if ok and align_ref is not None:
                    drift_penalty, drift_detail = composite_drift_penalty(
                        align_ref,
                        essay,
                        drift_w,
                        use_bertscore=drift_use_bertscore,
                        raw_slop=s,
                        bertscore_slop_gate=drift_bertscore_slop_gate,
                        embedder=embedder,
                        embedding_model_name=embedding_model_name,
                        alignment_scale=alignment_scale,
                    )
                opt_slop = s + drift_penalty

                if ok:
                    e_essay = embedder.encode(
                        clip_for_minilm(essay, max_chars=MINILM_ESSAY_CHARS),
                        convert_to_tensor=True,
                    )
                    e_top = embedder.encode(
                        clip_for_minilm(topic_desc, max_chars=MINILM_TOPIC_CHARS),
                        convert_to_tensor=True,
                    )
                    sim = float(util.cos_sim(e_essay.unsqueeze(0), e_top.unsqueeze(0))[0][0])
                    sims.append(sim)
                else:
                    failed_any = True
                    sims.append(0.0)

                raw_slops.append(s)
                opt_slops.append(opt_slop)
                rec = {
                    "topic": topic,
                    "essay": essay,
                    "slop_score": s,
                    "optimization_slop": opt_slop,
                    "drift_penalty": drift_penalty,
                    "constraint_ok": ok,
                    "constraint_reason": reason,
                    "generation": gen,
                    "prompt_id": cand.id,
                    "mutation_op": cand.mutation_op,
                    "domain": "deslop",
                    "chunked_scoring": chunked_scoring,
                }
                if alignment_reference_mode is not None:
                    rec["alignment_reference_mode"] = alignment_reference_mode
                    rec["alignment_scale"] = alignment_scale
                    if alignment_reference_mode == "source_passage" and align_ref is not None:
                        rec["alignment_source_len"] = len(align_ref)
                rec.update({k: v for k, v in slop_details.items() if isinstance(v, (int, float))})
                rec.update({k: float(v) for k, v in drift_detail.items() if isinstance(v, (int, float))})
                all_essays.append(rec)

            mean_slop = sum(opt_slops) / max(1, len(opt_slops))
            mean_raw_slop = sum(raw_slops) / max(1, len(raw_slops))
            mean_sim = sum(sims) / max(1, len(sims))
            fit = fitness_from_scores(
                mean_slop, mean_sim, lambda_semantic=lambda_semantic, failed=failed_any
            )
            cand.fitness = fit
            scored.append((cand, mean_slop, mean_sim, fit, mean_raw_slop))

            if fit > best_fitness:
                best_fitness = fit
                best_ever = cand

        scored.sort(key=lambda x: x[3], reverse=True)
        survivors = tournament_select(
            [t[0] for t in scored],
            max(1, population_size // 2),
        )

        # Log generation
        with log_path.open("a", encoding="utf-8") as f:
            for row in scored:
                cand, ms, sim, fit, mean_raw = row
                rowd: dict[str, Any] = {
                    "topic": topic,
                    "generation": gen,
                    "prompt_id": cand.id,
                    "mean_slop": ms,
                    "mean_sim": sim,
                    "fitness": fit,
                    "mutation_op": cand.mutation_op,
                }
                if align_ref is not None:
                    rowd["mean_raw_slop"] = mean_raw
                f.write(json.dumps(rowd, ensure_ascii=False) + "\n")

        if gen + 1 < generations:
            population = refill_population(
                survivors,
                population_size,
                mutator_groq_model=mutator_groq_model,
                cotrain_round=cotrain_round,
            )

    assert best_ever is not None
    return best_ever, all_essays
