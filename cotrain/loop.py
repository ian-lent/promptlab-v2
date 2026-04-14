"""
Adversarial co-training orchestration: deslop (prompt / essay optimizer) with optional
detector retraining. Default architecture treats the detector as a **static** reward; the
optimizer improves round-over-round via few-shot injection from winning essays.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import yaml

from cotrain.data_manager import CotrainDataManager
from cotrain.goodhart import goodhart_check
from cotrain.pair_logger import PairLogger
from cotrain.stopping import should_stop
from deslop.drift_coef_opt import (
    append_deslop_round_to_feature_pool,
    load_optimized_drift_weights,
    run_drift_coef_optimization_from_config_section,
)
from deslop.optimizer import optimize
from deslop.similarity import DriftWeights, drift_options_from_config
from deslop.prompt_bank import PromptCandidate
from deslop.run_topic import make_groq_essay_fn
from detector.model import SlopDetector


def _log(msg: str) -> None:
    print(msg, flush=True)


class _TeeStdout:
    """Mirror ``sys.stdout`` writes to a UTF-8 log file (flush per write for live tail)."""

    __slots__ = ("_orig", "_log")

    def __init__(self, original: TextIO, log_path: Path) -> None:
        self._orig = original
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("w", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        self._orig.write(data)
        self._orig.flush()
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._orig.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._orig.isatty()

    def fileno(self) -> int:
        return self._orig.fileno()

    def close_log(self) -> None:
        self._log.flush()
        self._log.close()


def load_topic_sources_index(path: Path) -> tuple[dict[str, str], dict[str, float]]:
    """
    Load ``topic`` → ``source_passage`` and best raw ``slop_score`` per topic.

    Lines may include optional ``slop_score`` (lower = better detector). When present for a
    topic, the passage with the **minimum** slop wins. Lines without ``slop_score`` use
    last-wins only for topics that have no scored line yet.

    Returns:
        (topic → passage, topic → best_slop). Missing scored history uses ``inf`` for that
        topic so the first scored append always ratchets in.
    """
    scored: dict[str, tuple[float, str]] = {}
    unscored_last: dict[str, str] = {}
    if not path.is_file():
        return {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t = str(r.get("topic", "")).strip()
        sp = str(r.get("source_passage", "")).strip()
        if not t or not sp:
            continue
        if r.get("slop_score") is not None:
            s = float(r["slop_score"])
            if t not in scored or s < scored[t][0]:
                scored[t] = (s, sp)
        else:
            unscored_last[t] = sp

    passages: dict[str, str] = {}
    best_slop: dict[str, float] = {}
    for t, sp in unscored_last.items():
        if t not in scored:
            passages[t] = sp
            best_slop[t] = float("inf")
    for t, (s, sp) in scored.items():
        passages[t] = sp
        best_slop[t] = s
    return passages, best_slop


def load_topic_sources_jsonl(path: Path) -> dict[str, str]:
    """
    Load ``topic`` → ``source_passage`` from JSONL (see ``load_topic_sources_index`` for
    ``slop_score`` / min-slop behavior).
    """
    return load_topic_sources_index(path)[0]


def _load_fewshot_pool(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _fewshot_slop(row: dict) -> float:
    return float(row.get("slop_score", 1.0))


def _merge_and_save_fewshot_pool(
    path: Path,
    existing: list[dict],
    new_rows: list[dict],
    cap: int,
) -> list[dict]:
    """Merge, dedupe by essay prefix hash, keep globally lowest-slop ``cap`` rows."""
    merged = list(existing) + list(new_rows)
    seen_set: set[int] = set()
    deduped: list[dict] = []
    for row in sorted(merged, key=_fewshot_slop):
        essay = str(row.get("essay", ""))
        h = hash(essay[:4000])
        if h in seen_set:
            continue
        seen_set.add(h)
        deduped.append(row)
    deduped = sorted(deduped, key=_fewshot_slop)[:cap]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as wf:
        for row in deduped:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
    return deduped


def _few_shot_texts_from_pool(pool: list[dict], n: int) -> list[str]:
    if n <= 0 or not pool:
        return []
    texts: list[str] = []
    for row in sorted(pool, key=_fewshot_slop)[:n]:
        t = str(row.get("essay", "")).strip()
        if t:
            texts.append(t)
    return texts


def _print_compute_estimate(
    *,
    num_rounds: int,
    topics_per_round: int,
    population_size: int,
    generations_per_topic: int,
    essays_per_candidate: int,
    few_shot_enabled: bool,
) -> None:
    """Rough Groq + detector call budget (order-of-magnitude) for sanity checks."""
    tr = max(0, num_rounds)
    tpr = max(0, topics_per_round)
    essay_calls_per_topic = max(0, population_size) * max(0, generations_per_topic) * max(
        0, essays_per_candidate
    )
    # Refill: up to (generations-1) * (pop - elite) Groq mutator calls per topic (upper bound).
    elite = max(1, int(0.2 * population_size)) if population_size else 1
    refill_slots = max(0, generations_per_topic - 1) * max(0, population_size - elite)
    mut_upper_per_topic = refill_slots
    per_round_topics = tpr
    total_essay = tr * per_round_topics * essay_calls_per_topic
    total_mut = tr * per_round_topics * mut_upper_per_topic
    det_scores = total_essay
    _log(
        json.dumps(
            {
                "cotrain_compute_estimate": {
                    "rounds": tr,
                    "topics_per_round": tpr,
                    "essay_groq_calls_approx": total_essay,
                    "mutator_groq_calls_upper_bound": total_mut,
                    "detector_forward_passes_approx": det_scores,
                    "few_shot_injection": bool(few_shot_enabled),
                    "note": "Mutator count is an upper bound (refill loop). Add rate limits / tokens for wall time.",
                }
            },
            indent=2,
        )
    )


def _write_colab_round_checkpoint(
    path: Path,
    *,
    round_num: int,
    num_rounds: int,
    detector_frozen: bool,
    checkpoint: str,
    fewshot_pool_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snap = {
        "round": round_num,
        "num_rounds": num_rounds,
        "detector_frozen": detector_frozen,
        "detector_checkpoint": checkpoint,
        "fewshot_pool_rows": fewshot_pool_size,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")


def _run_subprocess_teed(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    log_path: Path,
) -> None:
    """Stream child stdout/stderr to this process and to ``log_path`` (line-buffered)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        rc = process.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _deslop_slop_stats(scores: list[float], threshold: float) -> dict:
    if not scores:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stdev": None,
            "pct_below_threshold": None,
        }
    below = sum(1 for s in scores if s < threshold)
    stdev: float | None
    try:
        stdev = float(statistics.stdev(scores)) if len(scores) > 1 else 0.0
    except statistics.StatisticsError:
        stdev = None
    return {
        "count": len(scores),
        "mean": float(statistics.mean(scores)),
        "median": float(statistics.median(scores)),
        "min": float(min(scores)),
        "max": float(max(scores)),
        "stdev": stdev,
        "pct_below_threshold": float(below / len(scores)),
    }


def _resolve_topics_file_path(topics_file: str, cotrain_config_path: Path) -> Path:
    """Resolve ``topics_file`` relative to CWD or the co-training config file directory."""
    p = Path(topics_file)
    if p.is_absolute():
        return p
    rel = Path(topics_file)
    cwd_candidate = (Path.cwd() / rel).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    cfg_dir_candidate = (cotrain_config_path.resolve().parent / rel).resolve()
    if cfg_dir_candidate.is_file():
        return cfg_dir_candidate
    return cwd_candidate


def _load_topics_from_file(path: Path) -> list[str]:
    """
    Load topic strings from a plain-text file (one per line) or YAML with a top-level
    ``topics:`` list (e.g. ``configs/topics_alpaca_diverse.yaml``).
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise SystemExit(f"topics_file YAML {path} must be a mapping with a 'topics' list.")
        raw = doc.get("topics")
        if raw is None:
            raise SystemExit(f"topics_file YAML {path} has no 'topics' key.")
        if not isinstance(raw, list):
            raise SystemExit(f"topics_file {path}: 'topics' must be a list of strings.")
        return [str(t).strip() for t in raw if str(t).strip()]
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _default_topics(n: int) -> list[str]:
    pool = [
        "the ethics of AI in education",
        "climate policy and economic justice",
        "privacy in the age of smart cities",
        "whether college should be free",
        "the future of remote work",
        "social media and teenage mental health",
        "universal basic income",
        "nuclear power as a climate solution",
        "copyright and generative AI",
        "the role of luck in success",
    ]
    if n <= len(pool):
        return pool[:n]
    return pool + [f"topic variant {i}" for i in range(n - len(pool))]


def cotrain(
    topics: list[str],
    target_llm: Callable[[str], str],
    initial_detector_checkpoint: str,
    num_rounds: int = 5,
    topics_per_round: int = 20,
    population_size: int = 12,
    generations_per_topic: int = 10,
    essays_per_candidate: int = 3,
    *,
    detector_train_config: Path | None = None,
    base_train_jsonl: Path | None = None,
    binary_threshold: float = 0.5,
    pair_log_path: Path = Path("outputs/cotrain/prompt_pairs.jsonl"),
    summary_path: Path = Path("outputs/cotrain/round_summaries.jsonl"),
    detector_output_root: Path = Path("outputs/detector_cotrain"),
    stop_patience: int = 2,
    stop_epsilon: float = 0.02,
    recent_weight: float = 1.5,
    groq_model: str = "llama-3.3-70b-versatile",
    mutator_groq_model: str | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    semantic_similarity_weight: float = 0.3,
    constraint_kwargs: dict | None = None,
    chunked_scoring: bool = True,
    chunk_window_tokens: int | None = None,
    chunk_stride_tokens: int | None = None,
    chunk_max_chunks: int | None = None,
    chunk_aggregate: str = "weighted",
    chunk_weight_mean: float = 0.5,
    chunk_weight_max: float = 0.5,
    skip_detector_retrain: bool = False,
    resume_extra: bool = False,
    detector_num_epochs: float | None = None,
    detector_max_train_samples: int | None = None,
    drift_optimize_kwargs: dict | None = None,
    topic_sources: dict[str, str] | None = None,
    auto_update_topic_sources: bool = False,
    topic_sources_auto_path: Path | None = None,
    detector_mode: str = "static",
    detector_learning_rate_scale: float | None = None,
    few_shot_n: int = 5,
    few_shot_pool_max: int = 50,
    few_shot_pool_path: Path | None = None,
    colab_mode: bool = False,
    colab_checkpoint_path: Path | None = None,
    drift_coef_opt_config: dict | None = None,
    topic_source: str = "original",
) -> list[dict]:
    """
    Co-training loop: deslop optimization each round; optional detector retrain via
    ``detector/train.py`` when ``detector_mode`` is ``adaptive`` and retrain is not skipped.
    """
    detector_train_config = detector_train_config or Path("configs/detector.yaml")
    cfg = yaml.safe_load(detector_train_config.read_text(encoding="utf-8"))
    if base_train_jsonl is None:
        merged = cfg.get("merged_train_jsonl")
        if not merged:
            raise ValueError("Set merged_train_jsonl in detector config or pass base_train_jsonl")
        base_train_jsonl = Path(merged)

    mutator_groq_model = mutator_groq_model or groq_model
    constraint_kwargs = constraint_kwargs or {}
    drift_kw = dict(drift_optimize_kwargs or {})
    repo_root = Path(__file__).resolve().parent.parent
    dco_cfg = dict(drift_coef_opt_config or {})
    dco_enabled = bool(dco_cfg.get("enabled"))
    dco_out = Path(
        dco_cfg.get("optimized_output_path", "outputs/cotrain/optimized_drift_coefs.json")
    )
    mode_norm = str(detector_mode or "static").strip().lower()
    # Only explicit ``adaptive`` runs detector/train.py; unknown modes default to no retrain.
    detector_train_this_round = mode_norm == "adaptive" and not bool(skip_detector_retrain)
    effective_skip_detector = not detector_train_this_round
    few_shot_pool_path = few_shot_pool_path or Path("outputs/cotrain/fewshot_pool.jsonl")
    colab_checkpoint_path = colab_checkpoint_path or Path("outputs/cotrain/colab_checkpoint.json")
    few_shot_enabled = few_shot_n > 0
    topic_src = dict(topic_sources or {})
    auto_best_slop: dict[str, float] = {}
    auto_path = Path(topic_sources_auto_path) if topic_sources_auto_path is not None else None
    if auto_update_topic_sources and auto_path is not None and auto_path.is_file():
        auto_passages, auto_best_slop = load_topic_sources_index(auto_path)
        topic_src.update(auto_passages)

    data_manager = CotrainDataManager(base_train_jsonl)
    pair_log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pair_logger = PairLogger(pair_log_path)
    round_logs: list[dict] = []

    current_ckpt = initial_detector_checkpoint
    _log(
        "cotrain: loading initial SlopDetector (first HF download can take several minutes; "
        f"checkpoint={initial_detector_checkpoint!r}) ..."
    )
    det = SlopDetector(checkpoint=current_ckpt)
    _log(f"cotrain: detector ready (device={getattr(det, 'device', '?')}).")
    det.version = 0

    extra_accum_path = detector_output_root / "extra_cotrain.jsonl"
    detector_output_root.mkdir(parents=True, exist_ok=True)
    if extra_accum_path.exists() and not resume_extra:
        extra_accum_path.unlink()

    fewshot_pool: list[dict] = (
        _load_fewshot_pool(few_shot_pool_path) if few_shot_enabled else []
    )
    _log(
        json.dumps(
            {
                "cotrain_detector_policy": {
                    "detector_mode": mode_norm,
                    "skip_detector_retrain": bool(skip_detector_retrain),
                    "detector_train_subprocess": detector_train_this_round,
                    "initial_checkpoint": current_ckpt,
                    "detector_learning_rate_scale": detector_learning_rate_scale,
                }
            },
            indent=2,
        )
    )

    for r in range(1, num_rounds + 1):
        # Feedback loop (Idea 3 → future rounds): learned drift coefficients override YAML defaults.
        if dco_enabled:
            loaded_w = load_optimized_drift_weights(dco_out)
            if loaded_w:
                drift_kw = dict(drift_kw)
                drift_kw["drift_weights"] = DriftWeights(
                    alpha_semantic=loaded_w["alpha_semantic"],
                    alpha_rouge=loaded_w["alpha_rouge"],
                    alpha_bertscore=loaded_w["alpha_bertscore"],
                )
                _log(
                    json.dumps(
                        {
                            "drift_coef_round_start": {
                                "round": r,
                                "alpha_semantic": loaded_w["alpha_semantic"],
                                "alpha_rouge": loaded_w["alpha_rouge"],
                                "alpha_bertscore": loaded_w["alpha_bertscore"],
                            }
                        }
                    )
                )

        sampled = random.sample(topics, min(topics_per_round, len(topics)))
        fool_essays: list[dict] = []
        best_slops: list[float] = []
        fitness_all: list[float] = []
        round_slop_scores: list[float] = []
        round_best_fool: dict[str, tuple[float, str]] = {}
        round_all_essays: list[dict] = []
        fewshot_round_candidates: list[dict] = []

        if detector_train_this_round:
            _log(
                f"cotrain: round {r}/{num_rounds} — detector ADAPTIVE (will retrain after deslop); "
                f"current checkpoint: {current_ckpt!r}"
            )
        elif mode_norm == "static":
            _log(
                f"cotrain: round {r}/{num_rounds} — detector FROZEN (static reward); "
                f"checkpoint in use: {current_ckpt!r}"
            )
        else:
            _log(
                f"cotrain: round {r}/{num_rounds} — detector retrain SKIPPED "
                f"(mode={mode_norm!r}, skip_detector_retrain={bool(skip_detector_retrain)}); "
                f"checkpoint in use: {current_ckpt!r}"
            )

        few_shot_texts = (
            _few_shot_texts_from_pool(fewshot_pool, few_shot_n) if few_shot_enabled else []
        )

        _log(
            f"cotrain: round {r}/{num_rounds} — deslop on {len(sampled)} topics "
            "(Groq + sentence-transformers; can be quiet for minutes per topic) ..."
        )

        for ti, topic in enumerate(sampled):
            short = topic if len(topic) <= 100 else topic[:97] + "..."
            _log(f"cotrain: round {r} topic {ti + 1}/{len(sampled)}: {short!r}")
            call_drift = dict(drift_kw)
            ar_mode = call_drift.get("alignment_reference_mode")
            if ar_mode == "source_passage":
                src = topic_src.get(topic)
                if not src:
                    src = call_drift.get("alignment_source_passage")
                if not src or not str(src).strip():
                    raise ValueError(
                        "alignment_reference_mode is 'source_passage' but no source passage "
                        f"for topic {topic!r}. Use topic_sources_jsonl, prior-round "
                        "topic_sources_auto.jsonl (auto_update_topic_sources), or "
                        "alignment_source_passage in YAML."
                    )
                call_drift["alignment_source_passage"] = str(src).strip()

            best_cand, essays = optimize(
                topic,
                target_llm,
                det,
                population_size=population_size,
                generations=generations_per_topic,
                essays_per_candidate=essays_per_candidate,
                pair_logger=pair_logger,
                log_path=detector_output_root / f"deslop_r{r}.jsonl",
                cotrain_round=r,
                semantic_similarity_weight=semantic_similarity_weight,
                mutator_groq_model=mutator_groq_model,
                embedding_model_name=embedding_model,
                constraint_kwargs=constraint_kwargs,
                chunked_scoring=chunked_scoring,
                chunk_window_tokens=chunk_window_tokens,
                chunk_stride_tokens=chunk_stride_tokens,
                chunk_max_chunks=chunk_max_chunks,
                chunk_aggregate=chunk_aggregate,
                chunk_weight_mean=chunk_weight_mean,
                chunk_weight_max=chunk_weight_max,
                few_shot_examples=few_shot_texts or None,
                **call_drift,
            )
            round_all_essays.extend(essays)
            if best_cand is None:
                _log(
                    json.dumps(
                        {
                            "cotrain_optimizer_skip": True,
                            "round": r,
                            "topic": topic,
                            "essays_logged": len(essays),
                        }
                    )
                )
                _log(
                    f"cotrain:   topic {ti + 1}/{len(sampled)} — no valid candidate "
                    f"(all essays failed constraints); skipping pair log and continuing"
                )
                continue

            _log(
                f"cotrain:   finished topic {ti + 1}/{len(sampled)} "
                f"(best_fitness={best_cand.fitness:.4f}, essays_logged={len(essays)})"
            )
            slops = [e["slop_score"] for e in essays if e.get("constraint_ok")]
            if slops:
                best_slops.append(min(slops))
                round_slop_scores.extend(float(s) for s in slops)
            fitness_all.append(best_cand.fitness)

            for e in essays:
                if not e.get("constraint_ok"):
                    continue
                if e["slop_score"] < binary_threshold:
                    fool_essays.append(
                        {
                            "text": e["essay"],
                            "score": 1.0,
                            "source": f"deslop_round_{r}",
                            "domain": str(e.get("domain", "deslop")),
                            "split": "train",
                            "round": r,
                            "prompt_id": e.get("prompt_id", ""),
                        }
                    )

            seed_cand = PromptCandidate(
                id="seed",
                system_prompt="You are a helpful assistant.",
                user_template="Write an essay about {topic}.",
                style_instructions="",
            )
            pair_logger.log_improvement(
                topic,
                seed_cand,
                best_cand,
                parent_score=1.0,
                child_score=min((e["slop_score"] for e in essays if e.get("constraint_ok")), default=1.0),
                round_num=r,
                topic_source=topic_source,
            )

            fool_ok = [
                e
                for e in essays
                if e.get("constraint_ok") and float(e["slop_score"]) < binary_threshold
            ]
            if fool_ok:
                best_e = min(fool_ok, key=lambda x: float(x["slop_score"]))
                round_best_fool[topic] = (float(best_e["slop_score"]), str(best_e["essay"]))
            if few_shot_enabled:
                for e in essays:
                    if not e.get("constraint_ok"):
                        continue
                    if float(e["slop_score"]) >= binary_threshold:
                        continue
                    fewshot_round_candidates.append(
                        {
                            "essay": e["essay"],
                            "slop_score": float(e["slop_score"]),
                            "round": r,
                            "topic": topic,
                        }
                    )

        if few_shot_enabled and fewshot_round_candidates:
            fewshot_round_candidates.sort(key=_fewshot_slop)
            top_new = fewshot_round_candidates[: max(0, few_shot_n)]
            fewshot_pool = _merge_and_save_fewshot_pool(
                few_shot_pool_path, fewshot_pool, top_new, few_shot_pool_max
            )

        goodhart_result = goodhart_check(round_all_essays, det)

        if auto_update_topic_sources and auto_path is not None:
            auto_path.parent.mkdir(parents=True, exist_ok=True)
            with auto_path.open("a", encoding="utf-8") as af:
                for topic, (slop, text) in sorted(round_best_fool.items(), key=lambda x: x[0]):
                    prev = auto_best_slop.get(topic, float("inf"))
                    if slop < prev:
                        auto_best_slop[topic] = slop
                        rec = {
                            "topic": topic,
                            "source_passage": text,
                            "slop_score": slop,
                            "round": r,
                        }
                        af.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        af.flush()
                        topic_src[topic] = text

        data_manager.accumulate(r, fool_essays)

        # Append fool essays for next train — merge into extra JSONL
        with extra_accum_path.open("a", encoding="utf-8") as xf:
            for row in fool_essays:
                xf.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Build full training JSONL path for this round (base + extra)
        full_train_path = detector_output_root / f"train_merged_r{r}.jsonl"
        _write_merged_train(base_train_jsonl, extra_accum_path, full_train_path, recent_weight, r)
        # When retraining remotely, this is the file to copy to the GPU machine.
        _log(json.dumps({"round": r, "train_jsonl_for_retrain": str(full_train_path)}))

        out_dir = detector_output_root / f"round_{r}"
        val_jsonl = Path(cfg.get("merged_val_jsonl") or "data/merged/val.jsonl")

        if effective_skip_detector:
            current_ckpt = str(det.checkpoint)
        else:
            if not val_jsonl.is_file():
                raise FileNotFoundError(
                    f"Validation JSONL missing: {val_jsonl}. "
                    "Run data merge or set merged_val_jsonl in detector config."
                )
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent.parent / "detector" / "train.py"),
                "--config",
                str(detector_train_config),
                "--train-jsonl",
                str(full_train_path),
                "--val-jsonl",
                str(val_jsonl),
                "--output-dir",
                str(out_dir),
            ]
            if detector_num_epochs is not None:
                cmd.extend(["--num-epochs", str(detector_num_epochs)])
            if detector_max_train_samples is not None and detector_max_train_samples > 0:
                cmd.extend(["--max-train-samples", str(int(detector_max_train_samples))])
            if detector_learning_rate_scale is not None and float(
                detector_learning_rate_scale
            ) > 0:
                cmd.extend(["--lr-scale", str(float(detector_learning_rate_scale))])
            console_log = out_dir / "detector_train_console.log"
            _log(
                "cotrain: starting detector/train.py subprocess "
                f"(console + {console_log.name}; training_trace.jsonl inside output_dir) ..."
            )
            sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            _run_subprocess_teed(
                cmd,
                cwd=str(Path(__file__).resolve().parent.parent),
                env=sub_env,
                log_path=console_log,
            )
            _log("cotrain: detector retrain subprocess finished.")

            current_ckpt = str(out_dir / "best")
            if Path(current_ckpt).exists():
                det = SlopDetector(checkpoint=current_ckpt)
        det.version = r

        detector_metrics: dict | None = None
        if not effective_skip_detector:
            mf = out_dir / "metrics_final.json"
            if mf.is_file():
                try:
                    detector_metrics = json.loads(mf.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    detector_metrics = None

        summary = {
            "round": r,
            "optimizer_best_mean_slop": float(sum(best_slops) / max(1, len(best_slops))),
            "mean_fitness": float(sum(fitness_all) / max(1, len(fitness_all))),
            "n_fool": len(fool_essays),
            "detector_path": current_ckpt,
            "detector_frozen": bool(effective_skip_detector),
            "detector_mode": mode_norm,
            "binary_threshold": binary_threshold,
            "deslop_slop_stats": _deslop_slop_stats(round_slop_scores, binary_threshold),
            "detector_metrics": detector_metrics,
            "metrics_final_path": str(out_dir / "metrics_final.json")
            if not effective_skip_detector
            else None,
            "goodhart_check": goodhart_result,
            "few_shot_pool_size": len(fewshot_pool) if few_shot_enabled else 0,
        }
        round_logs.append(summary)
        with summary_path.open("a", encoding="utf-8") as sf:
            sf.write(json.dumps(summary) + "\n")
            sf.flush()

        if colab_mode:
            _write_colab_round_checkpoint(
                colab_checkpoint_path,
                round_num=r,
                num_rounds=num_rounds,
                detector_frozen=bool(effective_skip_detector),
                checkpoint=current_ckpt,
                fewshot_pool_size=len(fewshot_pool) if few_shot_enabled else 0,
            )

        if dco_enabled and dco_cfg.get("run_after_each_round", True):
            pool_path = Path(dco_cfg.get("features_jsonl", "outputs/cotrain/drift_features_holdout.jsonl"))
            round_log = detector_output_root / f"deslop_r{r}.jsonl"
            n_new = append_deslop_round_to_feature_pool(round_log, pool_path)
            _log(json.dumps({"drift_coef_opt_pool_append": n_new, "round": r}))
            run_drift_coef_optimization_from_config_section(dco_cfg, repo_root=repo_root)

        if should_stop(round_logs, patience=stop_patience, epsilon=stop_epsilon):
            break

    return round_logs


def _write_merged_train(
    base: Path,
    extra: Path,
    out: Path,
    recent_weight: float,
    up_to_round: int,
) -> None:
    from cotrain.data_manager import load_jsonl

    rows = load_jsonl(base) if base.exists() else []
    if extra.exists():
        extra_rows = load_jsonl(extra)
        rows.extend(extra_rows)
        last_round_rows = [r for r in extra_rows if int(r.get("round", 0)) == up_to_round]
        n_extra = max(0, int(len(last_round_rows) * (recent_weight - 1.0)))
        rows.extend(last_round_rows[:n_extra])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Phase 3: adversarial co-training (deslop ↔ detector).")
    ap.add_argument("--config", type=Path, default=Path("configs/cotrain.yaml"))
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override initial detector checkpoint (otherwise uses initial_detector_checkpoint from YAML).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny budget: 1 round, small deslop, detector 1 epoch + capped train rows (see smoke_* in cotrain.yaml).",
    )
    ap.add_argument(
        "--skip-detector-retrain",
        action="store_true",
        help="Run deslop rounds only; do not call detector/train.py (saves GPU/time).",
    )
    ap.add_argument(
        "--resume-extra",
        action="store_true",
        help="Do not delete outputs/detector_cotrain/extra_cotrain.jsonl; append to it across runs.",
    )
    ap.add_argument(
        "--detector-num-epochs",
        type=float,
        default=None,
        help="Override detector train epochs (also see cotrain.yaml smoke_detector_epochs when using --smoke).",
    )
    ap.add_argument(
        "--detector-max-train-samples",
        type=int,
        default=None,
        help="Cap detector training rows (<=0 disables). Smoke mode sets a default from cotrain.yaml.",
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        nargs="?",
        const=Path("outputs/cotrain/run.log"),
        default=None,
        help=(
            "Tee stdout to this file (line-buffered) for monitoring while the notebook cell runs. "
            "Pass a path, or use bare --log-file for outputs/cotrain/run.log."
        ),
    )
    args = ap.parse_args()

    tee: _TeeStdout | None = None
    if args.log_file is not None:
        tee = _TeeStdout(sys.stdout, args.log_file)
        sys.stdout = tee

    try:
        _cotrain_main_after_parse(args)
    finally:
        if tee is not None:
            sys.stdout = tee._orig
            tee.close_log()


def _cotrain_main_after_parse(args) -> None:
    cfg_path = args.config
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg: dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    topics = _default_topics(max(30, int(cfg.get("topics_per_round", 20)) * 2))
    tf = cfg.get("topics_file")
    if tf is not None and str(tf).strip():
        tpath = _resolve_topics_file_path(str(tf).strip(), cfg_path)
        if not tpath.is_file():
            raise SystemExit(f"topics_file not found: {tpath} (configured as {tf!r})")
        topics = _load_topics_from_file(tpath)
        _log(
            json.dumps(
                {
                    "cotrain_topics_file": str(tpath),
                    "n_topics_loaded": len(topics),
                }
            )
        )

    num_rounds = int(cfg.get("num_rounds", 5))
    topics_per_round = int(cfg.get("topics_per_round", 20))
    population_size = int(cfg.get("population_size", 12))
    generations_per_topic = int(cfg.get("generations_per_topic", 10))
    essays_per_candidate = int(cfg.get("essays_per_candidate", 3))

    if args.smoke:
        num_rounds = 1
        topics_per_round = min(2, len(topics))
        population_size = min(4, population_size)
        generations_per_topic = min(2, generations_per_topic)
        essays_per_candidate = min(1, essays_per_candidate) or 1

    det_num_epochs: float | None = args.detector_num_epochs
    det_max_samples: int | None = args.detector_max_train_samples
    if args.smoke:
        if det_num_epochs is None:
            det_num_epochs = float(cfg.get("smoke_detector_epochs", 1))
        if det_max_samples is None:
            raw_ms = cfg.get("smoke_max_train_samples", 4096)
            if raw_ms is None or int(raw_ms) <= 0:
                det_max_samples = None
            else:
                det_max_samples = int(raw_ms)
    if det_max_samples is not None and det_max_samples <= 0:
        det_max_samples = None

    if det_num_epochs is not None or det_max_samples is not None:
        _log(
            json.dumps(
                {
                    "detector_train_limits": True,
                    "num_epochs": det_num_epochs,
                    "max_train_samples": det_max_samples,
                }
            )
        )

    detector_mode = str(cfg.get("detector_mode", "static")).strip().lower()
    skip_detector_retrain = bool(cfg.get("skip_detector_retrain", True)) or bool(
        args.skip_detector_retrain
    )
    detector_lr_scale_raw = cfg.get("detector_learning_rate_scale", 0.0)
    detector_learning_rate_scale: float | None
    if detector_lr_scale_raw is None:
        detector_learning_rate_scale = None
    else:
        detector_learning_rate_scale = float(detector_lr_scale_raw)
    few_shot_n = int(cfg.get("few_shot_n", 5))
    few_shot_pool_max = int(cfg.get("few_shot_pool_max", 50))
    few_shot_pool_path = Path(
        cfg.get("few_shot_pool_path", "outputs/cotrain/fewshot_pool.jsonl")
    )
    colab_mode = bool(cfg.get("colab_mode", False))
    colab_checkpoint_path = Path(
        cfg.get("colab_checkpoint_path", "outputs/cotrain/colab_checkpoint.json")
    )

    _print_compute_estimate(
        num_rounds=num_rounds,
        topics_per_round=topics_per_round,
        population_size=population_size,
        generations_per_topic=generations_per_topic,
        essays_per_candidate=essays_per_candidate,
        few_shot_enabled=few_shot_n > 0,
    )
    _log(
        "cotrain: detector retrain subprocess (when enabled) uses cwd=repository root; "
        "on Colab, `cd` into the cloned repo before running so paths resolve."
    )

    det_cfg_path = Path(cfg.get("detector_train_config", "configs/detector.yaml"))
    det_yaml = yaml.safe_load(det_cfg_path.read_text(encoding="utf-8"))
    base_train = None
    bd = cfg.get("base_dataset_jsonl")
    if bd:
        bp = Path(bd)
        if bp.is_file():
            base_train = bp
        elif bp.exists():
            raise SystemExit(f"base_dataset_jsonl is not a file: {bp}")

    default_train = det_yaml.get("merged_train_jsonl")
    train_check = base_train or Path(default_train) if default_train else None
    if train_check is not None and not train_check.is_file():
        raise SystemExit(
            f"Training JSONL missing: {train_check}. "
            "Build merged data (see README) or set base_dataset_jsonl / merged_train_jsonl."
        )

    groq_model = str(cfg.get("groq_model", "llama-3.3-70b-versatile"))
    essay_temp = float(cfg.get("essay_temperature", 0.9))
    target_llm = make_groq_essay_fn(groq_model, essay_temp)

    constraint_kwargs = {
        "min_words": int(cfg.get("min_words", 150)),
        "max_words": int(cfg.get("max_words", 2000)),
        "min_topic_similarity": float(cfg.get("min_topic_similarity", 0.3)),
    }

    ts_path = cfg.get("topic_sources_jsonl")
    topic_sources_map: dict[str, str] = {}
    if ts_path:
        tsp = Path(ts_path)
        if not tsp.is_file():
            raise SystemExit(f"topic_sources_jsonl not found: {tsp}")
        topic_sources_map = load_topic_sources_jsonl(tsp)

    logs = cotrain(
        topics,
        target_llm,
        str(args.checkpoint or cfg.get("initial_detector_checkpoint", "pangram/editlens_roberta-large")),
        num_rounds=num_rounds,
        topics_per_round=topics_per_round,
        population_size=population_size,
        generations_per_topic=generations_per_topic,
        essays_per_candidate=essays_per_candidate,
        detector_train_config=det_cfg_path,
        base_train_jsonl=base_train,
        binary_threshold=float(cfg.get("binary_threshold", 0.5)),
        pair_log_path=Path(cfg.get("pair_log_path", "outputs/cotrain/prompt_pairs.jsonl")),
        summary_path=Path(cfg.get("round_summary_path", "outputs/cotrain/round_summaries.jsonl")),
        detector_output_root=Path(cfg.get("detector_output_root", "outputs/detector_cotrain")),
        stop_patience=int(cfg.get("stop_patience", 2)),
        stop_epsilon=float(cfg.get("stop_epsilon", 0.02)),
        recent_weight=float(cfg.get("recent_round_weight", 1.5)),
        groq_model=groq_model,
        mutator_groq_model=str(cfg.get("mutator_groq_model", groq_model)),
        embedding_model=str(cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        semantic_similarity_weight=float(
            cfg.get("semantic_similarity_weight", cfg.get("lambda_semantic", 0.3))
        ),
        constraint_kwargs=constraint_kwargs,
        chunked_scoring=bool(cfg.get("chunked_scoring", True)),
        chunk_window_tokens=cfg.get("chunk_window_tokens"),
        chunk_stride_tokens=cfg.get("chunk_stride_tokens"),
        chunk_max_chunks=cfg.get("chunk_max_chunks"),
        chunk_aggregate=str(cfg.get("chunk_aggregate", "weighted")),
        chunk_weight_mean=float(cfg.get("chunk_weight_mean", 0.5)),
        chunk_weight_max=float(cfg.get("chunk_weight_max", 0.5)),
        skip_detector_retrain=skip_detector_retrain,
        resume_extra=args.resume_extra or bool(cfg.get("resume_extra", False)),
        detector_num_epochs=det_num_epochs,
        detector_max_train_samples=det_max_samples,
        drift_optimize_kwargs=drift_options_from_config(cfg),
        topic_sources=topic_sources_map if topic_sources_map else None,
        auto_update_topic_sources=bool(cfg.get("auto_update_topic_sources", False)),
        topic_sources_auto_path=Path(
            cfg.get("topic_sources_auto_path", "outputs/cotrain/topic_sources_auto.jsonl")
        )
        if cfg.get("auto_update_topic_sources", False)
        else None,
        detector_mode=detector_mode,
        detector_learning_rate_scale=detector_learning_rate_scale,
        few_shot_n=few_shot_n,
        few_shot_pool_max=few_shot_pool_max,
        few_shot_pool_path=few_shot_pool_path,
        colab_mode=colab_mode,
        colab_checkpoint_path=colab_checkpoint_path,
        drift_coef_opt_config=cfg.get("drift_coef_opt"),
        topic_source=str(cfg.get("topic_source", "original")),
    )
    _log(json.dumps({"rounds_completed": len(logs), "last": logs[-1] if logs else None}))


if __name__ == "__main__":
    main()
