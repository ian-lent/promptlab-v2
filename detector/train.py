#!/usr/bin/env python3
"""
Train slop detector: K-class head with cross-entropy.

- **Fresh head** (``fresh_classification_head: true``): load ``FacebookAI/roberta-large``
  with ``num_labels=num_buckets`` (e.g. K=2 binary on mirror data).
- **EditLens checkpoint**: load gated ``pangram/editlens_roberta-large`` with matching K.

Supports ``--extra-data`` JSONL for co-training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from detector.training_trace import JsonlTrainingTraceCallback


def score_to_bucket(score: float, num_buckets: int) -> int:
    s = max(0.0, min(1.0, float(score)))
    b = int(math.floor(s * num_buckets))
    return max(0, min(num_buckets - 1, b))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rows_to_dataset(rows: list[dict], tokenizer, max_length: int, num_buckets: int) -> Dataset:
    texts = [str(r["text"]) for r in rows]
    labels = [score_to_bucket(r.get("score", 0.0), num_buckets) for r in rows]
    ds = Dataset.from_dict({"text": texts, "labels": labels})

    def tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = batch["labels"]
        return enc

    return ds.map(tok, batched=True, remove_columns=["text"])


def compute_metrics_factory(num_buckets: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels)
        preds = np.argmax(logits, axis=-1)
        macro = f1_score(labels, preds, average="macro", zero_division=0)
        acc = float((preds == labels).mean()) if len(labels) else 0.0
        out: dict[str, float] = {"macro_f1": float(macro), "accuracy": acc}
        if num_buckets == 2:
            m = np.max(logits, axis=-1, keepdims=True)
            exp = np.exp(logits - m)
            probs = exp / exp.sum(axis=-1, keepdims=True)
            pos_prob = probs[:, 1]
            try:
                out["roc_auc"] = float(roc_auc_score(labels, pos_prob))
            except ValueError:
                out["roc_auc"] = float("nan")
        return out

    return compute_metrics


def _try_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/detector.yaml"))
    p.add_argument("--extra-data", type=Path, default=None, help="Additional JSONL (co-training)")
    p.add_argument("--train-jsonl", type=Path, default=None, help="Override train path")
    p.add_argument("--val-jsonl", type=Path, default=None, help="Override val path")
    p.add_argument("--output-dir", type=Path, default=None, help="Override config output_dir")
    p.add_argument(
        "--num-epochs",
        type=float,
        default=None,
        help="Override num_epochs in YAML (quick tests / smoke)",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Shuffle (seed from YAML) and cap training rows after merging extra-data (quick tests)",
    )
    p.add_argument(
        "--lr-scale",
        type=float,
        default=None,
        help="Multiply YAML learning_rate (e.g. 0.1 from cotrain detector_learning_rate_scale).",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.num_epochs is not None:
        cfg["num_epochs"] = args.num_epochs
    checkpoint = cfg["checkpoint"]
    tokenizer_name = cfg.get("tokenizer_name") or "FacebookAI/roberta-large"
    num_buckets = int(cfg.get("num_buckets", 11))
    fresh_head = bool(cfg.get("fresh_classification_head", False))
    max_length = int(cfg.get("max_length", 512))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/detector"))

    train_path = args.train_jsonl or cfg.get("merged_train_jsonl")
    val_path = args.val_jsonl or cfg.get("merged_val_jsonl")
    if not train_path or not val_path:
        raise SystemExit("Set merged_train_jsonl / merged_val_jsonl in config or pass --train-jsonl/--val-jsonl")

    train_rows = load_jsonl(Path(train_path))
    if args.extra_data and args.extra_data.exists():
        train_rows.extend(load_jsonl(args.extra_data))
    val_rows = load_jsonl(Path(val_path))

    cap = args.max_train_samples
    if cap is not None and cap > 0 and len(train_rows) > cap:
        seed = int(cfg.get("seed", 42))
        rng = random.Random(seed)
        rng.shuffle(train_rows)
        train_rows = train_rows[:cap]
        print(
            f"Subsampled training rows to max_train_samples={cap} (shuffle seed={seed}); "
            f"val unchanged (n_val={len(val_rows)}).",
            flush=True,
        )

    hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_tok)

    train_ds = rows_to_dataset(train_rows, tokenizer, max_length, num_buckets)
    val_ds = rows_to_dataset(val_rows, tokenizer, max_length, num_buckets)

    if fresh_head:
        base_id = cfg.get("base_model") or tokenizer_name
        model = AutoModelForSequenceClassification.from_pretrained(
            base_id,
            num_labels=num_buckets,
            token=hf_tok,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, token=hf_tok)
        if model.config.num_labels != num_buckets:
            raise ValueError(
                f"Config num_buckets={num_buckets} but checkpoint has num_labels={model.config.num_labels}"
            )

    # MPS (Apple Silicon) often OOMs on roberta-large fine-tuning.
    # Default to CPU on MPS for reliability; you can override in YAML.
    is_mps = torch.backends.mps.is_available()
    force_cpu_on_mps = bool(cfg.get("force_cpu_on_mps", True))
    force_cpu = bool(cfg.get("force_cpu", False)) or (is_mps and force_cpu_on_mps)

    # If you do run on MPS, shorter sequences help a lot.
    if is_mps and not force_cpu and max_length > 256:
        max_length = 256

    # Reduce batch size and enable gradient checkpointing for a fighting chance.
    batch_size = int(cfg.get("batch_size", 8))
    if is_mps and batch_size > 2:
        batch_size = 2
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))

    if is_mps and not force_cpu:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # Transformers versions differ: some accept `no_cuda`, newer prefer `use_cpu`.
    # We set whichever exists for this environment.
    _ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    _device_kwargs: dict[str, Any] = {}
    if force_cpu:
        if "use_cpu" in _ta_params:
            _device_kwargs["use_cpu"] = True
        elif "no_cuda" in _ta_params:
            _device_kwargs["no_cuda"] = True

    trace_enabled = bool(cfg.get("training_trace", True))
    logging_steps = max(1, int(cfg.get("logging_steps", 20)))
    _eval_every = int(cfg.get("eval_every_steps", 0) or 0)
    use_step_eval = _eval_every > 0
    eval_steps = _eval_every if use_step_eval else None

    _ta_common: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(cfg.get("learning_rate", 2e-5)),
        **_device_kwargs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": min(batch_size, 2),
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": float(cfg.get("num_epochs", 3)),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.06)),
        "weight_decay": float(cfg.get("weight_decay", 0.01)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": int(cfg.get("save_total_limit", 3)),
        "seed": int(cfg.get("seed", 42)),
        "report_to": [],
    }
    if trace_enabled:
        _ta_common.update(
            {
                "logging_strategy": "steps",
                "logging_steps": logging_steps,
                "logging_first_step": True,
                "log_level": "info",
            }
        )
    else:
        _ta_common.update(
            {
                "logging_strategy": "epoch",
                "logging_steps": logging_steps,
                "logging_first_step": False,
            }
        )

    if use_step_eval and eval_steps is not None:
        _ta_common.update(
            {
                "eval_strategy": "steps",
                "eval_steps": eval_steps,
                "save_strategy": "steps",
                "save_steps": eval_steps,
            }
        )
    else:
        _ta_common.update({"eval_strategy": "epoch", "save_strategy": "epoch"})

    base_lr = float(_ta_common["learning_rate"])
    if args.lr_scale is not None:
        _ta_common["learning_rate"] = base_lr * float(args.lr_scale)

    _cuda_ok = torch.cuda.is_available() and not force_cpu
    _bf16_ok = _cuda_ok and getattr(torch.cuda, "is_bf16_supported", lambda: True)()
    _ta_common["bf16"] = bool(_bf16_ok)
    _ta_common["fp16"] = False

    training_args = TrainingArguments(**_ta_common)

    trace_cb: JsonlTrainingTraceCallback | None = None
    callbacks: list[Any] = []
    if trace_enabled:
        trace_path = output_dir / str(cfg.get("training_trace_filename", "training_trace.jsonl"))
        print(f"Training trace JSONL (append + flush each line): {trace_path.resolve()}", flush=True)
        trace_cb = JsonlTrainingTraceCallback(
            trace_path,
            meta={
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "num_buckets": num_buckets,
                "train_jsonl": _try_repo_relative(Path(train_path), repo_root),
                "val_jsonl": _try_repo_relative(Path(val_path), repo_root),
            },
            print_eval_summaries=bool(cfg.get("training_trace_print_eval", True)),
        )
        callbacks.append(trace_cb)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(num_buckets),
        callbacks=callbacks,
    )
    try:
        trainer.train()
        final_eval = trainer.evaluate()
    finally:
        if trace_cb is not None:
            trace_cb.close()

    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    metrics_final: dict[str, Any] = {
        "loss": final_eval.get("eval_loss"),
        "macro_f1": final_eval.get("eval_macro_f1"),
        "accuracy": final_eval.get("eval_accuracy"),
        "num_buckets": num_buckets,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "train_jsonl": _try_repo_relative(Path(train_path), repo_root),
        "val_jsonl": _try_repo_relative(Path(val_path), repo_root),
        "output_dir": _try_repo_relative(output_dir, repo_root),
        "best_model_path": _try_repo_relative(output_dir / "best", repo_root),
        "config_path": _try_repo_relative(Path(args.config), repo_root),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if num_buckets == 2 and "eval_roc_auc" in final_eval:
        metrics_final["roc_auc"] = final_eval.get("eval_roc_auc")

    out_metrics = output_dir / "metrics_final.json"
    out_metrics.write_text(
        json.dumps(_json_safe(metrics_final), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Colab / remote: zipping only `best/` still leaves metrics next to it as `round_N/metrics_final.json`.
    print(f"Saved to {output_dir / 'best'}")
    print(f"Wrote {out_metrics}")


if __name__ == "__main__":
    main()
