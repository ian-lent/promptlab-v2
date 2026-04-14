#!/usr/bin/env python3
"""
Seq2seq fine-tuning (T5 + LoRA) for prompt rewriting.

We minimize teacher-forcing cross-entropy against evolutionary ``output_prompt`` labels. The
point is **amortization**: one forward pass at inference should approximate hundreds of Groq
mutations. Training uses AdamW with cosine decay + warmup (standard for transformer fine-tuning)
because it balances fast convergence with stable late-stage loss — the same rationale as in the
course baseline for encoder–decoder models.

Weights & Biases logs train/val loss plus an optional **rewriter slop** metric: rewrite a held-out
prompt, generate an essay via Groq, score with SlopDetector. If ``WANDB_API_KEY`` or Groq/detector
are unavailable, those steps are skipped without failing the run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from rewriter.dataset import (
    SPLIT_MANIFEST_PATH_DEFAULT,
    RewriterDataset,
    T5Seq2SeqCollator,
    assert_split_manifest_matches_rewriter_and_drift_configs,
    ensure_split_manifest,
    load_pairs_jsonl,
    pairs_in_split,
)
from rewriter.inference import apply_topic_placeholder, rewrite_prompt


class RewriterSlopCallback(TrainerCallback):
    """
    Runs **only** when ``Seq2SeqTrainer`` calls ``evaluate()`` — i.e. on the schedule set by
    ``Seq2SeqTrainingArguments.eval_strategy`` and ``eval_steps`` (not every training step).

    Groq + detector calls are therefore bounded by ``~ (total_steps / eval_steps) * sample_n``,
    not by dataset length each step. Samples are drawn exclusively from the **validation** split
    (``val_rows``); if that split is empty, the callback never runs.
    """

    def __init__(
        self,
        *,
        val_rows: list[dict[str, Any]],
        tokenizer: Any,
        detector_checkpoint: str,
        groq_model: str,
        essay_temperature: float,
        sample_n: int,
        device: torch.device,
        eval_strategy: str,
        eval_steps: int,
        generation_max_new_tokens: int,
        max_input_length: int,
    ) -> None:
        self.val_rows = val_rows[:sample_n]
        self.tokenizer = tokenizer
        self.detector_checkpoint = detector_checkpoint
        self.groq_model = groq_model
        self.essay_temperature = essay_temperature
        self.device = device
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.generation_max_new_tokens = generation_max_new_tokens
        self.max_input_length = max_input_length
        self._detector = None
        self._groq_fn = None
        self._logged_schedule = False

    def _lazy_init(self) -> bool:
        if self._detector is not None and self._groq_fn is not None:
            return True
        if not os.environ.get("GROQ_API_KEY"):
            return False
        from deslop.run_topic import make_groq_essay_fn
        from detector.model import SlopDetector

        self._detector = SlopDetector(checkpoint=self.detector_checkpoint)
        self._groq_fn = make_groq_essay_fn(self.groq_model, self.essay_temperature)
        return True

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not self.val_rows:
            return control
        if not self._logged_schedule:
            print(
                json.dumps(
                    {
                        "rewriter_slop_callback": {
                            "note": "fires only when Trainer.evaluate() runs",
                            "eval_strategy": self.eval_strategy,
                            "eval_steps": self.eval_steps,
                            "val_samples_used": len(self.val_rows),
                        }
                    }
                ),
                flush=True,
            )
            self._logged_schedule = True
        if not self._lazy_init():
            return control
        model.eval()
        scores: list[float] = []
        for row in self.val_rows:
            topic = str(row.get("topic", "")).strip()
            src = str(row.get("input_prompt", row.get("input", ""))).strip()
            if not src:
                continue
            rw = rewrite_prompt(
                src,
                model,
                self.tokenizer,
                device=self.device,
                max_input_length=self.max_input_length,
                max_new_tokens=self.generation_max_new_tokens,
            )
            essay_prompt = apply_topic_placeholder(rw, topic or "the assigned topic")
            try:
                essay = self._groq_fn(essay_prompt)
            except Exception:
                continue
            scores.append(float(self._detector.score(essay)))
        if scores:
            mean_s = sum(scores) / len(scores)
            import wandb

            if os.environ.get("WANDB_API_KEY"):
                try:
                    wandb.log({"eval/rewriter_slop_mean": mean_s, "step": state.global_step})
                except Exception:
                    pass
            print(
                json.dumps(
                    {"rewriter_slop_mean": mean_s, "step": state.global_step, "n": len(scores)}
                ),
                flush=True,
            )
        return control


def _coerce_override_value(raw: str) -> Any:
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in raw or "e" in low:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def merge_cli_config_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    """``--override max_steps=10`` style; values are coerced to int/float/bool when possible."""
    for item in overrides:
        if "=" not in item:
            continue
        key, _, val = item.partition("=")
        key = key.strip()
        if not key:
            continue
        cfg[key] = _coerce_override_value(val.strip())


def train_from_config(cfg_path: Path, *, overrides: list[str] | None = None) -> None:
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if overrides:
        merge_cli_config_overrides(cfg, overrides)
    repo_root = cfg_path.resolve().parent.parent
    assert_split_manifest_matches_rewriter_and_drift_configs(cwd=repo_root)

    pairs_path = Path(cfg["pairs_jsonl"])
    if not pairs_path.is_absolute():
        pairs_path = (repo_root / pairs_path).resolve()
    manifest_path = Path(cfg.get("split_manifest_path", SPLIT_MANIFEST_PATH_DEFAULT))
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    ensure_split_manifest(
        pairs_path,
        manifest_path,
        seed=int(cfg.get("split_seed", 42)),
        train_frac=float(cfg.get("train_frac", 0.7)),
        val_frac=float(cfg.get("val_frac", 0.15)),
        force_rebuild=bool(cfg.get("force_rebuild_split", False)),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_pairs = load_pairs_jsonl(pairs_path)
    train_rows = pairs_in_split(all_pairs, manifest, "train")
    val_rows = pairs_in_split(all_pairs, manifest, "val")
    if not train_rows:
        raise SystemExit("No training pairs after split — check pair log path and manifest.")

    model_id = str(cfg.get("base_model", "t5-base"))
    tok = AutoTokenizer.from_pretrained(model_id, legacy=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = RewriterDataset(
        train_rows,
        tok,
        max_input_length=int(cfg.get("max_input_length", 512)),
        max_target_length=int(cfg.get("max_target_length", 512)),
    )
    val_ds = RewriterDataset(
        val_rows or train_rows[: max(1, len(train_rows) // 10)],
        tok,
        max_input_length=int(cfg.get("max_input_length", 512)),
        max_target_length=int(cfg.get("max_target_length", 512)),
    )

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    rank = int(cfg.get("lora_r", 8))
    lora = LoraConfig(
        r=rank,
        lora_alpha=int(cfg.get("lora_alpha", rank * 2)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=list(cfg.get("lora_target_modules", ["q", "v"])),
    )
    model = get_peft_model(base, lora)

    out_dir = Path(cfg.get("output_dir", "outputs/rewriter/t5_lora"))
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", str(cfg.get("wandb_project", "promptlab-v2")))
        ent = cfg.get("wandb_entity")
        if ent:
            os.environ.setdefault("WANDB_ENTITY", str(ent))

    collator = T5Seq2SeqCollator(tok, tok.pad_token_id or 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    eval_strategy = str(cfg.get("eval_strategy", "steps"))
    eval_steps = int(cfg.get("eval_steps", 200))
    max_in_len = int(cfg.get("max_input_length", 512))
    gen_max = int(cfg.get("generation_max_new_tokens", cfg.get("max_target_length", 512)))
    gen_max = max(256, gen_max)
    slop_callback = RewriterSlopCallback(
        val_rows=val_rows,
        tokenizer=tok,
        detector_checkpoint=str(cfg.get("detector_checkpoint", "pangram/editlens_roberta-large")),
        groq_model=str(cfg.get("groq_model", "llama-3.3-70b-versatile")),
        essay_temperature=float(cfg.get("essay_temperature", 0.9)),
        sample_n=int(cfg.get("rewriter_slop_eval_n", 4)),
        device=device,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        generation_max_new_tokens=gen_max,
        max_input_length=max_in_len,
    )

    ta: dict[str, Any] = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(cfg.get("per_device_eval_batch_size", 4)),
        "learning_rate": float(cfg.get("learning_rate", 3e-4)),
        "num_train_epochs": float(cfg.get("num_train_epochs", 5)),
        "weight_decay": float(cfg.get("weight_decay", 0.01)),
        "lr_scheduler_type": str(cfg.get("lr_scheduler_type", "cosine")),
        "logging_steps": int(cfg.get("logging_steps", 20)),
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "save_steps": int(cfg.get("save_steps", 500)),
        "save_total_limit": int(cfg.get("save_total_limit", 2)),
        "load_best_model_at_end": bool(cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": str(cfg.get("metric_for_best_model", "loss")),
        "greater_is_better": False,
        "bf16": bool(cfg.get("bf16", torch.cuda.is_available())),
        "report_to": "wandb" if use_wandb else "none",
        "run_name": str(cfg.get("wandb_run_name") or f"t5-lora-r{rank}"),
        "seed": int(cfg.get("seed", 42)),
    }
    ws = cfg.get("warmup_steps")
    if ws is not None and int(ws) > 0:
        ta["warmup_steps"] = int(ws)
    else:
        ta["warmup_ratio"] = float(cfg.get("warmup_ratio", 0.06))

    max_steps_raw = cfg.get("max_steps")
    if max_steps_raw is not None:
        ms = int(max_steps_raw)
        if ms > 0:
            ta["max_steps"] = ms
        else:
            ta["max_steps"] = -1
    args = Seq2SeqTrainingArguments(**ta)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tok,
        callbacks=[slop_callback],
    )
    trainer.train()
    adapter_dir = out_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(json.dumps({"saved_adapter": str(adapter_dir)}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="T5 rewriter training. Use --override key=value for smoke tests (e.g. max_steps=10)."
    )
    p.add_argument("--config", type=Path, default=Path("configs/rewriter.yaml"))
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Override a YAML key (repeatable), e.g. --override max_steps=10 --override eval_steps=5 --override lora_r=4",
    )
    args = p.parse_args()
    train_from_config(args.config, overrides=args.override)


if __name__ == "__main__":
    main()
