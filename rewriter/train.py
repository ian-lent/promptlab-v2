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
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from deslop import similarity as sim
from detector.model import SlopDetector
from rewriter.essay_dataset import EssayPairDataset
from rewriter.dataset import (
    SPLIT_MANIFEST_PATH_DEFAULT,
    RewriterDataset,
    T5Seq2SeqCollator,
    assert_split_manifest_matches_rewriter_and_drift_configs,
    ensure_split_manifest,
    load_pairs_jsonl,
    mix_sources,
    pair_row_id,
    pairs_in_split,
)
from rewriter.inference import apply_topic_placeholder, rewrite_prompt


class SlopEarlyStoppingCallback(TrainerCallback):
    """Saves best adapter checkpoint by rewriter_slop_mean (lower = better)."""

    def __init__(self, output_dir: str, patience: int = 8):
        self.best_slop = float("inf")
        self.best_step = 0
        self.patience = int(patience)
        self.steps_without_improvement = 0
        self.best_dir = os.path.join(output_dir, "best_slop_adapter")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        m = metrics or kwargs.get("metrics") or {}
        slop = m.get("rewriter_slop_mean")
        if slop is None:
            return control
        slop_f = float(slop)
        if slop_f < self.best_slop:
            self.best_slop = slop_f
            self.best_step = int(state.global_step)
            self.steps_without_improvement = 0
            src = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(src):
                if os.path.exists(self.best_dir):
                    shutil.rmtree(self.best_dir)
                shutil.copytree(src, self.best_dir)
            print(
                f"[SlopEarlyStopping] New best: {slop_f:.4f} at step {state.global_step}",
                flush=True,
            )
        else:
            self.steps_without_improvement += 1
            if self.steps_without_improvement >= self.patience:
                control.should_training_stop = True
                print(
                    f"[SlopEarlyStopping] No improvement for {self.patience} evals. "
                    f"Best: {self.best_slop:.4f} at step {self.best_step}. Stopping.",
                    flush=True,
                )
        return control


class EssayRewriterSlopCallback(TrainerCallback):
    """
    Evaluates rewriter_slop_mean for essay-to-essay rewriting by scoring generated essays directly.
    """

    def __init__(
        self,
        *,
        val_pairs: list[dict[str, Any]],
        tokenizer: Any,
        detector_checkpoint: str,
        n: int,
        device: torch.device,
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> None:
        import random

        self.val_pairs = random.sample(val_pairs, min(int(n), len(val_pairs)))
        self.tokenizer = tokenizer
        self.detector_checkpoint = detector_checkpoint
        self.device = device
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self._detector = None

    def _lazy_init(self) -> bool:
        if self._detector is None:
            self._detector = SlopDetector(checkpoint=self.detector_checkpoint)
        return True

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None or not self.val_pairs:
            return control
        if not self._lazy_init():
            return control
        model.eval()
        scores: list[float] = []
        with torch.no_grad():
            for p in self.val_pairs:
                base = str(p.get("baseline_essay", "")).strip()
                if not base:
                    continue
                enc = self.tokenizer(
                    "rewrite: " + base,
                    return_tensors="pt",
                    max_length=self.max_input_tokens,
                    truncation=True,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out_ids = model.generate(**enc, max_new_tokens=self.max_new_tokens, num_beams=4)
                rewritten = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
                scores.append(float(self._detector.score(rewritten)))
        mean_slop = sum(scores) / len(scores) if scores else 1.0
        print(
            json.dumps(
                {
                    "rewriter_slop_mean": mean_slop,
                    "step": int(state.global_step),
                    "n": len(scores),
                    "mode": "essay_rewriting",
                }
            ),
            flush=True,
        )
        m = metrics or kwargs.get("metrics")
        if isinstance(m, dict):
            m["rewriter_slop_mean"] = mean_slop
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                wandb.log({"eval/rewriter_slop_mean": mean_slop, "step": state.global_step})
            except Exception:
                pass
        return control


class WeightedMixTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer with an optional WeightedRandomSampler and an optional semantic loss term.
    """

    _minilm = None

    def __init__(self, *args, train_sampler=None, semantic_loss_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler
        self._semantic_loss_weight = float(semantic_loss_weight or 0.0)

    def get_train_dataloader(self):
        if self._train_sampler is None:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader

        args = self.args
        return DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=self._train_sampler,
            collate_fn=self.data_collator,
            drop_last=args.dataloader_drop_last,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
        )

    @classmethod
    def _get_minilm(cls, model_name: str):
        if cls._minilm is None:
            cls._minilm = sim._get_embedder(model_name)
        return cls._minilm

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        ce_loss = loss

        sem_w = self._semantic_loss_weight
        sem_loss = None
        if sem_w > 0 and outputs.logits is not None:
            try:
                logits = outputs.logits  # (b, t, vocab)
                pred_ids = torch.argmax(logits, dim=-1)
                labels = inputs.get("labels")
                if labels is None:
                    raise ValueError("Missing labels for semantic loss.")
                pad_id = self.processing_class.pad_token_id or 0
                tgt_ids = labels.detach().clone()
                tgt_ids[tgt_ids == -100] = pad_id
                pred_txt = self.processing_class.batch_decode(
                    pred_ids.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                tgt_txt = self.processing_class.batch_decode(
                    tgt_ids.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                minilm_name = "sentence-transformers/all-MiniLM-L6-v2"
                emb = self._get_minilm(minilm_name)
                with torch.no_grad():
                    e_pred = emb.encode(pred_txt, convert_to_tensor=True).float()
                    e_tgt = emb.encode(tgt_txt, convert_to_tensor=True).float()
                    e_pred = torch.nn.functional.normalize(e_pred, p=2, dim=1)
                    e_tgt = torch.nn.functional.normalize(e_tgt, p=2, dim=1)
                    cos = (e_pred * e_tgt).sum(dim=1).clamp(-1, 1)
                    mean_cos = cos.mean()
                    sem_loss = 1.0 - mean_cos
                loss = ce_loss + sem_w * sem_loss

                if os.environ.get("WANDB_API_KEY"):
                    try:
                        import wandb

                        wandb.log(
                            {
                                "loss/cross_entropy": float(ce_loss.detach().cpu().item()),
                                "loss/semantic": float(sem_loss.detach().cpu().item()),
                            }
                        )
                    except Exception:
                        pass
            except Exception:
                # Never fail training because semantic logging/decoding failed.
                pass

        return (loss, outputs) if return_outputs else loss


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
            m = kwargs.get("metrics")
            if isinstance(m, dict):
                m["rewriter_slop_mean"] = mean_s
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

    mode = str(cfg.get("rewriter_mode", "prompt")).strip().lower()
    mix_enabled = bool(cfg.get("mix_sources", True))
    pairs_path = Path(cfg.get("pairs_jsonl", "outputs/cotrain/prompt_pairs.jsonl"))
    if not pairs_path.is_absolute():
        pairs_path = (repo_root / pairs_path).resolve()
    manifest_path = Path(cfg.get("split_manifest_path", SPLIT_MANIFEST_PATH_DEFAULT))
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()

    # Split manifest is always based on organic pairs only (leakage control).
    ensure_split_manifest(
        pairs_path,
        manifest_path,
        seed=int(cfg.get("split_seed", 42)),
        train_frac=float(cfg.get("train_frac", 0.7)),
        val_frac=float(cfg.get("val_frac", 0.15)),
        force_rebuild=bool(cfg.get("force_rebuild_split", False)),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    organic_pairs = load_pairs_jsonl(pairs_path)
    train_rows = pairs_in_split(organic_pairs, manifest, "train")
    val_rows = pairs_in_split(organic_pairs, manifest, "val")
    if not train_rows:
        raise SystemExit("No training pairs after split — check pair log path and manifest.")

    # Optionally mix in synthetic sources for training only.
    train_sampler = None
    if mix_enabled:
        mixed_rows, _sampler = mix_sources(cfg, repo_root=repo_root)
        # Keep only organic split IDs for train/val; add synthetic rows to train only.
        train_ids = set(manifest.get("splits", {}).get("train", []))
        val_ids = set(manifest.get("splits", {}).get("val", []))
        mixed_train: list[dict[str, Any]] = []
        for r in mixed_rows:
            src = str(r.get("source", "organic"))
            if src == "organic":
                pid = pair_row_id(r)
                if pid in train_ids:
                    mixed_train.append(r)
            else:
                mixed_train.append(r)
        train_rows = mixed_train
        # Rebuild sampler to align exactly with the final train_rows list.
        weights_cfg = dict(cfg.get("data_mix_weights") or {})
        source_weight = {
            "organic": float(weights_cfg.get("organic", 1.0)),
            "alpaca": float(weights_cfg.get("alpaca", 0.4)),
            "cross_topic": float(weights_cfg.get("cross_topic", 0.7)),
            "augmented": float(weights_cfg.get("augmented", 0.3)),
        }
        w = [source_weight.get(str(r.get("source", "organic")), 1.0) for r in train_rows]
        train_sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)
        # val remains organic-only for comparability
        val_rows = [r for r in organic_pairs if pair_row_id(r) in val_ids]

    model_id = str(cfg.get("base_model", "t5-base"))
    tok = AutoTokenizer.from_pretrained(model_id, legacy=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if mode == "essay":
        essay_pairs_path = Path(cfg.get("essay_pairs_jsonl", "outputs/rewriter/essay_pairs_organic.jsonl"))
        if not essay_pairs_path.is_absolute():
            essay_pairs_path = (repo_root / essay_pairs_path).resolve()
        all_essay_pairs = load_pairs_jsonl(essay_pairs_path)
        if not all_essay_pairs:
            raise SystemExit(
                f"No essay pairs found at {essay_pairs_path}. "
                "Generate them first with `uv run python rewriter/generate_essay_pairs.py`."
            )
        # Use split IDs computed from organic prompt pairs but train on essay pairs filtered to matching topics.
        allowed_train_topics = {str(r.get("topic", "")).strip() for r in train_rows if str(r.get("topic", "")).strip()}
        allowed_val_topics = {str(r.get("topic", "")).strip() for r in val_rows if str(r.get("topic", "")).strip()}
        essay_train = [p for p in all_essay_pairs if str(p.get("topic", "")).strip() in allowed_train_topics]
        essay_val = [p for p in all_essay_pairs if str(p.get("topic", "")).strip() in allowed_val_topics]
        source_filter = list(cfg.get("essay_source_filter", ["organic"]) or ["organic"])
        train_ds = EssayPairDataset(
            essay_train,
            tok,
            max_input_len=int(cfg.get("max_input_tokens", 512)),
            max_target_len=int(cfg.get("max_target_tokens", 512)),
            source_filter=source_filter,
        )
        val_ds = EssayPairDataset(
            essay_val or essay_train[: max(1, len(essay_train) // 10)],
            tok,
            max_input_len=int(cfg.get("max_input_tokens", 512)),
            max_target_len=int(cfg.get("max_target_tokens", 512)),
            source_filter=source_filter,
        )
    else:
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
    if mode == "essay":
        slop_callback = EssayRewriterSlopCallback(
            val_pairs=getattr(val_ds, "examples", []),
            tokenizer=tok,
            detector_checkpoint=str(cfg.get("detector_checkpoint", "pangram/editlens_roberta-large")),
            n=int(cfg.get("rewriter_slop_eval_n", 20)),
            device=device,
            max_input_tokens=int(cfg.get("max_input_tokens", 512)),
            max_new_tokens=gen_max,
        )
    else:
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

    early = SlopEarlyStoppingCallback(
        output_dir=str(out_dir),
        patience=int(cfg.get("early_stopping_patience", 8)),
    )
    ta["load_best_model_at_end"] = False

    trainer = WeightedMixTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tok,
        callbacks=[slop_callback, early],
        train_sampler=train_sampler,
        semantic_loss_weight=float(cfg.get("semantic_loss_weight", 0.0)),
    )
    trainer.train()
    print(
        json.dumps({"best_slop": early.best_slop, "best_slop_step": early.best_step}),
        flush=True,
    )
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
