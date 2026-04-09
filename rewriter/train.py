#!/usr/bin/env python3
"""
QLoRA SFT for prompt→prompt rewriting (Llama-3.2-3B-Instruct).

Requires GPU, HF Llama access, and optional `pip install -e .[bnb]` for 4-bit (Linux/CUDA).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from trl import SFTConfig, SFTTrainer
except ImportError as e:
    raise SystemExit("Install trl>=0.8: pip install trl") from e


def load_pairs_jsonl(path: Path) -> Dataset:
    inputs, outputs = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            inputs.append(str(r["input"]))
            outputs.append(str(r["output"]))
    return Dataset.from_dict({"input": inputs, "output": outputs})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/rewriter.yaml"))
    args = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    raw = load_pairs_jsonl(Path(cfg["pairs_jsonl"]))
    split = raw.train_test_split(test_size=float(cfg.get("eval_split", 0.15)), seed=42)

    model_id = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_msg = (
        "You are a prompt rewriter. Given an essay prompt, rewrite it so that an LLM following "
        "the rewritten prompt will produce text that sounds naturally human-written. Preserve "
        "the original topic and intent."
    )

    def to_text(batch):
        texts = []
        for inp, out in zip(batch["input"], batch["output"], strict=True):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": inp},
                {"role": "assistant", "content": out},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    train_ds = split["train"].map(to_text, batched=True, remove_columns=["input", "output"])
    eval_ds = split["test"].map(to_text, batched=True, remove_columns=["input", "output"])

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    out_dir = Path(cfg["output_dir"])
    _cuda_ok = torch.cuda.is_available()
    _bf16_ok = _cuda_ok and getattr(torch.cuda, "is_bf16_supported", lambda: True)()
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 8)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        num_train_epochs=float(cfg.get("num_train_epochs", 4)),
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=bool(_bf16_ok),
        fp16=False,
        max_seq_length=int(cfg.get("max_seq_length", 1024)),
        report_to=[],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(out_dir / "lora_adapter"))
    print(f"Saved adapter to {out_dir / 'lora_adapter'}")


if __name__ == "__main__":
    main()
