#!/usr/bin/env python3
"""
Greedy decoding for the T5 rewriter (LoRA adapter on top of a frozen ``t5-*`` base).

The distillation target is the co-training ``output_prompt`` field, which may contain the
literal placeholder ``<topic>`` (see ``PromptCandidate.full_text``). Before sending a rewritten
prompt to an essay LLM, replace that placeholder with the real topic string.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def apply_topic_placeholder(prompt: str, topic: str) -> str:
    """Swap logged placeholders for the concrete topic before essay generation."""
    t = topic.strip()
    return prompt.replace("<topic>", t).replace("{topic}", t)


def rewrite_prompt(
    prompt: str,
    model: Any,
    tokenizer: Any,
    *,
    device: torch.device | str | None = None,
    max_input_length: int = 512,
    max_new_tokens: int = 512,
    num_beams: int = 4,
) -> str:
    """
    Map ``input_prompt → output_prompt`` in one forward pass (plus beam search).

    Optimization rationale: beam search is a cheap deterministic proxy for the evolutionary
    search that produced labels; it reduces variance when reporting rewriter slop in W&B.
    """
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    from rewriter.dataset import RewriterDataset

    enc = tokenizer(
        RewriterDataset.TASK_PREFIX + prompt.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)
    with torch.inference_mode():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
    text = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
    return text


def main() -> None:
    p = argparse.ArgumentParser(description="Run T5 rewriter (base + LoRA adapter).")
    p.add_argument("--base-model", type=str, default="t5-base")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt-file", type=Path, default=None)
    args = p.parse_args()

    text = args.prompt or ""
    if args.prompt_file:
        text = args.prompt_file.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Provide --prompt or --prompt-file")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base_model, legacy=False)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.to(device)
    model.eval()
    out = rewrite_prompt(text, model, tok, device=device)
    print(out)


if __name__ == "__main__":
    main()
