#!/usr/bin/env python3
"""One-shot prompt rewrite using merged base + LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, help="File containing naive prompt text")
    p.add_argument("--prompt", type=str, default=None)
    args = p.parse_args()

    text = args.prompt
    if args.prompt_file:
        text = args.prompt_file.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Provide --prompt or --prompt-file")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter))

    system_msg = (
        "You are a prompt rewriter. Given an essay prompt, rewrite it so that an LLM following "
        "the rewritten prompt will produce text that sounds naturally human-written."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": text},
    ]
    prompt_ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(prompt_ids, max_new_tokens=512, do_sample=False)
    decoded = tok.decode(out[0][prompt_ids.shape[1] :], skip_special_tokens=True)
    print(decoded.strip())


if __name__ == "__main__":
    main()
