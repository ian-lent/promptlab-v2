"""
Dataset for essay-to-essay rewriting.

Input:  high-slop essay (baseline_essay)
Target: low-slop essay (best_essay)
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class EssayPairDataset(Dataset):
    """
    Each example:
      input_text  = "rewrite: " + baseline_essay
      target_text = best_essay
    """

    TASK_PREFIX = "rewrite: "

    def __init__(
        self,
        pairs: list[dict[str, Any]],
        tokenizer: Any,
        max_input_len: int = 512,
        max_target_len: int = 512,
        source_filter: list[str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_len = int(max_input_len)
        self.max_target_len = int(max_target_len)

        if source_filter:
            allowed = set(source_filter)
            pairs = [p for p in pairs if str(p.get("source", "organic")) in allowed]

        self.examples = [
            p for p in pairs if p.get("baseline_essay") and p.get("best_essay")
        ]

        print(
            f"[EssayPairDataset] {len(self.examples)} usable pairs "
            f"(filtered from {len(pairs)})",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p = self.examples[idx]
        input_text = self.TASK_PREFIX + str(p["baseline_essay"])
        target_text = str(p["best_essay"])

        enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        labels = dec["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }

