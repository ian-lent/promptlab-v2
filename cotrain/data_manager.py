"""Accumulate base EditLens/mirror data plus deslopped essays across co-training rounds.

Topic-source JSONL for ``alignment_reference_mode: source_passage`` (including
``auto_update_topic_sources``) is written in ``cotrain/loop.py``, not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


class CotrainDataManager:
    def __init__(self, base_dataset_path: str | Path):
        self.base_path = Path(base_dataset_path)
        self.base_data: list[dict] = load_jsonl(self.base_path) if self.base_path.exists() else []
        self.round_data: dict[int, list[dict]] = {}

    def accumulate(self, round_num: int, essays: list[dict]) -> None:
        self.round_data[round_num] = list(essays)

    def build_training_set(self, up_to_round: int, recent_weight: float = 1.5) -> Dataset:
        rows: list[dict] = list(self.base_data)
        for r in range(1, up_to_round + 1):
            if r not in self.round_data:
                continue
            batch = self.round_data[r]
            if r == up_to_round:
                n_extra = max(0, int(len(batch) * (recent_weight - 1.0)))
                rows.extend(batch)
                rows.extend(batch[:n_extra] if n_extra else [])
            else:
                rows.extend(batch)
        if not rows:
            raise ValueError("No training rows — provide base_dataset_path with data.")
        return Dataset.from_dict(
            {
                "text": [r["text"] for r in rows],
                "labels": [float(r.get("score", 0.0)) for r in rows],
                "domain": [str(r.get("domain", "unknown")) for r in rows],
                "source": [str(r.get("source", "")) for r in rows],
            }
        )

    def get_stats(self) -> dict[str, Any]:
        from collections import Counter

        out: dict[str, Any] = {
            "base_n": len(self.base_data),
            "rounds": {},
        }
        for r, essays in self.round_data.items():
            out["rounds"][r] = {
                "n": len(essays),
                "sources": dict(Counter(e.get("source", "") for e in essays)),
            }
        return out
