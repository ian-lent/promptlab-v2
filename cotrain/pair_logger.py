"""JSONL log of (worse_prompt → better_prompt) pairs for rewriter distillation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from deslop.prompt_bank import PromptCandidate


class PairLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_improvement(
        self,
        topic: str,
        parent_prompt: PromptCandidate,
        child_prompt: PromptCandidate,
        parent_score: float,
        child_score: float,
        round_num: int,
        *,
        topic_source: str = "original",
    ) -> None:
        """Log when child slop score improves (lower is better)."""
        if child_score >= parent_score:
            return
        record = {
            "topic": topic,
            "topic_source": topic_source,
            "input_prompt": parent_prompt.full_text(),
            "output_prompt": child_prompt.full_text(),
            "input_slop_score": parent_score,
            "output_slop_score": child_score,
            "improvement": parent_score - child_score,
            "mutation_op": child_prompt.mutation_op,
            "round": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_trajectory(
        self,
        topic: str,
        trajectory: list[PromptCandidate],
        round_num: int,
        scores: list[float] | None = None,
        *,
        topic_source: str = "original",
    ) -> None:
        """
        Log consecutive pairs along a trajectory. If scores aligned with trajectory,
        only log steps that improved (lower slop).
        """
        if scores is None:
            scores = [0.0] * len(trajectory)
        for i in range(len(trajectory) - 1):
            a, b = trajectory[i], trajectory[i + 1]
            sa, sb = scores[i], scores[i + 1]
            self.log_improvement(
                topic, a, b, sa, sb, round_num, topic_source=topic_source
            )
        if len(trajectory) >= 2:
            self.log_improvement(
                topic,
                trajectory[0],
                trajectory[-1],
                scores[0],
                scores[-1],
                round_num,
                topic_source=topic_source,
            )

    def get_pairs(self, min_improvement: float = 0.05) -> list[dict]:
        if not self.log_path.exists():
            return []
        pairs = []
        with self.log_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if float(r.get("improvement", 0)) >= min_improvement:
                    pairs.append(r)
        return pairs
