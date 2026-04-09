"""Append-only JSONL trace for Hugging Face Trainer (live monitoring + post-hoc plots)."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


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


class JsonlTrainingTraceCallback(TrainerCallback):
    """
    Writes one JSON object per line to ``path``, flushed immediately so Colab / ``tail -f`` stay current.

    Events: ``train_begin``, ``log`` (per ``logging_steps``), ``eval`` (each validation), ``train_end``.
    """

    def __init__(
        self,
        path: Path,
        *,
        meta: dict[str, Any] | None = None,
        print_eval_summaries: bool = True,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta or {})
        self.print_eval_summaries = print_eval_summaries
        self._fp = self.path.open("a", encoding="utf-8", buffering=1)

    def _emit(self, record: dict[str, Any]) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        self._fp.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
        self._fp.flush()

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._emit(
            {
                "event": "train_begin",
                **self.meta,
                "max_steps": state.max_steps,
                "num_train_epochs": args.num_train_epochs,
            }
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if not logs:
            return
        self._emit(
            {
                "event": "log",
                "step": state.global_step,
                "epoch": round(float(state.epoch), 8),
                **{k: _json_safe(v) for k, v in logs.items()},
            }
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if not metrics:
            return
        rec: dict[str, Any] = {
            "event": "eval",
            "step": state.global_step,
            "epoch": round(float(state.epoch), 8),
            **_json_safe(metrics),
        }
        self._emit(rec)
        if self.print_eval_summaries:
            parts = []
            for k in sorted(metrics.keys()):
                v = metrics[k]
                if isinstance(v, float):
                    parts.append(f"{k}={v:.6g}")
                else:
                    parts.append(f"{k}={v}")
            print(f"[detector/train] step {state.global_step} eval | " + " ".join(parts[:20]), flush=True)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._emit(
            {
                "event": "train_end",
                "step": state.global_step,
                "epoch": round(float(state.epoch), 8),
            }
        )

    def close(self) -> None:
        fp = getattr(self, "_fp", None)
        if fp is not None and not fp.closed:
            fp.close()
        self._fp = None  # type: ignore[assignment]
