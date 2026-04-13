"""
Rewriter training data: prompt pairs from co-training logs, reproducible splits, T5 tokenization.

Split policy
------------
Rows are identified by a stable ``pair_row_id`` (hash of topic + input + output prompts) so
appending new lines to ``prompt_pairs.jsonl`` does not reshuffle old assignments. The manifest
stores which IDs belong to train / val / test.

Leakage control (Ideas 2 & 3)
-----------------------------
Drift-coefficient optimization should consume examples from the **test** split only (or a
dedicated export built from those rows). Rewriter fine-tuning must never train on pairs whose
IDs appear in that held-out set. This module writes ``rewriter_split_manifest.json`` so both
pipelines share one source of truth for who is held out.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

# Default manifest path — must match ``split_manifest_path`` defaults in configs/rewriter.yaml
# and ``drift_coef_opt.split_manifest_path`` in configs/deslop.yaml / cotrain.yaml.
SPLIT_MANIFEST_PATH_DEFAULT = "outputs/cotrain/splits/rewriter_split_manifest.json"


def resolve_split_manifest_path(value: str | Path | None, *, cwd: Path) -> Path:
    """Resolve a repo-relative or absolute split manifest path."""
    raw = str(value or SPLIT_MANIFEST_PATH_DEFAULT).strip()
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (cwd / p).resolve()


def assert_split_manifest_matches_rewriter_and_drift_configs(*, cwd: Path) -> None:
    """
    Fail fast if ``configs/rewriter.yaml`` and any ``drift_coef_opt`` block disagree on
    ``split_manifest_path`` (resolved). Prevents silent holdout drift between Ideas 2 and 3.
    """
    import yaml

    rw_yaml = cwd / "configs" / "rewriter.yaml"
    if not rw_yaml.is_file():
        return
    rw_cfg = yaml.safe_load(rw_yaml.read_text(encoding="utf-8")) or {}
    rw_resolved = resolve_split_manifest_path(rw_cfg.get("split_manifest_path"), cwd=cwd)
    for rel in ("configs/deslop.yaml", "configs/cotrain.yaml"):
        p = cwd / rel
        if not p.is_file():
            continue
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        dco = doc.get("drift_coef_opt")
        if not dco:
            continue
        drift_resolved = resolve_split_manifest_path(dco.get("split_manifest_path"), cwd=cwd)
        if drift_resolved != rw_resolved:
            raise AssertionError(
                f"split_manifest_path mismatch: {rw_yaml} resolves to {rw_resolved}, but "
                f"{p} drift_coef_opt resolves to {drift_resolved}. "
                "Use the same path in rewriter training and drift_coef_opt."
            )


def pair_row_id(record: dict[str, Any]) -> str:
    """Stable ID for a (topic, input_prompt, output_prompt) triple."""
    topic = str(record.get("topic", "")).strip()
    inp = str(record.get("input_prompt", record.get("input", ""))).strip()
    out = str(record.get("output_prompt", record.get("output", ""))).strip()
    payload = f"{topic}\n{inp}\n{out}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pairs_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_split_manifest(
    pairs: list[dict[str, Any]],
    *,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    source_path: str | None = None,
) -> dict[str, Any]:
    """
    Randomly partition pair row IDs into train / val / test (rest is test).

    Uses a fixed RNG seed so re-running with the same pairs file yields the same assignment.
    """
    if not pairs:
        return {
            "schema_version": 1,
            "seed": seed,
            "source_pairs_path": source_path,
            "source_sha256": None,
            "splits": {"train": [], "val": [], "test": []},
        }
    ids = [pair_row_id(r) for r in pairs]
    # Dedupe IDs while keeping one row per ID (last wins — rare collisions)
    id_to_row: dict[str, dict[str, Any]] = {}
    for r, pid in zip(pairs, ids, strict=True):
        id_to_row[pid] = r
    unique_ids = list(id_to_row.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    if n == 1:
        train_ids, val_ids, test_ids = unique_ids, [], []
    elif n == 2:
        train_ids, val_ids, test_ids = unique_ids[:1], [], unique_ids[1:]
    else:
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
        train_ids = unique_ids[:n_train]
        val_ids = unique_ids[n_train : n_train + n_val]
        test_ids = unique_ids[n_train + n_val :]
    return {
        "schema_version": 1,
        "seed": seed,
        "source_pairs_path": source_path,
        "source_sha256": _sha256_file(Path(source_path)) if source_path else None,
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
    }


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_split_manifest(
    pairs_jsonl: str | Path,
    manifest_path: str | Path,
    *,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """
    Load or create ``rewriter_split_manifest.json`` next to the pair log.

    If the manifest exists and ``force_rebuild`` is false, returns it after checking that the
    pairs file hash still matches (warns on stderr if not).
    """
    import warnings

    pairs_path = Path(pairs_jsonl)
    out_path = Path(manifest_path)
    pairs = load_pairs_jsonl(pairs_path)
    if out_path.is_file() and not force_rebuild:
        m = load_manifest(out_path)
        prev = m.get("source_sha256")
        cur = _sha256_file(pairs_path) if pairs_path.is_file() else None
        if prev and cur and prev != cur:
            warnings.warn(
                f"Pair log changed on disk since manifest was built ({pairs_path}); "
                "delete the manifest or pass force_rebuild=True to re-split.",
                stacklevel=2,
            )
        return m
    manifest = build_split_manifest(
        pairs,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        source_path=str(pairs_path.resolve()) if pairs_path.is_file() else None,
    )
    save_manifest(manifest, out_path)
    return manifest


def pairs_in_split(
    pairs: list[dict[str, Any]],
    manifest: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    allowed = set(manifest.get("splits", {}).get(split, []))
    if not allowed:
        return []
    return [r for r in pairs if pair_row_id(r) in allowed]


@dataclass
class T5Seq2SeqCollator:
    """Pad batches for causal language modeling loss on labels (ignore_index=-100)."""

    tokenizer: Any
    pad_token_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        import torch.nn.functional as F

        max_in = max(x["input_ids"].size(0) for x in batch)
        max_lab = max(x["labels"].size(0) for x in batch)
        input_ids = []
        attn = []
        labels = []
        for x in batch:
            pad_i = max_in - x["input_ids"].size(0)
            pad_l = max_lab - x["labels"].size(0)
            input_ids.append(F.pad(x["input_ids"], (0, pad_i), value=self.pad_token_id))
            attn.append(F.pad(x["attention_mask"], (0, pad_i), value=0))
            labels.append(F.pad(x["labels"], (0, pad_l), value=-100))
        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attn, dim=0),
            "labels": torch.stack(labels, dim=0),
        }


class RewriterDataset(Dataset):
    """
    T5 seq2seq pairs: encoder sees a task prefix + input prompt; decoder predicts output prompt.

    The prefix disambiguates the task for multi-task encoders and matches ``rewrite_prompt`` in
    ``rewriter.inference``.
    """

    TASK_PREFIX = "deslop rewrite: "

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_input_length: int = 512,
        max_target_length: int = 512,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.rows[idx]
        src = str(r.get("input_prompt", r.get("input", ""))).strip()
        tgt = str(r.get("output_prompt", r.get("output", ""))).strip()
        enc = self.tokenizer(
            self.TASK_PREFIX + src,
            truncation=True,
            max_length=self.max_input_length,
            padding=False,
            return_tensors="pt",
        )
        lab = self.tokenizer(
            tgt,
            truncation=True,
            max_length=self.max_target_length,
            padding=False,
            return_tensors="pt",
        )
        labels = lab["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
