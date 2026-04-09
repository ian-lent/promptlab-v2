"""SlopDetector: sequence classifier → continuous 0–1 score (any K buckets; K=2 binary supported)."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detector._device import default_torch_device


def _hf_token() -> str | bool | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _is_local_dir(checkpoint: str) -> bool:
    p = Path(checkpoint)
    return p.is_dir() and (p / "config.json").exists()


class SlopDetector:
    """
    Loads a HuggingFace ``AutoModelForSequenceClassification`` and maps softmax over
    K buckets to a score in ``[0, 1]``.

    **Decoding** (``score_decode``):

    - ``"expected_value"`` (default): weighted average of bucket indices scaled to ``[0, 1]``;
      spread mass → moderate score; mass on the last bucket → near 1. Matches the project spec.
    - ``"top_bucket_prob"``: ``max_k p_k`` (confidence in the predicted bucket); useful when the
      argmax class is often the same but peak probability differs.

    **Note:** The public ``pangram/editlens_roberta-large`` checkpoint uses **K = 4** (not 11).
    Bucket labels in the config are generic ``LABEL_0..3``; treat scores as *relative* until you
    calibrate on val data (``detector/calibrate.py``) or full-length essays, not tiny snippets.

    - **EditLens** (gated): ``pangram/editlens_roberta-large`` — set ``HF_TOKEN`` for reliable hub pulls.
    - **Binary mirror** (local): ``configs/detector_binary.yaml`` → ``K=2`` gives ``score`` = P(AI) under expected-value decode.
    """

    def __init__(
        self,
        checkpoint: str = "pangram/editlens_roberta-large",
        device: str | None = None,
        max_length: int = 512,
        token: str | bool | None = None,
        score_decode: str = "expected_value",
    ):
        self.checkpoint = checkpoint
        self.max_length = max_length
        if score_decode not in ("expected_value", "top_bucket_prob"):
            raise ValueError("score_decode must be 'expected_value' or 'top_bucket_prob'")
        self.score_decode = score_decode
        if device is None:
            device = str(default_torch_device())
        self.device = device
        local = _is_local_dir(checkpoint)
        auth: str | bool | None
        if token is not None:
            auth = token
        elif local:
            auth = False
        else:
            auth = _hf_token()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            token=auth,
        ).to(self.device)
        self.model.eval()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=auth)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "FacebookAI/roberta-large",
                token=False if local else auth,
            )

        self.num_buckets = int(self.model.config.num_labels)
        self.version = 0

    def score_proba(self, text: str) -> list[float]:
        """Return the softmax probability for each bucket label."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs.detach().cpu().tolist()

    def score(self, text: str) -> float:
        """Continuous 0–1 slop score (see class docstring for ``score_decode``)."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        if self.score_decode == "top_bucket_prob":
            return float(probs.max().item())
        bucket_values = torch.linspace(0, 1, self.num_buckets, device=self.device)
        return float((probs * bucket_values).sum().item())

    def score_top_bucket(self, text: str) -> float:
        """
        Alternate decode: probability of the *highest* bucket (often the most AI-like bucket).

        For K=2, this is exactly P(AI). For K>2, this can be a more discriminative scalar
        than the expected-value decode when probabilities are heavily skewed.
        """
        probs = torch.tensor(self.score_proba(text))
        return float(probs[-1].item())

    def score_batch(self, texts: list[str], batch_size: int = 16) -> list[float]:
        scores: list[float] = []
        self.model.eval()
        bucket_values = torch.linspace(0, 1, self.num_buckets, device=self.device)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            if self.score_decode == "top_bucket_prob":
                batch_scores = probs.max(dim=-1).values.tolist()
            else:
                batch_scores = (probs * bucket_values).sum(dim=-1).tolist()
            scores.extend(batch_scores)
        return scores

    def score_chunks(
        self,
        text: str,
        *,
        window_tokens: int | None = None,
        stride_tokens: int | None = None,
        max_chunks: int | None = None,
    ) -> list[float]:
        """
        Score a long document by sliding a token window over it.

        - **window_tokens**: window size in tokens (defaults to ``self.max_length``).
        - **stride_tokens**: step size in tokens (defaults to half window).
        - **max_chunks**: optional cap to bound runtime.

        Returns a list of per-chunk scores (same decode as ``score()``).
        """
        window_tokens = int(window_tokens or self.max_length)
        if stride_tokens is None:
            stride_tokens = max(1, window_tokens // 2)
        stride_tokens = int(stride_tokens)

        # Full-document encode is intentionally longer than model_max_length; we chunk below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            enc = self.tokenizer(
                text, add_special_tokens=True, truncation=False, return_tensors="pt"
            )
        input_ids = enc["input_ids"][0]  # (T,)
        attention_mask = enc["attention_mask"][0]

        T = int(input_ids.shape[0])
        if T <= window_tokens:
            return [self.score(text)]

        bucket_values = torch.linspace(0, 1, self.num_buckets, device=self.device)
        scores: list[float] = []

        start = 0
        n = 0
        while start < T:
            end = min(T, start + window_tokens)
            chunk_ids = input_ids[start:end].unsqueeze(0).to(self.device)
            chunk_mask = attention_mask[start:end].unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=chunk_ids, attention_mask=chunk_mask).logits
            probs = F.softmax(logits, dim=-1).squeeze(0)
            if self.score_decode == "top_bucket_prob":
                s = float(probs.max().item())
            else:
                s = float((probs * bucket_values).sum().item())
            scores.append(s)

            n += 1
            if max_chunks is not None and n >= int(max_chunks):
                break
            if end == T:
                break
            start += stride_tokens

        return scores

    def score_long(
        self,
        text: str,
        *,
        window_tokens: int | None = None,
        stride_tokens: int | None = None,
        max_chunks: int | None = None,
    ) -> dict[str, float | int]:
        """
        Chunked scoring for essay-length text.

        Returns dict with:
        - **mean**: mean chunk score
        - **max**: max chunk score (worst-case)
        - **n_chunks**: number of chunks scored
        """
        chunk_scores = self.score_chunks(
            text,
            window_tokens=window_tokens,
            stride_tokens=stride_tokens,
            max_chunks=max_chunks,
        )
        mean = float(sum(chunk_scores) / max(1, len(chunk_scores)))
        mx = float(max(chunk_scores)) if chunk_scores else float("nan")
        return {"mean": mean, "max": mx, "n_chunks": int(len(chunk_scores))}

    def save_versioned(self, output_dir: str | Path, extra_meta: dict[str, Any] | None = None) -> Path:
        output_dir = Path(output_dir)
        path = output_dir / f"detector_v{self.version}"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        meta = {"version": self.version, "checkpoint": self.checkpoint, "num_buckets": self.num_buckets}
        if extra_meta:
            meta.update(extra_meta)
        (path / "slop_detector_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_versioned(cls, path: str | Path, device: str | None = None) -> SlopDetector:
        path = Path(path)
        meta_file = path / "slop_detector_meta.json"
        version = 0
        if meta_file.exists():
            version = int(json.loads(meta_file.read_text(encoding="utf-8")).get("version", 0))
        det = cls(checkpoint=str(path), device=device, token=False)
        det.version = version
        return det
