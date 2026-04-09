"""Shared default device selection for inference and small scripts."""

from __future__ import annotations

import torch


def default_torch_device() -> torch.device:
    """Prefer CUDA when available; otherwise CPU (Colab / H100 friendly)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
