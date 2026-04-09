"""Secondary evaluation / Goodhart hooks (stub for forward compatibility)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from detector.model import SlopDetector


def goodhart_check(essays: list[dict], detector: SlopDetector) -> dict[str, Any]:
    """
    Optional second opinion when the primary detector is the only reward.

    Stub: returns ``{"enabled": false}``. Replace with a real evaluator (e.g. second
    model, human rubric proxy) without changing ``cotrain/loop.py`` call sites beyond
    reading richer fields from this dict.
    """
    del essays, detector
    return {"enabled": False}
