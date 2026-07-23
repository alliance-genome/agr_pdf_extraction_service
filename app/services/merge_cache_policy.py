"""Digest checks for the single replacement merge artifact."""

from __future__ import annotations

import hashlib
from typing import Mapping


def verify_merged_output_digest(text: str, metrics: Mapping | None) -> str:
    """Require merged bytes to match the digest recorded by finalization."""

    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    expected = None if metrics is None else metrics.get("output_digest")
    if not expected:
        raise ValueError("merged artifact is missing its output digest")
    if expected != actual:
        raise ValueError("merged artifact digest does not match metrics")
    return actual
