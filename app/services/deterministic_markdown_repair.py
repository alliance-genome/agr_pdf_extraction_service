"""Harmless final Markdown byte normalization."""

from __future__ import annotations

import hashlib


def normalize_trailing_newline(
    text: str,
    audit: list[dict],
) -> tuple[str, list[dict], bool]:
    """End with exactly one LF and retain byte-exact provenance."""

    raw = text.encode("utf-8")
    if raw.endswith(b"\n") and (len(raw) == 1 or raw[-2:] not in {b"\r\n", b"\n\n"}):
        return text, audit, False

    retained = raw.rstrip(b"\r\n")
    retained_end = len(retained)
    normalized_audit: list[dict] = []
    for entry in audit:
        start = entry.get("output_byte_start")
        end = entry.get("output_byte_end")
        if type(start) is not int or type(end) is not int:
            raise ValueError("newline normalization requires integer audit ranges")
        if start >= retained_end:
            continue
        clipped = dict(entry)
        if end > retained_end:
            removed = end - retained_end
            source_start = clipped.get("source_byte_start")
            source_end = clipped.get("source_byte_end")
            if (
                type(source_start) is not int
                or type(source_end) is not int
                or source_end - removed <= source_start
            ):
                raise ValueError("newline normalization cannot clip audit provenance")
            clipped["output_byte_end"] = retained_end
            clipped["source_byte_end"] = source_end - removed
        normalized_audit.append(clipped)

    newline = b"\n"
    normalized_audit.append(
        {
            "output_byte_start": retained_end,
            "output_byte_end": retained_end + 1,
            "source": "deterministic_markup",
            "artifact_digest": hashlib.sha256(newline).hexdigest(),
            "source_byte_start": 0,
            "source_byte_end": 1,
            "candidate_id": None,
            "region_id": None,
            "decision_method": "deterministic",
            "transformation": "trailing_newline_normalization",
        }
    )
    normalized = (retained + newline).decode("utf-8", errors="strict")
    return normalized, normalized_audit, True
