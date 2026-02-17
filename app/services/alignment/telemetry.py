"""Alignment telemetry helpers."""

from __future__ import annotations

from app.services.alignment.scoring import PairScoreComponents


def format_transition_reason(
    transition_name: str,
    pair_scores: dict[str, PairScoreComponents],
    gap_penalties: dict[str, float],
    local_score: float,
) -> str:
    """Render an audit-friendly reason string for one traceback column."""
    pair_bits = ", ".join(
        f"{name}={score.total:.3f}"
        for name, score in sorted(pair_scores.items())
    )
    gap_bits = ", ".join(
        f"{src}={penalty:.3f}"
        for src, penalty in gap_penalties.items()
        if penalty != 0.0
    )
    if not gap_bits:
        gap_bits = "none"
    return f"{transition_name} | pairs[{pair_bits or 'none'}] | gaps[{gap_bits}] | local={local_score:.3f}"
