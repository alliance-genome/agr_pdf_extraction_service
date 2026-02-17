"""Traceback reconstruction for 3-way DP alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.services.alignment.dp3 import MODE_COUNT, START_MODE, TRANSITIONS, DPResult, affine_gap_penalty
from app.services.alignment.scoring import transition_score
from app.services.alignment.telemetry import format_transition_reason
from app.services.consensus_models import Block


@dataclass
class AlignmentColumn:
    """One aligned output column with optional blocks from each source."""

    grobid_block: Block | None = None
    docling_block: Block | None = None
    marker_block: Block | None = None
    transition: str = ""
    local_score: float = 0.0
    cumulative_score: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _assert_monotonic(columns: list[AlignmentColumn]) -> None:
    """Defensive invariant check: each source index must be monotonic increasing."""
    last_index = {"grobid": -1, "docling": -1, "marker": -1}
    for col in columns:
        for source_name in ("grobid", "docling", "marker"):
            block = getattr(col, f"{source_name}_block")
            if block is None:
                continue
            if block.order_index <= last_index[source_name]:
                raise ValueError(f"Non-monotonic traceback for {source_name}")
            last_index[source_name] = block.order_index


def traceback_columns(result: DPResult, end_mode: int | None = None) -> list[AlignmentColumn]:
    """Reconstruct aligned columns from DP backpointers."""
    n_g = len(result.grobid_blocks)
    n_d = len(result.docling_blocks)
    n_m = len(result.marker_blocks)

    if n_g == 0 and n_d == 0 and n_m == 0:
        return []
    mode = result.best_mode if end_mode is None else end_mode
    if mode == START_MODE:
        return []

    i, j, k = n_g, n_d, n_m
    columns: list[AlignmentColumn] = []

    while i > 0 or j > 0 or k > 0:
        if mode < 0 or mode >= MODE_COUNT:
            raise ValueError(f"Invalid traceback mode at ({i},{j},{k}): {mode}")

        transition = TRANSITIONS[mode]
        di, dj, dk = transition.consume
        if i < di or j < dj or k < dk:
            raise ValueError(f"Traceback underflow for mode {transition.name} at ({i},{j},{k})")

        prev_i, prev_j, prev_k = i - di, j - dj, k - dk
        prev_mode = int(result.prev_mode[i, j, k, mode])
        if prev_mode < 0:
            raise ValueError(f"Missing backpointer at ({i},{j},{k}) mode={transition.name}")

        grobid_block = result.grobid_blocks[i - 1] if di else None
        docling_block = result.docling_blocks[j - 1] if dj else None
        marker_block = result.marker_blocks[k - 1] if dk else None
        pair_detail = transition_score(
            grobid_block,
            docling_block,
            marker_block,
            config=result.config,
        )
        _, gap_penalties = affine_gap_penalty(prev_mode, mode, config=result.config)
        local_score = float(result.move_score[i, j, k, mode])
        cumulative = float(result.scores[i, j, k, mode])

        columns.append(AlignmentColumn(
            grobid_block=grobid_block,
            docling_block=docling_block,
            marker_block=marker_block,
            transition=transition.name,
            local_score=local_score if np.isfinite(local_score) else 0.0,
            cumulative_score=cumulative if np.isfinite(cumulative) else 0.0,
            reason=format_transition_reason(
                transition.name,
                pair_detail.pair_scores,
                gap_penalties,
                local_score if np.isfinite(local_score) else 0.0,
            ),
            metadata={
                "pair_scores": {k: v.total for k, v in pair_detail.pair_scores.items()},
                "gap_penalties": gap_penalties,
            },
        ))

        i, j, k, mode = prev_i, prev_j, prev_k, prev_mode
        if i == 0 and j == 0 and k == 0 and mode == START_MODE:
            break

    columns.reverse()
    _assert_monotonic(columns)
    return columns
