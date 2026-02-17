"""Anchor-based partitioning before 3-way global DP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from app.services.alignment.dp3 import align_three_way_global
from app.services.alignment.scoring import ScoreConfig, transition_score
from app.services.alignment.traceback import AlignmentColumn, traceback_columns
from app.services.consensus_models import Block

logger = logging.getLogger(__name__)

_SOURCES = ("grobid", "docling", "marker")


@dataclass(frozen=True)
class AnchorPartitionConfig:
    """Controls for anchor partitioning."""

    enabled: bool = True
    min_anchor_score: float = 0.72
    ambiguity_delta: float = 0.03
    include_structural_secondary: bool = True
    llm_tiebreak_enabled: bool = True
    max_heading_level: int = 2  # only H1/H2 become partition anchors


def build_alignment_windows(
    grobid_blocks: list[Block],
    docling_blocks: list[Block],
    marker_blocks: list[Block],
    *,
    score_config: ScoreConfig,
    partition_config: AnchorPartitionConfig,
    llm: Any = None,
) -> list[dict[str, list[Block]]]:
    """Return deterministic windows partitioned by strong anchors."""
    all_blocks = {
        "grobid": grobid_blocks,
        "docling": docling_blocks,
        "marker": marker_blocks,
    }
    if not partition_config.enabled:
        return [all_blocks]

    anchor_blocks = {
        source: [b for b in blocks if _is_anchor_candidate(b, partition_config)]
        for source, blocks in all_blocks.items()
    }
    non_empty_sources = sum(1 for blocks in anchor_blocks.values() if blocks)
    if non_empty_sources < 2:
        return [all_blocks]

    anchor_dp = align_three_way_global(
        anchor_blocks["grobid"],
        anchor_blocks["docling"],
        anchor_blocks["marker"],
        config=score_config,
    )
    anchor_columns = traceback_columns(anchor_dp)
    strong_anchors = _select_strong_anchors(anchor_columns, score_config, partition_config, llm=llm)
    if not strong_anchors:
        return [all_blocks]

    windows = _windows_from_anchors(all_blocks, strong_anchors)

    # Step 3: conservation invariant — every source block must appear exactly once
    conservation_ok = True
    for source in _SOURCES:
        windowed_count = sum(len(w.get(source, [])) for w in windows)
        expected_count = len(all_blocks[source])
        if windowed_count != expected_count:
            logger.warning(
                "Conservation violated for %s: windowed=%d, expected=%d — "
                "falling back to single-window alignment",
                source, windowed_count, expected_count,
            )
            conservation_ok = False

    if not conservation_ok:
        return [all_blocks]

    return windows


def _is_anchor_candidate(block: Block, config: AnchorPartitionConfig) -> bool:
    if block.block_type == "heading":
        level = block.heading_level or 1
        return level <= config.max_heading_level
    if not config.include_structural_secondary:
        return False
    return block.block_type in {"table", "figure_ref"}


def _coerce_keep_decision(raw: Any) -> bool | None:
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"keep", "candidate_a", "a", "true"}:
            return True
        if lowered in {"drop", "candidate_b", "b", "false"}:
            return False
        return None
    if isinstance(raw, dict):
        for key in ("decision", "choice", "chosen_candidate"):
            if key in raw:
                parsed = _coerce_keep_decision(raw[key])
                if parsed is not None:
                    return parsed
    return None


def _llm_keep_anchor(
    column: AlignmentColumn,
    score: float,
    config: AnchorPartitionConfig,
    llm: Any,
) -> bool:
    if not config.llm_tiebreak_enabled:
        return score >= config.min_anchor_score
    if not callable(getattr(llm, "resolve_alignment_tiebreak", None)):
        return score >= config.min_anchor_score

    payload = {
        "task": "decide_anchor_boundary",
        "candidate_a": {
            "decision": "keep",
            "score": score,
            "texts": {
                "grobid": column.grobid_block.raw_text if column.grobid_block else "",
                "docling": column.docling_block.raw_text if column.docling_block else "",
                "marker": column.marker_block.raw_text if column.marker_block else "",
            },
        },
        "candidate_b": {
            "decision": "drop",
            "score": config.min_anchor_score,
            "texts": {},
        },
    }
    try:
        decision = llm.resolve_alignment_tiebreak(payload)
        parsed = _coerce_keep_decision(decision)
        if parsed is not None:
            return parsed
    except Exception:
        logger.exception("Anchor arbitration LLM call failed; keeping deterministic decision")
    return score >= config.min_anchor_score


def _select_strong_anchors(
    columns: list[AlignmentColumn],
    score_config: ScoreConfig,
    config: AnchorPartitionConfig,
    *,
    llm: Any = None,
) -> list[AlignmentColumn]:
    selected: list[AlignmentColumn] = []
    last_seen = {source: -1 for source in _SOURCES}

    for col in columns:
        present = [getattr(col, f"{source}_block") for source in _SOURCES if getattr(col, f"{source}_block") is not None]
        if len(present) < 2:
            continue

        detail = transition_score(col.grobid_block, col.docling_block, col.marker_block, config=score_config)
        pair_scores = [max(0.0, min(1.0, pair.total)) for pair in detail.pair_scores.values()]
        if not pair_scores:
            continue

        anchor_score = sum(pair_scores) / len(pair_scores)
        if anchor_score < (config.min_anchor_score - config.ambiguity_delta):
            continue
        if anchor_score < (config.min_anchor_score + config.ambiguity_delta):
            keep = _llm_keep_anchor(col, anchor_score, config, llm)
            if not keep:
                continue

        monotonic = True
        for source in _SOURCES:
            block = getattr(col, f"{source}_block")
            if block is None:
                continue
            if block.order_index <= last_seen[source]:
                monotonic = False
                break
        if not monotonic:
            continue

        for source in _SOURCES:
            block = getattr(col, f"{source}_block")
            if block is not None:
                last_seen[source] = block.order_index
        selected.append(col)

    return selected


def _get_anchor_text(anchor: AlignmentColumn) -> str:
    """Extract anchor text from whichever sources have it."""
    for source in _SOURCES:
        block = getattr(anchor, f"{source}_block")
        if block is not None:
            return block.raw_text
    return ""


def _find_best_split_point(
    blocks: list[Block],
    cursor: int,
    anchor_text: str,
    max_index: int | None = None,
) -> int:
    """Find the best split index in blocks[cursor:max_index] based on similarity to anchor_text."""
    if not anchor_text or cursor >= len(blocks):
        return cursor

    upper = max_index if max_index is not None else len(blocks)
    upper = min(upper, len(blocks))
    if cursor >= upper:
        return cursor

    best_idx = cursor
    best_sim = -1.0
    anchor_norm = anchor_text.lower().strip()

    for i in range(cursor, upper):
        block = blocks[i]
        sim = fuzz.token_sort_ratio(block.normalized_text, anchor_norm) / 100.0
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    # Only use the split point if the similarity is reasonable
    if best_sim < 0.40:
        return cursor  # no good match, don't split

    return best_idx


def _next_anchor_index(
    anchors: list[AlignmentColumn],
    anchor_idx: int,
    source: str,
) -> int | None:
    """Return the order_index of the next anchor that has an explicit block for *source*."""
    for future in anchors[anchor_idx + 1:]:
        block = getattr(future, f"{source}_block")
        if block is not None:
            return block.order_index
    return None


def _windows_from_anchors(
    all_blocks: dict[str, list[Block]],
    anchors: list[AlignmentColumn],
) -> list[dict[str, list[Block]]]:
    # assert order_index == list position
    for source in _SOURCES:
        for i, b in enumerate(all_blocks[source]):
            assert b.order_index == i, (
                f"{source} order_index mismatch at position {i}: "
                f"block {b.block_id} has order_index={b.order_index}"
            )

    cursors = {source: 0 for source in _SOURCES}
    windows: list[dict[str, list[Block]]] = []

    for anchor_idx, anchor in enumerate(anchors):
        pre_window: dict[str, list[Block]] = {}
        pre_has_content = False
        split_ends: dict[str, int] = {}

        for source in _SOURCES:
            anchor_block = getattr(anchor, f"{source}_block")
            if anchor_block is not None:
                # Step 2: skip if anchor is behind cursor (already consumed)
                end = max(anchor_block.order_index, cursors[source])
            else:
                # Step 1: bounded search — restrict to next known anchor index
                anchor_text = _get_anchor_text(anchor)
                upper = _next_anchor_index(anchors, anchor_idx, source)
                end = _find_best_split_point(
                    all_blocks[source], cursors[source], anchor_text,
                    max_index=upper,
                )
            split_ends[source] = end
            sliced = all_blocks[source][cursors[source]:end]
            pre_window[source] = sliced
            if sliced:
                pre_has_content = True

        if pre_has_content:
            windows.append(pre_window)

        anchor_window: dict[str, list[Block]] = {}
        for source in _SOURCES:
            anchor_block = getattr(anchor, f"{source}_block")
            if anchor_block is not None:
                # Step 2: never regress cursor
                if anchor_block.order_index >= cursors[source]:
                    anchor_window[source] = [anchor_block]
                    cursors[source] = anchor_block.order_index + 1
                else:
                    # Already consumed — treat as empty for this window
                    anchor_window[source] = []
            else:
                anchor_window[source] = []
                cursors[source] = max(cursors[source], split_ends[source])
        windows.append(anchor_window)

    tail_window: dict[str, list[Block]] = {}
    tail_has_content = False
    for source in _SOURCES:
        sliced = all_blocks[source][cursors[source]:]
        tail_window[source] = sliced
        if sliced:
            tail_has_content = True
    if tail_has_content:
        windows.append(tail_window)

    return windows or [all_blocks]
