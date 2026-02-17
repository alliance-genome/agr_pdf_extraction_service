"""Split/merge repair pass for traceback columns."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.services.alignment.arbitration import ArbitrationConfig, ArbitrationContext, choose_repair_candidate
from app.services.alignment.scoring import ScoreConfig, pair_score
from app.services.alignment.traceback import AlignmentColumn
from app.services.consensus_models import Block

_SOURCES = ("grobid", "docling", "marker")


def repair_split_merge_columns(
    columns: list[AlignmentColumn],
    *,
    config: ScoreConfig | None = None,
    arbitration_config: ArbitrationConfig | None = None,
    arbitration_context: ArbitrationContext | None = None,
    llm: Any = None,
    acceptance_threshold: float = 0.68,
    min_gain: float = 0.10,
) -> list[AlignmentColumn]:
    """Repair adjacent split/merge motifs using virtual concatenation scoring.

    Motifs handled:
    - match + adjacent gap
    - gap + adjacent match
    """
    cfg = config or ScoreConfig()
    arb_cfg = arbitration_config or ArbitrationConfig()
    arb_ctx = arbitration_context or ArbitrationContext()
    if len(columns) < 2:
        return columns

    repaired = list(columns)
    for idx in range(len(repaired) - 1):
        left = repaired[idx]
        right = repaired[idx + 1]
        _repair_adjacent_pair(
            left,
            right,
            config=cfg,
            arbitration_config=arb_cfg,
            arbitration_context=arb_ctx,
            llm=llm,
            acceptance_threshold=acceptance_threshold,
            min_gain=min_gain,
        )

    # Remove fully empty columns after repairs.
    return [col for col in repaired if _column_block_count(col) > 0]


def _column_block_count(column: AlignmentColumn) -> int:
    return sum(1 for source in _SOURCES if getattr(column, f"{source}_block") is not None)


def _get_block(column: AlignmentColumn, source: str) -> Block | None:
    return getattr(column, f"{source}_block")


def _set_block(column: AlignmentColumn, source: str, block: Block | None) -> None:
    setattr(column, f"{source}_block", block)


def _merge_blocks(left: Block, right: Block) -> Block:
    """Build a virtual concatenated block preserving source and order."""
    merged_raw = f"{left.raw_text}\n{right.raw_text}".strip()
    merged_norm = f"{left.normalized_text} {right.normalized_text}".strip()
    merged_md = f"{left.source_md}\n{right.source_md}".strip()
    return Block(
        block_id=f"{left.block_id}+{right.block_id}",
        block_type=left.block_type,
        raw_text=merged_raw,
        normalized_text=merged_norm,
        heading_level=left.heading_level,
        order_index=left.order_index,
        source=left.source,
        source_md=merged_md,
    )


def _annotate_repair(
    target: AlignmentColumn,
    *,
    split_source: str,
    anchor_source: str,
    motif: str,
    before_score: float,
    after_score: float,
    participating_block_ids: list[str],
) -> None:
    repairs = target.metadata.setdefault("repairs", [])
    repairs.append({
        "repair_type": "split_merge",
        "split_source": split_source,
        "anchor_source": anchor_source,
        "motif": motif,
        "before_score": round(before_score, 4),
        "after_score": round(after_score, 4),
        "participating_block_ids": participating_block_ids,
    })


def _repair_adjacent_pair(
    left: AlignmentColumn,
    right: AlignmentColumn,
    *,
    config: ScoreConfig,
    arbitration_config: ArbitrationConfig,
    arbitration_context: ArbitrationContext,
    llm: Any,
    acceptance_threshold: float,
    min_gain: float,
) -> None:
    for split_source in _SOURCES:
        split_left = _get_block(left, split_source)
        split_right = _get_block(right, split_source)
        if split_left is None or split_right is None:
            continue

        merged_candidate = _merge_blocks(split_left, split_right)
        repaired = False

        for anchor_source in _SOURCES:
            if anchor_source == split_source:
                continue

            anchor_left = _get_block(left, anchor_source)
            anchor_right = _get_block(right, anchor_source)
            target_column: AlignmentColumn | None = None
            clear_column: AlignmentColumn | None = None
            motif = ""
            anchor_block: Block | None = None

            if anchor_left is not None and anchor_right is None:
                motif = "match+gap"
                anchor_block = anchor_left
                target_column = left
                clear_column = right
            elif anchor_left is None and anchor_right is not None:
                motif = "gap+match"
                anchor_block = anchor_right
                target_column = right
                clear_column = left

            if anchor_block is None or target_column is None or clear_column is None:
                continue

            left_score = pair_score(split_left, anchor_block, config=config).total
            right_score = pair_score(split_right, anchor_block, config=config).total
            before_score = max(left_score, right_score)
            after_score = pair_score(merged_candidate, anchor_block, config=config).total

            if after_score < acceptance_threshold:
                continue
            if after_score < before_score + min_gain:
                keep_repair = choose_repair_candidate(
                    after_score=after_score,
                    before_score=before_score,
                    merged_text=merged_candidate.raw_text,
                    best_fragment_text=(
                        split_left.raw_text if left_score >= right_score else split_right.raw_text
                    ),
                    anchor_text=anchor_block.raw_text,
                    config=arbitration_config,
                    context=arbitration_context,
                    llm=llm,
                )
                if not keep_repair:
                    continue

            _set_block(target_column, split_source, replace(merged_candidate))
            _set_block(clear_column, split_source, None)
            _annotate_repair(
                target_column,
                split_source=split_source,
                anchor_source=anchor_source,
                motif=motif,
                before_score=before_score,
                after_score=after_score,
                participating_block_ids=[
                    split_left.block_id,
                    split_right.block_id,
                    anchor_block.block_id,
                ],
            )
            repaired = True
            break

        if repaired:
            continue
