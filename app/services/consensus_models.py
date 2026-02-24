"""Core data models and shared constants for the consensus pipeline."""

from __future__ import annotations

from dataclasses import dataclass

AGREE_EXACT = "AGREE_EXACT"
AGREE_NEAR = "AGREE_NEAR"
GAP = "GAP"
CONFLICT = "CONFLICT"


@dataclass
class Block:
    block_id: str
    block_type: str  # "heading", "paragraph", "table", "figure_ref", "equation", "citation_list"
    raw_text: str
    normalized_text: str
    heading_level: int | None  # 1-6 for headings, None otherwise
    order_index: int  # 0-based position in source document
    source: str  # "grobid", "docling", "marker"
    source_md: str = ""  # original markdown syntax; falls back to raw_text when empty
    page_no: int | None = None  # page hint from source extraction, if available


@dataclass
class AlignedTriple:
    segment_id: str  # immutable, e.g., "seg_001"
    grobid_block: Block | None = None
    docling_block: Block | None = None
    marker_block: Block | None = None
    classification: str = ""  # set during classify step
    agreed_text: str | None = None  # resolved text for non-CONFLICT
    confidence: float = 0.0  # alignment confidence for this triple


@dataclass
class MicroConflict:
    """A contiguous span of disagreement within a segment."""

    conflict_id: str
    segment_id: str
    grobid_tokens: list[str]
    docling_tokens: list[str]
    marker_tokens: list[str]
    context_before: list[str]
    context_after: list[str]
    output_position: int


@dataclass
class MicroConflictResult:
    """Result of micro-conflict extraction for one segment."""

    segment_id: str
    block_type: str
    agreed_tokens: list[str]
    conflicts: list[MicroConflict]
    majority_agree_ratio: float
