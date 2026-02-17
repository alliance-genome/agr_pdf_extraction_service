"""Micro-conflict extraction utilities.

This module isolates the word-level disagreement logic used to split a
segment into majority-agreed tokens plus minimal conflict spans for LLM
resolution.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.alignment.dp3 import align_three_way_tokens
from app.services.consensus_models import AGREE_EXACT, AGREE_NEAR, CONFLICT, GAP, AlignedTriple, MicroConflict, MicroConflictResult
from app.services.consensus_classification_assembly import _get_present_blocks, _pick_preferred_text

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Micro-conflict extraction (word-level, per segment)
# ---------------------------------------------------------------------------

_SENTENCE_END_TOKENS = {".", "!", "?"}
_MICRO_PLACEHOLDER_PREFIX = "__MC__"
_DELETION = "__MICRO_DELETION__"


def normalize_for_comparison(token: str) -> str:
    """Normalize a token for comparison, stripping markdown formatting.

    Used to prevent formatting-only differences (bold/italic markers, dashes,
    smart quotes) from registering as content conflicts in the 3-way DP and
    majority vote.  The original un-normalized tokens are preserved for output
    so formatting from the preferred source is kept intact.
    """
    if not token or token == _DELETION:
        return token

    t = token

    # Strip matched bold/italic marker pairs:  *word* -> word,  **word** -> word
    while len(t) > 1 and t[0] == "*" and t[-1] == "*":
        t = t[1:-1]
    # Strip unmatched leading/trailing markers:  **Fig -> Fig,  text** -> text
    t = t.lstrip("*").rstrip("*")

    # Strip heading markers:  ## -> ""  (let fallback handle it)
    if t.startswith("#"):
        t = t.lstrip("#").lstrip()

    # Unicode compatibility normalization (handles superscripts, ligatures, etc.)
    t = unicodedata.normalize("NFKC", t)

    # Normalize dashes/minus to ASCII hyphen
    t = t.replace("\u2013", "-")   # en-dash
    t = t.replace("\u2014", "-")   # em-dash
    t = t.replace("\u2212", "-")   # unicode minus

    # Normalize smart quotes to ASCII
    t = t.replace("\u2018", "'").replace("\u2019", "'")   # single
    t = t.replace("\u201c", '"').replace("\u201d", '"')   # double

    return t or token  # fallback to original if stripping emptied it

# 3-way token DP allocates O(n_a * n_b * n_c * 8) memory.  At 500 tokens per
# stream that's ~9 GB which is safe after extractors finish and free RAM.
# Blocks exceeding this (e.g. 1900-token reference lists) skip the DP entirely
# and the whole segment is treated as a single conflict for LLM resolution.
_TOKEN_DP_MAX = 500


def tokenize_for_diff(text: str) -> list[str]:
    """Tokenize text for word-level diff/majority alignment."""
    if not text:
        return []

    text = unicodedata.normalize("NFC", text)
    raw_tokens = text.split()
    tokens: list[str] = []
    for tok in raw_tokens:
        if len(tok) > 1 and tok[-1] in _SENTENCE_END_TOKENS and tok[-2] not in _SENTENCE_END_TOKENS:
            tokens.append(tok[:-1])
            tokens.append(tok[-1])
        else:
            tokens.append(tok)
    return tokens


def _build_majority_alignment(
    source_tokens: dict[str, list[str]],
) -> tuple[list[str], list[MicroConflict], float]:
    """Build majority-agreed token stream with placeholders at true conflicts.

    Uses full 3-way DP alignment (same algorithm family as block-level alignment)
    to simultaneously align all three token streams, then majority-votes each column.
    """
    _SOURCES = ("grobid", "docling", "marker")
    present_sources = [s for s in _SOURCES if source_tokens.get(s)]
    if not present_sources:
        return [], [], 1.0
    if len(present_sources) == 1:
        only = present_sources[0]
        return list(source_tokens.get(only, [])), [], 1.0

    # Guard: if any token stream exceeds the cap, skip DP and treat the whole
    # segment as one conflict so it goes straight to LLM resolution.
    max_len = max(len(source_tokens.get(s, [])) for s in present_sources)
    if max_len > _TOKEN_DP_MAX:
        logger.info(
            "Token DP skipped: max stream length %d > %d cap, treating as single conflict",
            max_len, _TOKEN_DP_MAX,
        )
        placeholder = f"{_MICRO_PLACEHOLDER_PREFIX}0__"
        conflict = MicroConflict(
            conflict_id="",
            segment_id="",
            grobid_tokens=list(source_tokens.get("grobid", [])),
            docling_tokens=list(source_tokens.get("docling", [])),
            marker_tokens=list(source_tokens.get("marker", [])),
            context_before=[],
            context_after=[],
            output_position=0,
        )
        return [placeholder], [conflict], 0.0

    # Run 3-way DP alignment on the token streams.
    # Pass normalize_fn so the DP scores matches on normalized forms
    # (ignoring markdown formatting) while returning original tokens.
    raw_columns = align_three_way_tokens(
        source_tokens.get("grobid", []),
        source_tokens.get("docling", []),
        source_tokens.get("marker", []),
        normalize_fn=normalize_for_comparison,
    )

    # Convert tuples to dicts keyed by source name.
    columns: list[dict[str, str | None]] = []
    for tok_g, tok_d, tok_m in raw_columns:
        columns.append({"grobid": tok_g, "docling": tok_d, "marker": tok_m})

    # Majority-vote each column to build agreed stream + conflict spans.
    agreed_tokens: list[str] = []
    conflicts: list[MicroConflict] = []
    active: MicroConflict | None = None
    agree_columns = 0
    total_columns = len(columns)

    def _start_conflict() -> MicroConflict:
        conflict_index = len(conflicts)
        placeholder = f"{_MICRO_PLACEHOLDER_PREFIX}{conflict_index}__"
        output_position = len(agreed_tokens)
        agreed_tokens.append(placeholder)
        return MicroConflict(
            conflict_id="",
            segment_id="",
            grobid_tokens=[],
            docling_tokens=[],
            marker_tokens=[],
            context_before=[],
            context_after=[],
            output_position=output_position,
        )

    for column in columns:
        # Build raw and normalized value maps per source.
        raw_values = {s: (column[s] if column[s] is not None else _DELETION) for s in present_sources}
        norm_values = {s: normalize_for_comparison(v) for s, v in raw_values.items()}

        # Majority-vote on NORMALIZED forms so formatting-only
        # differences (e.g. *lag-1* vs lag-1) count as agreement.
        counts = Counter(norm_values.values())
        majority_norm: str | None = None
        if counts:
            top_norm, top_count = counts.most_common(1)[0]
            if top_count >= 2:
                majority_norm = top_norm

        if majority_norm is not None:
            agree_columns += 1
            if active is not None:
                conflicts.append(active)
                active = None
            # Majority-agreed deletion: don't append to agreed_tokens
            if majority_norm != _DELETION:
                # Pick the ORIGINAL token with the most characters among
                # agreeing sources — this preserves markdown formatting
                # (e.g. *lag-1* is preferred over lag-1).
                candidates = [
                    raw_values[s]
                    for s in present_sources
                    if norm_values[s] == majority_norm and raw_values[s] != _DELETION
                ]
                best_token = max(candidates, key=len) if candidates else majority_norm
                agreed_tokens.append(best_token)
            continue

        if active is None:
            active = _start_conflict()

        for src in _SOURCES:
            tok = column[src]
            if tok is None:
                continue
            if src == "grobid":
                active.grobid_tokens.append(tok)
            elif src == "docling":
                active.docling_tokens.append(tok)
            else:
                active.marker_tokens.append(tok)

    if active is not None:
        conflicts.append(active)

    ratio = (agree_columns / total_columns) if total_columns > 0 else 1.0
    return agreed_tokens, conflicts, ratio


def _expand_to_sentence_boundary(
    agreed_tokens: list[str],
    conflict_start: int,
    conflict_end: int,
    cap: int = 30,
) -> tuple[int, int]:
    """Expand a conflict span to nearest sentence boundaries, with a hard cap."""
    left_limit = max(0, conflict_start - cap)
    left = left_limit
    for idx in range(conflict_start - 1, left_limit - 1, -1):
        if agreed_tokens[idx] in _SENTENCE_END_TOKENS:
            left = idx + 1
            break

    right_limit = min(len(agreed_tokens), conflict_end + cap)
    right = right_limit
    for idx in range(conflict_end, right_limit):
        if agreed_tokens[idx] in _SENTENCE_END_TOKENS:
            right = idx + 1
            break

    return left, right


def _is_micro_placeholder(token: str) -> bool:
    return token.startswith(_MICRO_PLACEHOLDER_PREFIX)


def _micro_cfg_int(value: object, default: int) -> int:
    if value is None or value.__class__.__module__.startswith("unittest.mock"):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _micro_cfg_float(value: object, default: float) -> float:
    if value is None or value.__class__.__module__.startswith("unittest.mock"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fill_micro_contexts(
    *,
    agreed_tokens: list[str],
    conflicts: list[MicroConflict],
    context_cap: int,
) -> None:
    """Populate context_before/context_after for each micro-conflict."""
    for conflict in conflicts:
        left, right = _expand_to_sentence_boundary(
            agreed_tokens,
            conflict.output_position,
            conflict.output_position + 1,
            cap=context_cap,
        )
        before = [
            tok
            for tok in agreed_tokens[left:conflict.output_position]
            if not _is_micro_placeholder(tok)
        ]
        after = [
            tok
            for tok in agreed_tokens[conflict.output_position + 1:right]
            if not _is_micro_placeholder(tok)
        ]
        conflict.context_before = before
        conflict.context_after = after


def extract_micro_conflicts_for_segment(triple: AlignedTriple) -> MicroConflictResult:
    """Extract micro-conflicts for one CONFLICT segment (all block types)."""
    from config import Config

    block_type = "paragraph"
    blocks = _get_present_blocks(triple)
    if blocks:
        block_type = blocks[0].block_type

    source_texts = {
        "grobid": (triple.grobid_block.source_md or triple.grobid_block.raw_text) if triple.grobid_block else "",
        "docling": (triple.docling_block.source_md or triple.docling_block.raw_text) if triple.docling_block else "",
        "marker": (triple.marker_block.source_md or triple.marker_block.raw_text) if triple.marker_block else "",
    }
    source_tokens = {
        src: tokenize_for_diff(text)
        for src, text in source_texts.items()
    }

    agreed_tokens, raw_conflicts, ratio = _build_majority_alignment(source_tokens)
    for idx, conflict in enumerate(raw_conflicts):
        conflict.conflict_id = f"{triple.segment_id}_mc_{idx}"
        conflict.segment_id = triple.segment_id

    context_cap = _micro_cfg_int(getattr(Config, "MICRO_CONFLICT_CONTEXT_CAP", 30), 30)
    _fill_micro_contexts(
        agreed_tokens=agreed_tokens,
        conflicts=raw_conflicts,
        context_cap=context_cap,
    )

    return MicroConflictResult(
        segment_id=triple.segment_id,
        block_type=block_type,
        agreed_tokens=agreed_tokens,
        conflicts=raw_conflicts,
        majority_agree_ratio=ratio,
    )


def _coalesce_micro_conflicts_if_high_divergence(
    result: MicroConflictResult,
) -> MicroConflictResult:
    """Merge nearby micro-conflicts for high-divergence segments."""
    from config import Config

    if not result.conflicts:
        return result

    ratio_threshold = _micro_cfg_float(
        getattr(Config, "MICRO_CONFLICT_HIGH_DIVERGENCE_RATIO_THRESHOLD", 0.40),
        0.40,
    )
    span_threshold = _micro_cfg_int(
        getattr(Config, "MICRO_CONFLICT_HIGH_DIVERGENCE_SPAN_THRESHOLD", 12),
        12,
    )
    coalesce_gap = _micro_cfg_int(getattr(Config, "MICRO_CONFLICT_COALESCE_GAP", 8), 8)
    min_tokens = _micro_cfg_int(
        getattr(Config, "MICRO_CONFLICT_HIGH_DIVERGENCE_MIN_TOKENS", 10),
        10,
    )

    token_len = len(result.agreed_tokens)
    should_coalesce = (
        (result.majority_agree_ratio < ratio_threshold and token_len >= min_tokens)
        or len(result.conflicts) > span_threshold
    )
    if not should_coalesce or len(result.conflicts) <= 1:
        return result

    merged = [
        MicroConflict(
            conflict_id=result.conflicts[0].conflict_id,
            segment_id=result.conflicts[0].segment_id,
            grobid_tokens=list(result.conflicts[0].grobid_tokens),
            docling_tokens=list(result.conflicts[0].docling_tokens),
            marker_tokens=list(result.conflicts[0].marker_tokens),
            context_before=[],
            context_after=[],
            output_position=result.conflicts[0].output_position,
        ),
    ]
    agreed_tokens = list(result.agreed_tokens)
    cumulative_offset = 0

    for nxt in result.conflicts[1:]:
        adjusted_pos = nxt.output_position - cumulative_offset
        cur = merged[-1]
        gap = adjusted_pos - cur.output_position - 1
        if gap <= coalesce_gap:
            gap_start = cur.output_position + 1
            gap_end = adjusted_pos
            gap_tokens = agreed_tokens[gap_start:gap_end]

            cur.grobid_tokens.extend(gap_tokens)
            cur.grobid_tokens.extend(nxt.grobid_tokens)
            cur.docling_tokens.extend(gap_tokens)
            cur.docling_tokens.extend(nxt.docling_tokens)
            cur.marker_tokens.extend(gap_tokens)
            cur.marker_tokens.extend(nxt.marker_tokens)

            del agreed_tokens[gap_start:gap_end + 1]
            removed = gap + 1
            cumulative_offset += removed
            continue

        merged.append(
            MicroConflict(
                conflict_id=nxt.conflict_id,
                segment_id=nxt.segment_id,
                grobid_tokens=list(nxt.grobid_tokens),
                docling_tokens=list(nxt.docling_tokens),
                marker_tokens=list(nxt.marker_tokens),
                context_before=[],
                context_after=[],
                output_position=adjusted_pos,
            ),
        )

    for idx, conflict in enumerate(merged):
        conflict.conflict_id = f"{result.segment_id}_mc_{idx}"

    from config import Config as _Config
    context_cap = _micro_cfg_int(getattr(_Config, "MICRO_CONFLICT_CONTEXT_CAP", 30), 30)
    _fill_micro_contexts(
        agreed_tokens=agreed_tokens,
        conflicts=merged,
        context_cap=context_cap,
    )
    return MicroConflictResult(
        segment_id=result.segment_id,
        block_type=result.block_type,
        agreed_tokens=agreed_tokens,
        conflicts=merged,
        majority_agree_ratio=result.majority_agree_ratio,
    )


def reconstruct_segment_from_micro_conflicts(
    agreed_tokens: list[str],
    conflicts: list[MicroConflict],
    resolved: dict[str, str],
) -> str:
    """Rebuild final text by splicing resolved micro-conflict spans."""
    by_position = {c.output_position: c for c in conflicts}
    output_parts: list[str] = []
    idx = 0
    while idx < len(agreed_tokens):
        conflict = by_position.get(idx)
        if conflict is not None:
            text = (resolved.get(conflict.conflict_id) or "").strip()
            if text:
                output_parts.append(text)
            idx += 1
            continue
        token = agreed_tokens[idx]
        if not _is_micro_placeholder(token):
            output_parts.append(token)
        idx += 1

    text = " ".join(output_parts)
    text = re.sub(r" ([.,;:!?)])", r"\1", text)
    text = re.sub(r"([(]) ", r"\1", text)
    return text.strip()


def extract_micro_conflicts(
    triples: list[AlignedTriple],
) -> dict[str, MicroConflictResult]:
    """Extract micro-conflicts for CONFLICT, AGREE_NEAR, and multi-source GAP segments.

    Runs 3-way token DP on each eligible segment to find word-level disagreements.
    Uses ThreadPoolExecutor for parallelism (Numba JIT releases the GIL).
    Segments with 0 conflicts get majority-agreed text directly (no LLM needed).
    Segments with conflicts get only the disagreeing spans sent to LLM.
    """
    # Collect eligible triples.
    eligible: list[AlignedTriple] = []
    for triple in triples:
        if triple.classification == CONFLICT:
            eligible.append(triple)
        elif triple.classification == AGREE_NEAR:
            eligible.append(triple)
        elif triple.classification == GAP:
            present = _get_present_blocks(triple)
            if len(present) >= 2:
                eligible.append(triple)

    if not eligible:
        return {}

    # Process segments in parallel — Numba-JIT kernel releases the GIL
    # so threads achieve true parallelism on the CPU-bound DP work.
    max_workers = min(len(eligible), os.cpu_count() or 4)
    results: dict[str, MicroConflictResult] = {}

    def _process_one(triple: AlignedTriple) -> tuple[str, MicroConflictResult]:
        result = extract_micro_conflicts_for_segment(triple)
        if triple.classification == CONFLICT:
            result = _coalesce_micro_conflicts_if_high_divergence(result)
        return triple.segment_id, result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in eligible}
        for future in as_completed(futures):
            triple = futures[future]
            segment_id, result = future.result()
            results[segment_id] = result

            # If 0 conflicts found, all sources agree at the word level — no LLM
            # needed. Use the original source text (preserves formatting like
            # newlines in tables) rather than reconstructing from tokens.
            if not result.conflicts:
                triple.classification = AGREE_EXACT
                blocks = _get_present_blocks(triple)
                triple.agreed_text = _pick_preferred_text(
                    blocks,
                    triple.grobid_block.block_type if triple.grobid_block else "paragraph",
                )

    logger.info(
        "Micro-conflict extraction: %d segments processed in parallel (workers=%d)",
        len(eligible), max_workers,
    )
    return results
