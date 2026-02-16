"""Micro-conflict extraction utilities.

This module isolates the word-level disagreement logic used to split a
segment into majority-agreed tokens plus minimal conflict spans for LLM
resolution.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from app.services.consensus_models import AGREE_EXACT, CONFLICT, AlignedTriple, MicroConflict, MicroConflictResult
from app.services.consensus_classification_assembly import _get_present_blocks, _pick_preferred_text

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Micro-conflict extraction (word-level, per segment)
# ---------------------------------------------------------------------------

_SENTENCE_END_TOKENS = {".", "!", "?"}
_MICRO_PLACEHOLDER_PREFIX = "__MC__"
_DELETION = "__MICRO_DELETION__"


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


def _build_position_map(
    opcodes: list[tuple[str, int, int, int, int]],
    ref_len: int,
) -> tuple[list[int | None], list[tuple[int, int, int]]]:
    """Map reference positions to another source and track insertions."""
    ref_to_other: list[int | None] = [None] * ref_len
    insertions: list[tuple[int, int, int]] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                ref_to_other[i1 + k] = j1 + k
        elif tag == "replace":
            overlap = min(i2 - i1, j2 - j1)
            for k in range(overlap):
                ref_to_other[i1 + k] = j1 + k
            if (j2 - j1) > overlap:
                insertions.append((i1 + overlap, j1 + overlap, j2))
        elif tag == "insert":
            insertions.append((i1, j1, j2))
        # delete => keep None mappings

    return ref_to_other, insertions


def _build_majority_alignment(
    source_tokens: dict[str, list[str]],
) -> tuple[list[str], list[MicroConflict], float]:
    """Build majority-agreed token stream with placeholders at true conflicts."""
    present_sources = [s for s in ("grobid", "docling", "marker") if source_tokens.get(s)]
    if not present_sources:
        return [], [], 1.0
    if len(present_sources) == 1:
        only = present_sources[0]
        return list(source_tokens.get(only, [])), [], 1.0

    ref_source = max(
        present_sources,
        key=lambda s: (len(source_tokens[s]), -("grobid", "docling", "marker").index(s)),
    )
    ref_tokens = source_tokens[ref_source]
    other_sources = [s for s in present_sources if s != ref_source]

    maps: dict[str, list[int | None]] = {}
    insertions_by_source: dict[str, dict[int, list[list[str]]]] = {}
    for other in other_sources:
        opcodes = SequenceMatcher(None, ref_tokens, source_tokens[other], autojunk=False).get_opcodes()
        ref_map, insertions = _build_position_map(opcodes, len(ref_tokens))
        maps[other] = ref_map
        grouped: dict[int, list[list[str]]] = {}
        for after_ref, j1, j2 in insertions:
            if j2 <= j1:
                continue
            grouped.setdefault(after_ref, []).append(source_tokens[other][j1:j2])
        insertions_by_source[other] = grouped

    columns: list[dict[str, str | None]] = []

    def _add_column(values: dict[str, str | None]) -> None:
        full = {"grobid": None, "docling": None, "marker": None}
        for src, val in values.items():
            full[src] = val
        columns.append(full)

    for ref_idx in range(len(ref_tokens) + 1):
        insertion_values: dict[str, str | None] = {}
        has_insertion = False
        for src in other_sources:
            pieces = insertions_by_source.get(src, {}).get(ref_idx, [])
            if pieces:
                insertion_values[src] = " ".join(" ".join(chunk) for chunk in pieces if chunk).strip()
                has_insertion = has_insertion or bool(insertion_values[src])
        if has_insertion:
            _add_column(insertion_values)

        if ref_idx == len(ref_tokens):
            break

        values: dict[str, str | None] = {ref_source: ref_tokens[ref_idx]}
        for src in other_sources:
            mapped = maps[src][ref_idx]
            values[src] = source_tokens[src][mapped] if mapped is not None else None
        _add_column(values)

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
        values = [column[s] if column[s] is not None else _DELETION for s in present_sources]
        counts = Counter(values)
        majority_token: str | None = None
        if counts:
            top_token, top_count = counts.most_common(1)[0]
            if top_count >= 2:
                majority_token = top_token

        if majority_token is not None:
            agree_columns += 1
            if active is not None:
                conflicts.append(active)
                active = None
            # Majority-agreed deletion: don't append to agreed_tokens
            if majority_token != _DELETION:
                agreed_tokens.append(majority_token)
            continue

        if active is None:
            active = _start_conflict()

        for src in ("grobid", "docling", "marker"):
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
    """Extract micro-conflicts for all CONFLICT segments and auto-reclassify 0-conflict cases."""
    results: dict[str, MicroConflictResult] = {}
    for triple in triples:
        if triple.classification != CONFLICT:
            continue
        result = extract_micro_conflicts_for_segment(triple)
        result = _coalesce_micro_conflicts_if_high_divergence(result)
        results[triple.segment_id] = result
        if not result.conflicts:
            triple.classification = AGREE_EXACT
            blocks = _get_present_blocks(triple)
            triple.agreed_text = _pick_preferred_text(
                blocks,
                triple.grobid_block.block_type if triple.grobid_block else "paragraph",
            )
    return results
