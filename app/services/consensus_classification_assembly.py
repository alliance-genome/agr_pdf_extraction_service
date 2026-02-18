"""Classification, conflict telemetry, and assembly helpers."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from app.services.consensus_models import AGREE_EXACT, AGREE_NEAR, CONFLICT, GAP, AlignedTriple, Block
from app.services.consensus_parsing_alignment import (
    _IMAGE_REF_RE,
    _normalize_for_comparison,
    clean_output_md,
    normalize_text,
)

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Step 3: CLASSIFY — Consensus state for each triple
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LINK_URL_RE = re.compile(r"\]\([^)]*\)")
_INTEGRITY_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"
    r"(?![A-Za-z0-9_])"
)
_CITATION_KEY_RE = re.compile(
    r"\["
    r"(?:"
    r"\d+(?:\s*[-–,]\s*\d+)*"  # [8], [8-10], [8–10], [6,7]
    r"|"
    r"[A-Za-z]+(?:\s+et\s+al\.?)?\s*\d{4}"  # [Smith et al. 2024]
    r")"
    r"\]"
)

# Per-block-type source preference for AGREE_NEAR
_SOURCE_PREFERENCE = {
    "heading": "grobid",
    "citation_list": "grobid",
    "table": "docling",
    "figure_ref": "marker",
    "paragraph": "marker",
    "equation": "docling",
}
_TEXTUAL_BLOCK_TYPES = {"heading", "paragraph", "figure_ref", "citation_list"}
_STRUCTURED_BLOCK_TYPES = {"table", "equation"}

# Structural heading whitelist — headings that should be preserved deterministically
# even when only one extractor (typically Grobid) produces them.
_STRUCTURAL_HEADING_RE = re.compile(
    r"^#{1,6}\s*(Abstract|Introduction|Methods|Materials\s+and\s+Methods|"
    r"Results|Discussion|Conclusion|Conclusions|References|"
    r"Acknowledgments?|Acknowledgements?|Supplementary|"
    r"Supplementary\s+Materials?|Appendix|Background|Keywords)\s*$",
    re.IGNORECASE,
)


def is_structural_heading(text: str | None) -> bool:
    """Return True if text matches a known structural heading pattern."""
    if not text:
        return False
    return bool(_STRUCTURAL_HEADING_RE.match(text.strip()))


def _extract_numeric_tokens(text: str) -> set[str]:
    """Extract numeric tokens, ignoring numbers inside tags and URLs."""
    cleaned = _HTML_TAG_RE.sub("", text)
    cleaned = _IMAGE_REF_RE.sub("", cleaned)
    cleaned = _LINK_URL_RE.sub("]", cleaned)
    return set(_NUMERIC_RE.findall(cleaned))


def _extract_citation_keys(text: str) -> set[str]:
    """Extract citation keys after removing HTML tags."""
    cleaned = _HTML_TAG_RE.sub("", text)
    return set(_CITATION_KEY_RE.findall(cleaned))


def _extract_numeric_tokens_integrity(text: str) -> set[str]:
    """Extract numeric tokens for integrity checks (more permissive + normalization).

    Goal: detect novel numbers introduced by the LLM even when formatting varies
    (e.g., `.001` vs `0.001`, `1e-3` vs `0.001`).
    """
    cleaned = _HTML_TAG_RE.sub("", text or "")
    cleaned = _IMAGE_REF_RE.sub("", cleaned)
    cleaned = _LINK_URL_RE.sub("]", cleaned)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = cleaned.replace("\u2212", "-")  # Unicode minus

    out: set[str] = set()
    for m in _INTEGRITY_NUM_RE.finditer(cleaned):
        raw = (m.group(0) or "").strip()
        if not raw:
            continue
        s = raw
        if s.startswith("+"):
            s = s[1:]
        if s.startswith("."):
            s = "0" + s
        if s.startswith("-."):
            s = s.replace("-.", "-0.", 1)
        try:
            d = Decimal(s)
        except (InvalidOperation, ValueError):
            continue

        txt = format(d, "f")
        if "." in txt:
            txt = txt.rstrip("0").rstrip(".")
        if txt == "-0":
            txt = "0"
        out.add(txt)
    return out


def _segment_allowed_numeric_tokens(seg: dict) -> set[str]:
    """Numeric tokens from THIS segment's extractor texts only.

    Checks only the source texts for one segment, preventing cross-segment
    substitutions (e.g. "Figure 5" → "Figure 8" where 8 exists in a different
    segment) from passing the guard.
    """
    allowed: set[str] = set()
    for key in ("grobid", "docling", "marker"):
        text = (seg.get(key) or "").strip()
        if text:
            allowed |= _extract_numeric_tokens_integrity(text)
    # For gap segments, also include the single-source text field
    if seg.get("status") == "gap" and seg.get("text"):
        allowed |= _extract_numeric_tokens_integrity(seg["text"])
    return allowed


def _numeric_integrity_novel_numbers(output_text: str, allowed_numbers: set[str]) -> list[str]:
    """Return sorted list of numeric tokens present in output but not in allowed_numbers."""
    out_nums = _extract_numeric_tokens_integrity(output_text or "")
    novel = out_nums - (allowed_numbers or set())
    return sorted(novel)


def _numeric_integrity_dropped_numbers(
    output_text: str, seg: dict,
) -> list[str]:
    """Return sorted list of numeric tokens present in >=2 sources but missing from output.

    Only flags numbers that appear in the *majority* of available sources,
    so legitimately choosing one source over another does not trigger false
    positives.  For gap segments (single source), any number present in that
    source counts.
    """
    out_nums = _extract_numeric_tokens_integrity(output_text or "")
    source_num_sets: list[set[str]] = []
    for key in ("grobid", "docling", "marker"):
        text = (seg.get(key) or "").strip()
        if text:
            source_num_sets.append(_extract_numeric_tokens_integrity(text))
    if not source_num_sets:
        # Gap segment — use the single-source text field
        gap_text = (seg.get("text") or "").strip()
        if gap_text:
            source_num_sets.append(_extract_numeric_tokens_integrity(gap_text))
    if not source_num_sets:
        return []

    # Count how many sources contain each number
    num_counts: Counter[str] = Counter()
    for nums in source_num_sets:
        for n in nums:
            num_counts[n] += 1

    n_sources = len(source_num_sets)
    # Threshold: number must appear in >=2 sources (or the only source for gaps)
    threshold = min(2, n_sources)
    expected = {n for n, count in num_counts.items() if count >= threshold}
    dropped = expected - out_nums
    return sorted(dropped)


def _content_similarity_check(
    resolved_text: str, source_texts: dict[str, str],
) -> float:
    """Return the maximum token-set similarity between resolved text and any source.

    A low max similarity indicates the resolved text may have come from the
    wrong section of the paper rather than from this segment's sources.
    """
    if not resolved_text or not source_texts:
        return 0.0
    max_sim = 0.0
    for src_text in source_texts.values():
        if not src_text:
            continue
        sim = fuzz.token_set_ratio(resolved_text, src_text) / 100.0
        if sim > max_sim:
            max_sim = sim
    return max_sim


def _numeric_count_ratio(
    resolved_text: str, source_texts: dict[str, str],
) -> tuple[int, int]:
    """Return (output_num_count, max_source_num_count).

    Used to detect truncation: if a source has many numeric tokens (e.g. a
    figure legend with p-values) but the output has far fewer, the LLM
    likely truncated the content.
    """
    out_count = len(_extract_numeric_tokens_integrity(resolved_text or ""))
    max_src = 0
    for text in source_texts.values():
        count = len(_extract_numeric_tokens_integrity(text or ""))
        if count > max_src:
            max_src = count
    return out_count, max_src


def _get_present_blocks(triple: AlignedTriple) -> list[Block]:
    """Return the non-None blocks from a triple."""
    blocks = []
    for attr in ("grobid_block", "docling_block", "marker_block"):
        b = getattr(triple, attr)
        if b is not None:
            blocks.append(b)
    return blocks


def _pick_preferred_text(
    blocks: list[Block], block_type: str, agreeing_indices: tuple[int, int] | None = None,
) -> str:
    """Pick the best source text, preferring an agreeing block over an outlier.

    When *agreeing_indices* is provided (the pair that matched), the preferred
    source is chosen only among the agreeing blocks so the outlier is never
    selected.  Falls back to source-preference among all blocks if no agreeing
    pair is supplied.
    """
    candidates = (
        [blocks[i] for i in agreeing_indices] if agreeing_indices is not None else blocks
    )
    preferred_source = _SOURCE_PREFERENCE.get(block_type, "marker")
    for b in candidates:
        if b.source == preferred_source:
            return b.source_md or b.raw_text
    return candidates[0].source_md or candidates[0].raw_text


def classify_triples(
    triples: list[AlignedTriple],
    near_threshold: float = 0.92,
    levenshtein_threshold: float = 0.90,
    always_escalate_tables: bool = True,
    strict_numeric_near: bool = True,
) -> None:
    """Classify each triple in-place. Mutates triple.classification and triple.agreed_text."""
    for triple in triples:
        blocks = _get_present_blocks(triple)

        if len(blocks) == 0:
            triple.classification = GAP
            triple.agreed_text = ""
            continue

        if len(blocks) == 1:
            triple.classification = GAP
            triple.agreed_text = blocks[0].source_md or blocks[0].raw_text
            continue

        # Tables and equations escalate to CONFLICT when configured (Phase 1 default).
        # Escalate if ANY block in the triple is table/equation, not just the first,
        # since extractors may disagree on block type.
        block_types = {b.block_type for b in blocks}
        if always_escalate_tables and block_types & {"table", "equation"}:
            triple.classification = CONFLICT
            continue

        # Use first block's type for source-preference routing
        block_type = blocks[0].block_type

        # Check all pairs for agreement
        normalized_texts = [_normalize_for_comparison(b.normalized_text) for b in blocks]
        raw_texts = [b.raw_text for b in blocks]

        # Check for AGREE_EXACT: any pair has identical normalized text
        exact_pair = None
        for i in range(len(normalized_texts)):
            for j in range(i + 1, len(normalized_texts)):
                if normalized_texts[i] == normalized_texts[j]:
                    exact_pair = (i, j)
                    break
            if exact_pair is not None:
                break

        if exact_pair is not None:
            triple.classification = AGREE_EXACT
            triple.agreed_text = _pick_preferred_text(blocks, block_type, exact_pair)
            continue

        # Check for AGREE_NEAR: pairwise similarity >= thresholds
        # AND no numeric/citation differences
        near_pair = None
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                token_ratio = fuzz.token_set_ratio(
                    normalized_texts[i], normalized_texts[j]
                ) / 100.0

                # Normalized Levenshtein distance
                max_len = max(len(normalized_texts[i]), len(normalized_texts[j]), 1)
                lev_dist = Levenshtein.distance(normalized_texts[i], normalized_texts[j])
                lev_sim = 1.0 - (lev_dist / max_len)

                if token_ratio >= near_threshold and lev_sim >= levenshtein_threshold:
                    # Check numeric/citation guardrail (normalized for comparison)
                    norm_raw_i = _normalize_for_comparison(raw_texts[i])
                    norm_raw_j = _normalize_for_comparison(raw_texts[j])
                    nums_i = _extract_numeric_tokens(norm_raw_i)
                    nums_j = _extract_numeric_tokens(norm_raw_j)
                    cites_i = _extract_citation_keys(norm_raw_i)
                    cites_j = _extract_citation_keys(norm_raw_j)

                    # If enabled, escalate to CONFLICT only when numeric tokens
                    # actually differ between extractors.  When numbers match,
                    # the deterministic AGREE_NEAR path is safe and avoids
                    # exposing already-correct content to LLM corruption.
                    if strict_numeric_near and nums_i != nums_j:
                        continue

                    if nums_i == nums_j and cites_i == cites_j:
                        near_pair = (i, j)
                        break
            if near_pair is not None:
                break

        if near_pair is not None:
            # Structural heading guard: when a non-paired block has a
            # structural heading (e.g. "## Abstract" from Grobid) that would
            # be overwritten by non-heading text in the agreeing pair,
            # preserve the heading instead.  This prevents mis-aligned
            # heading/non-heading triples from silently dropping structure.
            outlier_indices = set(range(len(blocks))) - set(near_pair)
            structural_heading_block = None
            for idx in outlier_indices:
                b = blocks[idx]
                if b.block_type == "heading" and is_structural_heading(
                    b.source_md or b.raw_text
                ):
                    structural_heading_block = b
                    break

            if structural_heading_block is not None:
                pair_has_heading = any(
                    blocks[i].block_type == "heading" for i in near_pair
                )
                if not pair_has_heading:
                    logger.info(
                        "AGREE_NEAR %s: preserving structural heading '%s' "
                        "from %s over non-heading agreeing pair",
                        triple.segment_id,
                        (structural_heading_block.source_md
                         or structural_heading_block.raw_text),
                        structural_heading_block.source,
                    )
                    triple.classification = AGREE_NEAR
                    triple.agreed_text = (
                        structural_heading_block.source_md
                        or structural_heading_block.raw_text
                    )
                    continue

            triple.classification = AGREE_NEAR
            triple.agreed_text = _pick_preferred_text(blocks, block_type, near_pair)
            continue

        # Everything else is CONFLICT
        triple.classification = CONFLICT

        # Debug diagnostics: why this triple fell through to CONFLICT
        if logger.isEnabledFor(logging.DEBUG):
            reasons: list[str] = []
            best_token = 0.0
            best_lev = 0.0
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    token_ratio = (
                        fuzz.token_set_ratio(normalized_texts[i], normalized_texts[j]) / 100.0
                    )
                    max_len = max(len(normalized_texts[i]), len(normalized_texts[j]), 1)
                    lev_dist = Levenshtein.distance(normalized_texts[i], normalized_texts[j])
                    lev_sim = 1.0 - (lev_dist / max_len)
                    best_token = max(best_token, token_ratio)
                    best_lev = max(best_lev, lev_sim)

                    if token_ratio < near_threshold:
                        reasons.append(f"token_ratio={token_ratio:.3f}<{near_threshold}")
                    if lev_sim < levenshtein_threshold:
                        reasons.append(f"lev_sim={lev_sim:.3f}<{levenshtein_threshold}")

                    diag_norm_i = _normalize_for_comparison(raw_texts[i])
                    diag_norm_j = _normalize_for_comparison(raw_texts[j])
                    nums_i = _extract_numeric_tokens(diag_norm_i)
                    nums_j = _extract_numeric_tokens(diag_norm_j)
                    if nums_i != nums_j:
                        reasons.append(f"numeric_diff={nums_i.symmetric_difference(nums_j)}")

                    cites_i = _extract_citation_keys(diag_norm_i)
                    cites_j = _extract_citation_keys(diag_norm_j)
                    if cites_i != cites_j:
                        reasons.append(f"citation_diff={cites_i.symmetric_difference(cites_j)}")

            logger.debug(
                "CONFLICT %s: best_token=%.3f best_lev=%.3f reasons=%s",
                triple.segment_id,
                best_token,
                best_lev,
                "; ".join(reasons[:5]),
            )


# ---------------------------------------------------------------------------
# Step 4: GUARD — Check fallback conditions
# ---------------------------------------------------------------------------

def _triple_block_family(triple: AlignedTriple) -> str:
    """Map a triple to a coarse block family for guard calibration."""
    blocks = _get_present_blocks(triple)
    if not blocks:
        return "unknown"

    block_types = {b.block_type for b in blocks}
    if block_types & _STRUCTURED_BLOCK_TYPES:
        return "structured"
    if block_types & _TEXTUAL_BLOCK_TYPES:
        return "textual"
    return "textual"


def _triple_source_count(triple: AlignedTriple) -> int:
    """Count how many extractors produced a block for this triple."""
    return sum(1 for b in (triple.grobid_block, triple.docling_block, triple.marker_block) if b is not None)


def _compute_conflict_telemetry(
    triples: list[AlignedTriple],
    conflict_ratio_threshold: float = 0.4,
    localized_conflict_span_max: float = 0.35,
    localized_conflict_relief: float = 0.15,
    localized_conflict_max_blocks: int = 25,
) -> dict:
    """Compute guard telemetry for calibration and adaptive thresholding.

    Two-source conflicts (where median-source is degenerate and we escalate
    to a single LLM call) are excluded from the conflict ratios that drive
    the nuclear full-LLM fallback.  They're still tracked for transparency.
    """
    total = len(triples)
    num_gap = sum(1 for t in triples if t.classification == GAP)
    denominator = total - num_gap
    num_conflict_all = sum(1 for t in triples if t.classification == CONFLICT)
    # Two-source conflicts are "handleable" — they just need one LLM call
    # to pick the better text, so they shouldn't trip the nuclear fallback.
    num_two_source_conflict = sum(
        1 for t in triples
        if t.classification == CONFLICT and _triple_source_count(t) < 3
    )
    num_conflict = num_conflict_all - num_two_source_conflict
    conflict_ratio = (num_conflict / denominator) if denominator > 0 else 0.0

    textual_denominator = 0
    textual_conflict = 0
    structured_denominator = 0
    structured_conflict = 0
    non_gap_triples = [t for t in triples if t.classification != GAP]
    conflict_positions: list[int] = []

    for idx, triple in enumerate(non_gap_triples):
        is_two_source = _triple_source_count(triple) < 3
        family = _triple_block_family(triple)
        if family == "structured":
            structured_denominator += 1
            if triple.classification == CONFLICT and not is_two_source:
                structured_conflict += 1
                conflict_positions.append(idx)
        elif family == "textual":
            textual_denominator += 1
            if triple.classification == CONFLICT and not is_two_source:
                textual_conflict += 1
                conflict_positions.append(idx)
        elif triple.classification == CONFLICT and not is_two_source:
            conflict_positions.append(idx)

    textual_conflict_ratio = (
        textual_conflict / textual_denominator if textual_denominator > 0 else 0.0
    )
    structured_conflict_ratio = (
        structured_conflict / structured_denominator if structured_denominator > 0 else 0.0
    )

    conflict_span_ratio = 0.0
    conflicts_localized = False
    adaptive_conflict_ratio_threshold = conflict_ratio_threshold
    if conflict_positions and denominator > 0:
        conflict_span = (max(conflict_positions) - min(conflict_positions) + 1)
        # Span over all triples (including GAPs) captures whether conflicts are
        # concentrated in one region of the full document timeline.
        conflict_span_ratio = conflict_span / max(total, 1)
        conflicts_localized = (
            len(conflict_positions) >= 2
            and len(conflict_positions) <= max(1, int(localized_conflict_max_blocks))
            and conflict_span_ratio <= max(0.0, float(localized_conflict_span_max))
        )
        if conflicts_localized:
            adaptive_conflict_ratio_threshold = min(
                0.95, conflict_ratio_threshold + max(0.0, float(localized_conflict_relief)),
            )

    return {
        "conflict_ratio": round(conflict_ratio, 4),
        "conflict_ratio_textual": round(textual_conflict_ratio, 4),
        "conflict_ratio_structured": round(structured_conflict_ratio, 4),
        "non_gap_blocks": denominator,
        "conflict_blocks": num_conflict,
        "conflict_blocks_two_source": num_two_source_conflict,
        "conflict_blocks_total": num_conflict_all,
        "textual_blocks": textual_denominator,
        "structured_blocks": structured_denominator,
        "conflicts_localized": conflicts_localized,
        "conflict_span_ratio": round(conflict_span_ratio, 4),
        "adaptive_conflict_ratio_threshold": round(adaptive_conflict_ratio_threshold, 4),
    }


def dedup_gap_triples(
    triples: list[AlignedTriple],
    window: int = 3,
    similarity_threshold: float = 0.85,
    length_ratio_threshold: float = 0.7,
) -> int:
    """Remove near-duplicate GAP blocks within a local window."""
    removed = 0

    for i, t_i in enumerate(triples):
        if t_i.classification != GAP or not t_i.agreed_text or not t_i.agreed_text.strip():
            continue

        norm_i = normalize_text(t_i.agreed_text)
        if not norm_i:
            continue

        for j in range(i + 1, min(i + window + 1, len(triples))):
            t_j = triples[j]
            if t_j.classification != GAP or not t_j.agreed_text or not t_j.agreed_text.strip():
                continue

            norm_j = normalize_text(t_j.agreed_text)
            if not norm_j:
                continue

            len_ratio = min(len(norm_i), len(norm_j)) / max(len(norm_i), len(norm_j))
            if len_ratio < length_ratio_threshold:
                continue

            similarity = fuzz.token_set_ratio(norm_i, norm_j) / 100.0
            if similarity >= similarity_threshold:
                if len(norm_i) <= len(norm_j):
                    logger.debug(
                        "GAP dedup: dropping %s (dup of %s, sim=%.2f)",
                        t_i.segment_id, t_j.segment_id, similarity,
                    )
                    t_i.agreed_text = ""
                    removed += 1
                    break

                logger.debug(
                    "GAP dedup: dropping %s (dup of %s, sim=%.2f)",
                    t_j.segment_id, t_i.segment_id, similarity,
                )
                t_j.agreed_text = ""
                removed += 1

    return removed


# ---------------------------------------------------------------------------
# Step 5: ASSEMBLE — Build final merged markdown
# ---------------------------------------------------------------------------

def assemble(
    triples: list[AlignedTriple],
    resolved_conflicts: dict[str, str],
) -> str:
    """Assemble final markdown and clean non-LLM blocks for output quality."""
    parts: list[str] = []

    for triple in triples:
        if triple.classification in (AGREE_EXACT, AGREE_NEAR, GAP):
            # Check if this segment was resolved by LLM resolution
            if triple.segment_id in resolved_conflicts:
                resolved_text = resolved_conflicts[triple.segment_id]
                if resolved_text:
                    parts.append(resolved_text)
                elif triple.classification == GAP:
                    pass  # LLM explicitly dropped this gap — do not include
                elif triple.agreed_text:
                    # Non-GAP (e.g. AGREE_NEAR) with empty LLM result — keep agreed_text
                    parts.append(clean_output_md(triple.agreed_text))
            elif triple.agreed_text:
                parts.append(clean_output_md(triple.agreed_text))
        elif triple.classification == CONFLICT:
            resolved = resolved_conflicts.get(triple.segment_id)
            if resolved:
                parts.append(resolved)
            else:
                # Fallback: use first available block text
                blocks = _get_present_blocks(triple)
                if blocks:
                    parts.append(clean_output_md(blocks[0].source_md or blocks[0].raw_text))

    return "\n\n".join(parts)


def ensure_abstract_heading(
    assembled_md: str,
    triples: list[AlignedTriple],
) -> tuple[str, bool]:
    """Post-assembly safety net: inject ``## Abstract`` if parsers had it but output doesn't.

    Only injects when there is evidence from at least one extractor that an
    Abstract heading existed in the source PDF.  Finds the first substantial
    paragraph (>200 chars) after a heading or rule to use as the injection point.

    Returns (possibly_modified_text, was_injected).
    """
    # Already present — nothing to do (allow optional space: "##Abstract" or "## Abstract")
    if re.search(r"^#{1,6}\s*Abstract\s*$", assembled_md, re.MULTILINE | re.IGNORECASE):
        return assembled_md, False

    # Check if any parser produced an Abstract heading
    parser_had_abstract = False
    for triple in triples:
        for block in _get_present_blocks(triple):
            if block.block_type == "heading":
                text = (block.source_md or block.raw_text).strip()
                if re.match(r"^#{1,6}\s*Abstract\s*$", text, re.IGNORECASE):
                    parser_had_abstract = True
                    break
        if parser_had_abstract:
            break

    if not parser_had_abstract:
        return assembled_md, False

    # Find injection point with fallback cascade:
    # 1. First substantial paragraph (>200 chars) after front matter (heading/rule)
    # 2. First non-heading/non-table paragraph (any length) — handles short abstracts
    # 3. Prepend ## Abstract at the start
    paragraphs = assembled_md.split("\n\n")
    past_front_matter = False
    first_body_paragraph_idx: int | None = None
    for i, para in enumerate(paragraphs):
        stripped = para.strip()
        if stripped.startswith("#") or stripped.startswith("---"):
            past_front_matter = True
            continue
        if stripped.startswith("|"):
            continue
        # Track first non-heading/non-table paragraph as fallback
        if first_body_paragraph_idx is None and stripped:
            first_body_paragraph_idx = i
        if past_front_matter and len(stripped) > 200:
            paragraphs.insert(i, "## Abstract")
            logger.info(
                "Post-assembly safety net: injected ## Abstract heading "
                "before paragraph %d",
                i,
            )
            return "\n\n".join(paragraphs), True

    # Fallback: first body paragraph (handles short abstracts / no front matter)
    if first_body_paragraph_idx is not None:
        paragraphs.insert(first_body_paragraph_idx, "## Abstract")
        logger.info(
            "Post-assembly safety net (fallback): injected ## Abstract heading "
            "before paragraph %d",
            first_body_paragraph_idx,
        )
        return "\n\n".join(paragraphs), True

    logger.warning(
        "Post-assembly safety net: parser had Abstract heading but could not "
        "find suitable injection point in assembled output"
    )
    return assembled_md, False


def dedup_assembled_paragraphs(
    text: str,
    similarity_threshold: float = 0.85,
    partial_threshold: float = 0.90,
    length_ratio_cap: float = 0.7,
    min_text_len: int = 50,
) -> tuple[str, int]:
    """Post-assembly global paragraph dedup — safety net for Layer 1.

    Splits assembled text on blank lines, does O(n²) pairwise comparison
    on qualifying paragraphs (>min_text_len chars, not headings), and removes
    the shorter duplicate.  Returns (cleaned_text, removed_count).
    """
    paragraphs = text.split("\n\n")
    removed_indices: set[int] = set()

    # Build list of (index, normalised, raw) for qualifying paragraphs
    qualifying: list[tuple[int, str, str]] = []
    for i, para in enumerate(paragraphs):
        stripped = para.strip()
        if not stripped or len(stripped) < min_text_len:
            continue
        # Skip headings
        if stripped.startswith("#"):
            continue
        # Skip table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        qualifying.append((i, normalize_text(stripped), stripped))

    # O(n²) pairwise
    for a in range(len(qualifying)):
        idx_a, norm_a, raw_a = qualifying[a]
        if idx_a in removed_indices:
            continue
        for b in range(a + 1, len(qualifying)):
            idx_b, norm_b, raw_b = qualifying[b]
            if idx_b in removed_indices:
                continue

            # Near-equal check
            tsr = fuzz.token_set_ratio(norm_a, norm_b) / 100.0
            near_equal_match = False
            if tsr >= similarity_threshold:
                # Numeric/citation guardrail: for containment (subset),
                # allow if shorter's values are a subset of longer's.
                nums_a = _extract_numeric_tokens(raw_a)
                nums_b = _extract_numeric_tokens(raw_b)
                cites_a = _extract_citation_keys(raw_a)
                cites_b = _extract_citation_keys(raw_b)
                shorter_nums = nums_a if len(norm_a) <= len(norm_b) else nums_b
                longer_nums = nums_b if len(norm_a) <= len(norm_b) else nums_a
                shorter_cites = cites_a if len(norm_a) <= len(norm_b) else cites_b
                longer_cites = cites_b if len(norm_a) <= len(norm_b) else cites_a
                if shorter_nums.issubset(longer_nums) and shorter_cites.issubset(longer_cites):
                    near_equal_match = True
                elif nums_a == nums_b and cites_a == cites_b:
                    near_equal_match = True

            if near_equal_match:
                # Drop the shorter paragraph
                victim = idx_a if len(norm_a) <= len(norm_b) else idx_b
                logger.info(
                    "Post-assembly dedup: dropping paragraph %d (near-equal to %d, tsr=%.2f)",
                    victim, idx_a if victim == idx_b else idx_b, tsr,
                )
                removed_indices.add(victim)
                if victim == idx_a:
                    break
                continue

            # Containment check
            shorter_norm, longer_norm = (norm_a, norm_b) if len(norm_a) <= len(norm_b) else (norm_b, norm_a)
            shorter_idx = idx_a if len(norm_a) <= len(norm_b) else idx_b
            if len(shorter_norm) < len(longer_norm) * length_ratio_cap:
                pr = fuzz.partial_ratio(shorter_norm, longer_norm) / 100.0
                if pr >= partial_threshold:
                    logger.info(
                        "Post-assembly dedup: dropping paragraph %d (contained in %d, pr=%.2f)",
                        shorter_idx,
                        idx_b if shorter_idx == idx_a else idx_a,
                        pr,
                    )
                    removed_indices.add(shorter_idx)
                    if shorter_idx == idx_a:
                        break

    cleaned = "\n\n".join(
        para for i, para in enumerate(paragraphs) if i not in removed_indices
    )
    return cleaned, len(removed_indices)


