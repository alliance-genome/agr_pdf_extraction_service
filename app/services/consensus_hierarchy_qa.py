"""Heading hierarchy post-processing, QA gates, and flanking context builders."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from app.services.consensus_classification_assembly import _extract_citation_keys, _extract_numeric_tokens
from app.services.consensus_models import CONFLICT, AlignedTriple
from app.services.consensus_parsing_alignment import (
    _FLANK_CHAR_BUDGET,
    _HTML_COMMENT_OUTPUT_RE,
    _SPAN_OPEN_RE,
    clean_output_md,
    normalize_text,
)

if TYPE_CHECKING:
    from app.services.llm_service import LLM

logger = logging.getLogger(__name__)
_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

def extract_heading_hierarchy(merged_md: str) -> list[dict]:
    """Extract the finalized heading hierarchy from merged markdown.

    Returns a document-order list of headings, each with:
    - text: the heading text (without leading '#' markers)
    - level: heading level (1-6)
    """
    matches = list(_HEADING_LINE_RE.finditer(merged_md or ""))
    headings: list[dict] = []
    for m in matches:
        headings.append({
            "text": m.group(2).strip(),
            "level": len(m.group(1)),
        })
    return headings


def resolve_header_hierarchy(merged_md: str, llm: "LLM") -> str:
    """Resolve flat heading levels into a proper hierarchy via LLM.

    Extracts all heading lines, asks the LLM to classify each one, validates
    the response, and rewrites heading levels in-place.  On any validation
    failure the original markdown is returned unchanged.
    """
    from config import Config

    # 1. Extract headings with their line positions
    matches = list(_HEADING_LINE_RE.finditer(merged_md))
    if not matches:
        logger.info("Header hierarchy: no headings found, skipping")
        return merged_md

    # 2. Build payload for LLM
    headers: list[dict] = []
    for idx, m in enumerate(matches):
        text = m.group(2).strip()
        # Content preview: grab ~100 chars of text following this heading
        line_end = m.end()
        preview = merged_md[line_end:line_end + 200].strip()
        # Trim to first ~100 meaningful chars (skip blank lines)
        preview_clean = " ".join(preview.split())[:100]
        headers.append({
            "index": idx,
            "text": text,
            "content_preview": preview_clean,
        })

    # 2b. Extract opening text (~1000 chars) for title detection
    opening_text = merged_md[:1000]

    logger.info(
        "Header hierarchy: extracted %d headings, calling LLM (model=%s, reasoning=%s)",
        len(headers), Config.HIERARCHY_LLM_MODEL, Config.HIERARCHY_LLM_REASONING,
    )

    # 3. Call LLM
    response = llm.resolve_header_hierarchy(
        headers,
        model=Config.HIERARCHY_LLM_MODEL,
        reasoning_effort=Config.HIERARCHY_LLM_REASONING,
        opening_text=opening_text,
    )

    # 4. Validate (includes text-fidelity check against what we sent)
    decisions = response.decisions
    detected_title = (response.detected_title or "").strip() or None
    original_levels = [len(m.group(1)) for m in matches]
    validation_error = _validate_hierarchy_decisions(
        decisions, len(headers), headers, original_levels,
        detected_title=detected_title,
    )
    if validation_error:
        logger.warning("Header hierarchy validation failed: %s — returning original markdown", validation_error)
        return merged_md

    # 5. Apply — build replacement map (match start → new line text)
    # Process in reverse order so character offsets stay valid
    result = merged_md
    for decision in sorted(decisions, key=lambda d: d.heading_index, reverse=True):
        match = matches[decision.heading_index]
        original_level = len(match.group(1))
        heading_text = match.group(2)

        if decision.action == "keep_level":
            logger.info(
                "  [%d] KEEP H%d: %s",
                decision.heading_index, original_level, heading_text[:80],
            )
            continue
        elif decision.action == "set_level":
            new_prefix = "#" * decision.new_level
            new_line = f"{new_prefix} {heading_text}"
            logger.info(
                "  [%d] SET H%d → H%d: %s",
                decision.heading_index, original_level, decision.new_level, heading_text[:80],
            )
            result = result[:match.start()] + new_line + result[match.end():]
        elif decision.action == "demote_to_text":
            logger.info(
                "  [%d] DEMOTE H%d → text: %s",
                decision.heading_index, original_level, heading_text[:80],
            )
            result = result[:match.start()] + heading_text + result[match.end():]

    # 6. Insert detected title as H1 if the LLM found one in the opening text
    if detected_title:
        title_stripped = detected_title.strip()
        # Try to find the title as a standalone line first
        lines = result.split("\n")
        inserted = False
        for i, line in enumerate(lines):
            if line.strip() == title_stripped:
                lines[i] = f"# {title_stripped}"
                logger.info("  TITLE MARKED as H1 (existing line): %s", title_stripped[:80])
                inserted = True
                break
        if inserted:
            result = "\n".join(lines)
        else:
            # Title exists only embedded in other text (e.g. citation line) —
            # prepend it as a new H1 at the top of the document.
            result = f"# {title_stripped}\n\n{result.lstrip()}"
            logger.info("  TITLE PREPENDED as H1: %s", title_stripped[:80])

    # Log summary
    actions = {"keep_level": 0, "set_level": 0, "demote_to_text": 0}
    for d in decisions:
        actions[d.action] = actions.get(d.action, 0) + 1
    logger.info(
        "Header hierarchy complete: %d headings (set_level=%d, demote=%d, keep=%d, title_detected=%s)",
        len(decisions), actions["set_level"], actions["demote_to_text"], actions["keep_level"],
        bool(detected_title),
    )

    return result


def _validate_hierarchy_decisions(
    decisions: list,
    expected_count: int,
    original_headers: list[dict] | None = None,
    original_levels: list[int] | None = None,
    detected_title: str | None = None,
) -> str | None:
    """Validate LLM hierarchy decisions. Returns error string or None if valid.

    Args:
        decisions: List of HeaderDecision from the LLM.
        expected_count: Number of headings we sent.
        original_headers: The header dicts we sent to the LLM (with 'text' keys).
            Used to verify the LLM didn't fabricate or modify heading text.
        original_levels: Original heading levels (1-6) from the markdown, indexed
            by heading position.  Used to resolve keep_level into an effective
            level for the H1 and jump checks.
        detected_title: If set, the LLM found the title in the opening text
            (not among headings), so 0 H1 headings is acceptable.
    """
    # Count check
    if len(decisions) != expected_count:
        return f"expected {expected_count} decisions, got {len(decisions)}"

    # Index uniqueness and completeness: must be exactly {0..n-1}
    indices = [d.heading_index for d in decisions]
    if sorted(indices) != list(range(expected_count)):
        seen = set()
        dupes = [i for i in indices if i in seen or seen.add(i)]
        if dupes:
            return f"duplicate heading_index values: {dupes}"
        return f"heading_index values {sorted(indices)} do not cover 0..{expected_count - 1}"

    # CRITICAL: Text fidelity check — the LLM must not modify heading text
    if original_headers:
        for d in decisions:
            if d.heading_index < 0 or d.heading_index >= len(original_headers):
                return f"heading_index {d.heading_index} out of range (0-{len(original_headers) - 1})"
            expected_text = original_headers[d.heading_index]["text"]
            if d.original_text.strip() != expected_text.strip():
                return (
                    f"LLM modified heading text at index {d.heading_index}: "
                    f"expected {expected_text!r}, got {d.original_text!r}"
                )

    # Validate new_level values for set_level actions
    for d in decisions:
        if d.action == "set_level":
            if d.new_level is None:
                return f"heading_index {d.heading_index}: set_level requires new_level but got None"
            if not (1 <= d.new_level <= 6):
                return f"heading_index {d.heading_index}: new_level {d.new_level} out of range (1-6)"

    # Build effective levels (resolving keep_level using original_levels)
    def _effective_level(d) -> int | None:
        if d.action == "set_level":
            return d.new_level
        if d.action == "keep_level" and original_levels:
            return original_levels[d.heading_index]
        return None  # unknown

    # H1 count: exactly 1 among headings, OR 0 if detected_title is set
    h1_count = sum(
        1 for d in decisions
        if _effective_level(d) == 1 and d.action != "demote_to_text"
    )
    if detected_title:
        # Title is in plain text — no heading should be H1
        if h1_count != 0:
            return f"detected_title is set but {h1_count} headings are also H1 (expected 0)"
    else:
        if h1_count != 1:
            return f"expected exactly 1 H1 (title), got {h1_count}"

    # Level jump check: consecutive headings should not skip levels (e.g. 1→3)
    sorted_decisions = sorted(decisions, key=lambda d: d.heading_index)
    prev_level = None
    for d in sorted_decisions:
        if d.action == "demote_to_text":
            continue
        eff = _effective_level(d)
        if eff is None:
            # Unknown level (keep_level without original_levels) — reset tracking
            prev_level = None
            continue
        if prev_level is not None and eff > prev_level + 1:
            return f"level jump from {prev_level} to {eff}"
        prev_level = eff

    # Too many demotions: more than half the headings demoted is suspicious
    demote_count = sum(1 for d in decisions if d.action == "demote_to_text")
    if demote_count > len(decisions) // 2:
        return f"too many demotions: {demote_count} of {len(decisions)} headings demoted"

    return None


def run_qa_gates(
    merged_md: str,
    similarity_threshold: float = 0.85,
    partial_threshold: float = 0.90,
    length_ratio_cap: float = 0.7,
    min_text_len: int = 50,
) -> dict:
    """Run post-assembly quality checks including **global** duplicate detection.

    The duplicate scan is now O(n²) over all paragraph pairs (not just
    adjacent) and includes containment checks, matching the thresholds used
    by the dedup layers.
    """
    results: dict[str, int | bool] = {}

    span_count = len(_SPAN_OPEN_RE.findall(merged_md))
    results["span_tag_count"] = span_count
    if span_count > 0:
        logger.warning("QA gate: found %d <span> tags in assembled output", span_count)

    comment_count = len(_HTML_COMMENT_OUTPUT_RE.findall(merged_md))
    results["html_comment_count"] = comment_count
    if comment_count > 0:
        logger.warning("QA gate: found %d HTML comments in assembled output", comment_count)

    # --- Global duplicate detection (replaces adjacent-only check) ----------
    paragraphs = [p.strip() for p in merged_md.split("\n\n") if p.strip()]

    # Build qualifying list (skip headings, tables, short paragraphs)
    qualifying: list[tuple[int, str, str]] = []  # (index, normalised, raw)
    for i, para in enumerate(paragraphs):
        if len(para) < min_text_len:
            continue
        if para.startswith("#"):
            continue
        if para.startswith("|") and para.endswith("|"):
            continue
        qualifying.append((i, normalize_text(para), para))

    global_dup_count = 0
    for a in range(len(qualifying)):
        idx_a, norm_a, raw_a = qualifying[a]
        for b in range(a + 1, len(qualifying)):
            idx_b, norm_b, raw_b = qualifying[b]

            # Near-equal
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
                logger.warning(
                    "QA gate: paragraphs %d and %d are %.0f%% similar (global duplicate)",
                    idx_a, idx_b, tsr * 100,
                )
                global_dup_count += 1
                continue

            # Containment
            shorter, longer = (norm_a, norm_b) if len(norm_a) <= len(norm_b) else (norm_b, norm_a)
            if len(shorter) < len(longer) * length_ratio_cap:
                pr = fuzz.partial_ratio(shorter, longer) / 100.0
                if pr >= partial_threshold:
                    shorter_idx = idx_a if len(norm_a) <= len(norm_b) else idx_b
                    longer_idx = idx_b if shorter_idx == idx_a else idx_a
                    logger.warning(
                        "QA gate: paragraph %d contained in %d (partial=%.0f%%, global duplicate)",
                        shorter_idx, longer_idx, pr * 100,
                    )
                    global_dup_count += 1

    results["global_duplicate_count"] = global_dup_count

    results["qa_passed"] = (span_count == 0 and comment_count == 0 and global_dup_count == 0)
    if results["qa_passed"]:
        logger.info("QA gates: all checks passed")
    else:
        logger.warning(
            "QA gates: %d issues detected (spans=%d, comments=%d, global_dupes=%d)",
            span_count + comment_count + global_dup_count,
            span_count, comment_count, global_dup_count,
        )

    return results


def _find_flanking_text(
    triples: list[AlignedTriple],
    conflict_index: int,
    direction: str,
) -> str:
    """Return nearest non-empty, non-CONFLICT agreed_text in the given direction."""
    if direction == "before":
        indices = range(conflict_index - 1, -1, -1)
    else:
        indices = range(conflict_index + 1, len(triples))

    for i in indices:
        triple = triples[i]
        if triple.classification != CONFLICT and triple.agreed_text and triple.agreed_text.strip():
            return triple.agreed_text
    return ""


def _truncate_to_token_budget(text: str, budget_chars: int = _FLANK_CHAR_BUDGET) -> str:
    """Truncate to a conservative character budget approximating token limits."""
    if len(text) <= budget_chars:
        return text
    return text[:budget_chars].rstrip() + "..."


def _build_flanking_context(
    triples: list[AlignedTriple],
    conflict_index: int,
) -> tuple[str, str]:
    """Build cleaned context_before/context_after around a conflict triple."""
    raw_before = _find_flanking_text(triples, conflict_index, "before")
    if raw_before:
        cleaned_before = clean_output_md(raw_before)
        if len(cleaned_before) > _FLANK_CHAR_BUDGET:
            cleaned_before = "..." + cleaned_before[-_FLANK_CHAR_BUDGET:].lstrip()
    else:
        cleaned_before = ""

    raw_after = _find_flanking_text(triples, conflict_index, "after")
    if raw_after:
        cleaned_after = _truncate_to_token_budget(clean_output_md(raw_after))
    else:
        cleaned_after = ""

    return cleaned_before, cleaned_after
