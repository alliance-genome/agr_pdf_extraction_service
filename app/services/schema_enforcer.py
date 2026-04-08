"""Post-merge schema enforcement for the PDFX consensus pipeline.

Normalizes merged markdown output to conform to the AGR ABC Markdown
Schema by running it through the ``agr_abc_document_parsers`` round-trip:
pre-process → ``read_markdown`` → Document model fixups → ``emit_markdown``
→ ``validate_markdown``.
"""

from __future__ import annotations

import logging
import re

from agr_abc_document_parsers import (
    emit_markdown,
    read_markdown,
    validate_markdown,
)
from agr_abc_document_parsers.models import Document, Section

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-processing patterns
# ---------------------------------------------------------------------------

# Section numbers at the start of headings: "## 2.1. Methods" or "## 2.1 Methods"
_SECTION_NUMBER_RE = re.compile(
    r"^(#{1,6})\s+\d+(?:\.\d+)*\.?\s+(.+)$",
    re.MULTILINE,
)

# Unicode escape literals that some extractors produce
_UNICODE_ESCAPE_RE = re.compile(r"/uni[0-9A-Fa-f]{4}")

# Common watermark / download notice patterns
_WATERMARK_PATTERNS = [
    re.compile(r"^Downloaded from .+$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Authorized licensed use limited to.+$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^This article is licensed under.+$", re.MULTILINE | re.IGNORECASE),
]

# DOI / PMID patterns for metadata relocation
_DOI_METADATA_RE = re.compile(
    r"^\(?\s*(?:doi\s*[: ]\s*|https?://(?:dx\.)?doi\.org/)?"
    r"(10\.\d{4,}/\S+?)\s*\)?[.,;:]*\s*$",
    re.IGNORECASE,
)
_PMID_METADATA_RE = re.compile(
    r"^\(?\s*PMID\s*[: ]\s*(\d+)\s*\)?[.,;:]*\s*$",
    re.IGNORECASE,
)

# Lines that should never be merged across paragraph boundaries
_NO_MERGE_LINE = re.compile(
    r"^(?:#{1,6}\s|[|>*\-]|\d+\.\s|\[\^|\*\*Figure|\*\*Table|<!-- )"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enforce_schema(merged_md: str) -> tuple[str, dict]:
    """Enforce ABC markdown schema on consensus-merged output.

    Returns ``(normalized_md, enforcement_metrics)``.

    If the round-trip fails or produces a degenerate document, returns the
    original *merged_md* unchanged with metrics indicating the skip reason.
    """
    if not merged_md or not merged_md.strip():
        return merged_md, {
            "enforcement_applied": False,
            "skip_reason": "empty_input",
        }

    try:
        return _enforce(merged_md)
    except Exception as e:
        logger.warning("Schema enforcement error: %s — returning original output", e)
        return merged_md, {
            "enforcement_applied": False,
            "skip_reason": "error",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _enforce(merged_md: str) -> tuple[str, dict]:
    """Core enforcement logic (no exception handling — caller wraps)."""

    # Phase 1: Pre-process
    preprocessed = _preprocess(merged_md)

    # Phase 2: Round-trip through Document model
    doc = read_markdown(preprocessed)

    # Degenerate check: parser couldn't meaningfully parse the input
    if not doc.sections and not doc.abstract:
        logger.warning("Schema enforcement: document has no sections and no abstract — skipping")
        return merged_md, {
            "enforcement_applied": False,
            "skip_reason": "unparseable",
        }

    # PDFX-specific fixups on the Document model
    adjusted_original_words = max(
        1,
        len(preprocessed.split()) - _estimate_intentional_word_loss(doc),
    )
    _normalize_authors(doc)
    _relocate_metadata_from_body(doc)

    normalized_md = emit_markdown(doc)

    # Content-preservation guard: reject if too much content was lost
    normalized_words = len(normalized_md.split())
    if normalized_words / adjusted_original_words < 0.90:
        logger.warning(
            "Schema enforcement: word count dropped from %d to %d (%.0f%%) — "
            "falling back to original output",
            adjusted_original_words,
            normalized_words,
            (normalized_words / adjusted_original_words) * 100,
        )
        return merged_md, {
            "enforcement_applied": False,
            "skip_reason": "content_loss",
            "original_words": adjusted_original_words,
            "normalized_words": normalized_words,
        }

    # Phase 3: Validate
    validation = validate_markdown(normalized_md)
    validation_metrics = _build_validation_metrics(validation)
    if validation.errors:
        logger.warning(
            "Schema enforcement: validation failed with %d error(s) and %d warning(s) "
            "— falling back to original output",
            validation_metrics["validation_errors"],
            validation_metrics["validation_warnings"],
        )
        return merged_md, {
            "enforcement_applied": False,
            "skip_reason": "validation_failed",
            **validation_metrics,
        }

    return normalized_md, {
        "enforcement_applied": True,
        **validation_metrics,
    }


# ---------------------------------------------------------------------------
# Phase 1: Pre-processing
# ---------------------------------------------------------------------------


def _preprocess(text: str) -> str:
    """Clean PDFX-specific content issues before the round-trip."""
    text = _strip_section_numbers(text)
    text = _strip_pdf_artifacts(text)
    text = _fix_mid_sentence_splits(text)
    return text


def _strip_section_numbers(text: str) -> str:
    """Remove leading section numbers from headings.

    ``## 2.1. Methods`` → ``## Methods``
    ``### 3.2.1 Analysis`` → ``### Analysis``
    """
    return _SECTION_NUMBER_RE.sub(r"\1 \2", text)


def _strip_pdf_artifacts(text: str) -> str:
    """Remove PDF-specific artifacts: unicode escapes, watermarks."""
    # Unicode escape literals (e.g., /uniFB01 already handled upstream,
    # but catch remaining /uniXXXX patterns)
    text = _UNICODE_ESCAPE_RE.sub("", text)

    # Watermark / download notice lines
    for pattern in _WATERMARK_PATTERNS:
        text = pattern.sub("", text)

    return text


def _fix_mid_sentence_splits(text: str) -> str:
    """Merge paragraphs that were split at page or column boundaries.

    Detects cases where a paragraph ends without sentence-ending punctuation
    and the next paragraph starts with a lowercase letter — a strong signal
    of a mid-sentence split from PDF column/page layout.
    """
    paragraphs = text.split("\n\n")
    merged: list[str] = []

    i = 0
    while i < len(paragraphs):
        current = paragraphs[i]
        # Try to merge with the next paragraph if conditions are met
        while (
            i + 1 < len(paragraphs)
            and _should_merge(current, paragraphs[i + 1])
        ):
            i += 1
            current = current.rstrip() + " " + paragraphs[i].lstrip()
        merged.append(current)
        i += 1

    return "\n\n".join(merged)


def _should_merge(preceding: str, following: str) -> bool:
    """Determine if two adjacent paragraphs should be merged."""
    preceding = preceding.strip()
    following = following.strip()

    if not preceding or not following:
        return False

    # Don't merge if either is a structural element
    if _NO_MERGE_LINE.match(preceding) or _NO_MERGE_LINE.match(following):
        return False

    # Preceding must be long enough to be a real paragraph fragment
    if len(preceding) < 40:
        return False

    # Preceding must NOT end with sentence-ending punctuation
    if preceding[-1] in ".!?:":
        return False

    # Following must start with a lowercase letter
    if not following[0].islower():
        return False

    return True


# ---------------------------------------------------------------------------
# Phase 2: Document model fixups
# ---------------------------------------------------------------------------


def _normalize_authors(doc: Document) -> None:
    """Emit plain comma-separated author names without affiliation superscripts."""
    for author in doc.authors:
        # PDFX author lines commonly encode affiliation markers as superscripts or
        # numbered affiliation lists. The ABC markdown schema emits plain
        # comma-separated names, so we intentionally drop affiliations here.
        author.affiliations = []


def _relocate_metadata_from_body(doc: Document) -> None:
    """Promote metadata-like body paragraphs to document fields and remove them."""
    for section in doc.sections:
        _relocate_metadata_from_section(section, doc)


def _relocate_metadata_from_section(section: Section, doc: Document) -> None:
    """Walk a section tree, relocating metadata paragraphs in place."""
    kept_paragraphs = []
    for para in section.paragraphs:
        text = para.text.strip()
        doi = _extract_metadata_value(text, _DOI_METADATA_RE)
        pmid = _extract_metadata_value(text, _PMID_METADATA_RE)

        if doi or pmid:
            if doi and not doc.doi:
                doc.doi = doi
            if pmid and not doc.pmid:
                doc.pmid = pmid
            continue

        kept_paragraphs.append(para)

    section.paragraphs = kept_paragraphs
    for subsection in section.subsections:
        _relocate_metadata_from_section(subsection, doc)


def _extract_metadata_value(text: str, pattern: re.Pattern[str]) -> str:
    """Return a metadata value only when the full paragraph is metadata-like."""
    stripped_text = text.strip()
    if stripped_text.startswith("(") and stripped_text.endswith(")"):
        stripped_text = stripped_text[1:-1].strip()

    match = pattern.match(stripped_text)
    if not match:
        return ""
    return match.group(1)


def _estimate_intentional_word_loss(doc: Document) -> int:
    """Estimate words intentionally removed by author and metadata normalization."""
    return _count_affiliation_line_words(doc) + _count_metadata_paragraph_words(doc.sections)


def _count_affiliation_line_words(doc: Document) -> int:
    """Count numbered affiliation-line words that the plain-author output removes."""
    seen_affiliations: set[str] = set()
    total = 0
    for author in doc.authors:
        for affiliation in author.affiliations:
            affiliation = affiliation.strip()
            if not affiliation or affiliation.isdigit() or affiliation in seen_affiliations:
                continue
            seen_affiliations.add(affiliation)
            total += len(affiliation.split()) + 1
    return total


def _count_metadata_paragraph_words(sections: list[Section]) -> int:
    """Count exact metadata paragraphs that will be relocated out of the body."""
    total = 0
    for section in sections:
        for para in section.paragraphs:
            text = para.text.strip()
            if _extract_metadata_value(text, _DOI_METADATA_RE) or _extract_metadata_value(
                text, _PMID_METADATA_RE,
            ):
                total += len(text.split())
        total += _count_metadata_paragraph_words(section.subsections)
    return total


def _build_validation_metrics(validation) -> dict:
    """Return serializable validation counts and issues."""
    return {
        "validation_valid": validation.valid,
        "validation_errors": len(validation.errors),
        "validation_warnings": len(validation.warnings),
        "validation_issues": [
            {
                "rule_id": issue.rule_id,
                "severity": issue.severity.value,
                "line": issue.line,
                "message": issue.message,
            }
            for issue in (validation.errors + validation.warnings)
        ],
    }
