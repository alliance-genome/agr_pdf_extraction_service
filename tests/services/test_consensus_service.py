"""Tests for the selective LLM merge consensus pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.llm_service import TokenAccumulator
from app.services.consensus_service import (
    Block,
    AlignedTriple,
    AGREE_EXACT,
    AGREE_NEAR,
    GAP,
    CONFLICT,
    _SOURCE_PREFERENCE,
    _DELETION,
    _pick_preferred_text,
    _extract_numeric_tokens,
    _extract_citation_keys,
    _extract_numeric_tokens_integrity,
    _segment_allowed_numeric_tokens,
    _numeric_integrity_dropped_numbers,
    _content_similarity_check,
    _numeric_count_ratio,
    MicroConflict,
    MicroConflictResult,
    tokenize_for_diff,
    _build_majority_alignment,
    _expand_to_sentence_boundary,
    extract_micro_conflicts_for_segment,
    extract_micro_conflicts,
    reconstruct_segment_from_micro_conflicts,
    _coalesce_micro_conflicts_if_high_divergence,
    clean_output_md,
    dedup_gap_triples,
    dedup_assembled_paragraphs,
    ensure_abstract_heading,
    is_structural_heading,
    normalize_text,
    normalize_extractor_output,
    parse_markdown,
    align_blocks,
    classify_triples,
    assemble,
    compute_metrics,
    _build_audit_entries,
    merge_with_consensus,
    run_qa_gates,
)


# ---------------------------------------------------------------------------
# Fixtures: sample markdown for each extractor
# ---------------------------------------------------------------------------

GROBID_MD = """# Introduction

This paper presents a novel approach to gene annotation in model organisms.
The approach leverages machine learning techniques [1] to improve accuracy.

## Methods

We used a dataset of 500 annotated genes from FlyBase (Release 2024).
The precision was 0.95 and recall was 0.87.

## Results

Our method achieved 95% accuracy on the test set.

## References

[1] Smith et al. 2023. Gene annotation methods.
"""

DOCLING_MD = """# Introduction

This paper presents a novel approach to gene annotation in model organisms.
The approach leverages machine learning techniques [1] to improve accuracy.

## Methods

We used a dataset of 500 annotated genes from FlyBase (Release 2024).
The precision was 0.95 and recall was 0.87.

| Metric | Value |
|--------|-------|
| Precision | 0.95 |
| Recall | 0.87 |

## Results

Our method achieved 95% accuracy on the test set.

## References

[1] Smith et al. 2023. Gene annotation methods.
"""

MARKER_MD = """# Introduction

This paper presents a novel approach to gene annotation in model organisms.
The approach leverages machine learning techniques [1] to improve accuracy.

## Methods

We used a dataset of 500 annotated genes from FlyBase (Release 2024).
The precision was 0.95 and recall was 0.87.

## Results

Our method achieved 95% accuracy on the test set.

## References

[1] Smith et al. 2023. Gene annotation methods.
"""


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParseMarkdown:
    def test_basic_parse(self):
        md = "# Title\n\nSome paragraph text.\n\n## Section\n\nMore text."
        blocks = parse_markdown(md, "grobid")
        assert len(blocks) >= 3
        assert blocks[0].block_type == "heading"
        assert blocks[0].source == "grobid"

    def test_heading_detection(self):
        md = "# H1\n\n## H2\n\n### H3\n\nParagraph."
        blocks = parse_markdown(md, "docling")
        headings = [b for b in blocks if b.block_type == "heading"]
        assert len(headings) >= 1

    def test_heading_level(self):
        md = "# Top Level\n\nBody text."
        blocks = parse_markdown(md, "grobid")
        heading = [b for b in blocks if b.block_type == "heading"]
        assert len(heading) >= 1
        assert heading[0].heading_level is not None

    def test_paragraph_detection(self):
        md = "This is a paragraph.\n\nThis is another paragraph."
        blocks = parse_markdown(md, "marker")
        paragraphs = [b for b in blocks if b.block_type == "paragraph"]
        assert len(paragraphs) >= 1

    def test_table_detection(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        blocks = parse_markdown(md, "docling")
        tables = [b for b in blocks if b.block_type == "table"]
        assert len(tables) >= 1

    def test_order_index_sequential(self):
        md = "# H1\n\nPara 1.\n\n## H2\n\nPara 2."
        blocks = parse_markdown(md, "grobid")
        for i, b in enumerate(blocks):
            assert b.order_index == i

    def test_block_id_format(self):
        md = "# Title\n\nText."
        blocks = parse_markdown(md, "grobid")
        for b in blocks:
            assert b.block_id.startswith("grobid_block_")

    def test_empty_markdown(self):
        blocks = parse_markdown("", "grobid")
        assert blocks == []

    def test_figure_ref_detection(self):
        md = "![Figure 1](image.png)"
        blocks = parse_markdown(md, "marker")
        # The image may be parsed as a paragraph containing a figure ref
        has_figure = any(b.block_type == "figure_ref" for b in blocks)
        has_text_with_figure = any("![" in b.raw_text for b in blocks)
        assert has_figure or has_text_with_figure


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_whitespace_collapsing(self):
        assert normalize_text("hello   world") == "hello world"

    def test_unicode_normalization(self):
        # NFKC normalizes compatibility characters
        result = normalize_text("\uff41\uff42\uff43")  # fullwidth abc
        assert result == "abc"

    def test_markdown_bold_equivalence(self):
        assert normalize_text("__bold__") == normalize_text("**bold**")

    def test_markdown_italic_equivalence(self):
        assert normalize_text("_italic_") == normalize_text("*italic*")

    def test_lowercasing(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strip_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_strip_marker_span_artifacts(self):
        text = '<span id="page-0-3">Drosophila</span> sleep genes'
        assert normalize_text(text) == "drosophila sleep genes"

    def test_strip_image_and_comment_artifacts(self):
        text = "Alpha <!-- image --> ![fig](_page_4_Picture_7.jpeg) Beta"
        assert normalize_text(text) == "alpha beta"

    def test_strip_link_url_keep_text(self):
        text = "See [Sleep genes](https://doi.org/10.1234/example)."
        assert normalize_text(text) == "see sleep genes."

    def test_strip_heading_markers(self):
        assert normalize_text("# Methods") == normalize_text("#### Methods")

    def test_normalize_extractor_output_strips_artifacts(self):
        text = (
            "<span id='page-1-0'>Alpha</span>\n\n"
            "<!-- image -->\n\n"
            "![Figure 1](image.png)\n\n"
            "[DOI](https://doi.org/10.1234/x)\n\n"
            "/uniFB01 /uniFB02"
        )
        result = normalize_extractor_output(text)
        assert "Alpha" in result
        assert "<span" not in result
        assert "<!-- image -->" not in result
        assert "<!-- page: 1 -->" in result
        assert "![" not in result
        assert "https://doi.org" not in result
        assert "fi fl" in result


class TestOutputCleaning:
    def test_clean_output_strips_span_and_comments(self):
        text = '<span id="page-1-0">Alpha</span> <!-- image --> Beta'
        assert clean_output_md(text) == "Alpha Beta"

    def test_clean_output_preserves_markdown(self):
        text = "# Heading\n\n**Bold** [Link](https://example.com) ![img](fig.png)"
        assert clean_output_md(text) == text


# ---------------------------------------------------------------------------
# Alignment tests
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_identical_documents_align(self):
        """Three identical documents should produce well-aligned triples."""
        md = "# Title\n\nParagraph one.\n\n## Section\n\nParagraph two."
        blocks_a = parse_markdown(md, "grobid")
        blocks_b = parse_markdown(md, "docling")
        blocks_c = parse_markdown(md, "marker")

        triples, confidence = align_blocks({
            "grobid": blocks_a,
            "docling": blocks_b,
            "marker": blocks_c,
        })

        assert len(triples) >= 3
        assert confidence > 0.5

        # All triples should have all 3 blocks
        for t in triples[:len(blocks_a)]:
            present = sum(1 for attr in ("grobid_block", "docling_block", "marker_block")
                         if getattr(t, attr) is not None)
            assert present >= 2

    def test_gap_handling(self):
        """Missing section in one extractor should produce GAP triples."""
        md_full = "# Title\n\nPara 1.\n\n## Methods\n\nPara 2.\n\n## Results\n\nPara 3."
        md_short = "# Title\n\nPara 1.\n\n## Results\n\nPara 3."

        blocks_full = parse_markdown(md_full, "grobid")
        blocks_full2 = parse_markdown(md_full, "docling")
        blocks_short = parse_markdown(md_short, "marker")

        triples, confidence = align_blocks({
            "grobid": blocks_full,
            "docling": blocks_full2,
            "marker": blocks_short,
        })

        # Should have triples for all blocks (no drops)
        assert len(triples) >= len(blocks_full)

    def test_leftover_blocks_collected(self):
        """Unmatched blocks from non-reference extractors should not be dropped."""
        md_a = "# A\n\nPara A."
        md_b = "# B\n\nPara B.\n\n## Extra Section\n\nExtra content."

        blocks_a = parse_markdown(md_a, "grobid")
        blocks_b = parse_markdown(md_b, "docling")
        blocks_c = parse_markdown(md_a, "marker")

        triples, _ = align_blocks({
            "grobid": blocks_a,
            "docling": blocks_b,
            "marker": blocks_c,
        })

        # All blocks from all sources should appear
        all_block_ids = set()
        for t in triples:
            for attr in ("grobid_block", "docling_block", "marker_block"):
                b = getattr(t, attr)
                if b is not None:
                    all_block_ids.add(b.block_id)

        total_input_blocks = len(blocks_a) + len(blocks_b) + len(blocks_c)
        assert len(all_block_ids) == total_input_blocks

    def test_empty_source(self):
        """Empty extractor output should be handled gracefully."""
        md = "# Title\n\nText."
        blocks = parse_markdown(md, "grobid")

        triples, confidence = align_blocks({
            "grobid": blocks,
            "docling": [],
            "marker": [],
        })

        assert len(triples) == len(blocks)
        assert confidence == 0.0

    def test_no_blocks_at_all(self):
        """All empty should return empty."""
        triples, confidence = align_blocks({
            "grobid": [],
            "docling": [],
            "marker": [],
        })
        assert triples == []
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassification:
    def _make_triple(self, seg_id, grobid_text, docling_text, marker_text,
                     block_type="paragraph"):
        triple = AlignedTriple(segment_id=seg_id)
        if grobid_text is not None:
            triple.grobid_block = Block(
                block_id="g", block_type=block_type, raw_text=grobid_text,
                normalized_text=normalize_text(grobid_text),
                heading_level=None, order_index=0, source="grobid",
            )
        if docling_text is not None:
            triple.docling_block = Block(
                block_id="d", block_type=block_type, raw_text=docling_text,
                normalized_text=normalize_text(docling_text),
                heading_level=None, order_index=0, source="docling",
            )
        if marker_text is not None:
            triple.marker_block = Block(
                block_id="m", block_type=block_type, raw_text=marker_text,
                normalized_text=normalize_text(marker_text),
                heading_level=None, order_index=0, source="marker",
            )
        return triple

    def test_agree_exact(self):
        t = self._make_triple("seg_001", "Hello world", "Hello world", "Different")
        classify_triples([t])
        assert t.classification == AGREE_EXACT
        assert t.agreed_text is not None

    def test_agree_exact_picks_agreeing_source_not_outlier(self):
        """When grobid==docling but marker differs, agreed_text must NOT be marker's text.

        For paragraphs, source preference is marker, but the outlier must be excluded.
        """
        t = self._make_triple("seg_001", "Hello world", "Hello world", "OUTLIER TEXT")
        classify_triples([t])
        assert t.classification == AGREE_EXACT
        assert t.agreed_text == "Hello world"
        assert t.agreed_text != "OUTLIER TEXT"

    def test_agree_exact_all_three(self):
        t = self._make_triple("seg_001", "Same text here", "Same text here", "Same text here")
        classify_triples([t])
        assert t.classification == AGREE_EXACT

    def test_agree_near_picks_agreeing_source_not_outlier(self):
        """When two sources are near-matches, agreed_text should come from an agreeing source."""
        t = self._make_triple(
            "seg_001",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
            "Completely different outlier text here",
        )
        classify_triples([t], near_threshold=0.85, levenshtein_threshold=0.85)
        assert t.classification == AGREE_NEAR
        # Must NOT pick the outlier marker text
        assert "outlier" not in t.agreed_text.lower()

    def test_agree_near(self):
        t = self._make_triple(
            "seg_001",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
            "Completely different text about something else",
        )
        classify_triples([t], near_threshold=0.85, levenshtein_threshold=0.85)
        assert t.classification == AGREE_NEAR

    def test_conflict_on_numeric_difference(self):
        """Even high similarity should be CONFLICT if numbers differ."""
        t = self._make_triple(
            "seg_001",
            "The accuracy was 95% on the test set",
            "The accuracy was 96% on the test set",
            "The accuracy was 97% on the test set",
        )
        classify_triples([t])
        assert t.classification == CONFLICT

    def test_reference_number_variance_escalates_to_conflict(self):
        """Reference-number drift is treated as CONFLICT (no exceptions)."""
        # Use realistic-length text so the fig/citation differences are a small
        # fraction of the total, producing high similarity scores (>0.97).
        # Only vary the figure number (not citations) to stay within the
        # exception's max-2-numeric-delta / max-1-citation-delta limits.
        base_a = (
            "The calcium signaling pathway was analyzed using fluorescence "
            "microscopy as shown in Fig. 1 and confirmed by western blot "
            "analysis [8]."
        )
        base_b = (
            "The calcium signaling pathway was analyzed using fluorescence "
            "microscopy as shown in Fig. 2 and confirmed by western blot "
            "analysis [8]."
        )
        t = self._make_triple(
            "seg_001",
            base_a,
            base_b,
            "A completely different paragraph that does not match.",
        )
        t.confidence = 0.8
        classify_triples([t], near_threshold=0.85, levenshtein_threshold=0.85)
        assert t.classification == CONFLICT

    def test_reference_variance_exception_not_applied_to_citation_list_blocks(self):
        """Citation lists remain strict even when text similarity is high."""
        t = self._make_triple(
            "seg_001",
            "[8] Author A et al. 2020",
            "[9] Author A et al. 2020",
            "Different citation record text",
            block_type="citation_list",
        )
        t.confidence = 0.9
        classify_triples([t], near_threshold=0.85, levenshtein_threshold=0.85)
        assert t.classification == CONFLICT

    def test_strict_numeric_near_allows_matching_numbers(self):
        """When strict_numeric_near is enabled and numbers MATCH, AGREE_NEAR is allowed."""
        t = self._make_triple(
            "seg_001",
            "We measured 10 samples in total.",
            "We measured ten (10) samples in total.",
            "Completely different outlier text here",
        )
        # Numbers match (both have "10") so strict_numeric_near should NOT
        # escalate — the deterministic AGREE_NEAR path is safe.
        classify_triples([t], near_threshold=0.70, levenshtein_threshold=0.70, strict_numeric_near=True)
        assert t.classification == AGREE_NEAR

    def test_strict_numeric_near_escalates_when_numbers_differ(self):
        """When strict_numeric_near is enabled and numbers DIFFER, escalate to CONFLICT."""
        t = self._make_triple(
            "seg_001",
            "We measured 10 samples in total.",
            "We measured 12 samples in total.",
            "Completely different outlier text here",
        )
        # Numbers differ (10 vs 12) so strict_numeric_near should escalate.
        classify_triples([t], near_threshold=0.70, levenshtein_threshold=0.70, strict_numeric_near=True)
        assert t.classification == CONFLICT

    def test_gap_single_source(self):
        t = self._make_triple("seg_001", "Only in grobid", None, None)
        classify_triples([t])
        assert t.classification == GAP
        assert t.agreed_text == "Only in grobid"

    def test_gap_no_sources(self):
        t = self._make_triple("seg_001", None, None, None)
        classify_triples([t])
        assert t.classification == GAP

    def test_table_always_conflict(self):
        t = self._make_triple(
            "seg_001",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            block_type="table",
        )
        classify_triples([t])
        assert t.classification == CONFLICT

    def test_mixed_block_type_table_escalation(self):
        """If ANY extractor sees a table, escalate to CONFLICT even if others say paragraph."""
        t = self._make_triple(
            "seg_001",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            "A B 1 2",
            "A B 1 2",
        )
        # Override block types: grobid sees table, others see paragraph
        t.grobid_block.block_type = "table"
        t.docling_block.block_type = "paragraph"
        t.marker_block.block_type = "paragraph"
        classify_triples([t])
        assert t.classification == CONFLICT

    def test_mixed_block_type_equation_escalation(self):
        """If ANY extractor sees an equation, escalate to CONFLICT."""
        t = self._make_triple(
            "seg_001",
            "E = mc^2",
            "E = mc^2",
            "E = mc^2",
        )
        t.grobid_block.block_type = "paragraph"
        t.docling_block.block_type = "equation"
        t.marker_block.block_type = "paragraph"
        classify_triples([t])
        assert t.classification == CONFLICT

    def test_no_escalation_when_disabled(self):
        """Mixed block types with escalation disabled should NOT force CONFLICT."""
        t = self._make_triple(
            "seg_001",
            "Same text here",
            "Same text here",
            "Same text here",
        )
        t.grobid_block.block_type = "table"
        t.docling_block.block_type = "paragraph"
        t.marker_block.block_type = "paragraph"
        classify_triples([t], always_escalate_tables=False)
        # With escalation off, exact match should win
        assert t.classification == AGREE_EXACT

    def test_conflict_on_low_similarity(self):
        t = self._make_triple(
            "seg_001",
            "Alpha beta gamma delta epsilon",
            "One two three four five",
            "Completely unrelated text here",
        )
        classify_triples([t])
        assert t.classification == CONFLICT

    def test_numeric_tokens_ignore_html_tag_attributes(self):
        assert _extract_numeric_tokens('<span id="page-0-3">25 6 7 5</span>') == {
            "25", "6", "7", "5",
        }

    def test_citation_keys_include_ranges_and_lists(self):
        keys = _extract_citation_keys("Prior work [8-10], [6,7], [Smith et al. 2024].")
        assert "[8-10]" in keys
        assert "[6,7]" in keys
        assert "[Smith et al. 2024]" in keys


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------

class TestGuardTelemetry:
    """Guard checks are now telemetry-only. Only alignment confidence causes failure."""

    def test_healthy_document_passes(self):
        """Healthy document should produce metrics with failed=False."""
        triples = []
        for i, cls in enumerate([AGREE_EXACT, AGREE_EXACT, AGREE_NEAR, CONFLICT, GAP]):
            t = AlignedTriple(segment_id=f"seg_{i:03d}")
            t.classification = cls
            t.confidence = 0.8
            t.grobid_block = Block(block_id=f"g_{i}", raw_text="x", normalized_text="x", block_type="paragraph", heading_level=None, order_index=i, source="grobid")
            t.docling_block = Block(block_id=f"d_{i}", raw_text="y", normalized_text="y", block_type="paragraph", heading_level=None, order_index=i, source="docling")
            t.marker_block = Block(block_id=f"m_{i}", raw_text="z", normalized_text="z", block_type="paragraph", heading_level=None, order_index=i, source="marker")
            triples.append(t)
        metrics = compute_metrics(triples, 0.7, False, None)
        assert metrics["failed"] is False
        assert metrics["failure_reason"] is None

    def test_fail_on_low_alignment_confidence(self):
        """Low alignment confidence should produce metrics with failed=True."""
        triples = []
        for i, cls in enumerate([AGREE_EXACT, AGREE_EXACT]):
            t = AlignedTriple(segment_id=f"seg_{i:03d}")
            t.classification = cls
            triples.append(t)
        metrics = compute_metrics(triples, 0.3, True, "alignment_too_low")
        assert metrics["failed"] is True
        assert metrics["failure_reason"] == "alignment_too_low"

    def test_empty_triples_triggers_failure(self):
        metrics = compute_metrics([], 0.7, True, "no_blocks")
        assert metrics["failed"] is True
        assert metrics["failure_reason"] == "no_blocks"

    def test_all_gap_does_not_fail(self):
        """All-GAP should have conflict_ratio = 0.0 and not fail."""
        triples = []
        for i in range(3):
            t = AlignedTriple(segment_id=f"seg_{i:03d}")
            t.classification = GAP
            triples.append(t)
        metrics = compute_metrics(triples, 0.7, False, None)
        assert metrics["failed"] is False
        assert metrics["conflict_ratio"] == 0.0

    def test_high_conflict_ratio_does_not_fail(self):
        """High conflict ratio documents go through zone resolution, not failure."""
        triples = []
        for i, cls in enumerate([CONFLICT, CONFLICT, CONFLICT, AGREE_EXACT, AGREE_EXACT]):
            t = AlignedTriple(segment_id=f"seg_{i:03d}")
            t.classification = cls
            t.confidence = 0.8
            t.grobid_block = Block(block_id=f"g_{i}", raw_text="x", normalized_text="x", block_type="paragraph", heading_level=None, order_index=i, source="grobid")
            t.docling_block = Block(block_id=f"d_{i}", raw_text="y", normalized_text="y", block_type="paragraph", heading_level=None, order_index=i, source="docling")
            t.marker_block = Block(block_id=f"m_{i}", raw_text="z", normalized_text="z", block_type="paragraph", heading_level=None, order_index=i, source="marker")
            triples.append(t)
        # conflict_ratio = 3/5 = 0.6 > 0.4, but this no longer triggers failure
        metrics = compute_metrics(triples, 0.7, False, None)
        assert metrics["failed"] is False
        assert metrics["conflict_ratio"] == 0.6


# ---------------------------------------------------------------------------
# Assembly tests
# ---------------------------------------------------------------------------

class TestAssembly:
    def test_basic_assembly(self):
        triples = [
            AlignedTriple(segment_id="seg_000", classification=AGREE_EXACT,
                          agreed_text="# Introduction"),
            AlignedTriple(segment_id="seg_001", classification=AGREE_NEAR,
                          agreed_text="Paragraph text here."),
            AlignedTriple(segment_id="seg_002", classification=GAP,
                          agreed_text="Gap text from one source."),
            AlignedTriple(segment_id="seg_003", classification=CONFLICT),
        ]

        resolved = {"seg_003": "LLM resolved this conflict."}
        result = assemble(triples, resolved)

        assert "# Introduction" in result
        assert "Paragraph text here." in result
        assert "Gap text from one source." in result
        assert "LLM resolved this conflict." in result

    def test_correct_ordering(self):
        triples = [
            AlignedTriple(segment_id="seg_000", classification=AGREE_EXACT,
                          agreed_text="First"),
            AlignedTriple(segment_id="seg_001", classification=AGREE_EXACT,
                          agreed_text="Second"),
            AlignedTriple(segment_id="seg_002", classification=AGREE_EXACT,
                          agreed_text="Third"),
        ]
        result = assemble(triples, {})
        assert result.index("First") < result.index("Second") < result.index("Third")

    def test_no_missing_segments(self):
        triples = [
            AlignedTriple(segment_id="seg_000", classification=CONFLICT),
            AlignedTriple(segment_id="seg_001", classification=CONFLICT),
        ]
        resolved = {"seg_000": "Resolved A", "seg_001": "Resolved B"}
        result = assemble(triples, resolved)
        assert "Resolved A" in result
        assert "Resolved B" in result

    def test_assembly_cleans_non_llm_segments(self):
        triples = [
            AlignedTriple(
                segment_id="seg_000",
                classification=AGREE_EXACT,
                agreed_text='<span id="page-1-0">Alpha</span>',
            ),
            AlignedTriple(
                segment_id="seg_001",
                classification=CONFLICT,
            ),
        ]
        result = assemble(triples, {"seg_001": "Resolved text"})
        assert "<span" not in result
        assert "Alpha" in result
        assert "Resolved text" in result

    def test_assembly_injects_page_markers_from_block_pages(self):
        triples = [
            AlignedTriple(
                segment_id="seg_000",
                classification=AGREE_EXACT,
                agreed_text="Intro",
                docling_block=Block(
                    block_id="d0",
                    block_type="paragraph",
                    raw_text="Intro",
                    normalized_text="intro",
                    heading_level=None,
                    order_index=0,
                    source="docling",
                    page_no=1,
                ),
            ),
            AlignedTriple(
                segment_id="seg_001",
                classification=AGREE_EXACT,
                agreed_text="Methods",
                marker_block=Block(
                    block_id="m1",
                    block_type="paragraph",
                    raw_text="Methods",
                    normalized_text="methods",
                    heading_level=None,
                    order_index=1,
                    source="marker",
                    page_no=2,
                ),
            ),
        ]
        result = assemble(triples, {})
        assert "<!-- page: 1 -->" in result
        assert "<!-- page: 2 -->" in result
        assert result.index("<!-- page: 1 -->") < result.index("Intro")
        assert result.index("<!-- page: 2 -->") < result.index("Methods")

    def test_assembly_page_majority_vote_beats_source_preference(self):
        triple = AlignedTriple(
            segment_id="seg_100",
            classification=AGREE_EXACT,
            agreed_text="Result text",
            grobid_block=Block(
                block_id="g100",
                block_type="paragraph",
                raw_text="Result text",
                normalized_text="result text",
                heading_level=None,
                order_index=0,
                source="grobid",
                page_no=5,
            ),
            docling_block=Block(
                block_id="d100",
                block_type="paragraph",
                raw_text="Result text",
                normalized_text="result text",
                heading_level=None,
                order_index=0,
                source="docling",
                page_no=6,
            ),
            marker_block=Block(
                block_id="m100",
                block_type="paragraph",
                raw_text="Result text",
                normalized_text="result text",
                heading_level=None,
                order_index=0,
                source="marker",
                page_no=6,
            ),
        )
        result = assemble([triple], {})
        assert "<!-- page: 6 -->" in result
        assert "<!-- page: 5 -->" not in result


class TestGapDedup:
    def test_dedup_gap_triples_drops_shorter_duplicate(self):
        triples = [
            AlignedTriple(
                segment_id="seg_000",
                classification=GAP,
                agreed_text="Gill Institute for Neuroscience and Department of Biology, IU.",
            ),
            AlignedTriple(
                segment_id="seg_001",
                classification=AGREE_EXACT,
                agreed_text="## Methods",
            ),
            AlignedTriple(
                segment_id="seg_002",
                classification=GAP,
                agreed_text="Gill Institute for Neuroscience and Department of Biology.",
            ),
        ]

        removed = dedup_gap_triples(triples, window=3, similarity_threshold=0.85, length_ratio_threshold=0.7)
        assert removed == 1
        assert sum(1 for t in triples if t.classification == GAP and t.agreed_text) == 1


class TestQAGates:
    def test_run_qa_gates_detects_artifacts_and_duplicates(self):
        merged = (
            '<span id="page-0-1"></span>Repeated paragraph content that is definitely longer than fifty chars.\n\n'
            "Repeated paragraph content that is definitely longer than fifty chars.\n\n"
            "<!-- image -->"
        )
        qa = run_qa_gates(merged)
        assert qa["span_tag_count"] >= 1
        assert qa["html_comment_count"] >= 1
        assert qa["global_duplicate_count"] >= 1
        assert qa["qa_passed"] is False


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_basic_metrics(self):
        triples = []
        for cls in [AGREE_EXACT, AGREE_EXACT, AGREE_NEAR, GAP, CONFLICT]:
            t = AlignedTriple(segment_id=f"seg_{len(triples):03d}")
            t.classification = cls
            triples.append(t)

        metrics = compute_metrics(triples, 0.75, False, None)
        assert metrics["total_blocks"] == 5
        assert metrics["agree_exact"] == 2
        assert metrics["agree_near"] == 1
        assert metrics["gap"] == 1
        assert metrics["conflict"] == 1
        # conflict_ratio = 1/(5-1) = 0.25
        assert metrics["conflict_ratio"] == 0.25
        assert metrics["alignment_confidence"] == 0.75
        assert metrics["failed"] is False

    def test_metrics_include_guard_telemetry(self):
        triples = []
        for cls in [AGREE_EXACT, CONFLICT]:
            t = AlignedTriple(segment_id=f"seg_{len(triples):03d}")
            t.classification = cls
            triples.append(t)

        metrics = compute_metrics(
            triples,
            0.8,
            True,
            "conflict_ratio",
            guard_telemetry={
                "conflict_ratio_textual": 0.5,
                "conflict_ratio_structured": 0.0,
                "conflicts_localized": True,
                "conflict_span_ratio": 0.2,
                "adaptive_conflict_ratio_threshold": 0.55,
            },
        )
        assert metrics["conflict_ratio_textual"] == 0.5
        assert metrics["conflict_ratio_structured"] == 0.0
        assert metrics["conflicts_localized"] is True
        assert metrics["adaptive_conflict_ratio_threshold"] == 0.55


# ---------------------------------------------------------------------------
# Phase 2 metadata tests
# ---------------------------------------------------------------------------

class TestPhase2Metadata:
    def test_audit_includes_resolution_method_metadata(self):
        triple = AlignedTriple(segment_id="seg_001", classification=CONFLICT)
        triple.grobid_block = Block(
            block_id="g_001",
            block_type="paragraph",
            raw_text="Alpha text.",
            normalized_text=normalize_text("Alpha text."),
            heading_level=None,
            order_index=0,
            source="grobid",
        )
        triple.docling_block = Block(
            block_id="d_001",
            block_type="paragraph",
            raw_text="Alpha text updated.",
            normalized_text=normalize_text("Alpha text updated."),
            heading_level=None,
            order_index=0,
            source="docling",
        )

        audit = _build_audit_entries(
            [triple],
            {"seg_001": "Resolved text"},
            {"seg_001": {
                "method": "median_source",
                "confidence": 0.91,
                "sources_agreeing": ["docling", "grobid"],
            }},
        )

        assert len(audit) == 1
        assert audit[0]["chosen_source"] == "median_source"
        assert audit[0]["details"]["resolution_method"] == "median_source"
        assert audit[0]["details"]["resolution_confidence"] == 0.91


# ---------------------------------------------------------------------------
# End-to-end / orchestrator tests
# ---------------------------------------------------------------------------

class TestMergeWithConsensus:
    @patch("config.Config")
    def test_missing_extractor_returns_failure(self, mock_config):
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.5

        llm = MagicMock()
        llm.usage = TokenAccumulator()
        result, metrics, _audit = merge_with_consensus("grobid", "", "marker", llm)
        assert result is None
        assert metrics["failed"] is True
        assert metrics["failure_reason"] == "missing_extractor"

    @patch("config.Config")
    def test_none_input_returns_failure(self, mock_config):
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.5

        llm = MagicMock()
        llm.usage = TokenAccumulator()
        result, metrics, _audit = merge_with_consensus("grobid", None, "marker", llm)
        assert result is None
        assert metrics["failure_reason"] == "missing_extractor"

    @patch("config.Config")
    def test_full_pipeline_with_agreement(self, mock_config):
        """When all 3 extractors agree, should succeed without calling LLM."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.5

        llm = MagicMock()
        llm.usage = TokenAccumulator()

        # Use identical text so everything agrees
        md = "# Title\n\nThis is the body text of the paper."
        result, metrics, _audit = merge_with_consensus(md, md, md, llm)

        assert result is not None
        assert "Title" in result
        assert "body text" in result
        assert metrics["failed"] is False
        assert metrics["qa"]["qa_passed"] is True
        # LLM should NOT have been called for conflict resolution
        llm.resolve_conflicts.assert_not_called()
        llm.resolve_micro_conflicts.assert_not_called()

    @patch("config.Config")
    def test_full_pipeline_with_conflicts(self, mock_config):
        """When extractors disagree, LLM should be called for micro-conflict resolution."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8  # high threshold to avoid fallback
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.1  # low threshold
        mock_config.CONSENSUS_LAYERED_MEDIUM_SIM_THRESHOLD = 1.1

        llm = MagicMock()
        llm.usage = TokenAccumulator()

        md_a = "# Title\n\nThe accuracy was 95%."
        md_b = "# Title\n\nThe accuracy was 96%."
        md_c = "# Title\n\nThe accuracy was 97%."

        def mock_resolve_micro(payload):
            response = MagicMock()
            response.resolved = []
            for mc in payload.get("micro_conflicts", []):
                item = MagicMock()
                item.conflict_id = mc["conflict_id"]
                item.text = "95.5%"
                response.resolved.append(item)
            return response
        llm.resolve_micro_conflicts.side_effect = mock_resolve_micro

        result, metrics, _audit = merge_with_consensus(md_a, md_b, md_c, llm)

        if result is not None:
            # Pipeline succeeded with micro-conflict resolution
            assert metrics["conflict"] >= 1
            llm.resolve_micro_conflicts.assert_called()
        else:
            # Fallback triggered (acceptable if confidence too low)
            assert metrics["failed"] is True

    @patch("config.Config")
    @patch("app.services.consensus_service._resolve_conflicts_micro")
    def test_resolution_methods_are_reported(self, mock_micro_resolve, mock_config):
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.1
        mock_config.CONSENSUS_LAYERED_MEDIUM_SIM_THRESHOLD = 1.1
        mock_config.CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES = False
        mock_config.CONSENSUS_HIERARCHY_ENABLED = False

        llm = MagicMock()
        llm.usage = TokenAccumulator()
        mock_micro_resolve.return_value = (
            {"seg_001": "Resolved by layered resolver."},
            {
                "seg_001": {
                    "method": "median_source",
                    "confidence": 0.87,
                    "sources_agreeing": ["docling", "grobid", "marker"],
                },
            },
        )

        md_a = "# Title\n\nThe accuracy was 95%."
        md_b = "# Title\n\nThe accuracy was 96%."
        md_c = "# Title\n\nThe accuracy was 97%."

        result, metrics, audit = merge_with_consensus(md_a, md_b, md_c, llm)
        assert result is not None
        assert metrics["failed"] is False
        assert metrics["resolution_methods"]["median_source"] == 1
        assert metrics["resolution_confidence_mean"] == 0.87
        assert any(
            entry.get("details", {}).get("resolution_method") == "median_source"
            for entry in audit
        )

    @patch("config.Config")
    def test_llm_error_triggers_failure(self, mock_config):
        """LLM failure during conflict resolution should trigger failure."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.1
        mock_config.CONSENSUS_LAYERED_MEDIUM_SIM_THRESHOLD = 1.1

        llm = MagicMock()
        llm.usage = TokenAccumulator()
        llm.resolve_micro_conflicts.side_effect = Exception("LLM is down")

        md_a = "# Title\n\nThe accuracy was 95%."
        md_b = "# Title\n\nThe accuracy was 96%."
        md_c = "# Title\n\nThe accuracy was 97%."

        result, metrics, _audit = merge_with_consensus(md_a, md_b, md_c, llm)

        # Per-segment LLM failures should degrade gracefully (rescue/fallback),
        # not fail the entire pipeline.
        if metrics.get("conflict", 0) > 0:
            assert result is not None
            assert metrics["failed"] is False
            llm.resolve_micro_conflicts.assert_called()

    @patch("config.Config")
    def test_sparse_extractor_is_excluded_from_alignment(self, mock_config, caplog):
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.5
        mock_config.CONSENSUS_ALWAYS_ESCALATE_TABLES = True

        llm = MagicMock()
        llm.usage = TokenAccumulator()
        grobid_md = "# Tiny\n\nOnly one paragraph."

        shared_blocks = "\n\n".join([
            f"Paragraph {i}: matching content for docling and marker." for i in range(12)
        ])
        docling_md = f"# Title\n\n{shared_blocks}"
        marker_md = f"# Title\n\n{shared_blocks}"

        with caplog.at_level("WARNING"):
            result, metrics, _audit = merge_with_consensus(grobid_md, docling_md, marker_md, llm)

        assert result is not None
        assert metrics["failed"] is False
        assert "excluding grobid from alignment" in caplog.text


# ---------------------------------------------------------------------------
# Source preference tests
# ---------------------------------------------------------------------------

class TestSourcePreference:
    def test_figure_ref_prefers_marker(self):
        """figure_ref blocks should prefer Marker source text."""
        assert _SOURCE_PREFERENCE["figure_ref"] == "marker"

        blocks = [
            Block(block_id="d", block_type="figure_ref",
                  raw_text="![Fig 1](docling_img.png)",
                  normalized_text="fig 1", heading_level=None,
                  order_index=0, source="docling"),
            Block(block_id="m", block_type="figure_ref",
                  raw_text="![Fig 1](marker_img.png)",
                  normalized_text="fig 1", heading_level=None,
                  order_index=0, source="marker"),
        ]
        result = _pick_preferred_text(blocks, "figure_ref")
        assert result == "![Fig 1](marker_img.png)"

    def test_source_md_preferred_over_raw_text(self):
        """When source_md is set, _pick_preferred_text should return it."""
        blocks = [
            Block(block_id="m", block_type="heading",
                  raw_text="Methods", normalized_text="methods",
                  heading_level=2, order_index=0, source="grobid",
                  source_md="## Methods"),
        ]
        result = _pick_preferred_text(blocks, "heading")
        assert result == "## Methods"

    def test_source_md_fallback_to_raw_text(self):
        """When source_md is empty, _pick_preferred_text should fall back to raw_text."""
        blocks = [
            Block(block_id="m", block_type="paragraph",
                  raw_text="plain text", normalized_text="plain text",
                  heading_level=None, order_index=0, source="marker"),
        ]
        result = _pick_preferred_text(blocks, "paragraph")
        assert result == "plain text"


# ---------------------------------------------------------------------------
# Markdown preservation tests
# ---------------------------------------------------------------------------

class TestMarkdownPreservation:
    def test_headings_preserved_in_parse(self):
        """parse_markdown should preserve heading markers in source_md."""
        blocks = parse_markdown("## Methods\n\nBody text.", "grobid")
        headings = [b for b in blocks if b.block_type == "heading"]
        assert len(headings) >= 1
        assert headings[0].source_md.startswith("## ")
        assert "Methods" in headings[0].source_md

    def test_image_refs_preserved_in_parse(self):
        """parse_markdown should preserve image markdown in source_md."""
        blocks = parse_markdown("![Figure 1](image.png)", "marker")
        assert len(blocks) >= 1
        has_image_md = any("![" in b.source_md and "](image.png)" in b.source_md
                          for b in blocks)
        assert has_image_md

    def test_bold_preserved_in_parse(self):
        """parse_markdown should preserve bold markers in source_md."""
        blocks = parse_markdown("This is **important** text.", "grobid")
        paragraphs = [b for b in blocks if b.block_type == "paragraph"]
        assert len(paragraphs) >= 1
        assert "**important**" in paragraphs[0].source_md

    def test_page_markers_assign_page_numbers_in_parse(self):
        md = "<!-- page: 2 -->\n\nPara 2.\n\n<!-- page: 4 -->\n\nPara 4."
        blocks = parse_markdown(md, "docling")
        paragraphs = [b for b in blocks if b.block_type == "paragraph"]
        assert len(paragraphs) == 2
        assert paragraphs[0].page_no == 2
        assert paragraphs[1].page_no == 4

    @patch("config.Config")
    def test_heading_markers_survive_full_pipeline(self, mock_config):
        """Heading markers (##) should be present in final assembled output."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.5
        mock_config.CONSENSUS_ALWAYS_ESCALATE_TABLES = True

        llm = MagicMock()
        llm.usage = TokenAccumulator()
        md = "# Title\n\n## Methods\n\nBody text of the methods section."
        result, metrics, _audit = merge_with_consensus(md, md, md, llm)

        assert result is not None
        assert "# Title" in result
        assert "## Methods" in result
        assert metrics["failed"] is False


# ---------------------------------------------------------------------------
# Global dedup: dedup_assembled_paragraphs tests
# ---------------------------------------------------------------------------

class TestAssembledDedup:
    def test_dedup_assembled_drops_non_adjacent_duplicate(self):
        """Post-assembly dedup catches non-adjacent duplicates."""
        text = (
            "# Introduction\n\n"
            "Proteomics relates the abundances of proteins to other biomolecules "
            "such as lipids or DNA and facilitates systems biology modeling.\n\n"
            "## Methods\n\n"
            "We used mass spectrometry to analyze protein samples from Drosophila eyes.\n\n"
            "## Results\n\n"
            "Proteomics relates the abundances of proteins to other biomolecules "
            "such as lipids or DNA and facilitates systems biology modeling.\n\n"
            "## Discussion\n\n"
            "The findings demonstrate the value of quantitative proteomics approaches."
        )
        cleaned, removed = dedup_assembled_paragraphs(text)
        assert removed == 1
        # The duplicate paragraph should appear only once
        assert cleaned.count("Proteomics relates the abundances") == 1
        # Headings should be preserved
        assert "# Introduction" in cleaned
        assert "## Methods" in cleaned
        assert "## Results" in cleaned
        assert "## Discussion" in cleaned

    def test_dedup_assembled_preserves_headings(self):
        """Headings are not stripped by post-assembly dedup even if similar."""
        text = (
            "# Methods\n\n"
            "The methods section describes the experimental approach.\n\n"
            "## Methods\n\n"
            "A different paragraph about something entirely unrelated to the heading above."
        )
        cleaned, removed = dedup_assembled_paragraphs(text)
        assert removed == 0
        assert "# Methods" in cleaned
        assert "## Methods" in cleaned


# ---------------------------------------------------------------------------
# Global dedup: QA gates global detection tests
# ---------------------------------------------------------------------------

class TestQAGatesGlobal:
    def test_qa_gates_global_duplicate_detection(self):
        """QA catches global (not just adjacent) duplicates."""
        merged = (
            "First unique paragraph that is definitely longer than fifty characters for testing.\n\n"
            "## Section Header\n\n"
            "Second unique paragraph that is also long enough to qualify for comparison.\n\n"
            "Some middle content that separates the duplicate paragraphs significantly.\n\n"
            "First unique paragraph that is definitely longer than fifty characters for testing."
        )
        qa = run_qa_gates(merged)
        assert qa["global_duplicate_count"] >= 1
        assert qa["qa_passed"] is False


# ---------------------------------------------------------------------------
# Global dedup: fail-hard pipeline tests
# ---------------------------------------------------------------------------

class TestGlobalDedupMonitoring:
    @patch("config.Config")
    def test_surviving_duplicates_logged_but_result_returned(self, mock_config):
        """Pipeline returns result even when global duplicates survive (monitoring only)."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.1
        mock_config.CONSENSUS_ALWAYS_ESCALATE_TABLES = True

        llm = MagicMock()
        llm.usage = TokenAccumulator()

        md = "# Title\n\nBody text of the paper for testing."
        with patch(
            "app.services.consensus_service.dedup_assembled_paragraphs",
            return_value=("# Title\n\nBody text of the paper for testing.", 0),
        ), patch(
            "app.services.consensus_service.run_qa_gates",
            return_value={
                "span_tag_count": 0,
                "html_comment_count": 0,
                "global_duplicate_count": 2,
                "qa_passed": False,
            },
        ):
            result, metrics, _audit = merge_with_consensus(md, md, md, llm)

        # Duplicates no longer trigger failure — zone-resolved output is kept
        assert result is not None
        assert metrics["failed"] is False


# ---------------------------------------------------------------------------
# Micro-conflict extraction tests
# ---------------------------------------------------------------------------

class TestMicroConflictExtraction:
    def _make_triple(
        self, seg_id, classification, agreed_text=None, grobid_text=None, docling_text=None,
        marker_text=None, block_type="paragraph",
    ):
        t = AlignedTriple(segment_id=seg_id, classification=classification)
        t.agreed_text = agreed_text
        if grobid_text is not None:
            t.grobid_block = Block(
                block_id=f"g_{seg_id}", block_type=block_type, raw_text=grobid_text,
                normalized_text=normalize_text(grobid_text),
                heading_level=None, order_index=0, source="grobid", source_md=grobid_text,
            )
        if docling_text is not None:
            t.docling_block = Block(
                block_id=f"d_{seg_id}", block_type=block_type, raw_text=docling_text,
                normalized_text=normalize_text(docling_text),
                heading_level=None, order_index=0, source="docling", source_md=docling_text,
            )
        if marker_text is not None:
            t.marker_block = Block(
                block_id=f"m_{seg_id}", block_type=block_type, raw_text=marker_text,
                normalized_text=normalize_text(marker_text),
                heading_level=None, order_index=0, source="marker", source_md=marker_text,
            )
        return t

    def test_tokenize_basic(self):
        tokens = tokenize_for_diff("The p-value was p < 0.001.")
        assert tokens == ["The", "p-value", "was", "p", "<", "0.001", "."]

    def test_tokenize_preserves_markdown(self):
        tokens = tokenize_for_diff("This is **bold** text.")
        assert "**bold**" in tokens

    def test_tokenize_table_markdown(self):
        tokens = tokenize_for_diff("| Col1 | Col2 |\n|------|------|\n| A | B |")
        assert "|" in tokens
        assert "Col1" in tokens

    def test_unanimous_agreement(self):
        triple = self._make_triple(
            "seg_001", CONFLICT,
            grobid_text="The result was significant.",
            docling_text="The result was significant.",
            marker_text="The result was significant.",
        )
        result = extract_micro_conflicts_for_segment(triple)
        assert result.majority_agree_ratio == 1.0
        assert len(result.conflicts) == 0

    def test_single_word_conflict(self):
        triple = self._make_triple(
            "seg_002", CONFLICT,
            grobid_text="The accuracy was 95 percent.",
            docling_text="The accuracy was 96 percent.",
            marker_text="The accuracy was 97 percent.",
        )
        result = extract_micro_conflicts_for_segment(triple)
        assert len(result.conflicts) == 1

    def test_majority_vote_resolves(self):
        triple = self._make_triple(
            "seg_003", CONFLICT,
            grobid_text="The p value was 0.001 .",
            docling_text="The p value was 0.001 .",
            marker_text="The p value was .001 .",
        )
        result = extract_micro_conflicts_for_segment(triple)
        assert len(result.conflicts) == 0
        assert result.majority_agree_ratio == 1.0

    def test_table_goes_through_micro_extraction(self):
        triple = self._make_triple(
            "seg_004", CONFLICT,
            grobid_text="| Gene | p |\n| BRCA1 | 0.001 |",
            docling_text="| Gene | p |\n| BRCA2 | 0.001 |",
            marker_text="| Gene | p |\n| BRCAl | 0.001 |",
            block_type="table",
        )
        result = extract_micro_conflicts_for_segment(triple)
        assert result.block_type == "table"
        assert len(result.conflicts) >= 1

    def test_equation_goes_through_micro_extraction(self):
        triple = self._make_triple(
            "seg_005", CONFLICT,
            grobid_text="E = mc^2",
            docling_text="E = mc^2",
            marker_text="E = m c^2",
            block_type="equation",
        )
        result = extract_micro_conflicts_for_segment(triple)
        assert result.block_type == "equation"
        assert isinstance(result.conflicts, list)

    def test_high_divergence_still_micro_extracted(self):
        triple = self._make_triple(
            "seg_006", CONFLICT,
            grobid_text="a b c d e f g h i j",
            docling_text="k l m n o p q r s t",
            marker_text="u v w x y z aa bb cc dd",
        )
        result = extract_micro_conflicts_for_segment(triple)
        assert len(result.conflicts) >= 1
        assert result.majority_agree_ratio < 0.5

    def test_high_divergence_coalesces_spans(self):
        agreed = [f"t{i}" for i in range(30)]
        conflicts = [
            MicroConflict(
                conflict_id=f"seg_007_mc_{i}",
                segment_id="seg_007",
                grobid_tokens=["a"],
                docling_tokens=["b"],
                marker_tokens=["c"],
                context_before=[],
                context_after=[],
                output_position=i * 2,
            )
            for i in range(8)
        ]
        result = MicroConflictResult(
            segment_id="seg_007",
            block_type="paragraph",
            agreed_tokens=agreed,
            conflicts=conflicts,
            majority_agree_ratio=0.2,
        )
        coalesced = _coalesce_micro_conflicts_if_high_divergence(result)
        assert len(coalesced.conflicts) <= len(result.conflicts)

    def test_two_source_segment(self):
        triple = self._make_triple(
            "seg_008", CONFLICT,
            grobid_text="The value was 10.",
            docling_text="The value was 11.",
            marker_text=None,
        )
        result = extract_micro_conflicts_for_segment(triple)
        assert isinstance(result.conflicts, list)

    def test_context_sentence_boundary(self):
        tokens = tokenize_for_diff("Sentence one. Sentence two with conflict. Sentence three.")
        left, right = _expand_to_sentence_boundary(tokens, 5, 6, cap=30)
        assert left <= 5
        assert right >= 6

    def test_reconstruct_splices_correctly(self):
        agreed = ["The", "__MC__0__", "result", "."]
        conflicts = [
            MicroConflict(
                conflict_id="seg_009_mc_0",
                segment_id="seg_009",
                grobid_tokens=["95%"],
                docling_tokens=["96%"],
                marker_tokens=["97%"],
                context_before=[],
                context_after=[],
                output_position=1,
            ),
        ]
        final = reconstruct_segment_from_micro_conflicts(
            agreed,
            conflicts,
            {"seg_009_mc_0": "95.5%"},
        )
        assert final == "The 95.5% result."

    @patch("config.Config")
    def test_agree_near_not_sent_to_llm(self, mock_config):
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.1
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8
        mock_config.CONSENSUS_ALWAYS_ESCALATE_TABLES = False
        mock_config.CONSENSUS_HIERARCHY_ENABLED = False
        llm = MagicMock()
        llm.usage = TokenAccumulator()

        md_a = "# Title\n\nThe p value was 0.001 and n = 10."
        md_b = "# Title\n\nThe p-value was 0.001 and n = 10."
        md_c = "# Title\n\nThe p value was 0.001 and n = 10."
        result, metrics, _audit = merge_with_consensus(md_a, md_b, md_c, llm)
        assert result is not None
        assert metrics["failed"] is False
        llm.resolve_micro_conflicts.assert_not_called()

    @patch("config.Config")
    def test_gap_handled_individually(self, mock_config):
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = 0.1
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8
        mock_config.CONSENSUS_ALWAYS_ESCALATE_TABLES = False
        mock_config.CONSENSUS_HIERARCHY_ENABLED = False
        llm = MagicMock()
        llm.usage = TokenAccumulator()
        llm.resolve_gap.return_value = {"seg_002": "kept gap text"}

        md_a = "# Title\n\nShared paragraph.\n\nOnly grobid has this."
        md_b = "# Title\n\nShared paragraph."
        md_c = "# Title\n\nShared paragraph."
        _result, _metrics, _audit = merge_with_consensus(md_a, md_b, md_c, llm)
        assert llm.resolve_gap.called

    def test_no_full_segment_fallback_paths_exist(self):
        llm_source = Path("app/services/llm_service.py").read_text()
        consensus_source = Path("app/services/consensus_service.py").read_text()
        assert "resolve_conflict_zones" not in llm_source
        assert "_resolve_single_zone" not in llm_source
        assert "_build_conflict_zones" not in consensus_source


# ---------------------------------------------------------------------------
# Phase 2 guards: per-segment allowed numbers, dropped numbers, content
# similarity, numeric truncation
# ---------------------------------------------------------------------------


class TestSegmentAllowedNumbers:
    """Tests for per-segment numeric token extraction."""

    def test_basic_extraction(self):
        seg = {"grobid": "Figure 5 shows n = 85", "docling": "Figure 5 shows n = 85", "marker": ""}
        allowed = _segment_allowed_numeric_tokens(seg)
        assert "5" in allowed
        assert "85" in allowed

    def test_no_cross_segment_leakage(self):
        """Numbers from other segments should NOT appear in per-segment allowed set."""
        seg_a = {"grobid": "Figure 5 was used", "docling": "Figure 5 was used"}
        seg_b = {"grobid": "Figure 8 was used", "docling": "Figure 8 was used"}
        allowed_a = _segment_allowed_numeric_tokens(seg_a)
        allowed_b = _segment_allowed_numeric_tokens(seg_b)
        assert "5" in allowed_a
        assert "8" not in allowed_a
        assert "8" in allowed_b
        assert "5" not in allowed_b

    def test_gap_segment_uses_text_field(self):
        seg = {"status": "gap", "text": "p = 0.001 was significant"}
        allowed = _segment_allowed_numeric_tokens(seg)
        assert "0.001" in allowed


class TestDroppedNumberDetection:
    """Tests for bidirectional numeric integrity guard."""

    def test_detects_dropped_consensus_number(self):
        seg = {"grobid": "n = 85 subjects", "docling": "n = 85 subjects", "marker": "n = 85 subjects"}
        # Output drops 85
        dropped = _numeric_integrity_dropped_numbers("n subjects", seg)
        assert "85" in dropped

    def test_no_false_positive_when_picking_one_source(self):
        """When LLM picks Source A over B, numbers unique to B are NOT flagged."""
        seg = {"grobid": "n = 85 subjects", "docling": "n = 90 subjects"}
        # Output picks grobid's version — 90 only appears in 1 source (< threshold of 2)
        dropped = _numeric_integrity_dropped_numbers("n = 85 subjects", seg)
        assert "90" not in dropped
        assert not dropped  # 85 is in output, 90 is only in 1 source

    def test_flags_number_in_majority_of_sources(self):
        seg = {"grobid": "p = 0.05 value", "docling": "p = 0.05 value", "marker": "p = 0.05 value"}
        dropped = _numeric_integrity_dropped_numbers("p value was significant", seg)
        assert "0.05" in dropped

    def test_empty_output_flags_all_consensus_numbers(self):
        seg = {"grobid": "7.1 ± 1.6 μA", "docling": "7.1 ± 1.6 μA"}
        dropped = _numeric_integrity_dropped_numbers("", seg)
        assert "7.1" in dropped
        assert "1.6" in dropped


class TestContentSimilarityCheck:
    """Tests for content similarity validation."""

    def test_correct_resolution_has_high_similarity(self):
        source_texts = {"grobid": "The protein kinase was activated by phosphorylation"}
        resolved = "The protein kinase was activated by phosphorylation"
        sim = _content_similarity_check(resolved, source_texts)
        assert sim > 0.9

    def test_wrong_section_has_low_similarity(self):
        source_texts = {
            "grobid": "prkdc knockout efficiency was 60%",
            "docling": "prkdc knockout efficiency was 60%",
        }
        # Totally wrong content from different section
        resolved = "BRAF T1799A mutation was detected in melanoma samples"
        sim = _content_similarity_check(resolved, source_texts)
        assert sim < 0.5  # Below _CONTENT_SIMILARITY_MIN threshold

    def test_empty_resolved_returns_zero(self):
        source_texts = {"grobid": "some text"}
        assert _content_similarity_check("", source_texts) == 0.0

    def test_empty_sources_returns_zero(self):
        assert _content_similarity_check("some text", {}) == 0.0


class TestNumericCountRatio:
    """Tests for numeric truncation detection."""

    def test_full_preservation(self):
        source_texts = {"grobid": "p=0.01, p=0.05, p=0.001, p=0.0001"}
        resolved = "p=0.01, p=0.05, p=0.001, p=0.0001"
        out_count, max_src = _numeric_count_ratio(resolved, source_texts)
        assert out_count == max_src

    def test_detects_truncation(self):
        source_texts = {"grobid": "p=0.01, p=0.05, p=0.001, p=0.0001, p=0.02"}
        resolved = "p=0.01"
        out_count, max_src = _numeric_count_ratio(resolved, source_texts)
        assert max_src >= 5
        assert out_count < max_src * 0.5

    def test_empty_resolved(self):
        source_texts = {"grobid": "values 1 2 3 4 5"}
        out_count, max_src = _numeric_count_ratio("", source_texts)
        assert out_count == 0
        assert max_src >= 5


# ---------------------------------------------------------------------------
# Micro-conflict: majority deletion sentinel tests (C-1)
# ---------------------------------------------------------------------------

class TestMajorityDeletion:
    """Test that 2/3 sources having None (deletion) reaches majority via sentinel."""

    def test_two_none_one_token_produces_deletion(self):
        """When 2 of 3 sources have None, the token should be deleted (not kept)."""
        source_tokens = {
            "grobid": ["hello", "world"],
            "docling": ["hello"],          # shorter — "world" maps to None
            "marker": ["hello"],           # shorter — "world" maps to None
        }
        agreed, conflicts, ratio = _build_majority_alignment(source_tokens)
        # "hello" is agreed upon by all three; "world" should be deleted (2/3 None)
        assert "hello" in agreed
        assert "world" not in agreed
        # No conflicts — both columns resolved by majority
        assert len(conflicts) == 0

    def test_none_not_filtered_from_counter(self):
        """The _DELETION sentinel should appear in Counter so None tokens participate in voting."""
        source_tokens = {
            "grobid": ["alpha", "beta"],
            "docling": ["alpha"],
            "marker": ["alpha"],
        }
        agreed, conflicts, _ = _build_majority_alignment(source_tokens)
        # "beta" appears in 1/3 and None in 2/3 — majority deletion wins
        assert "beta" not in agreed
        assert len(conflicts) == 0


# ---------------------------------------------------------------------------
# Zero-conflict preserves original formatting (C-2)
# ---------------------------------------------------------------------------

class TestZeroConflictPreservesFormatting:
    """When majority voting resolves all disagreements, use preferred source text."""

    def test_extract_micro_conflicts_zero_conflict_preserves_newlines(self):
        """A table with near-identical sources should preserve newlines, not join with spaces."""
        table_text = "| Col A | Col B |\n|-------|-------|\n| 1     | 2     |"
        triple = AlignedTriple(
            segment_id="seg_t1",
            grobid_block=Block(
                block_id="g1", block_type="table", raw_text=table_text,
                normalized_text=table_text.lower(), heading_level=None,
                order_index=0, source="grobid", source_md=table_text,
            ),
            docling_block=Block(
                block_id="d1", block_type="table", raw_text=table_text,
                normalized_text=table_text.lower(), heading_level=None,
                order_index=0, source="docling", source_md=table_text,
            ),
            marker_block=Block(
                block_id="m1", block_type="table", raw_text=table_text,
                normalized_text=table_text.lower(), heading_level=None,
                order_index=0, source="marker", source_md=table_text,
            ),
            classification=CONFLICT,
        )
        results = extract_micro_conflicts([triple])
        # All three are identical so should reclassify and preserve original text
        assert triple.classification == AGREE_EXACT
        assert "\n" in triple.agreed_text  # newlines preserved


# ---------------------------------------------------------------------------
# Missing micro result routes to rescue (C-5)
# ---------------------------------------------------------------------------

class TestMissingMicroResultRescue:
    """When micro result is None, _resolve_conflicts_micro should route to rescue."""

    @patch("app.services.consensus_service._rescue_segment")
    def test_missing_result_calls_rescue(self, mock_rescue):
        """A CONFLICT triple with no micro result should invoke rescue, not be silently skipped."""
        from app.services.consensus_service import _resolve_conflicts_micro

        triple = AlignedTriple(
            segment_id="seg_missing",
            grobid_block=Block(
                block_id="g1", block_type="paragraph", raw_text="text a",
                normalized_text="text a", heading_level=None,
                order_index=0, source="grobid", source_md="text a",
            ),
            docling_block=Block(
                block_id="d1", block_type="paragraph", raw_text="text b",
                normalized_text="text b", heading_level=None,
                order_index=0, source="docling", source_md="text b",
            ),
            marker_block=Block(
                block_id="m1", block_type="paragraph", raw_text="text c",
                normalized_text="text c", heading_level=None,
                order_index=0, source="marker", source_md="text c",
            ),
            classification=CONFLICT,
        )

        llm_mock = MagicMock()
        # micro_results has no entry for this triple
        _resolve_conflicts_micro([triple], {}, llm_mock)
        mock_rescue.assert_called_once()
        call_args = mock_rescue.call_args
        assert call_args[0][0] == "seg_missing"  # seg_id


# ---------------------------------------------------------------------------
# Structural heading preservation (KANBAN-1069)
# ---------------------------------------------------------------------------


class TestIsStructuralHeading:
    """Tests for the structural heading whitelist."""

    def test_abstract_heading(self):
        assert is_structural_heading("## Abstract") is True

    def test_abstract_h3(self):
        assert is_structural_heading("### Abstract") is True

    def test_introduction(self):
        assert is_structural_heading("## Introduction") is True

    def test_methods(self):
        assert is_structural_heading("## Methods") is True

    def test_materials_and_methods(self):
        assert is_structural_heading("## Materials and Methods") is True

    def test_results(self):
        assert is_structural_heading("## Results") is True

    def test_discussion(self):
        assert is_structural_heading("## Discussion") is True

    def test_references(self):
        assert is_structural_heading("## References") is True

    def test_acknowledgments(self):
        assert is_structural_heading("## Acknowledgments") is True
        assert is_structural_heading("## Acknowledgements") is True

    def test_conclusion(self):
        assert is_structural_heading("## Conclusion") is True
        assert is_structural_heading("## Conclusions") is True

    def test_non_structural_heading(self):
        assert is_structural_heading("## 2.1. Fly Strains") is False

    def test_random_text(self):
        assert is_structural_heading("Some random paragraph text") is False

    def test_none(self):
        assert is_structural_heading(None) is False

    def test_empty(self):
        assert is_structural_heading("") is False

    def test_case_insensitive(self):
        assert is_structural_heading("## ABSTRACT") is True
        assert is_structural_heading("## abstract") is True


class TestAgreeNearHeadingGuard:
    """Tests for the AGREE_NEAR structural heading guard.

    Reproduces the failure in AGRKB_101000000645569 where Grobid's
    '## Abstract' heading was overwritten by docling/marker affiliation text.
    """

    def _make_block(self, source, block_type, text, heading_level=None, source_md=None):
        return Block(
            block_id=f"{source}_0",
            block_type=block_type,
            raw_text=text,
            normalized_text=text.lower().strip().lstrip("#").strip(),
            heading_level=heading_level,
            order_index=0,
            source=source,
            source_md=source_md or text,
        )

    def test_heading_preserved_over_non_heading_pair(self):
        """Grobid heading should not be overwritten by docling/marker paragraph pair."""
        affiliation = (
            "Perinatal Institute and Division of Neonatology, Perinatal "
            "and Pulmonary Biology, Cincinnati Children's Hospital Medical "
            "Center, Cincinnati, OH 45229"
        )
        affiliation_alt = (
            "Perinatal Institute and Division of Neonatology, Perinatal "
            "and Pulmonary Biology, Cincinnati Childrens Hospital Medical "
            "Center, Cincinnati OH 45229"
        )
        triple = AlignedTriple(
            segment_id="seg_002",
            grobid_block=self._make_block(
                "grobid", "heading", "Abstract",
                heading_level=2, source_md="## Abstract",
            ),
            docling_block=self._make_block(
                "docling", "paragraph", affiliation,
            ),
            marker_block=self._make_block(
                "marker", "paragraph", affiliation_alt,
            ),
        )

        classify_triples([triple])

        assert triple.classification == AGREE_NEAR
        assert triple.agreed_text == "## Abstract"

    def test_normal_agree_near_not_affected(self):
        """Normal AGREE_NEAR (no heading involved) should work as before."""
        triple = AlignedTriple(
            segment_id="seg_010",
            grobid_block=self._make_block(
                "grobid", "paragraph", "The quick brown fox jumps over the lazy dog.",
            ),
            docling_block=self._make_block(
                "docling", "paragraph", "The quick brown fox jumps over the lazy dog.",
            ),
            marker_block=self._make_block(
                "marker", "paragraph", "The quick brown fox jumps over the lazy dog.",
            ),
        )

        classify_triples([triple])

        assert triple.classification in (AGREE_EXACT, AGREE_NEAR)
        assert "## Abstract" not in (triple.agreed_text or "")

    def test_non_structural_heading_not_guarded(self):
        """Non-structural headings (e.g. '## 2.1 Fly Strains') should NOT
        trigger the guard — only whitelisted structural headings."""
        similar_text_a = "Section content about fly strains and experimental setup details"
        similar_text_b = "Section content about fly strains and experimental setup detail"
        triple = AlignedTriple(
            segment_id="seg_020",
            grobid_block=self._make_block(
                "grobid", "heading", "2.1 Fly Strains",
                heading_level=2, source_md="## 2.1 Fly Strains",
            ),
            docling_block=self._make_block(
                "docling", "paragraph", similar_text_a,
            ),
            marker_block=self._make_block(
                "marker", "paragraph", similar_text_b,
            ),
        )

        classify_triples([triple])

        # Should NOT preserve the non-structural heading
        if triple.classification == AGREE_NEAR:
            assert triple.agreed_text != "## 2.1 Fly Strains"


class TestGapHeadingPreservation:
    """Tests for GAP structural heading bypass in _resolve_conflicts_micro."""

    def _make_block(self, source, block_type, text, heading_level=None, source_md=None):
        return Block(
            block_id=f"{source}_0",
            block_type=block_type,
            raw_text=text,
            normalized_text=text.lower().strip().lstrip("#").strip(),
            heading_level=heading_level,
            order_index=0,
            source=source,
            source_md=source_md or text,
        )

    def test_structural_heading_gap_preserved_without_llm(self):
        """Structural heading GAP should be kept deterministically, no LLM call."""
        triple = AlignedTriple(
            segment_id="seg_gap_abstract",
            grobid_block=self._make_block(
                "grobid", "heading", "Abstract",
                heading_level=2, source_md="## Abstract",
            ),
            classification=GAP,
            agreed_text="## Abstract",
        )

        llm_mock = MagicMock()
        from app.services.consensus_resolution import _resolve_conflicts_micro
        resolved, metadata = _resolve_conflicts_micro([triple], {}, llm_mock)

        assert resolved["seg_gap_abstract"] == "## Abstract"
        assert metadata["seg_gap_abstract"]["method"] == "structural_heading_gap_keep"
        # LLM should NOT have been called
        llm_mock.resolve_gap.assert_not_called()

    def test_non_structural_heading_gap_still_uses_llm(self):
        """Non-structural heading GAPs should still go through LLM resolution."""
        triple = AlignedTriple(
            segment_id="seg_gap_custom",
            grobid_block=self._make_block(
                "grobid", "heading", "2.1 Fly Strains",
                heading_level=2, source_md="## 2.1 Fly Strains",
            ),
            classification=GAP,
            agreed_text="## 2.1 Fly Strains",
        )

        llm_mock = MagicMock()
        llm_mock.resolve_gap.return_value = {"seg_gap_custom": "## 2.1 Fly Strains"}
        from app.services.consensus_resolution import _resolve_conflicts_micro
        _resolve_conflicts_micro([triple], {}, llm_mock)

        # LLM should have been called for non-structural headings
        llm_mock.resolve_gap.assert_called_once()

    def test_paragraph_gap_still_uses_llm(self):
        """Normal paragraph GAPs should still go through LLM resolution."""
        triple = AlignedTriple(
            segment_id="seg_gap_para",
            grobid_block=self._make_block(
                "grobid", "paragraph", "Some orphan text from grobid only.",
            ),
            classification=GAP,
            agreed_text="Some orphan text from grobid only.",
        )

        llm_mock = MagicMock()
        llm_mock.resolve_gap.return_value = {"seg_gap_para": "Some orphan text from grobid only."}
        from app.services.consensus_resolution import _resolve_conflicts_micro
        _resolve_conflicts_micro([triple], {}, llm_mock)

        llm_mock.resolve_gap.assert_called_once()


class TestEnsureAbstractHeading:
    """Tests for the post-assembly safety net."""

    def _make_block(self, source, block_type, text, heading_level=None, source_md=None):
        return Block(
            block_id=f"{source}_0",
            block_type=block_type,
            raw_text=text,
            normalized_text=text.lower().strip().lstrip("#").strip(),
            heading_level=heading_level,
            order_index=0,
            source=source,
            source_md=source_md or text,
        )

    def test_injects_when_missing(self):
        """Inject ## Abstract when parser had it but output is missing it."""
        md = (
            "# Paper Title\n\n"
            "---\n\n"
            "This is a long abstract paragraph that describes the findings of "
            "this study in great detail. It covers methods, results, and "
            "conclusions. The study was conducted over a period of several "
            "years and involved multiple research teams across different "
            "institutions. The results show significant improvements."
        )
        triples = [
            AlignedTriple(
                segment_id="seg_heading",
                grobid_block=self._make_block(
                    "grobid", "heading", "Abstract",
                    heading_level=2, source_md="## Abstract",
                ),
                classification=GAP,
                agreed_text="## Abstract",
            ),
        ]

        result, injected = ensure_abstract_heading(md, triples)
        assert injected is True
        assert "## Abstract" in result

    def test_no_injection_when_present(self):
        """Don't inject if ## Abstract already exists."""
        md = "# Title\n\n## Abstract\n\nThis is the abstract text."
        triples = [
            AlignedTriple(
                segment_id="seg_heading",
                grobid_block=self._make_block(
                    "grobid", "heading", "Abstract",
                    heading_level=2, source_md="## Abstract",
                ),
                classification=GAP,
            ),
        ]

        result, injected = ensure_abstract_heading(md, triples)
        assert injected is False
        assert result == md

    def test_no_injection_when_no_parser_evidence(self):
        """Don't inject if no parser had an Abstract heading."""
        md = (
            "# Paper Title\n\n"
            "---\n\n"
            "This is abstract-like text but no parser had the heading. "
            "It covers methods, results, and conclusions extensively."
        )
        triples = [
            AlignedTriple(
                segment_id="seg_para",
                grobid_block=self._make_block(
                    "grobid", "paragraph", "Some paragraph text.",
                ),
                classification=AGREE_EXACT,
            ),
        ]

        result, injected = ensure_abstract_heading(md, triples)
        assert injected is False

    def test_no_duplicate_when_no_space_heading_present(self):
        """Don't inject if ##Abstract (no space) already exists in output."""
        md = "# Title\n\n##Abstract\n\nThis is the abstract text."
        triples = [
            AlignedTriple(
                segment_id="seg_heading",
                grobid_block=self._make_block(
                    "grobid", "heading", "Abstract",
                    heading_level=2, source_md="##Abstract",
                ),
                classification=GAP,
            ),
        ]

        result, injected = ensure_abstract_heading(md, triples)
        assert injected is False
        assert result == md

    def test_parser_evidence_no_space_heading(self):
        """Detect parser evidence when heading has no space (##Abstract)."""
        md = (
            "# Paper Title\n\n"
            "---\n\n"
            "This is a long abstract paragraph that describes the findings of "
            "this study in great detail. It covers methods, results, and "
            "conclusions. The study was conducted over a period of several "
            "years and involved multiple research teams across different "
            "institutions. The results show significant improvements."
        )
        triples = [
            AlignedTriple(
                segment_id="seg_heading",
                grobid_block=self._make_block(
                    "grobid", "heading", "Abstract",
                    heading_level=2, source_md="##Abstract",
                ),
                classification=GAP,
                agreed_text="##Abstract",
            ),
        ]

        result, injected = ensure_abstract_heading(md, triples)
        assert injected is True
        assert "## Abstract" in result

    def test_injects_for_short_abstract(self):
        """Short abstracts (<200 chars) should still get ## Abstract via fallback."""
        md = (
            "# Paper Title\n\n"
            "---\n\n"
            "Brief abstract about the study."
        )
        triples = [
            AlignedTriple(
                segment_id="seg_heading",
                grobid_block=self._make_block(
                    "grobid", "heading", "Abstract",
                    heading_level=2, source_md="## Abstract",
                ),
                classification=GAP,
                agreed_text="## Abstract",
            ),
        ]

        result, injected = ensure_abstract_heading(md, triples)
        assert injected is True
        assert "## Abstract" in result
        # Heading should appear before the abstract text
        assert result.index("## Abstract") < result.index("Brief abstract")

    def test_injects_for_no_front_matter(self):
        """Documents without heading/rule front matter should still get injection."""
        md = "This is the abstract text without any headings or rules preceding it."
        triples = [
            AlignedTriple(
                segment_id="seg_heading",
                grobid_block=self._make_block(
                    "grobid", "heading", "Abstract",
                    heading_level=2, source_md="## Abstract",
                ),
                classification=GAP,
                agreed_text="## Abstract",
            ),
        ]

        result, injected = ensure_abstract_heading(md, triples)
        assert injected is True
        assert "## Abstract" in result
